"""Step 5 - Phase 1a: supervised warm start of mT5-small (Hindi question -> English keyword query).

What changed and why (the old policy mapped "रूस की राजधानी" to "japan capital city tokyo"):
  * The 6 hand-written anchor pairs repeated 20x are gone - they were memorised.
  * Training data: all 1,190 XQuAD-hi questions (was the first 600, ~10 articles) PLUS ~4,000 SQuAD-train
    questions spread over all 442 SQuAD articles, translated to Hindi with NLLB-200 and filtered by
    cross-lingual similarity. SQuAD-train is disjoint from XQuAD (built from SQuAD-dev) and TruthfulQA.
  * A real held-out split (by article / paragraph) -> data/sft_val.jsonl, used for early stopping and
    for Evaluation.py. The old evaluation scored training items.
  * 3e-4 learning rate, early stopping on validation loss (was 1e-3 for 19 epochs).
  * Shared input embeddings are frozen: less overfitting and ~1.5 GB less GPU memory.

    python -m spacy download en_core_web_sm      # once
    python Phase_1_SFT_pipeline.py
    python Phase_1_SFT_pipeline.py --skip-data   # retrain only
"""
import argparse
import hashlib
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch

from crag_utils import (ENCODER_NAME, MAX_PROMPT_TOKENS, MAX_QUERY_TOKENS, POLICY_BASE, QUERY_PREFIX, SFT_DIR,
                        SFT_TRAIN, SFT_VAL, generate_queries, get_device, read_jsonl, set_seed, write_jsonl)

NLLB_NAME = "facebook/nllb-200-distilled-600M"


# ------------------------------------------------------------------ data
def clean_keywords(english_text, nlp, max_words=8):
    """Nouns, proper nouns, numbers and adjectives, stop words removed (reference query)."""
    words = []
    for t in nlp(english_text):
        if t.pos_ in ("NOUN", "PROPN", "NUM", "ADJ") and not t.is_stop:
            w = (t.text if t.pos_ == "PROPN" else t.lemma_).lower().strip()
            if len(w) > 1 and w.replace("-", "").isalnum() and w not in words:
                words.append(w)
    return " ".join(words[:max_words])


def translate_en_hi(texts, device, batch=32):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLLB_NAME, src_lang="eng_Latn")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_NAME, torch_dtype=dtype).to(device).eval()
    bos = tok.convert_tokens_to_ids("hin_Deva")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos, max_length=96, num_beams=3)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t.strip()
        if (s // batch) % 20 == 0:
            print(f"    translated {min(s + batch, len(order))}/{len(order)}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def prepare_sft_data(n_squad, val_frac, min_sim, seed, device):
    import spacy
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rng = random.Random(seed)
    nlp = spacy.load("en_core_web_sm")
    records = []

    # --- XQuAD: professional Hindi translations, aligned with English by id
    print("[*] Loading XQuAD hi/en ...")
    xq_en = {r["id"]: r for r in load_dataset("google/xquad", "xquad.en", split="validation")}
    for r in load_dataset("google/xquad", "xquad.hi", split="validation"):
        en = xq_en.get(r["id"])
        if not en:
            continue
        records.append({"hindi_prompt": r["question"].strip(), "question_en": en["question"].strip(),
                        "answer_en": en["answers"]["text"][0] if en["answers"]["text"] else "",
                        "group": "xq_" + hashlib.md5(en["context"].encode()).hexdigest()[:10], "source": "xquad"})
    print(f"    {len(records)} XQuAD pairs")

    # --- SQuAD-train: diverse topics, round-robin across articles
    if n_squad:
        print(f"[*] Sampling {n_squad} SQuAD-train questions across all articles ...")
        by_title = defaultdict(list)
        for r in load_dataset("rajpurkar/squad", split="train"):
            by_title[r["title"]].append(r)
        titles = sorted(by_title)
        for t in titles:
            rng.shuffle(by_title[t])
        per = math.ceil(n_squad / len(titles))
        squad = [r for t in titles for r in by_title[t][:per]][:n_squad]
        print(f"[*] Translating {len(squad)} questions to Hindi with NLLB-200 on {device} ...")
        hi = translate_en_hi([r["question"] for r in squad], device)
        enc = SentenceTransformer(ENCODER_NAME, device=device)
        e_en = enc.encode([r["question"] for r in squad], normalize_embeddings=True, batch_size=128)
        e_hi = enc.encode(hi, normalize_embeddings=True, batch_size=128)
        sims = (e_en * e_hi).sum(1)
        kept = 0
        for r, h, s in zip(squad, hi, sims):
            if s < min_sim:  # drop translations that changed the meaning
                continue
            records.append({"hindi_prompt": h, "question_en": r["question"].strip(),
                            "answer_en": r["answers"]["text"][0] if r["answers"]["text"] else "",
                            "group": "sq_" + r["title"], "source": "squad_nllb", "xling_sim": round(float(s), 3)})
            kept += 1
        print(f"    kept {kept}/{len(squad)} SQuAD pairs (xling_sim >= {min_sim})")

    # --- reference queries
    for r in records:
        r["target_query"] = clean_keywords(r["question_en"], nlp)
    records = [r for r in records if len(r["target_query"].split()) >= 2]

    # --- split by paragraph/article so validation topics are unseen
    groups = sorted({r["group"] for r in records})
    rng.shuffle(groups)
    val_groups = set(groups[:max(1, int(len(groups) * val_frac))])
    train = [r for r in records if r["group"] not in val_groups]
    val = [r for r in records if r["group"] in val_groups]
    rng.shuffle(train)
    write_jsonl(train, SFT_TRAIN)
    write_jsonl(val, SFT_VAL)
    print(f"[+] {len(train)} train / {len(val)} held-out val pairs -> {SFT_TRAIN}, {SFT_VAL}")
    for r in train[:3]:
        print(f"    {r['hindi_prompt']}  ->  {r['target_query']}")


# ------------------------------------------------------------------ training
def batches(pairs, bs, shuffle, rng):
    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [pairs[j] for j in idx[i:i + bs]]


def step_loss(model, tok, batch, device):
    enc = tok([QUERY_PREFIX + r["hindi_prompt"] for r in batch], return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_PROMPT_TOKENS, return_token_type_ids=False).to(device)
    lab = tok(text_target=[r["target_query"] for r in batch], return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_QUERY_TOKENS).input_ids.to(device)
    lab[lab == tok.pad_token_id] = -100
    return model(**enc, labels=lab).loss


@torch.no_grad()
def val_loss(model, tok, val, device, bs=32):
    model.eval()
    return float(np.mean([step_loss(model, tok, b, device).item() for b in batches(val, bs, False, None)]))


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b) if a | b else 0.0


def train(a, device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    train_set, val_set = read_jsonl(SFT_TRAIN), read_jsonl(SFT_VAL)
    print(f"[*] SFT on {len(train_set)} pairs, validating on {len(val_set)} held-out pairs, device={device}")
    tok = AutoTokenizer.from_pretrained(POLICY_BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(POLICY_BASE).to(device)
    if not a.train_embeddings:
        model.shared.weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train_set) / a.batch)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, 1 - s / steps))
    rng = random.Random(a.seed)
    best, bad, history = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for b in batches(train_set, a.batch, True, rng):
            loss = step_loss(model, tok, b, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        vl = val_loss(model, tok, val_set, device)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vl})
        print(f"[*] epoch {ep}: train loss {np.mean(losses):.4f} | held-out val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            model.save_pretrained(SFT_DIR)
            tok.save_pretrained(SFT_DIR)
            print(f"    saved best -> {SFT_DIR}")
        else:
            bad += 1
            if bad >= a.patience:
                print("[*] early stopping")
                break
    write_jsonl(history, os.path.join(SFT_DIR, "sft_history.jsonl"))


def test(device, n=10):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SFT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SFT_DIR).to(device)
    val = read_jsonl(SFT_VAL)
    preds = generate_queries(model, tok, [r["hindi_prompt"] for r in val], device)
    print(f"\n[+] Held-out keyword Jaccard: {np.mean([jaccard(p, r['target_query']) for p, r in zip(preds, val)]):.3f}")
    probes = ["रूस की राजधानी क्या है?", "दुनिया में कितने महाद्वीप हैं?", "माउंट एवरेस्ट की ऊंचाई कितनी है?",
              "च्युइंग गम निगलने से क्या होता है?"]
    print("--- unseen probes ---")
    for p, q in zip(probes, generate_queries(model, tok, probes, device)):
        print(f"  {p}  ->  {q}")
    print("--- held-out examples ---")
    for r, p in list(zip(val, preds))[:n]:
        print(f"  {r['hindi_prompt']}\n    ref: {r['target_query']}\n    gen: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--n-squad", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--train-embeddings", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    device = get_device()
    if not a.skip_data or not os.path.exists(SFT_TRAIN):
        prepare_sft_data(a.n_squad, a.val_frac, a.min_sim, a.seed, device)
    train(a, device)
    test(device)


if __name__ == "__main__":
    main()
