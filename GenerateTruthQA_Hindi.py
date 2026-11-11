"""Step 1 - Build TruthfulQA-Hindi (v1) from the original English TruthfulQA.

v0 (truthfulqa_hindi_raw.jsonl) was translated with Helsinki-NLP/opus-mt-en-hi and failed a manual
check: about 23 of 40 random items changed meaning (watermelon -> "water", cats -> monkeys,
"land on the moon" -> "think about the moon"). v1 re-translates every question and every reference
answer with IndicTrans2 (as specified in the proposal), falling back to NLLB-200 if IndicTrans2 cannot
be loaded, and adds an automatic screening score (xling_sim) for the human reviewers.

    python GenerateTruthQA_Hindi.py                 # IndicTrans2 (needs HF access + transformers<5 + IndicTransToolkit)
    python GenerateTruthQA_Hindi.py --mt nllb       # facebook/nllb-200-distilled-600M, no gating
"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from crag_utils import (ENCODER_NAME, TQA_CSV, TQA_CSV_URL, TQA_DIR, TQA_FILE, get_device, set_seed,
                        write_jsonl)

IT2_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
NLLB_NAME = "facebook/nllb-200-distilled-600M"
OPUS_NAME = "Helsinki-NLP/opus-mt-en-hi"


class IndicTrans2:
    name = IT2_NAME

    def __init__(self, device, beams):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor
        self.tok = AutoTokenizer.from_pretrained(IT2_NAME, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(IT2_NAME, trust_remote_code=True).to(device).eval()
        self.ip, self.device, self.beams = IndicProcessor(inference=True), device, beams

    @torch.no_grad()
    def __call__(self, sents):
        batch = self.ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="hin_Deva")
        enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        gen = self.model.generate(**enc, use_cache=True, max_length=256, num_beams=self.beams)
        dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(dec, lang="hin_Deva")


class NLLB:
    name = NLLB_NAME

    def __init__(self, device, beams):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(NLLB_NAME, src_lang="eng_Latn")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_NAME, torch_dtype=dtype).to(device).eval()
        self.bos = self.tok.convert_tokens_to_ids("hin_Deva")
        self.device, self.beams = device, beams

    @torch.no_grad()
    def __call__(self, sents):
        enc = self.tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        gen = self.model.generate(**enc, forced_bos_token_id=self.bos, max_length=256, num_beams=self.beams)
        return self.tok.batch_decode(gen, skip_special_tokens=True)


class Opus:
    name = OPUS_NAME

    def __init__(self, device, beams):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(OPUS_NAME)
        self.model = MarianMTModel.from_pretrained(OPUS_NAME).to(device).eval()
        self.device, self.beams = device, beams

    @torch.no_grad()
    def __call__(self, sents):
        enc = self.tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        gen = self.model.generate(**enc, num_beams=self.beams, max_length=256)
        return self.tok.batch_decode(gen, skip_special_tokens=True)


def load_translator(kind, device, beams):
    if kind == "indictrans2":
        try:
            return IndicTrans2(device, beams)
        except Exception as e:  # gated repo, transformers 5.x, toolkit missing ...
            print(f"[!] IndicTrans2 could not be loaded ({type(e).__name__}: {str(e)[:160]})")
            print("[!] Falling back to NLLB-200 (recorded in the mt_system field of every item).")
            kind = "nllb"
    return {"nllb": NLLB, "opus": Opus}[kind](device, beams)


def split_answers(s):
    return [a.strip() for a in str(s).split(";") if a.strip() and a.strip().lower() != "nan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", choices=["indictrans2", "nllb", "opus"], default="indictrans2")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="translate only the first N questions (testing)")
    a = ap.parse_args()
    set_seed(42)
    device = get_device()
    if device == "mps":
        device = "cpu"
    os.makedirs(TQA_DIR, exist_ok=True)

    if not os.path.exists(TQA_CSV):
        print("[*] Downloading the original TruthfulQA.csv ...")
        pd.read_csv(TQA_CSV_URL).to_csv(TQA_CSV, index=False)
    df = pd.read_csv(TQA_CSV).head(a.limit)

    rows = []
    for i, r in df.iterrows():
        rows.append({"id": f"tqa_hi_{i + 1:04d}", "category": str(r["Category"]).strip(), "type": str(r["Type"]),
                     "source_question_en": str(r["Question"]).strip(), "best_answer_en": str(r["Best Answer"]).strip(),
                     "correct_answers_en": split_answers(r["Correct Answers"]),
                     "incorrect_answers_en": split_answers(r["Incorrect Answers"])})

    strings = []
    for r in rows:
        strings += [r["source_question_en"], r["best_answer_en"]] + r["correct_answers_en"] + r["incorrect_answers_en"]
    uniq = list(dict.fromkeys(strings))

    mt = load_translator(a.mt, device, a.beams)
    print(f"[*] Translating {len(uniq)} unique strings with {mt.name} on {device} (beams={a.beams})")
    uniq_sorted = sorted(uniq, key=len)  # similar lengths per batch = much faster
    hi = {}
    for i in tqdm(range(0, len(uniq_sorted), a.batch), desc="Translating"):
        batch = uniq_sorted[i:i + a.batch]
        for src, tgt in zip(batch, mt(batch)):
            hi[src] = tgt.strip()

    print("[*] Screening translations (cross-lingual similarity) ...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device=device)
    e_en = enc.encode([r["source_question_en"] for r in rows], normalize_embeddings=True, batch_size=64)
    e_hi = enc.encode([hi[r["source_question_en"]] for r in rows], normalize_embeddings=True, batch_size=64)

    for r, en_v, hi_v in zip(rows, e_en, e_hi):
        r["question_hi"] = hi[r["source_question_en"]]
        r["best_answer_hi"] = hi[r["best_answer_en"]]
        r["correct_answers_hi"] = [hi[x] for x in r["correct_answers_en"]]
        r["incorrect_answers_hi"] = [hi[x] for x in r["incorrect_answers_en"]]
        r["xling_sim"] = round(float(en_v @ hi_v), 4)
        r["mt_system"] = mt.name
        r["verification_status"] = "unverified"
        r["cultural_adaptation_flag"] = False
        r["notes"] = ""

    write_jsonl(rows, TQA_FILE)
    sims = sorted(r["xling_sim"] for r in rows)
    print(f"[+] Wrote {len(rows)} items to {TQA_FILE}")
    print(f"[+] Cross-lingual similarity: median {sims[len(sims) // 2]:.3f}; below 0.6: {sum(s < 0.6 for s in sims)} items")
    for r in rows[:3]:
        print(f"    {r['source_question_en']}  ->  {r['question_hi']}")


if __name__ == "__main__":
    main()
