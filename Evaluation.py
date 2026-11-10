"""Step 8 - Evaluation (proposal Sec. 3.5 + 4.2). Two parts, both on data the model never trained on.

Part A  --part query   (fast, no Mistral; ~5 min)
    Held-out XQuAD/SQuAD questions (data/sft_val.jsonl). For the SFT policy, the PPO policy and the
    Translate-then-Retrieve baseline: keyword Jaccard vs the reference query, and Hit@3 = the gold
    English answer string appears in the top-3 retrieved passages.

Part B  --part e2e     (Mistral; ~1 h for 100 items)
    TruthfulQA-Hindi, end to end, systems:
        no_retrieval        Mistral answers from its own knowledge
        translate_retrieve  MT (opus-mt-hi-en) of the prompt used as the query
        llm_rewrite         Mistral writes the English query
        sft_query           mT5 after supervised warm start only
        crag_rl             mT5 after PPO (proposed)
    Metrics: hallucination rate + truthfulness (TruthfulQA similarity protocol against the English
    reference answers, multilingual encoder), abstention, FActScore-lite (NLI claim precision against the
    same gold-question evidence for every system), AlignScore-proxy (NLI faithfulness to the system's own
    passages), Self-Contrast delta-SC with exact McNemar test, per-stage latency (mean, p95), bootstrap CIs.

    python Evaluation.py --part query
    python Evaluation.py --part e2e --n 100
    python Evaluation.py --part e2e --n 0          # whole benchmark (several hours)
    python Evaluation.py --part e2e --verified-only  # only the human-verified items
    python Evaluation.py --part e2e --systems no_retrieval --policy crag_rl_no_cos=./models/ppo_no_cos   # ablation
Results are written to ./results/ and resume automatically if interrupted.
"""
import argparse
import json
import math
import os
import random
import re
from collections import defaultdict

import numpy as np

from crag_utils import (ABSTAIN_HI, MT_HI_EN, NLI_MODEL, PPO_DIR, RESULTS_DIR, SFT_DIR, SFT_VAL, TQA_FILE,
                        append_jsonl, generate_queries, get_device, load_tqa_english_refs,
                        read_jsonl, set_seed)

SYSTEMS = ["no_retrieval", "translate_retrieve", "llm_rewrite", "sft_query", "crag_rl"]


# ------------------------------------------------------------------ helpers
class HiEnTranslator:
    def __init__(self, device="cpu"):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(MT_HI_EN)
        self.model = MarianMTModel.from_pretrained(MT_HI_EN).to(device).eval()
        self.device = device

    def __call__(self, texts, bs=32):
        import torch
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, num_beams=4, max_length=128)
            out += [t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True)]
        return out


class NLI:
    def __init__(self, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent, self.device = labels.get("entailment", 0), device

    def entail(self, premises, hypotheses):
        import torch
        pairs = [(p, h) for h in hypotheses for p in premises]
        probs = []
        for i in range(0, len(pairs), 16):
            b = pairs[i:i + 16]
            enc = self.tok([p for p, _ in b], [h for _, h in b], return_tensors="pt", truncation="only_first",
                           max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                probs.append(self.model(**enc).logits.softmax(-1)[:, self.ent].cpu())
        return torch.cat(probs).view(len(hypotheses), len(premises)).numpy()


def sentences(text):
    return [s for s in re.split(r"(?<=[।\.\?\!])\s+", text.strip()) if len(s.strip()) > 2]


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b) if a | b else 0.0


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def bootstrap_ci(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None], float)
    if not len(xs):
        return [None, None]
    m = np.random.default_rng(seed).choice(xs, (n, len(xs))).mean(1)
    return [round(float(np.quantile(m, 0.025)), 3), round(float(np.quantile(m, 0.975)), 3)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


# ------------------------------------------------------------------ Part A
def part_query(n):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from phase2_retriever import KnowledgeRetriever

    device = get_device()
    val = read_jsonl(SFT_VAL)[: n or None]
    prompts = [r["hindi_prompt"] for r in val]
    retr = KnowledgeRetriever(device=device)
    queries = {"translate_retrieve": HiEnTranslator(device if device == "cuda" else "cpu")(prompts)}
    for name, path in (("sft_query", SFT_DIR), ("crag_rl", PPO_DIR)):
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        queries[name] = generate_queries(model, tok, prompts, device)
        del model
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, summary = [], {}
    for name, qs in queries.items():
        hits = retr.search(qs, 3)
        hit = [bool(r["answer_en"]) and any(r["answer_en"].lower() in h["text"].lower() for h in hs)
               for r, hs in zip(val, hits)]
        summary[name] = {"n": len(val), "jaccard": mean([jaccard(q, r["target_query"]) for q, r in zip(qs, val)]),
                         "hit@3": mean(hit), "hit@3_ci95": bootstrap_ci(hit),
                         "query_words": mean([len(q.split()) for q in qs])}
        for r, q, h in zip(val, qs, hit):
            rows.append({"system": name, "hindi_prompt": r["hindi_prompt"], "ref": r["target_query"], "query": q, "hit@3": h})
    json.dump(summary, open(f"{RESULTS_DIR}/query_eval_summary.json", "w"), indent=2)
    with open(f"{RESULTS_DIR}/query_eval_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# Query generation on held-out questions (n={len(val)})", "",
             "| System | Hit@3 ↑ (95% CI) | Keyword Jaccard ↑ | Query words |", "|---|---|---|---|"]
    for k, m in summary.items():
        lines.append(f"| {k} | {fmt(m['hit@3'])} ({m['hit@3_ci95'][0]}–{m['hit@3_ci95'][1]}) | {fmt(m['jaccard'])} | {fmt(m['query_words'], 1)} |")
    open(f"{RESULTS_DIR}/query_eval.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nExamples:")
    for i in range(min(6, len(val))):
        print(f"  {val[i]['hindi_prompt']}")
        for k in queries:
            print(f"    {k:<19} {queries[k][i]}")


# ------------------------------------------------------------------ Part B
def load_benchmark(n, seed, verified_only=False):
    """Verified benchmark (Annotation_stats.py output) if present, else the machine-translated v1."""
    from Annotation_stats import TQA_VERIFIED

    if os.path.exists(TQA_VERIFIED):
        items = {r["id"]: r for r in read_jsonl(TQA_VERIFIED)}
        st = defaultdict(int)
        for r in items.values():
            st[r.get("verification_status", "unverified")] += 1
        print(f"[*] Using the human-verified benchmark {TQA_VERIFIED}: {dict(st)}")
    else:
        items = {r["id"]: r for r in read_jsonl(TQA_FILE)}
        print(f"[!] {TQA_VERIFIED} not found (run Annotation_stats.py); using unverified {TQA_FILE}")
    rows = list(items.values())
    if verified_only:
        rows = [r for r in rows if r.get("verification_status", "unverified") != "unverified"]
        print(f"[*] --verified-only: evaluating on the {len(rows)} human-verified items")
    if n and n < len(rows):  # category-stratified, seeded sample
        by = defaultdict(list)
        for r in rows:
            by[r["category"]].append(r)
        rng = random.Random(seed)
        picked = []
        for cat in sorted(by):
            g = by[cat][:]
            rng.shuffle(g)
            picked += g[:max(1, round(n * len(g) / len(rows)))]
        rng.shuffle(picked)
        rows = sorted(picked[:n], key=lambda r: r["id"])
    return rows


def part_e2e(n, seed, systems, top_k, extra_policies=None, verified_only=False):
    import torch
    from sentence_transformers import SentenceTransformer  # noqa: F401 (encoder comes from the retriever)
    from run_full_system import CrossLingualRAGPipeline
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    items = load_benchmark(n, seed, verified_only)
    refs = load_tqa_english_refs()
    out_dir = f"{RESULTS_DIR}/e2e_{'verified_' if verified_only else ''}n{len(items)}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] TruthfulQA-Hindi: {len(items)} items, systems: {', '.join(systems)} -> {out_dir}")

    pipe = CrossLingualRAGPipeline(PPO_DIR, top_k)       # loads retriever + Mistral once
    retr = pipe.retriever
    policies = {"crag_rl": (pipe.ptok, pipe.policy)}
    if "sft_query" in systems:
        policies["sft_query"] = (AutoTokenizer.from_pretrained(SFT_DIR),
                                 AutoModelForSeq2SeqLM.from_pretrained(SFT_DIR).to("cpu").eval())
    for name, path in (extra_policies or {}).items():
        policies[name] = (AutoTokenizer.from_pretrained(path), AutoModelForSeq2SeqLM.from_pretrained(path).to("cpu").eval())
    mt = HiEnTranslator("cpu") if "translate_retrieve" in systems else None
    nli = NLI("cpu")

    # fixed FActScore evidence: top-5 passages for the GOLD English question (same for every system)
    ev_path = f"{out_dir}/factscore_evidence.json"
    evidence = json.load(open(ev_path, encoding="utf-8")) if os.path.exists(ev_path) else {}
    miss = [it for it in items if it["id"] not in evidence]
    if miss:
        for it, hs in zip(miss, retr.search([it["source_question_en"] for it in miss], 5)):
            evidence[it["id"]] = [h["text"] for h in hs]
        json.dump(evidence, open(ev_path, "w", encoding="utf-8"), ensure_ascii=False)

    # truthfulness judge: original English references, plus the human-checked Hindi best answer
    def truth_score(answer, it):
        r = refs[it["id"]]
        cor = [r["best"]] + r["correct"]
        if it.get("verification_status", "unverified") != "unverified" and it.get("best_answer_hi"):
            cor.append(it["best_answer_hi"])
        inc = r["incorrect"] or ["-"]
        E = retr.encode([answer] + cor + inc)
        a = E[0]
        return float((E[1:1 + len(cor)] @ a).max() - (E[1 + len(cor):] @ a).max())

    import time
    for system in systems:
        path = f"{out_dir}/pred_{system}.jsonl"
        done = {r["id"] for r in read_jsonl(path)} if os.path.exists(path) else set()
        todo = [it for it in items if it["id"] not in done]
        print(f"\n[*] {system}: {len(todo)} items to run ({len(done)} cached)")
        for k, it in enumerate(todo, 1):
            q_hi = it["question_hi"]
            t0 = time.perf_counter()
            if system == "no_retrieval":
                res = {"query": None, "passages": [], "evidence_sim": None, "gated": False, "t_retrieval": 0.0}
                t1 = time.perf_counter()
                res["answer"] = pipe.gen.generate_answer(q_hi, None)
                res["t_synthesis"] = time.perf_counter() - t1
                t_query = 0.0
            else:
                if system == "translate_retrieve":
                    query = mt([q_hi])[0]
                elif system == "llm_rewrite":
                    query = pipe.gen.rewrite_query(q_hi)
                else:
                    tok, model = policies[system]
                    query = generate_queries(model, tok, [q_hi], "cpu")[0]
                t_query = time.perf_counter() - t0
                res = pipe.answer_from_query(q_hi, query)
            ans = res["answer"]
            abstain = ans.strip() == ABSTAIN_HI
            ts = truth_score(ans, it)
            claims = [] if abstain else sentences(ans)
            fs = align = None
            if claims:
                fs = float((nli.entail(evidence[it["id"]], claims).max(1) > 0.5).mean())
                if res["passages"]:
                    align = float(nli.entail([p["text"] for p in res["passages"]], claims).max(1).mean())
            rec = {"id": it["id"], "category": it["category"], "system": system, "question_hi": q_hi,
                   "question_en": it["source_question_en"], "query": res["query"],
                   "passages": [{"title": p.get("title", ""), "score": round(p["score"], 3), "text": p["text"][:400]}
                                for p in res["passages"]],
                   "evidence_sim": res["evidence_sim"], "gated": res["gated"], "answer": ans, "abstain": abstain,
                   "truth_score": round(ts, 4), "hallucinated": int((not abstain) and ts <= 0),
                   "factscore": fs, "align": align, "cultural": bool(it.get("cultural_adaptation_flag")),
                   "latency": {"query": t_query, "retrieval": res["t_retrieval"], "synthesis": res["t_synthesis"],
                               "total": t_query + res["t_retrieval"] + res["t_synthesis"]}}
            append_jsonl(rec, path)
            if k % 10 == 0 or k == len(todo):
                print(f"    {k}/{len(todo)}  last answer: {ans[:70]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summarise(out_dir, systems)


def summarise(out_dir, systems):
    preds = {s: {r["id"]: r for r in read_jsonl(f"{out_dir}/pred_{s}.jsonl")} for s in systems
             if os.path.exists(f"{out_dir}/pred_{s}.jsonl")}
    base = preds.get("no_retrieval", {})
    summary = {}
    for s, rows in preds.items():
        rs = list(rows.values())
        h = [r["hallucinated"] for r in rs]
        tot = [r["latency"]["total"] for r in rs]
        m = {"n": len(rs), "hallucination_rate": mean(h), "hallucination_ci95": bootstrap_ci(h),
             "truthful_rate": mean([r["truth_score"] > 0 for r in rs]), "abstain_rate": mean([r["abstain"] for r in rs]),
             "informative_truthful_rate": mean([(not r["abstain"]) and r["truth_score"] > 0 for r in rs]),
             "factscore_lite": mean([r["factscore"] for r in rs]), "factscore_ci95": bootstrap_ci([r["factscore"] for r in rs]),
             "align_proxy": mean([r["align"] for r in rs]),
             "gated_rate": mean([r["gated"] for r in rs]) if s != "no_retrieval" else None,
             "query_words": mean([len(r["query"].split()) for r in rs if r["query"]]),
             "latency": {k: {"mean": mean([r["latency"][k] for r in rs]),
                             "p95": round(float(np.percentile([r["latency"][k] for r in rs], 95)), 3)}
                         for k in ("query", "retrieval", "synthesis", "total")},
             "hallucination_rate_without_cultural": mean([r["hallucinated"] for r in rs if not r["cultural"]])}
        if base and s != "no_retrieval":
            ids = [i for i in rows if i in base]
            hn = np.array([base[i]["hallucinated"] for i in ids])
            hs = np.array([rows[i]["hallucinated"] for i in ids])
            corr, intro = int(((hn == 1) & (hs == 0)).sum()), int(((hn == 0) & (hs == 1)).sum())
            m["self_contrast"] = {"delta_sc": round(float((hn - hs).mean()), 3), "corrected": corr,
                                  "newly_introduced": intro, "unchanged_hallucinated": int(((hn == 1) & (hs == 1)).sum()),
                                  "mcnemar_p": round(mcnemar_exact(corr, intro), 4)}
            bl = np.mean([base[i]["latency"]["total"] for i in ids])
            m["latency_overhead_s"] = round(float(np.mean([rows[i]["latency"]["total"] for i in ids]) - bl), 3)
        summary[s] = m
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2, ensure_ascii=False)

    n = max(m["n"] for m in summary.values())
    L = [f"# TruthfulQA-Hindi end-to-end results (n={n})", "",
         "| System | Halluc. rate ↓ (95% CI) | Truthful ↑ | Informative & truthful ↑ | Abstain | FActScore-lite ↑ | AlignScore-proxy ↑ | ΔSC ↑ (McNemar p) | Query words |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, m in summary.items():
        sc = m.get("self_contrast")
        ci = m["hallucination_ci95"]
        L.append(f"| {s} | {fmt(m['hallucination_rate'])} ({ci[0]}–{ci[1]}) | {fmt(m['truthful_rate'])} | "
                 f"{fmt(m['informative_truthful_rate'])} | {fmt(m['abstain_rate'])} | {fmt(m['factscore_lite'])} | "
                 f"{fmt(m['align_proxy'])} | " + (f"{sc['delta_sc']:+.3f} ({sc['mcnemar_p']})" if sc else "–")
                 + f" | {fmt(m['query_words'], 1)} |")
    L += ["", "## Self-Contrast breakdown vs no_retrieval", "", "| System | Corrected | Newly introduced | Unchanged hallucinated |",
          "|---|---|---|---|"]
    for s, m in summary.items():
        if "self_contrast" in m:
            sc = m["self_contrast"]
            L.append(f"| {s} | {sc['corrected']} | {sc['newly_introduced']} | {sc['unchanged_hallucinated']} |")
    L += ["", "## Latency (seconds per question)", "", "| System | Query | Retrieval | Synthesis | Total mean | Total p95 | Overhead vs no_retrieval |",
          "|---|---|---|---|---|---|---|"]
    for s, m in summary.items():
        l = m["latency"]
        L.append(f"| {s} | {fmt(l['query']['mean'], 2)} | {fmt(l['retrieval']['mean'], 2)} | {fmt(l['synthesis']['mean'], 2)} | "
                 f"{fmt(l['total']['mean'], 2)} | {fmt(l['total']['p95'], 2)} | {fmt(m.get('latency_overhead_s'), 2)} |")
    open(f"{out_dir}/results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\n[+] Wrote {out_dir}/results.md, summary.json and pred_<system>.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["query", "e2e", "summary"], default="query")
    ap.add_argument("--n", type=int, default=None, help="items (e2e default 100; 0 = all). Part A uses all held-out by default")
    ap.add_argument("--policy", action="append", default=[], help="extra policy for ablations: crag_rl_<name>=<path>")
    ap.add_argument("--systems", nargs="*", default=SYSTEMS)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--verified-only", action="store_true", help="evaluate only on human-verified items")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    extra = dict(p.split("=", 1) for p in a.policy)
    systems = a.systems + [k for k in extra if k not in a.systems]
    if a.part == "query":
        part_query(a.n or 0)
    elif a.part == "e2e":
        part_e2e((0 if a.verified_only else 100) if a.n is None else a.n, a.seed, systems, a.top_k, extra, a.verified_only)
    else:
        d = sorted(p for p in os.listdir(RESULTS_DIR) if p.startswith("e2e_"))[-1]
        summarise(f"{RESULTS_DIR}/{d}", systems)


if __name__ == "__main__":
    main()
