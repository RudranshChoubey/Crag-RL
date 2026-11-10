"""Step 9 - Quality control for TruthfulQA-Hindi and the final verified dataset.

Reads the annotated review sheet (.csv or .xlsx), checks BOTH the Hindi question and the Hindi best
answer, and reports:
  * coverage (how many items each annotator labelled) and Cohen's kappa on the doubly labelled items
  * error rate with a 95% Wilson interval, overall and split into question / answer errors
  * failures per category, cultural flags
  * sanity warnings (e.g. a failed item without a fix, an answer typed into the question column)
  * whether the protocol requires extending the review to the full dataset

It then writes the verified benchmark, data/truthfulqa_hindi/truthfulqa_hindi_verified.jsonl, with the
human-corrected question and best answer applied (machine translations kept as *_mt fields).
Evaluation.py uses that file automatically.

    python Annotation_stats.py                                   # default CSV path
    python Annotation_stats.py --file VERIFICATION.xlsx          # also saved as the standard CSV
"""
import argparse
import json
import math
import os

import pandas as pd

from crag_utils import TQA_DIR, TQA_FILE, TQA_SAMPLE_CSV, read_jsonl, write_jsonl

TQA_VERIFIED = os.path.join(TQA_DIR, "truthfulqa_hindi_verified.jsonl")


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def text_col(df, col):
    if col not in df:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str).str.strip()


def load_sheet(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, encoding="utf-8-sig")
    for c in ("annotator_1_valid", "annotator_2_valid"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cultural_flag"] = pd.to_numeric(df.get("cultural_flag", 0), errors="coerce").fillna(0).astype(int)
    df["corrected_question_hi"] = text_col(df, "corrected_question_hi")
    df["corrected_best_answer_hi"] = text_col(df, "corrected_best_answer_hi")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=TQA_SAMPLE_CSV, help="annotated sheet (.csv or .xlsx)")
    ap.add_argument("--threshold", type=float, default=0.10, help="error rate that triggers a full review")
    a = ap.parse_args()

    df = load_sheet(a.file)
    if a.file.lower().endswith((".xlsx", ".xls")):
        df.to_csv(TQA_SAMPLE_CSV, index=False, encoding="utf-8-sig")
        print(f"[*] Saved the annotated sheet as {TQA_SAMPLE_CSV}")

    a1, a2 = df["annotator_1_valid"], df["annotator_2_valid"]
    if a1.isna().all() and a2.isna().all():
        raise SystemExit("No annotations yet: fill annotator_1_valid / annotator_2_valid (1 = valid, 0 = needs fix).")
    cq = df["corrected_question_hi"] != ""
    ca = df["corrected_best_answer_hi"] != ""
    labelled = a1.notna() | a2.notna()
    fail = (a1 == 0) | (a2 == 0)
    lab = df[labelled]
    k, n = int(fail[labelled].sum()), int(labelled.sum())

    out = {"sample_size": len(df), "categories": int(df["category"].nunique()),
           "labelled_by_annotator_1": int(a1.notna().sum()), "labelled_by_annotator_2": int(a2.notna().sum()),
           "labelled_by_anyone": n, "unlabelled": int((~labelled).sum())}
    both = df[a1.notna() & a2.notna()]
    if len(both):
        x, y = both["annotator_1_valid"].astype(int), both["annotator_2_valid"].astype(int)
        out["doubly_labelled"] = len(both)
        out["raw_agreement"] = round(float((x == y).mean()), 3)
        out["cohens_kappa"] = round(cohen_kappa(x, y), 3)

    def rate(mask):
        kk = int((mask & labelled).sum())
        lo, hi = wilson(kk, n)
        return {"items": kk, "rate": round(kk / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}

    out["items_failed"] = rate(fail)
    out["question_errors"] = rate(cq)
    out["answer_errors"] = rate(ca)
    out["question_and_answer_errors"] = rate(cq & ca)
    out["question_only_errors"] = rate(cq & ~ca)
    out["answer_only_errors"] = rate(ca & ~cq)
    out["cultural_flags"] = int(df["cultural_flag"].sum())
    out["failures_by_category"] = lab[fail[labelled]].groupby("category").size().sort_values(ascending=False).to_dict()
    out["full_review_required"] = bool(n and k / n > a.threshold)

    # ---------------------------------------------------------------- sanity checks
    warnings = []
    for _, r in df[fail & ~cq & ~ca].iterrows():
        warnings.append(f"{r['id']}: marked invalid but has no corrected question or answer")
    for _, r in df[cq & ~df["corrected_question_hi"].str.contains(r"[?？]")].iterrows():
        warnings.append(f"{r['id']}: corrected question has no '?': is it an answer typed into the question column? "
                        f"-> {r['corrected_question_hi'][:60]}")
    for _, r in df[(cq | ca) & ~fail & labelled].iterrows():
        warnings.append(f"{r['id']}: has a correction but both annotators marked it valid")
    for _, r in df[(cq | ca) & ~labelled].iterrows():
        warnings.append(f"{r['id']}: has a correction but no validity label")
    out["warnings"] = warnings

    print(json.dumps({k_: v for k_, v in out.items() if k_ != "warnings"}, indent=2, ensure_ascii=False))
    if warnings:
        print(f"\n[!] {len(warnings)} things to check in the sheet:")
        for w in warnings:
            print("   - " + w)
    if out["unlabelled"]:
        print(f"\n[!] {out['unlabelled']} items have no label from either annotator.")
    if out["full_review_required"]:
        print(f"\n[!] Error rate {k / n:.1%} is above {a.threshold:.0%}: the protocol extends review to the full dataset.")
    stats_path = os.path.join(TQA_DIR, "truthfulqa_verification_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- verified dataset
    items = read_jsonl(TQA_FILE)
    sheet = df.set_index("id")
    counts = {}
    for it in items:
        it["question_hi_mt"], it["best_answer_hi_mt"] = it["question_hi"], it["best_answer_hi"]
        status = "unverified"
        if it["id"] in sheet.index:
            r = sheet.loc[it["id"]]
            fq, fa = r["corrected_question_hi"], r["corrected_best_answer_hi"]
            if fq:
                it["question_hi"] = fq
            if fa:
                it["best_answer_hi"] = fa
            if fq and fa:
                status = "corrected_question_and_answer"
            elif fq:
                status = "corrected_question"
            elif fa:
                status = "corrected_answer"
            elif pd.notna(r["annotator_1_valid"]) or pd.notna(r["annotator_2_valid"]):
                status = "verified"
            if int(r["cultural_flag"]) == 1:
                it["cultural_adaptation_flag"] = True
        it["verification_status"] = status
        counts[status] = counts.get(status, 0) + 1
    write_jsonl(items, TQA_VERIFIED)
    print(f"\n[+] Stats -> {stats_path}")
    print(f"[+] Verified benchmark -> {TQA_VERIFIED} ({len(items)} items: {counts})")


if __name__ == "__main__":
    main()
