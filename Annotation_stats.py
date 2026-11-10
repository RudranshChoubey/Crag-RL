"""Step 3 - Quality control for TruthfulQA-Hindi after both annotators have filled the sheet.

Reports Cohen's kappa, the correction rate with a 95% Wilson interval (estimate of residual error
in the unreviewed 80%), failures per category, and whether the protocol requires a full review.
The corrections and cultural flags are applied automatically by Evaluation.py.
"""
import argparse
import json
import math

import pandas as pd

from crag_utils import TQA_SAMPLE_CSV


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=TQA_SAMPLE_CSV)
    ap.add_argument("--threshold", type=float, default=0.10, help="error rate that triggers a full review")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, encoding="utf-8-sig")
    a1, a2 = df["annotator_1_valid"], df["annotator_2_valid"]
    if a1.isna().all() and a2.isna().all():
        raise SystemExit("No annotations yet: fill annotator_1_valid / annotator_2_valid (1 = valid, 0 = needs fix).")

    out = {"sample_size": len(df), "categories": int(df["category"].nunique())}
    both = df[a1.notna() & a2.notna()]
    if len(both):
        x, y = both["annotator_1_valid"].astype(int), both["annotator_2_valid"].astype(int)
        out["doubly_annotated"] = len(both)
        out["raw_agreement"] = round(float((x == y).mean()), 3)
        out["cohens_kappa"] = round(cohen_kappa(x, y), 3)
    labelled = df[a1.notna() | a2.notna()]
    fail = (labelled["annotator_1_valid"] == 0) | (labelled["annotator_2_valid"] == 0)
    k, n = int(fail.sum()), len(labelled)
    lo, hi = wilson(k, n)
    out.update({"failed_items": k, "labelled_items": n, "error_rate": round(k / n, 3),
                "error_rate_ci95": [round(lo, 3), round(hi, 3)],
                "corrections_written": int(df["corrected_question_hi"].fillna("").astype(str).str.strip().ne("").sum()),
                "cultural_flags": int(pd.to_numeric(df["cultural_flag"], errors="coerce").fillna(0).sum()),
                "failures_by_category": labelled[fail].groupby("category").size().sort_values(ascending=False).to_dict(),
                "full_review_required": bool(k / n > a.threshold)})
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(a.csv.replace(".csv", "_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if k > out["corrections_written"]:
        print(f"[!] {k - out['corrections_written']} failed items still have no corrected_question_hi.")


if __name__ == "__main__":
    main()
