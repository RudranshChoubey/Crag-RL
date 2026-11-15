"""Step 2 - Draw the 20% category-stratified sample for human verification.

Rows are sorted with the lowest cross-lingual similarity first, so reviewers see the most
likely broken translations while they are fresh. Will not overwrite a sheet that already has
annotations unless you pass --force.
"""
import argparse
import os

import pandas as pd

from crag_utils import TQA_FILE, TQA_SAMPLE_CSV, read_jsonl

SAMPLE_FRACTION = 0.20
RANDOM_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if os.path.exists(TQA_SAMPLE_CSV) and not a.force:
        old = pd.read_csv(TQA_SAMPLE_CSV, encoding="utf-8-sig")
        if old[["annotator_1_valid", "annotator_2_valid"]].notna().any().any():
            raise SystemExit(f"[!] {TQA_SAMPLE_CSV} already has annotations. Use --force to overwrite it.")

    df = pd.DataFrame(read_jsonl(TQA_FILE))
    parts = []
    for _, g in df.groupby("category"):
        k = max(1, round(len(g) * SAMPLE_FRACTION))  # at least one item per category
        parts.append(g.sample(n=min(k, len(g)), random_state=RANDOM_SEED))
    sample = pd.concat(parts)
    if "xling_sim" in sample:
        sample = sample.sort_values("xling_sim")

    sample = sample.assign(annotator_1_valid="", annotator_2_valid="", corrected_question_hi="",
                           cultural_flag=0, notes="")
    cols = ["id", "category", "xling_sim", "source_question_en", "question_hi", "best_answer_en",
            "best_answer_hi", "annotator_1_valid", "annotator_2_valid", "corrected_question_hi",
            "cultural_flag", "notes"]
    sample[[c for c in cols if c in sample]].to_csv(TQA_SAMPLE_CSV, index=False, encoding="utf-8-sig")
    print(f"[+] Saved {len(sample)} items across {sample.category.nunique()} categories to {TQA_SAMPLE_CSV}")
    print("    Annotate with two independent annotators (criteria in README.md), then run: python Annotation_stats.py")


if __name__ == "__main__":
    main()
