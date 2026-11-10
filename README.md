# CRAG-RL: Cross-Lingual RAG with RL-Optimised Query Generation

Rudransh Choubey · Saisish Patnaik · Pranav Srivastava — Dept. of AIML, CSE, JAIN (Deemed-to-be University)

A Hindi question is turned into a short English search query by an mT5-small policy trained with
supervised warm start + PPO. The query retrieves English Wikipedia passages from ChromaDB, and a frozen
Mistral-7B-Instruct answers in Hindi using only that evidence. Only the query policy is trained.

```
Hindi question ──► mT5 query policy (SFT → PPO) ──► English query ──► MiniLM + ChromaDB (Wikipedia)
                                                                          │ top-k passages
                     relevance gate (abstain if no passage is about the question)
                                                                          ▼
                                                      Mistral-7B-Instruct (4-bit) ──► Hindi answer
```

## Files

| File | Step | What it does | Main output |
|---|---|---|---|
| `crag_utils.py` | — | shared paths, model names, constants, helpers | — |
| `GenerateTruthQA_Hindi.py` | 1 | builds **TruthfulQA-Hindi v1** (IndicTrans2, falls back to NLLB-200) + screening score | `data/truthfulqa_hindi/truthfulqa_hindi.jsonl` |
| `Sample_data_verification.py` | 2 | 20% category-stratified review sheet, likely-broken items first | `data/truthfulqa_hindi/truthfulqa_verification_sample.csv` |
| `Annotation_stats.py` | 3 | Cohen's κ, correction rate + 95% CI, full-review rule | `..._sample_stats.json` |
| `Knowledege_build.py` | 4 | English Wikipedia knowledge base in ChromaDB | `data/chroma_db/`, `data/kb_cache/` |
| `phase2_retriever.py` | — | Phase 2 retriever (run it alone to test the KB) | — |
| `Phase_1_SFT_pipeline.py` | 5 | builds SFT data (XQuAD + SQuAD-train, held-out split) and warm-starts mT5 | `data/sft_*.jsonl`, `models/mt5_sft_warmstart/` |
| `Phase_4.py` | 6 | PPO alignment with the prompt-conditioned reward | `models/mt5_ppo_aligned/` (+ `train_log.jsonl`, `ppo_curves.png`) |
| `phase3_synthesis.py` | — | Phase 3 Mistral reader (run alone to test it) | — |
| `run_full_system.py` | 7 | full system, demo + interactive Hindi questions | — |
| `Evaluation.py` | 8 | Part A: held-out query quality · Part B: TruthfulQA-Hindi vs baselines | `results/` |
| `docs/` | — | dataset documentation, annotation guidelines, design notes | — |

## Setup (~15 min)

Windows + NVIDIA GPU (6 GB+). Use **Windows Terminal or the VS Code terminal** — the old PowerShell
console draws Hindi incorrectly (that is why the earlier screenshots looked garbled).

```bash
python -m venv RLEnv
RLEnv\Scripts\activate                     # macOS/Linux: source RLEnv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
huggingface-cli login                      # Mistral and IndicTrans2 are gated: accept their terms on huggingface.co first
```

Optional, for IndicTrans2 (proposal's translator). It needs `transformers<5`, so use a separate venv:
```bash
python -m venv IT2Env && IT2Env\Scripts\activate
pip install "transformers==4.46.3" IndicTransToolkit sentencepiece sentence-transformers pandas tqdm torch
python GenerateTruthQA_Hindi.py            # then switch back to RLEnv for everything else
```
If you skip this, step 1 falls back to NLLB-200 automatically and says so.

## Run order

| # | Command | Time (laptop GPU) | Check that it worked |
|---|---|---|---|
| 1 | `python GenerateTruthQA_Hindi.py` | 10–20 min | 790 items written; first line mentions तरबूज (watermelon); median similarity printed (≈0.75+ is healthy) |
| 2 | `python Sample_data_verification.py` | seconds | 160 items across 37 categories |
| — | **annotate** the CSV (two people, independently) — `docs/ANNOTATION_GUIDELINES.md` | ~1.5 h each | runs in parallel with steps 4–7 |
| 4 | `python Knowledege_build.py --targeted` | 45–75 min | ends with "Collection 'wiki_passages_v2' now holds N passages" (N in the hundreds of thousands) |
| 4b | `python phase2_retriever.py` | 1 min | "capital of Russia" returns a Russia/Moscow passage |
| 5 | `python Phase_1_SFT_pipeline.py` | 20–30 min | held-out val loss falls; the **unseen probes** print sensible queries (रूस की राजधानी → something with "russia capital", not Japan) |
| 6 | `python Phase_4.py` | 15–30 min | held-out reward rises above the step-0 value; `kl` stays small (< ~0.5); queries stay readable English |
| 7 | `python run_full_system.py` | 3 min | Russia question → a Russia query and Moscow in the answer; no English text in brackets |
| 8a | `python Evaluation.py --part query` | ~5 min | `results/query_eval.md`; crag_rl Hit@3 vs sft_query vs translate_retrieve |
| 3 | `python Annotation_stats.py` (after annotation) | seconds | κ and error rate printed |
| 8b | `python Evaluation.py --part e2e --n 100` | 60–90 min | `results/e2e_n100/results.md` with all tables |

Steps 4 and 5 can run at the same time if your GPU has 8 GB or more. Run 8b **after** annotation:
the human corrections and cultural flags are applied automatically. Interrupted? Rerun the same
command: the KB fetch and the evaluation both resume.

If PPO queries turn into repeated words or gibberish while reward rises, the policy is gaming the reward:
rerun with `python Phase_4.py --kl-coef 0.2`.

### Reward ablations (proposal §3.5, item 7) — optional, ~20 min each

```bash
python Phase_4.py --beta 0  --out ./models/ppo_no_cos
python Phase_4.py --gamma 0 --out ./models/ppo_no_len
python Phase_4.py --alpha 0 --out ./models/ppo_no_evidence
python Evaluation.py --part e2e --n 100 --systems no_retrieval \
   --policy crag_rl_no_cos=./models/ppo_no_cos --policy crag_rl_no_len=./models/ppo_no_len \
   --policy crag_rl_no_evidence=./models/ppo_no_evidence
```

## What to commit to GitHub

Commit: all `.py` files, `README.md`, `requirements.txt`, `docs/`, `.gitignore`, and from `data/`:
`truthfulqa_hindi/*` (the benchmark, v0 file, filled review sheet, stats), `sft_train.jsonl`, `sft_val.jsonl`,
`kb_cache/` (frozen Wikipedia snapshot); from `models/`: `*/train_log.jsonl`, `*/ppo_curves.png`,
`*/sft_history.jsonl`, `*/ppo_config.json`; and the whole `results/` folder.

Not committed (`.gitignore`): `data/chroma_db/` (rebuildable) and model weights (`*.safetensors`, over
GitHub's 100 MB limit). Share the trained policy separately (Google Drive / Hugging Face) if asked.

## Systems compared (Evaluation.py Part B)

| System | How the English query is made |
|---|---|
| `no_retrieval` | none — Mistral answers from its own knowledge |
| `translate_retrieve` | machine translation of the question (opus-mt-hi-en) |
| `llm_rewrite` | Mistral prompted to write keywords |
| `sft_query` | mT5 after supervised warm start only |
| `crag_rl` | mT5 after PPO (proposed) |

The Hindi-RAG baseline (Hindi Wikipedia index) from the proposal is not implemented here; say so in the report.
