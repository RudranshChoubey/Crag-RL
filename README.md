# CRAG-RL: Cross-Lingual RAG with RL-Optimised Query Generation

**Mitigating fact-conflicting hallucination in low-resource languages (Hindi)**

Rudransh Choubey · Saisish Patnaik · Pranav Srivastava
Department of AIML, CSE, JAIN (Deemed-to-be University), Bengaluru

## Overview

Large language models hallucinate more when asked questions in low-resource languages, and
retrieval-augmented generation (RAG) is limited by how little encyclopedic text exists in those
languages. CRAG-RL retrieves evidence from English Wikipedia instead, while the user asks and
receives answers in Hindi.

A small query generator, trained with supervised fine-tuning (SFT) followed by reinforcement learning
(PPO), turns a Hindi question into a short English search query. It is optimised for retrieving
evidence about the question rather than for translation fidelity.

```
Hindi question ──► mT5-small query policy (SFT → PPO) ──► English keyword query
                                                              │
                                     MiniLM encoder + ChromaDB (English Wikipedia)
                                                              │ top-k passages
                      relevance gate (abstain if no passage is about the question)
                                                              │
                                     Mistral-7B-Instruct (frozen) ──► Hindi answer
```

Only the query policy is trained; the retriever and the answer generator stay frozen.

| Component | Choice |
|---|---|
| Query policy | `google/mt5-small` + value head; supervised warm start, then PPO |
| RL algorithm | PPO: clipped surrogate, GAE, entropy bonus, KL penalty to the SFT policy |
| Encoder | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | ChromaDB, cosine similarity (531,473 passages) |
| Answer generator | `mistralai/Mistral-7B-Instruct-v0.3`, 4-bit NF4, greedy decoding |
| Relevance gate | abstain when max cos(question, passage) < 0.35 |
| Benchmark | TruthfulQA-Hindi (790 items, introduced in this project) |
| Training questions | XQuAD-hi + SQuAD-train translated to Hindi (no overlap with the benchmark) |

### Reward

With `e = max_j cos(x, c_j)` (the similarity between the Hindi question `x` and the best retrieved
passage `c_j`, both embedded with the multilingual encoder):

```
R = e + p(e) − ℓ(q)
p(e) = −0.5          if e < 0.45
       0.5 · e       otherwise
ℓ(q) = 0.3           if the query q has fewer than 2 or more than 8 words, else 0
```

Empty or repetitive queries (fewer than 70% unique words) receive −1. An earlier reward that combined
question–evidence, query–evidence and question–query similarity with a length penalty was replaced
after the first training run (see [Development notes](#development-notes)).

## Results

On 100 TruthfulQA-Hindi questions (category-stratified sample):

| System | Hallucination rate ↓ (95% CI) | Informative & truthful ↑ | Abstain | FActScore-lite ↑ | ΔSC vs No Retrieval ↑ |
|---|---|---|---|---|---|
| No Retrieval | 0.49 (0.39–0.59) | **0.51** | 0.00 | 0.44 | – |
| Translate-then-Retrieve | 0.22 (0.14–0.31) | 0.28 | 0.50 | 0.69 | +0.27 |
| LLM Rewrite | 0.24 (0.16–0.33) | 0.21 | 0.55 | 0.63 | +0.25 |
| SFT Query | **0.08** (0.03–0.13) | 0.11 | 0.81 | 0.79 | **+0.41** |
| CRAG-RL (PPO) | **0.08** (0.03–0.14) | 0.13 | 0.79 | **0.86** | **+0.41** |

All ΔSC values are significant (exact McNemar test, p < 0.001).

- Every retrieval-augmented system reduces fact-conflicting answers compared with the same model without retrieval.
- CRAG-RL's low hallucination rate comes mostly from abstaining: it refuses 79% of questions, often
  because its query retrieved off-topic evidence and the relevance gate fired.
- Plain machine translation of the question retrieves relevant evidence more often (held-out Hit@3:
  0.126 versus 0.038 for CRAG-RL) and gives more informative truthful answers.
- PPO did not measurably improve on the supervised policy.

Full tables, including Self-Contrast breakdowns and per-stage latency, are in `results/`.

## Repository structure

| File | Purpose |
|---|---|
| `crag_utils.py` | shared paths, model names, constants and helpers |
| `GenerateTruthQA_Hindi.py` | builds TruthfulQA-Hindi (IndicTrans2, NLLB-200 fallback) with a translation-quality score |
| `Sample_data_verification.py` | draws the 20% category-stratified sample for human verification |
| `Annotation_stats.py` | inter-annotator agreement, error rates with 95% CIs, and the verified benchmark file |
| `Knowledege_build.py` | builds the English Wikipedia knowledge base in ChromaDB |
| `phase2_retriever.py` | Phase 2: dense retrieval (run alone to test the index) |
| `Phase_1_SFT_pipeline.py` | builds the training pairs and runs the supervised warm start |
| `Phase_4.py` | PPO training of the query policy |
| `phase3_synthesis.py` | Phase 3: Hindi answer generation with Mistral |
| `run_full_system.py` | end-to-end system with demo questions and an interactive prompt |
| `Evaluation.py` | held-out query evaluation (Part A) and end-to-end benchmark evaluation (Part B) |
| `dataset/` | released data (see below) |
| `results/` | evaluation outputs: tables, summaries and per-question predictions |

### Released data (`dataset/`)

| File | Contents |
|---|---|
| `TruthfulQA.csv` | original English TruthfulQA (Lin et al., 2022) |
| `truthfulqa_hindi.jsonl` | TruthfulQA-Hindi, machine-translated with NLLB-200, with English fields and a cross-lingual similarity score per item |
| `truthfulqa_hindi_verified.jsonl` | the same with human corrections applied (`verification_status` per item; machine translations kept as `*_mt`) |
| `truthfulqa_verification_sample.csv` | the 160-item annotated review sheet |
| `truthfulqa_verification_stats.json` | agreement and error-rate statistics |
| `sft_train.jsonl`, `sft_val.jsonl` | 3,620 / 396 Hindi question → English keyword query pairs, split by source article |
| `truthfulqa_verification_sample_just for testing.csv` | an earlier test copy of the review sheet; not used by the scripts |

The scripts read and write under `./data/`. To reuse the released files instead of regenerating them:

```bash
mkdir -p data/truthfulqa_hindi
cp dataset/TruthfulQA.csv dataset/truthfulqa_* data/truthfulqa_hindi/
cp dataset/sft_*.jsonl data/
```

## Requirements

- Python 3.10–3.12
- An NVIDIA GPU with at least 6 GB of memory (4-bit loading of Mistral requires CUDA)
- A Hugging Face account with access accepted for `mistralai/Mistral-7B-Instruct-v0.3`
  (and optionally `ai4bharat/indictrans2-en-indic-dist-200M`)

## Installation

```bash
python -m venv RLEnv
# Windows: RLEnv\Scripts\activate      Linux/macOS: source RLEnv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
huggingface-cli login
```

IndicTrans2 requires `transformers<5`. Without a separate environment for it, step 1 falls back to
NLLB-200 automatically and records the system used in every item (the released dataset used NLLB-200).

## Usage

Run the steps in order from the repository root. Steps 1–3 can be skipped by copying the released data
as shown above.

| # | Command | Approx. time | Expected result |
|---|---|---|---|
| 1 | `python GenerateTruthQA_Hindi.py` | 10–20 min | 790 items; median cross-lingual similarity printed |
| 2 | `python Sample_data_verification.py` | seconds | 160-item review sheet across 37 categories |
| 3 | `python Annotation_stats.py --file <annotated sheet>` | seconds | agreement statistics and `truthfulqa_hindi_verified.jsonl` |
| 4 | `python Knowledege_build.py` | 45–75 min | collection `wiki_passages_v2` and its passage count |
| 5 | `python phase2_retriever.py` | 1 min | relevant passages for sample queries |
| 6 | `python Phase_1_SFT_pipeline.py` | 20–30 min | falling held-out loss; queries for unseen questions |
| 7 | `python Phase_4.py` | 15–30 min | held-out reward per checkpoint; `ppo_curves.png` |
| 8 | `python run_full_system.py` | minutes | demo answers, then interactive mode |
| 9 | `python Evaluation.py --part query` | ~5 min | `results/query_eval.md` |
| 10 | `python Evaluation.py --part e2e --n 100` | 60–120 min | `results/e2e_n100/results.md` |

Notes:
- Step 3 needs the review sheet from step 2 annotated by two people (see [Dataset](#dataset-truthfulqa-hindi)).
  Step 10 uses the verified benchmark automatically when it exists.
- `--targeted` in step 4 also fetches full English Wikipedia pages for the TruthfulQA questions. In our
  runs most of these requests were rejected by the Wikipedia API, so the released results do not depend on them.
- `run_full_system.py` retrieves the top 5 passages; `Evaluation.py` uses the top 3 (`--top-k`). Both use the 0.35 gate.
- `Evaluation.py --part e2e --verified-only` evaluates only on the human-verified items; `--n 0` runs the full benchmark.
- Steps 4 and 10 resume where they stopped if interrupted.
- On Windows, use Windows Terminal or the VS Code terminal so that Devanagari renders correctly.
- The `--alpha`, `--beta`, `--delta` and `--gamma` options of `Phase_4.py` belong to the earlier reward and
  have no effect on the current one.

## Evaluation

| System | How the English query is produced |
|---|---|
| `no_retrieval` | none; the model answers from its own knowledge |
| `translate_retrieve` | machine translation of the question (`Helsinki-NLP/opus-mt-hi-en`) |
| `llm_rewrite` | Mistral prompted to write a keyword query |
| `sft_query` | query policy after the supervised warm start only |
| `crag_rl` | query policy after PPO (proposed) |

Metrics:

- **Hallucination rate and truthfulness:** similarity of the answer to TruthfulQA's correct versus
  incorrect references (English references, plus the human-checked Hindi best answer where available).
- **Abstention rate**, and the share of answers that are both informative and truthful.
- **FActScore-lite:** NLI-based claim precision against fixed gold-question evidence.
- **AlignScore-proxy:** NLI faithfulness to the system's own retrieved passages.
- **Self-Contrast ΔSC:** paired comparison against `no_retrieval`, with an exact McNemar test.
- **Latency:** per stage, mean and p95.
- **Retrieval Hit@3** on held-out training-pool questions.

## Dataset: TruthfulQA-Hindi

- **Construction:** a first translation with `opus-mt-en-hi` was rejected after inspection (about 23 of 40
  sampled items changed meaning). The released version was translated with NLLB-200: median English–Hindi
  similarity 0.917, with 29 of 790 items below 0.6.
- **Verification:** two bilingual annotators independently checked a 160-item category-stratified sample.
  An item was marked valid only if (a) the factual proposition is preserved, (b) the distinction between the
  correct answer and the misconception is preserved, and (c) the Hindi is natural and unambiguous. Failed items
  received a corrected question and/or best answer, and items whose premise is US/UK-specific were flagged.
  Agreement was very high (Cohen's κ = 0.975), and errors were found in 45.6% of items (questions 37.5%,
  reference answers 33.8%).
- **Coverage:** this error rate is above the protocol's 10% threshold for a full review, which has not yet been
  done. The 630 unverified items should be treated as noisy.

## Development notes

**Fixes to the first implementation.** The first version memorised its training questions (a question about
Russia's capital produced `japan capital city tokyo`), embedded queries in its reward with the wrong model,
ignored the Hindi question in the reward, scored log-probabilities with shifted decoder inputs, used REINFORCE
instead of PPO, and evaluated on training items. These were fixed before the runs reported here.

**First full run.** SFT data filtered at similarity ≥ 0.6 (4,303 / 478 pairs), the earlier reward, PPO learning
rate 2e-5 and KL coefficient 0.05. The held-out reward was highest at step 0 (the SFT policy, 0.819) and fell to
0.72–0.81 over 100 steps while the KL to the SFT policy rose to 0.2–0.7, so the saved checkpoint was the SFT policy.

**Final run.** SFT data filter raised to 0.78 to remove translations that changed entities; the stricter reward
above; learning rate 2e-6 and KL coefficient 0.15; relevance gate raised from 0.25 to 0.35. The final policy changes
27 of 100 test queries relative to the SFT policy, without improving retrieval or end-to-end metrics.

**Deviations from the original proposal.**

| Proposal | Implementation | Reason |
|---|---|---|
| Reward on answer consistency (AlignScore) | question–evidence similarity reward | running Mistral-7B per PPO rollout does not fit in GPU memory alongside the policy |
| IndicTrans2 translation | NLLB-200 (automatic fallback) | IndicTrans2 is gated and needs `transformers<5` |
| English Wikipedia snapshot | Simple English Wikipedia + XQuAD paragraphs | the full dump is ~50 GB; targeted page fetches were rate-limited |
| ~500-token chunks | 120 words | the encoder reads at most 128 word-pieces |
| FActScore | FActScore-lite (sentence-level claims, multilingual NLI) | no LLM fact-decomposer budget |
| Full review if error rate > 10% | 20% sample reviewed (error rate 45.6%) | time |
| Hindi-RAG baseline, reward ablations | not run | time |

## Limitations

- Most of CRAG-RL's reduction in hallucination comes from abstention, not from better answers.
- Both rewards used for PPO are proxies (evidence relevance), not answer correctness; the answer-consistency
  reward of the original proposal did not fit in GPU memory alongside the policy.
- Only 20% of the benchmark is human-verified, and the end-to-end evaluation uses 100 questions with one seed.
- Automatic judges (embedding similarity, NLI) approximate human judgement; reworded abstentions are not detected.
- The Hindi-RAG baseline and the reward ablations from the proposal were not run.
- English Wikipedia cannot answer questions about facts specific to Hindi-speaking communities.

## Acknowledgements

TruthfulQA (Lin et al., 2022), XQuAD (Artetxe et al., 2020), SQuAD (Rajpurkar et al., 2016), mT5 (Xue et al., 2021),
Sentence-BERT multilingual models (Reimers and Gurevych, 2020), Mistral 7B (Jiang et al., 2023), NLLB (NLLB Team, 2022),
IndicTrans2 (Gala et al., 2023), PPO (Schulman et al., 2017).
