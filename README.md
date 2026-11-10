# CRAG-RL: Cross-Lingual RAG with RL-Optimised Query Generation

**Mitigating fact-conflicting hallucination in low-resource languages (Hindi)**

Rudransh Choubey · Pranav Srivastava · Saisish Patnaik 
Department of AIML, CSE, JAIN (Deemed-to-be University), Bengaluru

## Overview

Large language models hallucinate more when asked questions in low-resource languages, and
retrieval-augmented generation (RAG) is limited by how little encyclopedic text exists in those
languages. CRAG-RL retrieves evidence from English Wikipedia instead, while the user asks and
receives answers in Hindi.

The key component is a small query generator, trained with reinforcement learning, that turns a
Hindi question into a short English search query. It is optimised for retrieving evidence that
answers the question rather than for translation fidelity.

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
| Vector store | ChromaDB, cosine similarity, top-k = 3 |
| Answer generator | `mistralai/Mistral-7B-Instruct-v0.3`, 4-bit NF4 |
| Benchmark | TruthfulQA-Hindi (790 items, introduced in this project) |
| Training questions | XQuAD-hi + SQuAD-train translated to Hindi (disjoint from the benchmark) |

### Reward

```
R = α · max_j cos(x, c_j)  +  β · mean_j cos(q, c_j)  +  δ · cos(x, q)  −  γ · min(1, |q| / L_max)
```

`x` is the Hindi question, `q` the generated English query, and `c_j` the retrieved passages, all embedded
with the same multilingual encoder. Empty or repetitive queries receive −1. Defaults: α = 1.0, β = 0.3,
δ = 0.5, γ = 0.2, L_max = 16. See `docs/DESIGN_NOTES.md` for the rationale.

## Repository structure

| File | Purpose | Output |
|---|---|---|
| `crag_utils.py` | shared paths, model names, constants and helpers | — |
| `GenerateTruthQA_Hindi.py` | builds TruthfulQA-Hindi (IndicTrans2, NLLB-200 fallback) with a translation-quality score | `data/truthfulqa_hindi/truthfulqa_hindi.jsonl` |
| `Sample_data_verification.py` | draws a 20% category-stratified sample for human verification | `data/truthfulqa_hindi/truthfulqa_verification_sample.csv` |
| `Annotation_stats.py` | inter-annotator agreement (Cohen's κ) and error rate with 95% CI | `..._sample_stats.json` |
| `Knowledege_build.py` | builds the English Wikipedia knowledge base | `data/chroma_db/`, `data/kb_cache/` |
| `phase2_retriever.py` | Phase 2: dense retrieval | — |
| `Phase_1_SFT_pipeline.py` | builds the training data and runs the supervised warm start | `data/sft_*.jsonl`, `models/mt5_sft_warmstart/` |
| `Phase_4.py` | PPO training of the query policy | `models/mt5_ppo_aligned/` |
| `phase3_synthesis.py` | Phase 3: Hindi answer generation | — |
| `run_full_system.py` | end-to-end system with an interactive prompt | — |
| `Evaluation.py` | held-out query evaluation and end-to-end benchmark evaluation | `results/` |
| `docs/DATASET.md` | dataset creation process | — |
| `docs/ANNOTATION_GUIDELINES.md` | human verification protocol | — |
| `docs/DESIGN_NOTES.md` | implementation details and deviations from the original proposal | — |

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

IndicTrans2 requires `transformers<5`. To use it for building the benchmark, run step 1 from a separate
environment:

```bash
python -m venv IT2Env
pip install "transformers==4.46.3" IndicTransToolkit sentencepiece sentence-transformers pandas tqdm torch
python GenerateTruthQA_Hindi.py
```

Without it, step 1 falls back to NLLB-200 automatically and records the system used in every item.

## Usage

Run the steps in order from the repository root.

| # | Command | Approx. time | Expected result |
|---|---|---|---|
| 1 | `python GenerateTruthQA_Hindi.py` | 10–20 min | 790 items; median cross-lingual similarity printed |
| 2 | `python Sample_data_verification.py` | seconds | 160-item review sheet across 37 categories |
| 3 | `python Knowledege_build.py --targeted` | 45–75 min | collection `wiki_passages_v2` with its passage count |
| 4 | `python phase2_retriever.py` | 1 min | relevant passages for sample queries |
| 5 | `python Phase_1_SFT_pipeline.py` | 20–30 min | falling held-out loss; sample queries for unseen questions |
| 6 | `python Phase_4.py` | 15–30 min | held-out reward per checkpoint; training curves |
| 7 | `python run_full_system.py` | minutes | demo answers, then interactive mode |
| 8 | `python Evaluation.py --part query` | ~5 min | `results/query_eval.md` |
| 9 | `python Annotation_stats.py` | seconds | agreement and error-rate statistics |
| 10 | `python Evaluation.py --part e2e --n 100` | 60–90 min | `results/e2e_n100/results.md` |

Notes:
- Step 9 requires the review sheet from step 2 to be annotated by two people following
  `docs/ANNOTATION_GUIDELINES.md`. Corrections and cultural flags in the sheet are applied
  automatically in step 10.
- Steps 3 and 10 resume where they stopped if interrupted.
- `--n 0` in step 10 evaluates the full benchmark (several hours).
- Training curves are written to `models/mt5_ppo_aligned/ppo_curves.png`.
- On Windows, use Windows Terminal or the VS Code terminal so Devanagari renders correctly.
- If PPO queries degenerate into repeated words while the reward rises, increase the KL penalty:
  `python Phase_4.py --kl-coef 0.2`.

### Reward ablations

```bash
python Phase_4.py --beta 0  --out ./models/ppo_no_cos
python Phase_4.py --gamma 0 --out ./models/ppo_no_len
python Phase_4.py --alpha 0 --out ./models/ppo_no_evidence
python Evaluation.py --part e2e --n 100 --systems no_retrieval \
   --policy crag_rl_no_cos=./models/ppo_no_cos --policy crag_rl_no_len=./models/ppo_no_len \
   --policy crag_rl_no_evidence=./models/ppo_no_evidence
```

## Evaluation

| System | How the English query is produced |
|---|---|
| `no_retrieval` | none; the model answers from its own knowledge |
| `translate_retrieve` | machine translation of the question (`Helsinki-NLP/opus-mt-hi-en`) |
| `llm_rewrite` | Mistral prompted to write a keyword query |
| `sft_query` | query policy after the supervised warm start only |
| `crag_rl` | query policy after PPO (proposed) |

Metrics:

- **Hallucination rate and truthfulness:** similarity of the answer to TruthfulQA's correct versus incorrect reference answers
- **Abstention rate**
- **FActScore-lite:** NLI-based claim precision against fixed gold-question evidence
- **AlignScore-proxy:** NLI faithfulness to the system's own retrieved passages
- **Self-Contrast ΔSC:** paired comparison against `no_retrieval`, with an exact McNemar test
- **Latency:** per stage, mean and p95

Bootstrap 95% confidence intervals are reported, and results are given with and without culturally flagged items.

## Dataset: TruthfulQA-Hindi

A bilingual (English–Hindi) version of TruthfulQA. Every question and reference answer is machine-translated,
screened automatically, and a 20% category-stratified sample is verified by two bilingual annotators. Items whose
premise depends on US/UK-specific context are flagged. Full details: `docs/DATASET.md`.

The benchmark is used only for evaluation. The query policy is trained on XQuAD-hi and translated SQuAD-train
questions, which share no questions with it.

## Limitations

- English Wikipedia cannot answer questions about facts specific to Hindi-speaking communities that it does not cover.
- The knowledge base includes pages retrieved for the benchmark's topics, which makes retrieval easier than in an
  open-world setting (equally for all systems).
- Automatic metrics rely on multilingual embeddings and NLI models and are approximations of human judgement.
- The Hindi-RAG baseline (retrieval from Hindi Wikipedia) described in the proposal is not implemented.

## Acknowledgements

TruthfulQA (Lin et al., 2022), XQuAD (Artetxe et al., 2020), SQuAD (Rajpurkar et al., 2016), mT5 (Xue et al., 2021),
Sentence-BERT multilingual models (Reimers and Gurevych, 2020), Mistral 7B (Jiang et al., 2023), IndicTrans2
(Gala et al., 2023), NLLB (NLLB Team, 2022), PPO (Schulman et al., 2017).
