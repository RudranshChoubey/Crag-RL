# Design notes: what was fixed, and every deviation from the proposal

## Problems found in the first implementation, and the fixes

| # | Problem | Evidence | Fix |
|---|---|---|---|
| 1 | Query policy memorised its training data | "रूस की राजधानी क्या है?" → `japan capital city tokyo`; "कितने महाद्वीप हैं?" → `continent canarian islands coast` (a training question) | removed the 6 anchor pairs repeated 20× (17% of the data); all 1,190 XQuAD + ~4,000 SQuAD-train questions over 442 articles; held-out split by article; lr 3e-4 with early stopping instead of 1e-3 × 19 epochs |
| 2 | Reward embedded queries with the wrong model | `collection.query(query_texts=…)` uses ChromaDB's default English encoder; the index was built with multilingual MiniLM (both 384-d, so no error) | every query embedded with the index's own encoder (`phase2_retriever.py`) |
| 3 | Reward ignored the Hindi question | any query that matched *some* passage scored high, so generic queries were rewarded for every prompt | reward terms anchored on the question: cos(x_hi, evidence) and cos(x_hi, q) |
| 4 | Log-probs scored with shifted decoder inputs | passing `labels=` makes the model shift the sequence again; each token was scored after a pad | teacher-forced scoring with `decoder_input_ids = seq[:, :-1]` |
| 5 | REINFORCE instead of PPO | no clipping, value head or GAE | full PPO: clipped surrogate, value head + GAE, entropy, KL to the frozen SFT policy |
| 6 | Evaluation used training items | `Evaluation.py` read `sft_training_pairs.jsonl` | Part A on the held-out split; Part B on TruthfulQA-Hindi with baselines |
| 7 | Most of each passage was never embedded | 350-word chunks, MiniLM reads 128 word-pieces | 120-word chunks, 20 overlap |
| 8 | Narrow knowledge base | first 25,000 Simple Wikipedia articles; Kuechly question retrieved "Rugby football" | all Simple Wikipedia + XQuAD paragraphs + targeted English Wikipedia pages |
| 9 | Confident answers from irrelevant passages | Everest demo answered "5,030 m" from the generic "Mountain" article | relevance gate: abstain when max cos(question, passage) < 0.25 |
| 10 | English explanations in Hindi answers | "(The answer is not available…)" | stricter few-shot prompt + bracket stripping |
| 11 | Broken benchmark translation | opus-mt-en-hi changed the meaning of ~23/40 sampled items | IndicTrans2 (NLLB-200 fallback) + screening + verification |

## Deviations from the proposal (state these in the report)

| Proposal | Implementation | Reason |
|---|---|---|
| Reward α·AlignScore(y, c) on the generated answer | α·max cos(x_hi, c_j): cross-lingual similarity between the question and the retrieved evidence | running Mistral-7B for every PPO rollout does not fit a 6–8 GB laptop GPU alongside the policy; this term rewards retrieving evidence about the question, which the answer must be grounded in. AlignScore-style faithfulness is still measured at evaluation (NLI). |
| Reward β·CosSim(q, c) − γ·Len | kept, plus δ·cos(x_hi, q) | without a question-anchored term the policy found generic high-similarity queries |
| English Wikipedia snapshot | Simple English Wikipedia (all) + XQuAD paragraphs + full English Wikipedia pages fetched for each TruthfulQA question (cached) | full dump is ~50 GB; targeted pages make retrieval easier than open-world, for all systems equally |
| ~500-token chunks | 120 words | encoder window is 128 word-pieces |
| Training data: translated open-domain QA | XQuAD-hi (professional) + SQuAD-train translated with NLLB-200, filtered at xling_sim ≥ 0.6 | gives gold English questions for the SFT targets; disjoint from TruthfulQA |
| FActScore | FActScore-lite: Hindi answer split into sentences, each checked by multilingual NLI (mDeBERTa-v3 XNLI) against 5 passages retrieved with the gold English question | no LLM fact-decomposer budget |
| TruthfulQA truthfulness judge | similarity protocol: max cos(answer, correct refs) − max cos(answer, incorrect refs), English refs, multilingual encoder | no fine-tuned judge for Hindi |
| Hindi-RAG baseline | not run | no Hindi Wikipedia index built in time |
| Mistral-7B | Mistral-7B-Instruct-v0.3, 4-bit NF4 | GPU memory |

## Hyperparameters

SFT: AdamW lr 3e-4, batch 16, ≤ 8 epochs, early stopping (patience 2) on held-out loss, shared embeddings frozen.
PPO: 100 steps × 16 rollouts, 4 PPO epochs, minibatch 8, lr 2e-5, clip 0.2, GAE λ 0.95 (γ = 1), c_v 0.1,
c_e 0.01, λ_KL 0.05 (k3 estimator), rollout temperature 0.8 / top-k 50, reward α 1.0, β 0.3, δ 0.5, γ 0.2,
L_max 16. Retrieval top-k 3. Reader: greedy decoding, ≤ 120 new tokens, gate 0.25.
