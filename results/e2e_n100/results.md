# TruthfulQA-Hindi end-to-end results (n=100)

| System | Halluc. rate ↓ (95% CI) | Truthful ↑ | Informative & truthful ↑ | Abstain | FActScore-lite ↑ | AlignScore-proxy ↑ | ΔSC ↑ (McNemar p) | Query words |
|---|---|---|---|---|---|---|---|---|
| no_retrieval | 0.490 (0.39–0.59) | 0.510 | 0.510 | 0.000 | 0.440 | – | – | – |
| translate_retrieve | 0.220 (0.14–0.31) | 0.650 | 0.280 | 0.500 | 0.690 | 0.545 | +0.270 (0.0) | 10.9 |
| llm_rewrite | 0.240 (0.16–0.33) | 0.640 | 0.210 | 0.550 | 0.633 | 0.684 | +0.250 (0.0002) | 7.1 |
| sft_query | 0.080 (0.03–0.13) | 0.730 | 0.110 | 0.810 | 0.789 | 0.552 | +0.410 (0.0) | 3.0 |
| crag_rl | 0.080 (0.03–0.14) | 0.730 | 0.130 | 0.790 | 0.857 | 0.576 | +0.410 (0.0) | 3.1 |

## Self-Contrast breakdown vs no_retrieval

| System | Corrected | Newly introduced | Unchanged hallucinated |
|---|---|---|---|
| translate_retrieve | 33 | 6 | 16 |
| llm_rewrite | 34 | 9 | 15 |
| sft_query | 43 | 2 | 6 |
| crag_rl | 44 | 3 | 5 |

## Latency (seconds per question)

| System | Query | Retrieval | Synthesis | Total mean | Total p95 | Overhead vs no_retrieval |
|---|---|---|---|---|---|---|
| no_retrieval | 0.00 | 0.00 | 7.05 | 7.05 | 10.45 | – |
| translate_retrieve | 0.44 | 0.11 | 7.07 | 7.62 | 11.73 | 0.57 |
| llm_rewrite | 1.46 | 0.05 | 19.36 | 20.86 | 42.79 | 13.81 |
| sft_query | 0.34 | 0.07 | 3.24 | 3.65 | 10.44 | -3.40 |
| crag_rl | 0.35 | 0.07 | 9.16 | 9.58 | 30.85 | 2.53 |
