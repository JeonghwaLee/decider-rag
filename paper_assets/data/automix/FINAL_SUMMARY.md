# AutoMix-style cascade vs Decider (Ours): final live results (2026-11-10, all runs done 10:01 UTC)
Cost scope: cascade = every cloud call charged (answer + verification + escalation); LLM selectors = answer + selector tokens (complete pipeline).
Cached-input price NOT used for the cascade: vLLM prefix_cache_hits_total counts the shared prompt once per sample for n>1 (e.g. 32,928 hits for a 4,943-token request), so the discount is inflated.

## HotpotQA (1,600 q, D = 0.722 s)
| policy | F1 | $/1k | wall mean/p95 | DQ-F1 |
| Decider (Ours) [k8 run] | 0.628 | 0.239 | 0.51/0.80 | 0.580 |
| cascade k8 | 0.619 | 0.213 | 1.36/1.83 | 0.000 |
| Decider (Ours) [k4 run] | 0.628 | 0.239 | 0.51/0.80 | 0.582 |
| cascade k4 (validation-chosen low-cost) | 0.613 | 0.186 | 1.32/1.79 | 0.001 |
Decider - k8: F1 +0.009 [-0.004,+0.021], $ +0.026, DQ +0.579. Decider - k4: F1 +0.015 [+0.001,+0.028], $ +0.053, wall -0.82 s, DQ +0.581.
Verification (edge, local): k8 754 ms, k4 725 ms. DQ-F1 advantage vanishes at >= 3-4 D.

## MuSiQue (400 q, D = 23.871 s)
| policy | F1 | $/1k | wall mean/p95 | DQ-F1 |
| Decider (Ours) | 0.495 | 2.01 | 8.1/34.8 | 0.478 |
| cascade k8 (x1 infeasible -> IBC point) | 0.467 | 4.26 | 10.7/37.2 | 0.436 |
| Qwen3.8 shared-prefix | 0.456 | 3.16 | 9.3/35.0 | 0.425 |
| Qwen3.8 thinking | 0.446 | 5.45 | 19.0/53.2 | 0.394 |
| cascade k1 (never escalates; separate run, Decider 0.498 / 2.03 / DQ 0.477) | 0.388 | 1.25 | 1.5/2.3 | 0.388 |
Decider - k8: F1 +0.028 [+0.006,+0.051], $ -2.25, wall -2.6 s, DQ +0.042 [+0.020,+0.066] (positive over 0.6-2 D).
Decider - k1: F1 +0.110 [+0.081,+0.141], $ +0.78, DQ +0.089. All-unparsable: k8 1.8%, k1 21.2%. Verification: k8 2.03 s, k1 1.05 s.
Files: hotpot_live_{summary,diag}.txt, hotpot_k4_{summary,diag}.txt, musique_live_{summary,diag}.txt, musique_k1_{summary,diag}.txt, offline_*.
