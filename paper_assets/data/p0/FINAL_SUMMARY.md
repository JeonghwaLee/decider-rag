# P0: learned pre-generation vs learned post-generation

All comparisons below are pre-generation minus learned post-generation. Frozen validation policies; reused tests; one interleaved live campaign. Costs are cloud-token $/1,000 requests, without cache discounts or local hardware pricing.

## hotpot (1600 queries; D=0.721979 s)

| Arm | F1 | Cloud $/1k | Mean wall s | p95 s | DQ-F1 | E/C/R |
|---|---:|---:|---:|---:|---:|---|
| pre | 0.6264 | 0.2360 | 0.5391 | 0.8325 | 0.5622 | 835/765/0 |
| post | 0.6562 | 0.2858 | 0.6934 | 1.1561 | 0.4582 | 663/937/0 |

Paired question bootstrap (10,000 draws), pre minus post:

- f1: -0.02980 [-0.04074, -0.01923]
- cost_per_1k: -0.04980 [-0.06721, -0.03180]
- wall_mean: -0.15431 [-0.16396, -0.14464]
- dq_f1: +0.10403 [+0.08611, +0.12164]

Block-bootstrap sensitivity for timing:

- wall_mean: -0.15431 [-0.16602, -0.14280]
- dq_f1: +0.10403 [+0.08467, +0.12351]

## musique (400 queries; D=23.871410 s)

| Arm | F1 | Cloud $/1k | Mean wall s | p95 s | DQ-F1 | E/C/R |
|---|---:|---:|---:|---:|---:|---|
| pre | 0.4905 | 1.9875 | 7.9953 | 34.8011 | 0.4760 | 0/166/234 |
| post | 0.4723 | 1.8301 | 6.3324 | 34.7362 | 0.4578 | 0/193/207 |

Paired question bootstrap (10,000 draws), pre minus post:

- f1: +0.01825 [-0.00139, +0.03815]
- cost_per_1k: +0.15738 [-0.07411, +0.39495]
- wall_mean: +1.66293 [+0.58890, +2.77191]
- dq_f1: +0.01825 [-0.00209, +0.03960]

Block-bootstrap sensitivity for timing:

- wall_mean: +1.66293 [+0.29771, +3.11598]
- dq_f1: +0.01825 [-0.00493, +0.04252]

