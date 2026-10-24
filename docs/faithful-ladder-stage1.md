# Faithful ladder Stage 1 results

Status: 2026-10-22. Complete. This is an internal ablation record, not part of
the paper.

Protocol: `v4.1-faithful-ladder`, 6,649 fit tasks, 53 task-level probe tasks
from `harmbench`, `agi_elo`, `emoji_attack`, and `afrieval`, one full pair epoch
(2,494 steps), BCE objective, three seeds, 104 validation tasks, five-fold
out-of-fold calibration. No test labels were used.

Hardware note: Stage 1 ran on A100-80GB before the switch to H100. The protocol
does not depend on the GPU.

## Calibrated Brier ALC (OOF)

Lower is better.

| rung | seed 20261018 | seed 20261019 | seed 20261020 | mean |
|---|---:|---:|---:|---:|
| `2pl` | 0.223371 | 0.223280 | 0.223322 | 0.223324 |
| `2pl_anp` | 0.223329 | 0.223293 | 0.223307 | 0.223310 |
| `gp_2pl` | 0.223333 | 0.223345 | 0.223354 | 0.223344 |
| `gp_anp` | 0.223280 | 0.223246 | 0.223273 | 0.223266 |
| Beta | 0.222838 | 0.222838 | 0.222838 | 0.222838 |

## Paired deltas vs 2PL

Negative means the rung is better. MDE80 is the paired 80%-power minimum
detectable effect computed from the observed task-level standard error. A delta
smaller than MDE80 is `can't tell`, not `no effect`.

| rung | delta | 95% CI | MDE80 | verdict | task wins |
|---|---:|---:|---:|---|---:|
| `2pl_anp` | -0.000014 | [-0.000048, +0.000019] | 0.000048 | can't tell | 55/104 |
| `gp_2pl` | +0.000020 | [-0.000055, +0.000095] | 0.000107 | can't tell | 57/104 |
| `gp_anp` | -0.000058 | [-0.000186, +0.000070] | 0.000183 | can't tell | 63/104 |

Per-seed mean deltas vs 2PL:

| rung | seed 18 | seed 19 | seed 20 | direction |
|---|---:|---:|---:|---|
| `2pl_anp` | -0.000042 | +0.000013 | -0.000014 | 2/3 better |
| `gp_2pl` | -0.000038 | +0.000064 | +0.000032 | 1/3 better |
| `gp_anp` | -0.000091 | -0.000034 | -0.000049 | 3/3 better |

## Component deltas

| contrast | delta | 95% CI | MDE80 | verdict | task wins |
|---|---:|---:|---:|---|---:|
| `gp_anp - gp_2pl` | -0.000078 | [-0.000254, +0.000098] | 0.000251 | can't tell | 53/104 |
| `gp_anp - 2pl_anp` | -0.000044 | [-0.000167, +0.000080] | 0.000176 | can't tell | 60/104 |

## Benchmark-cluster sensitivity

Task intervals are primary. Cluster means are shown only to check whether the
`gp_anp` point estimate is stable across benchmarks.

| cluster | tasks | `gp_anp - 2pl` mean |
|---|---:|---:|
| `afrimedqa` | 32 | -0.000383 |
| `appworld` | 18 | +0.000356 |
| `cmmlu` | 22 | -0.000543 |
| `cruxeval` | 31 | +0.000355 |
| `devbench` | 1 | +0.000735 |

The sign flips across clusters. A five-cluster bootstrap for `gp_anp - 2pl`
gives `[-0.000445, +0.000367]`.

## Verdict

No rung clears its MDE against `2pl`. `gp_anp` has the best point estimate and
is better in all three seeds, but the effect is about one third of its MDE and
the task interval includes zero. The cluster means flip sign. Treat this as a
weak hint, not a promotion.

The Beta reference has a better point estimate than every neural rung
(0.222838 vs 0.223266 to 0.223344). The paired Beta difference is also
underpowered on 104 tasks and is reported as `can't tell`, not as proof that
Beta wins.

Per the frozen stop rules:

- do not promote any Stage 1 rung;
- do not run the objective ladder on a non-winner;
- do not run the second pair epoch;
- do not run `gp_anp_prior` or the latent controls yet.

Any of those would be an exploratory override, not the pre-registered ladder.

## Artifacts

Results live under
`/data/results/faithful-ladder/ladder-v41-{rung}-bce-{seed}-masked.json`.
Paired analysis is produced by `scripts/analyze_faithful_ladder.py`.
