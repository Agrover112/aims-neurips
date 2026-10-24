# Uniform ablation audit

Status: validation grid audited, test readout separated, report rebuilt locally.
Date: 2026-10-20.

## Scope

This audit covers the four uniform ablation families:

| Family | Artifact | Seed/mode configurations |
| --- | --- | --- |
| Encoder ladder | `/tmp/opencode/uniform-v1/ladder` | `20261018` masked/visible, `20261019` masked/visible |
| Calibration decomposition | `/tmp/opencode/uniform-v1/calibration` | same four |
| Disjoint-episode control | `/tmp/opencode/uniform-v1/disjoint` | same four |
| Acquisition ranking | `/tmp/opencode/uniform-v1/acquisition` | `20261018` masked, `20261019` masked |
| Budget shrinkage | `/tmp/opencode/uniform-v1/budget-shrinkage` | five-fold out-of-fold |

Every family uses the same 104 validation tasks, ten frozen orders, budgets
`0,1,3,7,15,31`, weights `0.1,0.2,0.2,0.2,0.2,0.1`, Beta-Binomial baseline,
Brier ALC as the primary metric, and an equal-weight task average. Test rows are
read only for the final selected model and the Beta baseline.

The consolidated output is
`/tmp/opencode/uniform-v1/analysis/summary.json`. The table builder is
`scripts/analyze_uniform_ablations.py`.

## Checks that pass

- 104 validation tasks in every family and in every seed/mode configuration.
- 10 frozen orders 0-9 for the ladder, calibration, and disjoint families.
- 10 fixed candidate streams 0-9 for acquisition.
- Six budgets, no missing task/order/budget cells.
- No duplicate rows on task/order/budget within an arm or variant.
- Five folds cover every validation task; each task has one fold label.
- Every validation fold has held-out tasks (19-24 tasks per fold).
- Test rows carry fold `-1` and never enter a validation table.
- Calibration fits use validation rows only. Each out-of-fold prediction uses a
  calibrator fitted without that task's fold.
- Acquisition arms share the task set and stream set. At budget zero the four
  CNP arms have identical Brier scores, as they must before any label is
  revealed.
- Acquisition streams have at least 136 candidates, so the budget-31 step never
  exhausts the pool.

## Primary validation results

Selected configuration: seed `20261018`, masked mode. Deltas are paired task
differences, first arm minus Beta, with a 95% task bootstrap interval. Negative
is better.

### Encoder ladder, selected configuration

| Arm | Raw ALC | OOF-calibrated ALC | Delta vs Beta | 95% task CI | Wins |
| --- | ---: | ---: | ---: | --- | ---: |
| Beta-Binomial | 0.222838 | 0.222838 | 0.000000 | fixed | 0/104 |
| CNP full | 0.234915 | 0.222551 | -0.000287 | [-0.001463, +0.000905] | 47/104 |
| CNP zero correction | 0.232362 | 0.222524 | -0.000314 | [-0.001487, +0.000883] | 48/104 |
| CNP retrained, no correction | 0.235495 | 0.222849 | +0.000011 | [-0.001051, +0.001096] | 42/104 |
| 1PL | 0.236439 | 0.223539 | +0.000701 | [-0.000324, +0.001754] | 45/104 |
| 2PL | 0.267809 | 0.223298 | +0.000460 | [-0.000617, +0.001563] | 53/104 |

The selected CNP full model is 0.0003 better than Beta on validation and the
four-configuration mean is 0.0003 worse. Neither difference is distinguishable
from zero. Raw neural forecasts are 0.0095 to 0.0450 worse than Beta.

### Encoder ladder, four-configuration mean

| Arm | Raw ALC | OOF-calibrated ALC | Delta vs Beta | 95% task CI |
| --- | ---: | ---: | ---: | --- |
| Beta-Binomial | 0.222838 | 0.222838 | 0.000000 | fixed |
| CNP full | 0.233139 | 0.223129 | +0.000291 | [-0.000715, +0.001314] |
| CNP zero correction | 0.232140 | 0.223115 | +0.000277 | [-0.000735, +0.001294] |
| CNP retrained, no correction | 0.234086 | 0.223215 | +0.000377 | [-0.000577, +0.001333] |
| 1PL | 0.236091 | 0.223518 | +0.000680 | [-0.000393, +0.001764] |
| 2PL | 0.268009 | 0.223502 | +0.000664 | [-0.000436, +0.001770] |

The learned ability correction does not improve calibrated ALC. Full CNP minus
zero correction is `+0.000014` in the four-configuration mean, with interval
`[-0.000029, +0.000056]`. The correction hurts raw ALC by `+0.002553` in the
selected configuration and `+0.000999` in the four-configuration mean.

### Calibration decomposition, selected CNP

Baseline for the delta column is full BCE calibration (`0.222559`).

| Variant | ALC | Delta vs full BCE | 95% task CI | Wins |
| --- | ---: | ---: | --- | ---: |
| Raw | 0.234915 | +0.012356 | [+0.008917, +0.015836] | 31/104 |
| Context rate only | 0.224485 | +0.001926 | [+0.000533, +0.003087] | 28/104 |
| Anchor only, BCE | 0.224607 | +0.002048 | [+0.000661, +0.003238] | 26/104 |
| Platt only, BCE | 0.227554 | +0.004995 | [+0.003205, +0.006921] | 40/104 |
| Full BCE | 0.222559 | 0.000000 | fixed | 0/104 |
| Full BCE, task-balanced | 0.222436 | -0.000123 | [-0.000567, +0.000295] | 37/104 |
| Full Brier | 0.220980 | -0.001579 | [-0.002581, -0.000477] | 82/104 |
| Full Brier, task-balanced | 0.220844 | -0.001715 | [-0.002460, -0.000937] | 82/104 |
| Full Brier, shared fit | 0.222337 | -0.000223 | [-0.001178, +0.000728] | 65/104 |
| Full BCE, fixed 0.5 slope | 0.224017 | +0.001458 | [+0.000600, +0.002267] | 42/104 |
| Full Brier, fixed 0.5 slope | 0.222437 | -0.000122 | [-0.001328, +0.001160] | 71/104 |

The full BCE calibrator is the best of the submitted-family variants. Context
only and anchor only are worse, and Platt only is much worse. Task balancing is
a small improvement that is not significant: full BCE task minus full BCE is
`-0.000123` with interval `[-0.000567, +0.000295]`.

The Brier-loss fit is the one clear out-of-fold improvement over BCE:
`-0.001579` for the selected CNP and `-0.001618` for the four-configuration
mean, with benchmark-cluster intervals that also exclude zero. It is not
adopted. **Calibration note (2026-10-22): these are v1 numbers. The current
`AnchorCalibrator` refits scale/bias per anchor and constrains scale to be
nonnegative; do not mix these rows with a calibration-v2 rerun.** The fitted
Brier scales are negative at budgets 0, 1, 3, and 7
(for example `-3.03`, `-3.47`, `-6.00`, and `-3.81`), and the anchor weights
rise to `0.79-0.96`. The gain comes from inverting or nearly inverting the
validation item ranking and leaning on the context rate. Validation raw AUC is
near `0.48`, so the inversion helps validation Brier. Test raw AUC is above
`0.50` and rises to `0.56`, so the inversion would not transfer. The submitted
archive keeps the BCE calibrator.

The calibration mean over four configurations tells the same story. Full Brier
minus full BCE is `-0.001618` with task interval
`[-0.002511, -0.000692]`; full Brier task is `-0.001759` with interval
`[-0.002436, -0.001066]`. The selected CNP full Brier would score `0.220980`
against Beta `0.222838`, but the benchmark-cluster interval for that comparison
is `[-0.005389, +0.001753]`, so it is not robust to benchmark clustering.
The four-configuration mean full Brier comparison with Beta is `-0.001324` with
task interval `[-0.002794, +0.000173]`.

Online-AIRT shows the same direction but a much smaller Brier-fit gain:
selected `0.223180` vs full BCE `0.223303` (`-0.000123`, task interval
`[-0.000530, +0.000335]`), and four-configuration mean `0.223067` vs
`0.223504` (`-0.000437`, task interval `[-0.000859, +0.000022]`).

### Disjoint-episode control

Selected configuration:

| Arm | Raw ALC | Calibrated ALC | Delta vs original | 95% task CI |
| --- | ---: | ---: | ---: | --- |
| Original CNP full | 0.234915 | 0.222551 | 0.000000 | fixed |
| Disjoint training | 0.233434 | 0.222681 | +0.000130 calibrated | [-0.000002, +0.000261] |
| Disjoint training | 0.233434 | 0.222681 | -0.001482 raw | [-0.002069, -0.000891] |

The calibrated difference is a near tie. Raw ALC improves in the selected
masked configuration, but it worsens in the other three configurations. The
four-configuration mean is `0.223015` calibrated vs `0.223129` original
(`-0.000113`, task interval `[-0.000192, -0.000039]`) and `0.234647` raw vs
`0.233139` (`+0.001508`, task interval `[+0.000787, +0.002301]`). The old claim
that disjoint training is worse in all four raw cells is false.

### Acquisition ranking

Deltas are arm minus random CNP (`C-R`). Negative is better.

| Seed | Arm | ALC | Delta vs C-R | 95% task CI | Wins |
| --- | --- | ---: | ---: | --- | ---: |
| 20261018 | C-R random | 0.221289 | 0.000000 | fixed | 0/104 |
| 20261018 | C-U uncertainty | 0.222321 | +0.001032 | [-0.002220, +0.004585] | 62/104 |
| 20261018 | C-F Fisher | 0.228710 | +0.007421 | [+0.002671, +0.012779] | 54/104 |
| 20261018 | C-D disagreement | 0.222873 | +0.001584 | [-0.001900, +0.005184] | 63/104 |
| 20261018 | beta-R random | 0.221638 | +0.000349 | [-0.000847, +0.001549] | 55/104 |
| 20261019 | C-R random | 0.221893 | 0.000000 | fixed | 0/104 |
| 20261019 | C-U uncertainty | 0.223009 | +0.001116 | [-0.002918, +0.005479] | 66/104 |
| 20261019 | C-F Fisher | 0.226159 | +0.004266 | [-0.001100, +0.010884] | 61/104 |
| 20261019 | C-D disagreement | 0.222952 | +0.001059 | [-0.002799, +0.005176] | 66/104 |
| 20261019 | beta-R random | 0.221638 | -0.000255 | [-0.001327, +0.000812] | 56/104 |

No active rule beats random. Fisher information is worse in both seeds and the
seed mean is `+0.005844` with task interval `[+0.001427, +0.011075]` and
benchmark-cluster interval `[+0.000672, +0.013015]`. Uncertainty and
disagreement are both slightly worse than random, with intervals that span
zero. The Beta predictor with random order is statistically indistinguishable
from the CNP predictor with random order.

### Budget shrinkage

The out-of-fold kappa grid selects `4.0` at every budget at or above 1 in every
fold. The flat prior is exactly the kappa `4.0` Beta-Binomial prior, so the tuned
arm reduces to the flat arm:

- flat ALC: `0.222837959`
- tuned ALC: `0.222837959`
- delta: `0.000000`, interval `[0.000000, 0.000000]`

This ablation is a null by construction. It does not support a budget-specific
shrinkage claim.

### Final test

Test is not an ablation source. Only the final selected model and Beta are
reported.

| Model | Raw ALC | Calibrated ALC | Delta vs Beta | 95% task CI |
| --- | ---: | ---: | ---: | --- |
| Beta-Binomial | 0.218185 | 0.218185 | 0.000000 | fixed |
| CNP full, selected | 0.221057 | 0.219939 | +0.001754 | [+0.000401, +0.002998] |
| CNP full, two-seed masked mean | 0.226548 | 0.219685 | +0.001500 | [+0.000385, +0.002605] |

The selected calibrated model is behind Beta by 0.0018. The task interval
excludes zero, but the six-benchmark cluster interval is
`[-0.000716, +0.002914]` for the selected seed and `[+0.000072, +0.002892]` for
the two-seed mean. Test has only 22 tasks and six benchmark clusters, so the
task interval is the primary readout and the cluster interval is a sensitivity
check.

The test Brier curve is not monotone after calibration at budget 0: raw
`0.258828` to calibrated `0.259418`. Budget 0 has a negative calibration scale,
so it flips the validation item ranking. At budgets 1-31 the calibration slope
is positive, and calibration improves Brier at four of five nonzero budgets.
The final ALC gap is therefore not driven by a single budget.

## Corrections to the earlier report

The earlier report mixed validation and test, selected-seed and four-config
means, and threshold/reserve acquisition with the new ranking policies. The
following claims are retired:

1. "Context-only calibration beats full BCE in all four configurations." On
   out-of-fold validation, full BCE is `0.0019` better than context only in the
   selected CNP and `0.0014` better in the four-config mean.
2. "Task-balanced fitting beats row weighting." It is slightly better but not
   significant: `-0.000123` for selected full BCE and `-0.000137` for selected
   full Brier, with intervals that span zero.
3. "The budget-0 calibrated Brier is 0.1634." The uniform value is `0.235450`
   on validation and `0.259418` on test.
4. "The correction improves raw test ALC by 0.0009." The selected test raw
   difference is `+0.002872` (full minus zero is not the same quantity; see the
   ladder table).
5. "Disjoint training is slightly worse in all four raw cells." It improves
   the selected masked raw cell by `-0.001482` and worsens the other three.
6. "No active acquisition rule beats random by more than 0.0002." The new
   ranking policies are different. No active rule beats random, but Fisher is
   `+0.0074` worse in seed `20261018` and `+0.0043` worse in seed `20261019`.
7. Hidden leaderboard scores are not ablation evidence. They remain
   descriptive and unpaired.

## Remaining limitations

- Validation has only five benchmark clusters, one of which has one task.
  Benchmark-cluster intervals are reported as sensitivity, not as the primary
  interval.
- The Brier-fit calibrator is a post hoc diagnostic. Its negative scales at
  low budgets exploit a validation-specific ranking inversion, so the
  improvement does not justify changing the submitted archive.
- The third full-scale seed `20261020` is still deferred. The current
  four-configuration mean uses seeds `20261018` and `20261019`.
- Test has 22 tasks and six benchmark clusters. The test table is a final
  readout, not a model-selection grid.
