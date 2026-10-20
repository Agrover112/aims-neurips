# Active-acquisition ablation notation

The offline simulator is `modal_active_ablation.py`; its results are aggregated by
`scripts/analyze_active_ablation.py`. Both use the same arm names.

## Predictors

| symbol | predictor |
|---|---|
| `P_β` | Laplace-smoothed Beta-Binomial group mean, `p = (2 + successes) / (4 + n)`. |
| `P_C` | CNP-AIRT, masked subject block, seed 20261018, with the validation-fitted per-budget anchor calibrator. |

## Acquisition rules

| symbol | rule |
|---|---|
| `A_R` | Random stream order. Reveal the first `b` candidates in a random permutation. |
| `A_U(P)` | Uncertainty from predictor `P`: buy when `|p_P - 0.5| <= 0.15`. |
| `A_V(β)` | Beta posterior variance: buy when `Var(β) >= 0.01`. |
| `A_D` | Disagreement: buy when `|p_C - p_β| >= 0.15`. |

All rules reserve enough candidates to fill the remaining label budget with
`items_remaining <= labels_remaining`. One acquisition trajectory runs to 31
labels (or until the stream is exhausted), and evaluation uses nested prefixes
at budgets 0, 1, 3, 7, 15, and 31.

Entropy is not a separate arm. For a fixed threshold, `H(p)` is a monotone
transform of `|p - 0.5|`, so entropy acquisition makes the same decisions as
`A_U` with a matched threshold.

## Arms

| arm | study | interpretation |
|---|---|---|
| `β-R` | Priority-1 | Beta predictor, random acquisition. |
| `β-Uβ` | Priority-1 | Beta predictor, Beta-mean uncertainty. This is the policy in the first submission. |
| `β-Vβ` | Priority-1 | Beta predictor, Beta posterior variance. |
| `C-R` | Priority-1 | CNP predictor, random acquisition. |
| `C-UC` | Priority-1 | CNP predictor, CNP uncertainty. This is the deployed active CNP policy. |
| `C-Uβ` | Priority-2 | CNP predictor, Beta-mean acquisition. Acquisition ignores the neural prediction. |
| `C-Vβ` | Priority-2 | CNP predictor, Beta posterior variance. Acquisition ignores the neural prediction. |
| `C-D` | Priority-2 | CNP predictor, CNP/Beta disagreement. |

Omitted arms:

- `P_β + A_U(C)` is identical to `P_β + A_U(β)` because the Beta predictor has no item-level signal.
- `P_β + A_D` is identically zero disagreement for the same reason.
- `P_C + A_H` is monotone-equivalent to `P_C + A_U(C)` at the fixed threshold.

## Competition files

| file | policy | hook calls `predict`? |
|---|---|---|
| `labeling_beta.py` | `P_C + A_U(β)` | No. The hook accepts only `input`, `labeled`, and `context`. |
| `labeling_disagreement.py` | `P_C + A_D` | Yes. It needs the current CNP prediction. |

Both files filter revealed evidence to the same subject and benchmark before
computing the Beta mean. They do not call external services and do not read
files outside the submission.
