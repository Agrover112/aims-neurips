# Uniform ablation protocol

Status: binding for every ablation table in the report.

The earlier report mixed validation-only, test-only, selected-seed, four-configuration
means, task-level standard errors, seed standard deviations, and one hidden
leaderboard table. That was not acceptable for a NeurIPS competition report.
This file fixed one protocol and one reporting contract. The uniform rebuild now
follows it.

## 1. Split roles

- **Train:** 6,702 tasks. Model fitting only.
- **Validation:** 104 tasks. Every ablation is evaluated here.
- **Test:** 22 tasks. Final selected model and the Beta baseline only. Test is
  not an ablation source.

Calibration and any tuned component use out-of-fold predictions inside the 104
validation tasks. The default is five-fold cross-fitting stratified by
benchmark. Each task is scored exactly once, by a calibrator that did not see
that task. This removes the current validation-fit/test-evaluate asymmetry.

## 2. Fixed evaluation grid

Every ablation table uses:

- the same 104 validation tasks;
- the same 10 frozen acquisition orders for the architecture, calibration, and
  disjoint tables;
- for the acquisition policy table, the same 10 fixed candidate streams per
  task, with the out-of-fold calibrators fitted on the frozen orders;
- budgets `0, 1, 3, 7, 15, 31`;
- the same disjoint acquisition and evaluation pools;
- the same Beta-Binomial baseline `p = (2 + successes) / (4 + n)`;
- Brier ALC as the primary metric, ECE ALC as the secondary metric;
- task as the independent unit, with orders averaged within task;
- paired differences against the same baseline on the same task and order.

No table may mix validation and test rows. No table may mix selected-seed and
four-configuration means without a column that says which is which.

## 3. Uncertainty

- Primary interval: paired task bootstrap over the 104 validation tasks,
  10,000 draws, percentile 95% interval.
- Sensitivity interval: benchmark-clustered bootstrap, resampling the five
  validation benchmarks. Report it as a sensitivity check, not as the primary
  interval.
- Do not report a seed standard deviation as if it were a task-level standard
  error. If a table averages seeds, say so and report the seed spread
  separately.

## 4. Ablation families

### Encoder ladder

Arms: Beta baseline, 1PL plus MAP ability, 2PL plus MAP ability, CNP item head
with the correction zeroed, CNP-AIRT with the learned correction, and the
correction-free retrained CNP. All arms use the same validation tasks, orders,
pools, and out-of-fold calibrators. Report the selected seed and mode as the
main row. Report the four-configuration mean only as a separate robustness
table.

### Calibration

Arms: raw, full BCE, full Brier, context-only, anchor-only, Platt-only,
task-balanced, shared fit, and fixed 0.5 at budget 0. Fit each variant
out-of-fold on validation. Evaluate all variants on the same held-out
validation tasks. Do not use test for the calibration ablation.

### Disjoint-episode training control

Arms: original training condition and disjoint context/target training
condition. Same architecture, seed set, step count, and sampler. Fit
calibrators out-of-fold on validation. Evaluate on the same validation tasks.
Test is not part of this ablation.

### Acquisition

Arms: random, predictive uncertainty `|p - 0.5|`, IRT Fisher information
`a_i^2 p_i (1 - p_i)`, CNP-Beta disagreement, and an optional oracle. Every
policy sees the same 104 validation tasks and the same 10 candidate streams.
Every policy ranks the remaining candidate pool at each step. The old
threshold-plus-reserve rule is not an item-ranking policy and must not be
compared with Fisher information as if it were.

The Beta posterior variance arm is not an item-ranking policy. It is constant
across candidates when item features are ignored. It may appear only as a
stopping-rule control, clearly labeled, not as an acquisition policy.

### Budget shrinkage

Arms: flat Beta and budget-specific shrinkage. Fit the concentration on
out-of-fold validation tasks. Evaluate on the same held-out tasks. Test is not
part of this ablation.

## 5. Forbidden sources

- Hidden Codabench scores are never ablation evidence.
- Hidden scores are never used to choose a component, calibrator, acquisition
  policy, or ablation arm.
- The only permitted hidden table is a separate submission description that
  states the hidden groups differ and the comparison is unpaired.

## 6. Table contract

Every ablation table must state, in the caption or a nearby sentence:

- split and number of tasks;
- number of orders;
- budget grid;
- baseline;
- whether the table uses selected seed, four-configuration mean, or both;
- uncertainty unit;
- whether the comparison is paired.

If an arm was not run under this protocol, it is marked `not run` and excluded
from claims. Reformatting an old mixed table is not a fix.

## 7. Audit outcome

The uniform rebuild is complete. The audit record is `docs/uniform-audit.md`,
and the consolidated source of truth is
`/tmp/opencode/uniform-v1/analysis/summary.json`.

| Table | Source | Uniform? | Outcome |
| --- | --- | --- | --- |
| Main validation ladder | selected seed and four-configuration mean, validation only | Yes | Rebuilt with out-of-fold calibration; test moved to a separate final table |
| Calibration decomposition | 104 validation tasks, five folds, both models, four configurations | Yes | Rebuilt with out-of-fold fitting; test excluded |
| Disjoint-episode control | 104 validation tasks, four configurations, out-of-fold calibration | Yes | Rebuilt; reported as a near tie |
| Acquisition | 104 validation tasks, 10 fixed candidate streams, two seeds | Yes | Ranking policies on identical streams; old threshold/reserve rule removed |
| Budget shrinkage | 104 validation tasks, out-of-fold kappa selection | Yes | Null by construction; kappa 4.0 at every budget at or above 1 |
| Final test | 22 test tasks, selected model and Beta only | Separate readout | Not an ablation; reported after selection |
| Hidden Codabench scores | different unpaired groups | Not an ablation | Descriptive only; never used for a claim about a component |

The audit confirms 104 tasks, 10 orders, 6 budgets, five folds with every task
held out once, no duplicate task-order-budget rows, and no test rows in any
ablation table. Every acquisition policy has identical budget-0 rows, and every
policy sees the same candidate stream within a task and seed.

The suite is compliant with this protocol. Two limitations remain outside it:
validation has only five benchmark clusters, and only two full-scale seeds are
run. Task intervals are primary, benchmark-cluster intervals are a sensitivity
check, and the third seed remains deferred until the harness and report are
fully audited.

## 8. Corrected training ladder (v4.1, specified)

The v4.1 campaign keeps the validation grid, folds, budgets, and calibration
contract above, but changes the training protocol: per-visit rotating disjoint
acquisition/evaluation splits, uniform context draws from the acquisition half,
evaluation-half targets only, padded 16-task batches, a global task-budget pair
epoch, and the corrected latent KL normalization. The full rung list and
decision rules are in
[`faithful-ablation-ladder.md`](faithful-ablation-ladder.md). No v4/v4.1 table
may be mixed with a v3 table.
