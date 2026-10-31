# Online testing

This document specifies an online evaluation pipeline for probabilistic response
prediction with test-time label acquisition. It keeps the useful parts of the
CodaBench protocol, especially hidden evaluation items, sequential acquisition,
and nested label budgets. It fixes the main problem with using CodaBench
submissions as experiments: different submissions can be scored on different
subject and benchmark groups.

The intended use is model development and ablation. Every model and acquisition
policy should face the same tasks, evaluation rows, acquisition pools, and
candidate orders. Differences can then be interpreted as method differences
rather than differences in the sampled test set.

## 1. The testing contract

The following rules are non-negotiable for the main comparison.

1. Freeze one data snapshot before model comparison.
2. Split complete benchmarks for the main unseen-benchmark experiment.
3. Keep repeated outcomes and duplicate content groups on one side of every
   split.
4. Freeze one test panel of eligible subject and benchmark tasks.
5. Freeze one acquisition pool and one evaluation pool for every task.
6. Give every method the same candidate streams.
7. Reveal only the current acquisition candidate, never future candidates.
8. Reveal a candidate's outcome only after the policy accepts it.
9. Never reveal evaluation-pool outcomes to the predictor or policy.
10. Run one acquisition trajectory to 31 labels and score nested prefixes at
    budgets 0, 1, 3, 7, 15, and 31.
11. Reconstruct online state from the complete acquired context at every
    checkpoint. Do not depend on mutable process state.
12. Score the same fixed evaluation rows at every budget.
13. Compute metrics per subject and benchmark task first, then average tasks
    equally.
14. Compare methods with paired task-level differences and shared order seeds.
15. Fit models, thresholds, priors, and calibrators without final-test outcomes.
16. Save every acquisition decision and every unrounded prediction.

If any rule changes, assign the run a new protocol version. Do not place results
from two protocol versions in one comparison table.

## 2. What CodaBench gets right, and what the local pipeline changes

CodaBench already implements several sound online-testing constraints. Its
acquisition and evaluation pools are disjoint. Candidates arrive sequentially.
Predictions are committed at each budget. Later labels cannot revise earlier
predictions. A single acquisition trajectory produces nested label prefixes.

The problem is experimental comparability. Our six past submissions received
8 or 9 hidden subject and benchmark groups each, but almost all groups differed
between submissions. Submission `1005155` and `1005187`, for example, shared no
exact group. Their aggregate score difference therefore mixes model quality
with test-set difficulty.

The platform selects complete subject and benchmark groups under an
approximately 1,000-source-item draw limit. It prefers less represented
subjects and benchmarks and uses random tie-breaking, so a later submission
need not receive the earlier submission's groups. Anonymous subject and
benchmark IDs are stable. Only an exact match of both IDs is paired evidence;
matching a benchmark ID with a different subject is not a paired comparison.

| Property | CodaBench submission | Paired local testing |
| --- | --- | --- |
| Groups | Platform draws groups for each submission | One frozen panel for every method |
| Evaluation pool | Hidden and fixed within a group | Hidden from the method and fixed in a manifest |
| Acquisition pool | Disjoint from evaluation | Same |
| Candidate access | Current candidate only for streaming hooks | Same in the primary protocol |
| Candidate order | Hidden fixed order | Several frozen shared orders |
| Budget trajectory | One nested trajectory | Same |
| Predictions | Hidden after submission | Stored row by row |
| Uncertainty | No paired interval in the feedback | Paired task bootstrap, plus benchmark-cluster sensitivity |
| Reproducibility | Private coordinator state | Snapshot, manifests, hashes, and event logs |

The local pipeline should support three named modes. Results from these modes
must not be mixed.

### 2.1 Competition replay

This mode reproduces the public CodaBench contract as closely as local data
allows:

- 50/50 split by distinct source item within a task;
- one revealed response for each acquired item;
- all response rows for evaluation items are scored;
- current candidate only;
- budgets 0, 1, 3, 7, 15, and 31;
- weights 0.1, 0.2, 0.2, 0.2, 0.2, and 0.1;
- group-level scoring followed by an unweighted mean across groups.

Use this mode to estimate submission behavior.

### 2.2 Paired research mode

This is the default for model and acquisition claims. It keeps the same online
information restrictions but uses:

- all eligible validation or test tasks, or one frozen stratified panel;
- several shared candidate-order seeds;
- paired comparisons on identical task and order keys;
- complete prediction and decision logs;
- task-level and benchmark-clustered uncertainty;
- separate model seeds.

Use this mode to decide whether one method is better than another.

### 2.3 Full-pool diagnostic mode

This mode lets a policy inspect every remaining acquisition candidate and take
the global argmax. It is useful for answering whether a score such as Fisher
information identifies useful items when the whole pool is visible. It is not
a faithful streaming result.

Name these arms explicitly, for example `fisher_full_pool`, and never compare
them in the same policy table as streaming Fisher without a mode column.

## 3. Units and terminology

Use one term for each level of the data.

| Term | Meaning |
| --- | --- |
| Benchmark | A dataset or evaluation collection. |
| Subject | A base model together with provider, harness, reasoning effort, version, and other evaluation settings. |
| Task or group | One subject and benchmark combination. This is the primary independent unit. |
| Source item | One distinct question or task before repeated response runs. |
| Input | The exact subject and item information passed to the predictor. |
| Response row | One observed binary outcome for an input. One input may have several response rows. |
| Acquisition pool | Items whose outcomes may be purchased. These items are never scored. |
| Evaluation pool | Items whose outcomes are scored and never revealed. |
| Candidate stream | A fixed order of acquisition items. |
| Budget | Maximum acquired labels available at a checkpoint. |
| Model seed | Randomness used to train a model. |
| Order seed | Randomness used only to order acquisition candidates. |

The task, not a response row and not a candidate-order replicate, is the main
statistical unit.

## 4. Prepare the data before building episodes

Online evaluation cannot repair a damaged response matrix. Data cleaning,
split construction, and online replay must be separate stages with separate
artifacts.

### 4.1 Freeze and identify the source snapshot

Record:

- source dataset version and retrieval time;
- hashes of every source table;
- Git commit of the preprocessing code;
- row counts before and after each filter;
- all exclusion reasons;
- schema and data types;
- the split seed and protocol version.

Never rebuild a named test panel silently from newer data. New data requires a
new snapshot and a new panel identifier.

### 4.2 Keep only valid observed binary outcomes

Accepted labels are exactly 0 and 1. Missing outcomes are absent observations.
They are never filled with zero, the subject mean, the benchmark mean, or a
model prediction.

Reject or quarantine:

- labels outside `{0, 1}`;
- non-finite labels;
- responses with unresolved subject or item identifiers;
- malformed item or subject records;
- response rows that cannot be assigned to a source snapshot.

Save the number and identity of removed rows.

### 4.3 Distinguish duplicates from repeated trials

These are different cases.

1. An exact duplicated database row is usually an ingestion error. Remove it
   after identifying the columns that define exact identity.
2. Several independently recorded outcomes for the same subject and item are
   repeated trials. Preserve them and assign them to the same pool.
3. Different item identifiers with identical normalized content are duplicate
   questions. Group them by a stable content hash before splitting.

For competition-compatible scoring, every preserved evaluation response row
contributes to Brier and ECE. The predictor returns one probability for an
identical input, and that probability is reused across its response rows.

Also report a unique-input-weighted sensitivity score. This shows whether a few
items with many trials dominate the row-weighted official score. Keep the
row-weighted score as the primary competition-replay result.

### 4.4 Prevent content leakage

Normalize content before hashing. The normalization policy should be fixed and
documented. At minimum, normalize Unicode, line endings, and surrounding
whitespace. Avoid aggressive transformations that merge genuinely different
questions.

All records with one content hash must stay in the same outer split. If the
same question appears in several benchmarks, benchmark splitting alone does
not prevent leakage. A conservative pipeline assigns the whole connected
component of benchmark IDs and content hashes to one split.

### 4.5 Apply training filters without peeking at test outcomes

The current data investigation found that 38 percent of items have only one
response and that the old fixed 128-item cap retained only 7.9 percent of
eligible rows. That setup removed much of the cross-subject overlap needed to
learn item difficulty. The details and plots are in
[the data-filtering decision log](README.md#data-filtering-changes-everything-2026-10-26-in-progress--not-yet-applied-to-any-submitted-model),
[model-results-visualization.pdf](docs/model-results-visualization.pdf), and
[item-passrate-distribution.pdf](docs/item-passrate-distribution.pdf).

For model fitting, track these filters separately:

- duplicate content removal;
- zero-variance items;
- minimum distinct subjects per item, with `n=15` and `n=30` as documented
  sensitivity settings rather than universal constants;
- minimum responses or items per subject and benchmark task;
- missing text and missing metadata indicators.

Outcome-dependent filters such as zero variance must not be recomputed from a
sealed final test set after model development starts. Either freeze the test
eligibility rule before development or leave such items in the evaluation set.
The training matrix may use stricter filters than the test panel, but the
difference must be stated.

Do not use a fixed first-128-items cap. Prefer full coverage. If compute forces
a cap, use a deterministic random cap, rotate coverage across epochs, record
the retained fraction per benchmark, and run a cap sensitivity analysis.

### 4.6 Keep evaluation questions separate

Different splits answer different questions:

- Split by complete benchmark to test transfer to unseen benchmarks.
- Split by complete subject to test transfer to unseen subjects.
- Split by content-hash group to test unseen-item transfer.
- Mask random observed cells only to test matrix interpolation.

The main online competition simulation should split by complete benchmark.
Random cell masking is a useful diagnostic, but it is not evidence of
unseen-benchmark transfer.

Use train data for model fitting, validation data for architecture,
hyperparameters, acquisition thresholds, and calibration, and final test data
for one selected comparison. If validation is repeatedly inspected, use outer
benchmark folds or keep a second sealed test panel.

## 5. Freeze a paired test panel

### 5.1 Define task eligibility

A competition-shaped task needs at least 80 distinct source items. A 50/50
split then provides about 40 acquisition items and 40 evaluation items, enough
for a maximum budget of 31 while retaining a useful scoring pool.

Eligibility must be deterministic. Store, for each task:

- task ID;
- subject ID;
- benchmark ID;
- number of distinct source items;
- number of unique inputs;
- number of response rows;
- acquisition and evaluation counts;
- split assignment;
- exclusion reason, if excluded.

Use all eligible tasks when practical. If a smaller panel is required, sample
it once before comparing methods. Stratify on information available without
test outcomes, such as benchmark identity and task size. Save the chosen IDs.

### 5.2 Split by source item

For each task, assign distinct source items to acquisition or evaluation using
a stable hash of:

```text
protocol_version, snapshot_id, task_id, item_uid, pool_seed
```

Assign all repeated trials, input variants, and exact aliases of that item to
the same pool. For odd item counts, define in advance which pool receives the
extra item. The current competition convention gives it to acquisition.

The evaluation pool must never be passed to the acquisition policy. Its labels
must remain inaccessible until scoring.

### 5.3 Freeze several candidate streams

Generate candidate orders with stable hashes of:

```text
protocol_version, task_id, item_uid, order_seed
```

Ten shared order seeds per task are a useful default for acquisition studies.
All policies must receive the same streams. Architecture studies that use
random acquisition can use fewer orders for speed, but the number must be
reported.

Do not regenerate candidate orders from a model seed. Training randomness and
stream randomness measure different things.

### 5.4 Persist the panel

A practical frozen panel contains:

```text
panel/
  manifest.json
  tasks.parquet
  episode_items.parquet
  evaluation_rows.parquet
  acquisition_labels.parquet
  streams.parquet
```

`manifest.json` should contain hashes of every table and the exact budgets,
weights, seeds, filtering rules, and source snapshot.

The current repository already has most of the required building blocks in
[data.py](src/aims_starter/data.py). Its `build_episode_index` assigns complete
items to pools, retains all evaluation response rows, and selects one
deterministic acquisition response per item. The manifest warns that this is a
local deterministic approximation of CodaBench's private split.

## 6. The online interface

### 6.1 Predictor

Use the submission-compatible interface:

```python
def predict(input: list, labeled: list | None = None) -> float:
    ...
```

The return value is a finite probability in `[0, 1]`. `labeled` contains only
accepted acquisition examples available at that checkpoint.

The predictor must be stateless with respect to prior calls. It may load frozen
weights and immutable caches, but it must reconstruct task-specific state from
`labeled`. This catches models that work in one process but fail when workers
restart between budgets.

### 6.2 Acquisition policy

Use the streaming interface:

```python
def acquisition_function(
    input: list,
    prediction: float | None,
    labeled: list | None,
    context: dict,
) -> bool:
    ...
```

The context should contain:

```python
{
    "subject_id": "subject_582731",
    "benchmark_id": "benchmark_274926",
    "labels_remaining": 31,
    "labels_acquired": 0,
    "max_labels": 31,
    "items_remaining": 40,
}
```

`items_remaining` includes the current candidate. The policy returns `True` to
purchase the current label and `False` to reject the candidate permanently.
The outcome is added to `labeled` only after a `True` decision.

The public coordinator implementation is in
[streaming_ingestion.py](reference/starting_kit/tools/streaming_ingestion.py),
and the full interface is described in the
[starting-kit README](reference/starting_kit/README.md#streaming-test-time-adaptation).

### 6.3 One faithful trajectory

The coordinator should follow this order:

1. Score budget 0 with an empty context.
2. Present acquisition candidates one at a time.
3. Ask the policy to accept or reject the current candidate.
4. Reveal its outcome only after acceptance.
5. When the accepted-label count reaches 1, freeze the context and score the
   evaluation pool.
6. Continue the same stream to counts 3, 7, 15, and 31.
7. Commit each checkpoint before revealing more labels.

The contexts are nested:

```text
C_0  = {}
C_1  = {selected label 1}
C_3  = C_1  plus selected labels 2 and 3
C_7  = C_3  plus selected labels 4 through 7
C_15 = C_7  plus selected labels 8 through 15
C_31 = C_15 plus selected labels 16 through 31
```

Never rerun acquisition independently for each budget. That produces six
different, non-nested tests and gives a policy several chances to exploit the
same candidate pool.

### 6.4 Streaming pseudocode

```python
budgets = [0, 1, 3, 7, 15, 31]
selected = []
score_checkpoint(task, budget=0, labeled=selected)

for position, candidate in enumerate(frozen_stream):
    if len(selected) == 31:
        break

    prediction = predictor.predict(candidate.input, selected)
    context = {
        "subject_id": task.subject_id,
        "benchmark_id": task.benchmark_id,
        "labels_acquired": len(selected),
        "labels_remaining": 31 - len(selected),
        "max_labels": 31,
        "items_remaining": len(frozen_stream) - position,
    }
    query = policy.decide(candidate.input, prediction, selected, context)
    log_decision(candidate, prediction, context, query)

    if query:
        selected.append((candidate.input, hidden_acquisition_label(candidate)))
        if len(selected) in {1, 3, 7, 15, 31}:
            score_checkpoint(task, budget=len(selected), labeled=selected.copy())

for missing_budget in unscored_budgets:
    score_checkpoint(task, budget=missing_budget, labeled=selected.copy())
```

The final loop handles policies that exhaust the stream before reaching a cap.
Store both the nominal budget and `labels_acquired`.

### 6.5 Checkpoint barriers and cross-task evidence

CodaBench completes a checkpoint for all tasks before any task advances to a
higher budget. Reproduce that barrier in competition-replay mode.

For clean task-level research, pass only evidence from the current subject and
benchmark to the predictor. CodaBench supplies the accumulated evidence list
across sampled groups, so submission code must filter it correctly. Cross-task
online learning is a separate method. If tested, declare it as its own arm and
preserve the global task schedule because results then depend on task order.

## 7. Online model updates

Acquisition and adaptation are separate components. Do not change both in one
ablation and call the difference an acquisition gain.

### 7.1 No-update control

Ignore all acquired labels. The curve should be flat except for model
randomness. This checks the coordinator and scoring code.

### 7.2 Beta-Binomial update

With prior `Beta(alpha, beta)`, successes `s`, and `n` acquired labels:

```text
p = (alpha + s) / (alpha + beta + n)
```

The current baseline uses `alpha=beta=2`, so `p=(2+s)/(4+n)`. This is a strong
control for small budgets because it cannot overreact as sharply as the raw
empirical mean.

### 7.3 MAP ability update

For fixed item parameters, recompute ability from the complete acquired
context:

```text
theta_hat = argmax_theta [log p(theta)
                          + sum over acquired items log p(y_i | theta, item_i)]
```

Do this from scratch or from a deterministic initializer at every checkpoint.
The previous process's in-memory `theta` must not be required. Log convergence,
gradient norm, Hessian condition, clamp hits, and the final objective.

### 7.4 Posterior-predictive update

A MAP point discards uncertainty. A posterior-predictive arm integrates over
the ability posterior:

```text
p(y=1 | x, C) = integral p(y=1 | x, theta) p(theta | C) d theta
```

Use quadrature, Laplace sampling, or another predeclared approximation. Compare
it with MAP on the same contexts and streams.

### 7.5 Joint local updates

If the model updates both ability and a task-local difficulty correction, the
correction must have a prior and must reset for every task. Do not update the
global item network on final-test labels. Log the separate movement in ability
and difficulty so a flexible local adapter cannot silently absorb everything.

### 7.6 Gradient updates

Gradient-based test-time training is allowed only as a separate declared arm.
Reset model parameters to the frozen checkpoint for every task and stream.
Never carry gradients, optimizer state, or adapted weights into another task.

## 8. Acquisition policies

Every acquisition study should include a random baseline and should first hold
the predictor, calibration, tasks, pools, and streams fixed.

### 8.1 Random

Random selection is the primary acquisition baseline. A deterministic
implementation can hash the current task, stream seed, and candidate, or query
with probability:

```text
min(1, labels_remaining / items_remaining)
```

using a fixed per-candidate uniform value. This samples without replacement
and fills the budget when enough candidates remain.

### 8.2 Predictive uncertainty

For binary outcomes, score a candidate by closeness to 0.5:

```text
score_uncertainty = -abs(p - 0.5)
```

Binary entropy is a monotone transformation of the same quantity. With matched
thresholds, it is not a distinct policy.

### 8.3 Fisher information

For scalar 2PL IRT with discrimination `a_i` and current success probability
`p_i`:

```text
Fisher_i = a_i^2 * p_i * (1 - p_i)
```

For multidimensional ability with posterior covariance `Sigma`:

```text
Fisher-D_i = p_i * (1 - p_i) * a_i^T Sigma a_i
```

This is the rank-one D-optimal information gain. The implementation and its
derivation are in [modal_mirt_acquisition.py](modal_mirt_acquisition.py).

The existing MIRT experiment takes an argmax over all remaining candidates.
That is a full-pool diagnostic. A streaming Fisher policy must score only the
current candidate and accept it using a threshold, a validation-fitted
position-dependent percentile, or an online quantile based only on scores seen
so far.

### 8.4 Disagreement

Score candidates where two predictors disagree:

```text
score_disagreement = abs(p_model - p_reference)
```

The reference can be Beta-Binomial, another model family, or a posterior mean.
Both probabilities must be computed from the same currently available context.

### 8.5 Diversity and coverage

A streaming max-min policy can score the distance from the current candidate's
embedding to the nearest already selected embedding. It may use selected
candidates, but not unseen future candidates. Per-task k-means over the full
pool is not streaming-deployable.

### 8.6 Expected information gain

An expected-information policy estimates the expected reduction in posterior
entropy or target predictive loss under both possible labels. It must use only
the current candidate, current posterior, and frozen model. The candidate's
real outcome cannot enter the estimate.

### 8.7 Oracle policies

An oracle may use hidden outcomes or the complete candidate pool only as an
upper-bound diagnostic. Prefix its name with `oracle_`, exclude it from
deployable rankings, and never use it to tune a submitted policy.

### 8.8 Threshold and reserve logic

A streaming score needs a decision rule. Fit thresholds on validation tasks,
not final test tasks. A simple policy is:

```python
must_query = items_remaining <= labels_remaining
query = must_query or score >= threshold
```

This guarantees that the policy can fill the budget. It also creates a known
failure mode: a threshold that almost never fires becomes a late-stream reserve
policy. Always log:

- number and fraction selected by the score condition;
- number and fraction forced by reserve;
- stream position of every selected candidate;
- mean and median selected position;
- score distribution for accepted and rejected candidates;
- actual labels acquired at every checkpoint.

If most selections are forced, report that the acquisition rule was inactive.
Do not attribute the resulting curve to Fisher, uncertainty, or disagreement.

## 9. Calibration

Calibration affects both evaluation and acquisition when a policy uses
predicted probabilities.

1. Fit calibration on training or cross-fitted validation predictions only.
2. Fit budget-specific calibration only if every budget has enough held-out
   tasks.
3. Score each validation task with a calibrator that did not see that task.
4. Freeze the final calibrator before final-test replay.
5. Store raw and calibrated probabilities.
6. Record the actual acquired-label count used to select a calibration entry.
7. Never fit a calibrator on final-test outcomes.

For a policy-only ablation, use one fixed predictor and one fixed calibrator for
all policies. Fitting a separate calibrator per policy measures the complete
policy and calibration system, not acquisition alone. Both views can be useful,
but they belong in separate tables.

The binding local ablation rules are in
[docs/ablation-protocol.md](docs/ablation-protocol.md). They require five-fold
out-of-fold validation calibration and exclude final test from ablation grids.

## 10. Metric definitions

All metric calculations use unrounded probabilities. Round only when rendering
a report.

Let `g` identify a subject and benchmark task, `b` a label budget, and `j` a
scored evaluation response row. Let `y_gbj` be its binary outcome and `p_gbj`
the committed probability.

### 10.1 Brier score for one response

```text
Brier(y, p) = (p - y)^2
```

It ranges from 0 to 1. Lower is better. A constant probability of 0.5 has
Brier 0.25 for every binary outcome.

### 10.2 Group Brier at one budget

For `N_g` evaluation response rows:

```text
B_g,b = (1 / N_g) * sum_j (p_g,b,j - y_g,j)^2
```

The evaluation rows are fixed across budgets. If an input has repeated
response trials, the same probability is reused and every preserved response
row contributes to the mean.

Example:

```text
y = [1, 0]
p = [0.75, 0.25]
Brier = ((0.75 - 1)^2 + (0.25 - 0)^2) / 2 = 0.0625
```

### 10.3 Per-budget total Brier

If there are `G` tasks:

```text
B_b,total = (1 / G) * sum_g B_g,b
```

Every task has equal weight. Do not concatenate all response rows and compute
one global mean, because large tasks would then dominate.

Also report the number of tasks, unique inputs, and response rows. The
CodaBench log column `Number of Unique Subject-Item Pairs` is not necessarily
the number of response rows used by the scorer.

### 10.4 Expected calibration error for one group and budget

Use ten fixed equal-width bins:

```text
[0.0, 0.1), [0.1, 0.2), ..., [0.8, 0.9), [0.9, 1.0]
```

A probability exactly on an interior edge enters the upper bin. A probability
of exactly 1.0 enters the final bin.

For each nonempty bin `k`:

```text
confidence_k = mean predicted probability in bin k
accuracy_k   = mean observed outcome in bin k
weight_k     = number of rows in bin k / N_g
```

Then:

```text
ECE_g,b = sum_k weight_k * abs(confidence_k - accuracy_k)
```

Compute ECE within each task first. Do not pool predictions from tasks with
different base rates and call that the official ECE.

ECE is unstable for small groups and depends on binning. Keep it as a
diagnostic. Brier is the primary proper scoring rule. Reliability diagrams,
the Brier decomposition, and a pooled ECE sensitivity view can be added, but
they do not replace the specified group ECE.

The per-budget total ECE is the unweighted task mean:

```text
ECE_b,total = (1 / G) * sum_g ECE_g,b
```

This is different from placing every task's predictions into one shared set of
bins. The shared-bin calculation may be reported as a separate pooled
diagnostic, but it is not the task-balanced score.

### 10.5 Brier ALC for one group

The budgets are:

```text
b = [0, 1, 3, 7, 15, 31]
```

After transforming the horizontal axis with `x=log2(1+b)`, the locations are
`[0, 1, 2, 3, 4, 5]`. A trapezoidal area on this equally spaced axis gives
half weight to the endpoints and full weight to the interior points. After
normalization, the weights are:

```text
w = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
```

The group Brier ALC is:

```text
Brier_ALC_g = 0.1 * B_g,0
            + 0.2 * B_g,1
            + 0.2 * B_g,3
            + 0.2 * B_g,7
            + 0.2 * B_g,15
            + 0.1 * B_g,31
```

The weighted group calibration summary is calculated the same way:

```text
ECE_ALC_g = sum_b w_b * ECE_g,b
```

### 10.6 Overall Brier ALC and ECE

Average group curves equally:

```text
Brier_ALC_total = (1 / G) * sum_g Brier_ALC_g
ECE_ALC_total   = (1 / G) * sum_g ECE_ALC_g
```

Equivalently:

```text
Brier_ALC_total = sum_b w_b * B_b,total
```

This equality is a useful scoring test. It holds because both operations are
linear. ECE itself is nonlinear within a bin, so calculate group ECE first and
only then average the group values.

### 10.7 Worked example from a real log

In the [submission 1005155 scoring log](results/codabench-1005155/scoring_stdout.txt),
`subject_120099` and `benchmark_869474` have these Brier scores:

```text
budget:  0         1         3         7         15        31
Brier:   0.250000  0.257967  0.268184  0.248174  0.250314  0.251654
weight:  0.1       0.2       0.2       0.2       0.2       0.1
```

The weighted sum is `0.255093`, which is the group's Brier ALC in the summary.
Applying the same weights to its six ECE values gives `0.071067`, the summary
ECE. The overall submission score is the unweighted mean of the nine group
ALCs, `0.148605`.

The metric implementation and unit tests are in
[metrics.py](src/aims_starter/metrics.py) and
[test_metrics.py](tests/test_metrics.py).

## 11. Aggregation and uncertainty

### 11.1 Preserve the hierarchy

For each model seed and acquisition policy:

1. Compute metrics for every task, stream seed, and budget.
2. Compute ALC within each task and stream.
3. Average stream seeds within each task.
4. Compare methods by paired task-level differences.
5. Average tasks equally for the headline result.
6. Report model-seed results separately, then their mean and spread.

Do not treat ten streams for one task as ten independent tasks. Do not report
seed standard deviation as a task-level standard error.

With `S_g` stream orders for task `g`, the paired-research per-budget total is:

```text
B_b,total = (1 / G) * sum_g [(1 / S_g) * sum_s B_g,s,b]
```

Use the same order of operations for ECE and ALC: average repeated stream
measurements within a task, then average tasks.

### 11.2 Paired differences

For policy `A` and baseline `R` on task `g` and stream `s`:

```text
delta_g,s = Brier_ALC_g,s,A - Brier_ALC_g,s,R
```

Negative is better. Average `delta_g,s` over streams to get one `delta_g` per
task. Report:

- mean paired delta;
- 95 percent paired task-bootstrap interval;
- task win, tie, and loss counts;
- median task delta;
- budget-specific paired deltas.

As a sensitivity check, resample benchmarks and include all their tasks. This
benchmark-cluster bootstrap can be wide when only a few benchmarks are
available, which is itself useful information.

### 11.3 Model seeds

Use the same tasks, pools, and streams for every model seed. Three model seeds
are preferable when training is stochastic. Report each seed. If only two are
available, call the difference a two-seed spread, not a stable variance
estimate.

### 11.4 Multiple comparisons

Choose one primary baseline and a small predeclared set of policy comparisons.
Large policy grids will produce apparent winners by chance. Use validation to
select a policy, then make one final paired comparison on the sealed test
panel.

## 12. Logging and artifacts

A result should be reproducible without rerunning the model.

### 12.1 Run manifest

Store:

- run ID and timestamp;
- protocol version;
- data snapshot and panel hashes;
- Git commit and dirty-worktree state;
- model name, checkpoint hash, model seed, and configuration;
- calibration artifact hash;
- policy name, threshold source, and configuration;
- budgets, weights, pool seed, and stream seeds;
- software and hardware versions;
- start, end, runtime, and failure status.

### 12.2 Acquisition event log

Write one immutable row per candidate event:

```text
run_id
task_id
benchmark_id
subject_id
stream_seed
candidate_position
item_uid
labels_before
items_remaining
labels_remaining
candidate_prediction_raw
candidate_prediction_calibrated
acquisition_score
threshold
decision
decision_reason        # score, random, reserve, budget_full
revealed_label         # null when rejected
context_hash_before
context_hash_after
```

Never write a rejected candidate's hidden label. That is a useful audit test.

### 12.3 Prediction log

Write one row per scored response row and checkpoint:

```text
run_id
model_seed
policy
task_id
benchmark_id
subject_id
stream_seed
budget
labels_acquired
evaluation_input_uid
evaluation_item_uid
response_row_uid
outcome
probability_raw
probability_calibrated
context_hash
prediction_runtime_ms
```

If repeated response rows share one input, retain one prediction ID and link
all scored rows to it.

### 12.4 Metric tables

Produce at least:

```text
metrics_by_task_budget.parquet
metrics_by_task_alc.parquet
metrics_by_budget.csv
paired_deltas.csv
summary.json
```

`metrics_by_task_budget` should include Brier, ECE, unique inputs, response
rows, nominal budget, and actual labels acquired. `summary.json` should be
generated from those rows, not maintained by hand.

### 12.5 Human-readable scoring log

Mirror the CodaBench structure:

1. ALC summary with one row per task.
2. One section for every budget.
3. Aggregate means after each table.
4. Notes for tasks that acquired fewer labels than the nominal cap.
5. Protocol and artifact hashes at the top.

The repository parser in
[plot_codabench_scoring.py](scripts/plot_codabench_scoring.py) demonstrates how
the existing text format is read and checked.

## 13. Required validation checks

Fail the run rather than warn when an invariant is violated.

### 13.1 Data and split checks

- No response row appears in more than one outer split.
- No content-hash group crosses an outer split.
- No source item appears in both acquisition and evaluation pools.
- All repeated trials of one item remain in one pool.
- Every task satisfies the declared eligibility rule.
- Missing outcomes are absent, not imputed.
- Every stream contains each acquisition item exactly once.
- Evaluation rows are identical across methods and budgets.

### 13.2 Online-state checks

- Budget contexts are nested prefixes.
- Context size never exceeds the nominal budget.
- Rejected labels remain unavailable.
- Future candidate features are unavailable in streaming mode.
- Evaluation outcomes are inaccessible until scoring.
- Predictor output is finite and in `[0, 1]`.
- Repeated identical inputs receive one reused probability per checkpoint.
- Restarting the prediction worker does not change deterministic output.
- Every task is reset before a new stream seed.
- No gradient or optimizer state crosses tasks unless the protocol explicitly
  tests cross-task adaptation.

### 13.3 Comparison checks

- Every compared arm has identical task and stream keys.
- Every arm using the same predictor has identical budget-0 predictions unless
  calibration or random model inference is intentionally different.
- Every acquisition arm uses the same candidate stream within a task and seed.
- ALC recomputed from budget scores matches stored task ALC.
- Overall ALC computed from task ALC matches ALC computed from aggregate budget
  means.
- Raw prediction tables reproduce the published summary exactly.
- Test outcomes did not fit a model, threshold, prior, or calibrator.

## 14. Recommended experiment sequence

Do not start with a large end-to-end grid. Build evidence in this order.

### Stage 1: Validate the pipeline

Run:

- constant 0.5 predictor;
- no-update predictor;
- Beta-Binomial `Beta(2,2)`;
- raw empirical mean.

Expected checks:

- constant 0.5 has Brier 0.25 at every budget;
- no-update curves are flat;
- every policy has identical budget-0 results for a fixed predictor;
- the empirical mean can overreact at one label;
- ALC identities hold to numerical precision.

### Stage 2: Compare predictors under random acquisition

Use the same random streams for Beta-Binomial, MAP IRT, posterior-predictive
IRT, MIRT, CNP, ANP, or other models. This isolates prediction and online
updating from acquisition.

Report raw and calibrated curves. Inspect budget 0 separately because it tests
offline transfer before any labels are available.

### Stage 3: Compare acquisition policies with one frozen predictor

Compare random, uncertainty, streaming Fisher, disagreement, and diversity on
the same predictor and calibration. Use several shared streams and report
forced-reserve fractions.

### Stage 4: Compare full systems

Only after Stages 2 and 3 should each model use its preferred update,
calibration, and acquisition policy. Call this an end-to-end comparison, not an
acquisition ablation.

### Stage 5: Read the sealed test panel once

Choose the final system on validation. Evaluate that system and the predeclared
baseline on the final test panel. Keep exploratory policy grids off the final
test set.

## 15. Known failure modes from this project

### Unpaired CodaBench scores

Different submissions saw different groups. Aggregate gaps cannot identify the
cause. Fix this with one frozen panel and paired deltas.

### Full-pool Fisher presented as streaming Fisher

Taking the global argmax after viewing all remaining candidate features solves
an easier problem. Keep it as a diagnostic and add a current-candidate
threshold replay for deployment evidence.

### Threshold policies that reduce to reserve

If the threshold rarely fires, almost every label is forced near the end of the
stream. Log decision reasons and forced fractions.

### Non-nested budgets

Independent acquisition runs at each budget can make a method look better and
inflate compute. Use one trajectory and prefixes.

### Context and target overlap

Training on target items that include revealed context items while evaluating
on disjoint targets creates a train-test mismatch. Keep episodic training
context and target items disjoint for the faithful configuration.

### Fixed 128-item training cap

A fixed cap can remove most rows from large benchmarks and destroy the overlap
needed to estimate item difficulty. Use full coverage or rotating deterministic
subsamples.

### Duplicate questions across splits

Benchmark-level splitting does not catch the same text appearing under several
benchmark or item IDs. Group by normalized content hash before splitting.

### Calibration leakage

Fitting budget calibrators on the same tasks being scored gives optimistic
results. Use out-of-fold validation calibration and a frozen final artifact.

### Pseudo-replication

Ten streams from one task are repeated measurements, not ten tasks. Average
streams within task before uncertainty estimation.

### Global pooled scoring

Pooling all response rows weights large tasks more heavily than small tasks.
Compute each task's metric first and average tasks equally.

### ECE overinterpretation

Ten-bin ECE from 40 to 70 inputs is noisy. Use it as a calibration diagnostic,
not the model-selection target.

## 16. Existing examples and source files

### Real CodaBench logs

| Submission | Method | Brier ALC | Useful context |
| --- | --- | ---: | --- |
| [1005155](results/codabench-1005155/scoring_stdout.txt) | Two-stage Beta-Binomial shrinkage | 0.148605 | Smooth improvement from budget 0 to 31; worked ALC example above. |
| [1005187](results/codabench-1005187/scoring_stdout.txt) | CNP-AIRT, random acquisition | 0.186848 | Neural predictor with random acquisition. |
| [1005210](results/codabench-1005210/scoring_stdout.txt) | Online-AIRT, active acquisition | 0.201107 | MAP-style online adaptation with the active policy. |
| [1005225](results/codabench-1005225/scoring_stdout.txt) | CNP-AIRT, active acquisition | 0.214205 | Same model family as 1005187, but mostly different groups, so not a clean policy comparison. |
| [1005650](results/codabench-1005650/scoring_stdout.txt) | Unsmoothed empirical mean | 0.219004 | Shows one-label overreaction: aggregate Brier rises from 0.2500 to 0.3076. |
| [1005839](results/codabench-1005839/scoring_stdout.txt) | Near-flat budget shrinkage | 0.185556 | Example of validation-selected symmetric shrinkage. |

These logs contain a task-level ALC table followed by one table per budget.
They are good report-format examples, but their aggregate scores are not paired
ablation evidence.

### Protocol and implementation references

- [Starting-kit protocol](reference/starting_kit/README.md)
- [Streaming coordinator client](reference/starting_kit/tools/streaming_ingestion.py)
- [Canonical metric functions](src/aims_starter/metrics.py)
- [Metric tests](tests/test_metrics.py)
- [Episode and stream construction](src/aims_starter/data.py)
- [Simple competition-shaped simulator](src/aims_starter/evaluation.py)
- [Evaluation tests](tests/test_evaluation.py)
- [Binding ablation protocol](docs/ablation-protocol.md)
- [Streaming threshold and reserve policy](organized/submission/legacy/submission/labeling.py)
- [Acquisition-arm definitions](organized/submission/legacy/submission_variants/README.md)
- [Older streaming active ablation](modal_active_ablation.py)
- [Full-pool MIRT acquisition diagnostics](modal_mirt_acquisition.py)
- [CodaBench scoring-log parser](scripts/plot_codabench_scoring.py)

## 17. Suggested implementation modules

A clean implementation can use these boundaries:

```text
online_testing/
  schema.py             # validated records and IDs
  snapshot.py           # source hashes and manifests
  filtering.py          # explicit data-cleaning stages
  splits.py             # benchmark, subject, and content grouping
  panel.py              # task eligibility and fixed panel
  episodes.py           # acquisition/evaluation pools
  streams.py            # shared order seeds
  coordinator.py        # sequential reveal and checkpoint barriers
  predictors.py         # adapter around predict(input, labeled)
  policies.py           # random, uncertainty, Fisher, disagreement, diversity
  calibration.py        # out-of-fold fit and frozen transforms
  metrics.py            # Brier, ECE, ALC, aggregation
  statistics.py         # paired bootstrap and cluster sensitivity
  logging.py            # events, predictions, manifests
  report.py             # tables, curves, and score reproduction
  tests/
```

Keep the coordinator independent of model code. A policy should not receive a
dataframe containing hidden labels merely because it promises not to inspect
the label column. Pass a sanitized candidate object instead.

## 18. Definition of done

The pipeline is ready for scientific comparisons when all of the following are
true:

- A frozen panel can be replayed twice with byte-identical deterministic logs.
- Every model sees the same tasks, evaluation rows, pools, and stream orders.
- Streaming policies cannot inspect future candidates or rejected labels.
- One trajectory produces nested contexts at all six budgets.
- Worker restarts do not change predictions.
- Raw predictions reproduce task, budget, and total metrics exactly.
- Constant, no-update, Beta-Binomial, and empirical-mean controls pass.
- Policy tables use paired task differences against random acquisition.
- Threshold and reserve decisions are distinguishable in the logs.
- Calibration is out of fold on validation and frozen on test.
- Final-test outcomes cannot affect filtering, selection, calibration, or
  policy thresholds.
- The report states the snapshot, task count, order count, budgets, baseline,
  model seeds, uncertainty unit, and whether the comparison is paired.

Meeting these conditions gives us something CodaBench alone cannot provide: a
repeatable explanation of why one online model or acquisition policy wins.
