# Offline benchmark-holdout replay

This is a local evaluation protocol, not a CodaBench submission. It makes
checkpoint and policy comparisons paired: every method sees the same filtered
tasks, acquisition/evaluation partition, and one-at-a-time candidate orders.

The panel is created from the filtered benchmark-holdout split:

- training benchmarks are never included;
- validation benchmarks are for choosing a checkpoint, shrinkage weight, or
  acquisition policy;
- test benchmarks are for one final, frozen choice;
- each subject-benchmark task has disjoint acquisition and evaluation items;
- a complete task is selected only if it fits within a 1,000-source-item cap;
- the acquisition stream is fixed before a model or policy runs;
- scores use nested budgets `0, 1, 3, 7, 15, 31`, Brier score, ECE, and the
  competition ALC weights `(0.1, 0.2, 0.2, 0.2, 0.2, 0.1)`.

Candidate selection receives one current candidate, predictions based on
previously accepted labels, and remaining counts. It cannot inspect a future
candidate or a rejected candidate's label. Evaluation labels are accessed only
after predictions for that budget have been made.

## Create the frozen panel

The panel is written once to the `aims-neurips-data` volume. A second run with
the same parameters reuses it rather than replacing it.

```bash
.venv/bin/modal run --profile aims-test \
  organized/testing/online/modal_benchmark_holdout_replay.py::main \
  --action build-panel
```

The default panel name is:

```text
benchmark-holdout-online-v1-s20261018-n15-m80-cap1000
```

Its manifest and frozen parquet files are under:

```text
/data/results/online-replay/panels/<panel-name>/
```

`manifest.json` records the data-filter audit, split seed, selected task
counts, source-item cap, order seeds, hashes of all frozen parquet files, and
the scoring protocol.

## Compare methods on validation first

Start with the no-model Beta-binomial baseline. It verifies panel loading,
nested prefixes, and scoring without a checkpoint.

```bash
.venv/bin/modal run --profile aims-test \
  organized/testing/online/modal_benchmark_holdout_replay.py::main \
  --action replay \
  --panel-name benchmark-holdout-online-v1-s20261018-n15-m80-cap1000 \
  --role validation \
  --prediction-mode beta_binomial \
  --policy random_prefix \
  --run-name validation-beta-random-v1
```

Run the filtered benchmark-holdout ANP checkpoint against exactly the same
panel and random stream:

```bash
.venv/bin/modal run --profile aims-test \
  organized/testing/online/modal_benchmark_holdout_replay.py::main \
  --action replay \
  --panel-name benchmark-holdout-online-v1-s20261018-n15-m80-cap1000 \
  --checkpoint anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029-bge_m3_long-seed20261018.pt \
  --role validation \
  --prediction-mode anp \
  --policy random_prefix \
  --run-name validation-anp-filtered-random-v1
```

For an ANP/Beta interpolation, set `--prediction-mode anp_beta_shrinkage` and
choose `--shrinkage-weight` on validation only. The default `--neutral-at-zero`
returns 0.5 before any acquired label, so a shrinkage sweep cannot silently
change the budget-zero score.

The uncertainty policy uses the current candidate's probability only. It has a
reserve rule, so it accepts when the number of candidates left equals the
number of labels still needed to reach 31:

```bash
.venv/bin/modal run --profile aims-test \
  organized/testing/online/modal_benchmark_holdout_replay.py::main \
  --action replay \
  --panel-name benchmark-holdout-online-v1-s20261018-n15-m80-cap1000 \
  --checkpoint anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029-bge_m3_long-seed20261018.pt \
  --role validation \
  --prediction-mode anp \
  --policy uncertainty \
  --uncertainty-threshold 0.15 \
  --run-name validation-anp-uncertainty-t015-v1
```

`confidence_extremes` is a streaming control that selects current candidates
far from 0.5 with the same reserve rule. It tests whether any gain from
uncertainty acquisition is specific to uncertain labels rather than merely a
consequence of rejecting candidates.

### Rank-based policies and expected Brier reduction

`bernoulli_variance` scores the current candidate by
\(p(1-p)\). `expected_brier_reduction` (EBR) instead asks how much buying
that candidate's label would change predictions on a fixed train-only reference
set:

\[
\operatorname{EBR}(x)=
\sum_{y\in\{0,1\}} p(y\mid x,C)
\frac{1}{|R|}\sum_{r\in R}
\left[p(r\mid C\cup\{(x,y)\})-p(r\mid C)\right]^2.
\]

Both policies compare the current score with scores of train-only probe items
under the same context, then apply the reserve rule. They do not inspect later
candidates. EBR is therefore faithful to the streaming information constraint,
but it is not the true expected Brier reduction on hidden evaluation items.
Using evaluation items as \(R\) would leak the scored distribution into the
policy.

EBR is expensive because each candidate requires predictions under both
possible labels. The current implementation uses 32 reference items and 32
calibration probes. It is appropriate for a small controlled replay, not yet
for a CodaBench submission.

## Final test replay

Change only `--role test` after freezing the method on validation. Do not use
test output to choose a threshold, checkpoint, or model family.

Each replay creates a new directory under:

```text
/data/results/online-replay/runs/<panel-name>/<run-name>/
```

It includes `summary.json`, per-task and per-benchmark Brier/ECE scores,
macro-task and micro-response summaries, per-task ALC, every predicted
evaluation probability, and every acquisition decision. Accepted labels are
recorded after acceptance. Rejected labels remain null.

The code is split into:

- `organized/testing/online/benchmark_holdout_replay.py`: deterministic panel,
  metric, and policy helpers. It is safe to unit-test locally.
- `organized/testing/online/modal_benchmark_holdout_replay.py`: CPU-only Modal
  panel creation and checkpoint replay against the volume.

## TabPFN-v2 model × acquisition-policy baseline

`organized/testing/online/modal_tabpfn_v2_benchmark_holdout_replay.py` adds
the off-the-shelf **TabPFN-v2** classifier to this exact panel. It has no
task-specific training: at each acquisition decision it may use only the
labels previously accepted for that subject×benchmark task.

The BGE-M3 item vectors are 1,024-D, while TabPFN-v2 is pre-trained for at
most 500 feature columns. To avoid silently operating outside that limit, the
runner first fits a PCA-500 transform on **unique item vectors from the
filtered training benchmarks only**. It uses no labels, and the frozen
transform is then applied unchanged to validation/test candidates.

Prepare that transform once:

```bash
.venv/bin/modal run --profile aims-test \
  organized/testing/online/modal_tabpfn_v2_benchmark_holdout_replay.py::main \
  --action prepare-transform
```

Then run all three policy cells on validation. The contexts contain at most 31
labels, so this TabPFN-v2 replay is CPU-only; the PCA preparation is CPU-only
as well:

```bash
for policy in random_prefix uncertainty confidence_extremes; do
  .venv/bin/modal run --detach --profile aims-test \
    organized/testing/online/modal_tabpfn_v2_benchmark_holdout_replay.py::main \
    --action replay --policy "$policy" \
    --run-name "validation-tabpfn-v2-pca500-${policy}-v1"
done
```

TabPFN cannot establish a binary class mapping before both labels have been
observed. At those contexts the runner records a symmetric Beta(2,2) fallback
instead of pretending a TabPFN prediction exists. The replay summary reports
how often this fallback occurred, the number of model fits, and all per-budget
Brier/ECE values. The existing plotting script reads the registered
`model_id`/`model_label` fields, so after the three runs it will include TabPFN
as a fourth model panel automatically:

```bash
.venv/bin/python organized/utils/plot_benchmark_holdout_online_replay.py \
  --runs-root /tmp/aims-online-replay-XXXXXX/benchmark-holdout-online-v1-s20261018-n15-m80-cap1000 \
  --output docs/online-replay-acquisition-validation.pdf
```

Use the validation cells to choose one policy/threshold. Only then run the
single corresponding `--role test` cell; do not select it from test results.
