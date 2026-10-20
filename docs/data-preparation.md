# Fast data preparation

Use one prepared dataset for every baseline, IRT model, and Neural Process
variant. Do not let individual experiments parse the raw Hugging Face files.

## 1. Install the data tools

```bash
python -m pip install -e '.[data]'
```

## 2. Start with a two-benchmark pilot

The lean downloader skips traces and embeddings and chooses the two smallest
eligible benchmarks:

```bash
python scripts/download_measurement_tables.py \
  --out data/raw/measurement-db-pilot \
  --limit 2

python scripts/prepare_measurement_data.py \
  --source data/raw/measurement-db-pilot \
  --out data/processed/measurement-db-pilot
```

Use this pilot to debug every model's training and evaluation loop.

### Run the same pilot on Modal

The Modal pipeline uses the persistent `aims-neurips-data` Volume. It requires
a workspace Secret named `huggingface-secret` containing `HF_TOKEN`:

```bash
modal run modal_ingest.py --dataset-name pilot --limit 2
modal run modal_app.py --action inspect
```

The remote paths are `/data/datasets/pilot/raw` and
`/data/datasets/pilot/processed`. The download manifest records the exact Hub
revision; the processing manifest records row counts, null counts, skipped
benchmarks, and the split policy.

### Current Modal snapshot (17 October 2026)

The first full table-only snapshot completed successfully on Modal and is
pinned to Measurement Data Bank revision
`54074453da7b9911be0f48e4c560be547bc84d21`.

This snapshot intentionally applies a **temporary benchmark-level minimum of
80 items** to keep the first training and evaluation iteration focused and
fast. The official competition rule is narrower: the 80-distinct-item
eligibility requirement applies to each evaluated **subject--benchmark pair**,
not to every public benchmark that may be used for training. Therefore:

- The current snapshot is sufficient for the first competition-shaped
  experiments.
- Reported offline evaluation must still enforce at least 80 observed distinct
  items separately for every subject--benchmark pair.
- If additional training diversity is useful, create a second, non-overwriting
  corpus containing all binary item-level benchmarks, including benchmarks
  with fewer than 80 total items. Those smaller tasks may support item
  representations, subject priors, or meta-learning, but they must not enter
  the official-style Brier ALC evaluation unless the pair itself is eligible.

The current snapshot should not be described as the complete Measurement Data
Bank or as the complete set of binary item-level training benchmarks.

| Artifact | Count or size |
|---|---:|
| Downloaded benchmark directories | 55 |
| Usable binary item-level benchmarks | 54 |
| Items | 1,356,312 |
| Observed response rows | 16,278,271 |
| Distinct prediction inputs | 11,802,108 |
| Subject--benchmark tasks | 8,046 |
| Raw tables | 794,312,301 bytes |
| Processed tables | 2,372,545,618 bytes |

The corresponding public trace inventory contains files for 54 of these 55
downloaded benchmark directories (`agentic_review_perturbation` has no trace
file) and occupies approximately 21.66 GiB compressed. Traces are persisted in
a separate resumable store under `/data/datasets/full/traces`; they are not
joined into the canonical observation table. They may support offline subject
and failure-mode representation learning, but hidden Codabench traces are not
available at test time.

`chartmuseum` was retained in the immutable raw snapshot but excluded from the
processed corpus because all 162 response values were missing. Fifteen other
benchmarks contained some missing response values. The processor drops only
those unobserved rows and records their counts under `dropped_observations` in
the manifest; observed non-binary values still make a benchmark ineligible.

The initial smoothed Beta-Binomial control on the fixed benchmark test split
covered 22 competition-eligible subject--benchmark tasks and obtained Brier ALC
`0.2151811873`. Its full task-level curves are stored at:

```text
/data/results/beta-binomial-full-test-min80-seed20261017/metrics.json
```

This is an infrastructure/control result, not a tuned result or the final
cross-validation estimate.

## 3. Download the full eligible public tables

For the full table-only corpus, omit `--limit`:

```bash
python scripts/download_measurement_tables.py \
  --out data/raw/measurement-db
```

On Modal, use a separate immutable snapshot name:

```bash
modal run modal_ingest.py --dataset-name full --limit 0
modal run modal_app.py --action evaluate-full
```

Prepare the final all-benchmark protocol without training any model:

```bash
modal run modal_app.py --action prepare-all-benchmark-protocol
```

This freezes the benchmark-level 80/10/10 train/validation/test split, every
eligible subject--benchmark task, and ten shared deterministic label-reveal
orders. Tune only on validation benchmarks; inspect the test split once for
the final comparison. Every model must use identical item pools and orders.

Alternatively, the existing PAIEC utility also builds its retrieval payload,
including traces:

The existing PAIEC utility selects only binary, item-level benchmarks, records
the exact Hub revision, and leaves the source tables intact:

```bash
python reference/paiec_baseline/tools/prepare_data.py \
  --out data/raw/measurement-db \
  --no-embeddings
```

The Hugging Face token is read from the existing `hf auth login` configuration.
Omit `--no-embeddings` later if the published embedding payload is wanted.

## 4. Build the shared experiment tables

```bash
python scripts/prepare_measurement_data.py \
  --source data/raw/measurement-db/data \
  --out data/processed/measurement-db
```

The processor writes:

| File | Purpose |
|---|---|
| `benchmarks.parquet` | Benchmark descriptions and metadata |
| `subjects.parquet` | Subject metadata plus stable cross-benchmark signatures |
| `items.parquet` | Namespaced item IDs, raw text, metadata, and `model_text` |
| `observations.parquet` | One row per trial; the canonical binary training table |
| `pairs.parquet` | Repeated trials aggregated per distinct prediction input |
| `tasks.parquet` | One row per subject--benchmark task for episodic sampling |
| `manifest.json` | Source files, row counts, skipped data, nulls, and split policy |

Raw data are never rewritten. Item IDs are namespaced by benchmark, and
repeated trials for an identical prediction input receive the same split.

## 5. Choose the split that answers the experiment

- `split_by_benchmark`: primary test of transfer to an unseen benchmark.
- `split_by_task`: held-out subject--benchmark episodes; closest to NP training.
- `split_by_subject`: transfer to unseen model configurations.
- `split_by_input`: interpolation control only; never use it to claim
  unseen-benchmark generalization.

Fit preprocessing, calibration, vocabularies, and dimensionality reduction on
the training partition only. Never split individual trials from the same
`input_id` across train and validation.

## 6. Train models from the same artifacts

- 1PL/2PL/3PL and AmortizedIRT use `observations.parquet` joined to item and
  subject features.
- NCF uses integer indices derived from `subject_uid` and `item_uid` on the
  training split.
- CNP/NP/ANP/RAA group observations by `task_id`. Sample context prefixes at
  budgets `0,1,3,7,15,31`, and score the remaining rows as targets.
- Acquisition experiments reuse the same task and target sets but change only
  how context rows are selected.

`aims_starter.data.iter_episodes` supplies a deterministic first-pass episode
iterator. During training, replace its fixed order with seeded resampling so a
model sees multiple context/target partitions without duplicating the dataset.
The expected NP-AIRT model interface and training objective are documented in
[`docs/np-airt-implementation.md`](np-airt-implementation.md).

## Fast iteration order

1. Run the processor once.
2. Prove the complete loop on two benchmarks and one seed.
3. Cache item embeddings by `item_uid` once.
4. Run A0 frozen AIRT and A1 MAP update.
5. Add deterministic CNP; only then try latent NP, attention, and relations.
6. Keep acquisition random until the prediction ablation is stable.

This avoids spending days repeatedly tokenizing, joining, splitting, or
embedding the same records for each model.
