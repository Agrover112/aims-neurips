# Submission workspace

This folder builds ZIP files for the online evaluation interface. It does not
upload anything. The `submission` Git branch is where we keep these packages
separate from training and analysis work.

## Start here

There are two current benchmark-holdout ANP candidate packages. Both use the
same architecture, item embeddings, and random streaming acquisition rule.
They differ only in how the training loop sampled the number of observed
labels.

| Candidate ZIP | What differs during training | Selected checkpoint | Status |
| --- | --- | --- | --- |
| `benchmark-holdout-anp-w64-filtered.zip` | ordinary episodic training | step 1,200; fixed-validation Brier 0.16749 | candidate |
| `benchmark-holdout-anp-w64-bhold-budgetladder.zip` | samples the official budgets 0, 1, 3, 7, 15, 31 with their Brier-ALC weights | step 50; fixed-validation Brier 0.21051 | candidate |

Neither row is a declared final submission. The checkpoint score selects a
checkpoint *within its own run*. It does not replace a matched, held-out replay
comparison between the two packages.

## What both candidates contain

- **Model:** latent Attentive Neural Process, width 64, latent dimension 32,
  one context self-attention block, and four target cross-attention heads.
- **Item representation:** 1,024-dimensional BGE-M3 embedding of `item_content`
  only. The submission recomputes this embedding from the incoming item text.
- **Online update:** it rebuilds the prediction from the labels acquired for
  the current subject--benchmark pair. It does not change model weights at
  inference time.
- **Acquisition:** seeded random streaming order with a reserve rule, so it
  cannot inspect future candidates and still fills the label budget.
- **Training data:** benchmark holdout with the retained-response rules
  `min_item_responders=15` and `min_items_per_subject=80`.

## Build and check

Build the four saved BGE-M3 ANP packages. The first two archives listed below
are the benchmark-holdout candidates; the other two are older random-subject
split ablations.

```bash
.venv/bin/python organized/submission/build_bge_anp_submissions.py
```

The archives are written to `dist/`. Check a ZIP before any upload:

```bash
.venv/bin/python reference/paiec_baseline/check_submission_zip.py \
  dist/benchmark-holdout-anp-w64-filtered.zip
```

The checker verifies the ZIP layout, imports `model.py`, calls `predict()`, and
checks the acquisition hook. It uses a local synthetic embedding for that smoke
test; it does not measure the real BGE-M3 runtime or score quality.

Both benchmark-holdout archives were built and passed this checker on this
branch. They have not been uploaded by this preparation step.

## How the ZIP chooses its model

`build_bge_anp_submissions.py` makes one ZIP per entry in its `VARIANTS`
dictionary. Each ZIP contains:

```text
model.py                    evaluation entry point
labeling.py                 random streaming acquisition
aims_predictor/             ANP code and text embedder
assets/<checkpoint>.pt      one trained checkpoint
assets/metadata.json        architecture and embedding contract for that checkpoint
models.txt                  BGE-M3 model to pre-download
```

`model.py` is generic. At runtime it reads `assets/metadata.json`, loads the
checkpoint named there, and constructs the matching ANP. This is why one
predictor implementation can package several checkpoints safely.

## Other files in this folder

- `build_anp_ablations.py`: packages older Qwen ANP ablations. They are useful
  for debugging architecture changes, not current benchmark-holdout candidates.
- `build_submission.py`: packages an older latent CNP with Qwen-8B embeddings.
  Its checkpoint came from an in-training run, not the completed deterministic
  CNP width sweep. Keep it only for reproducibility.
- `modal_router_check.py`, `modal_per_budget_brier.py`, and the latency scripts:
  local diagnostics. They do not upload a ZIP.

## Before choosing or uploading a candidate

1. Compare the two ANPs on the same frozen replay panel, candidate orders, and
   acquisition policy.
2. Check per-budget Brier, not only one checkpoint-selection Brier.
3. Build the selected archive again and run the ZIP checker.
4. Record the archive filename, checkpoint filename, Git commit, and replay
   result next to the upload record.
