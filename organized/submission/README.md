# Latent CNP-128 submission staging area

This directory stages a CodaBench submission around the selected checkpoint
below. It uses a random streaming acquisition policy.

## Checkpoint provenance

- File: `assets/latent-cnp128-qwen3-8b-seed20261018.pt`
- Source workspace / volume: `aims-neurips:aims-neurips-data`
- Source run label: `latent-cnp128-cpu10-diagnostics-100k-s20261018`
- Architecture: latent CNP, linear pair encoder, mean pooling, width 128,
  latent dimension 128, Qwen3-Embedding-8B item embeddings truncated to 1,024
  dimensions.
- Selected fixed-validation checkpoint: step 41,350, unseen-target Brier
  0.1272059134.

**Important:** this is **not** the completed deterministic CNP d=128 width
sweep. That run wrote result/curve Parquets but did not save its model weights.
This package instead uses the available *in-training latent-CNP-128* checkpoint
above. Do not label it as the completed deterministic CNP d=128 result or
compare its submission outcome to that run's offline table.

## Runtime contract and status

The checkpoint expects the exact Qwen3-Embedding-8B content-only embedding:
last-token pooling, first 1,024 Matryoshka dimensions, then L2 normalisation.
`models.txt` declares that model so CodaBench pre-downloads it before the
offline runtime starts. `prediction_config.json` sets concurrency to one because
the 8B encoder must not be loaded by several CPU workers at once.

Slicing to 1,024 and then renormalising is equivalent to the store's
normalise-then-slice-then-renormalise, because both drop a positive scalar, so
the predictor reproduces `allbench-embed-qwen3_8b-v2` exactly.

### Embedding performance

Embedding is the entire cost of this submission. Three things address it, and
none of them change the vectors:

- **A disk cache** under `$AIMS_EMB_CACHE_DIR` (default `$TMPDIR/aims-np-embcache`).
  The platform restarts prediction workers between budget checkpoints, so an
  in-process dictionary is discarded about six times per run and every item is
  re-encoded. On disk, each distinct item text is encoded once per run. Entries
  are fixed-size float32 written to a temporary name and `os.replace`d into
  place, so concurrent workers need no locking; verified with eight processes
  contending over one directory. Any filesystem error disables the cache
  instead of failing the run.
- **Batched, length-sorted encoding.** Padding costs real compute, so batches
  are capped by a *token* budget rather than a row count and are filled longest
  first. One long item forms its own batch instead of padding short ones out to
  its length. This mirrors how the store itself was built
  (`modal_embed_v2.py::_encode`), including left padding, which is what makes
  batched last-token pooling correct.
- **Separate CUDA and CPU tuning** in `DEVICE_LIMITS`. GPU allocation is
  automatic on CodaBench, so the CUDA path is the expected one and uses a
  32,768-token budget matching the store build. The CPU path is a
  correctness-preserving fallback: 4,096 tokens, every core via
  `torch.set_num_threads`, and optional dynamic int8 behind `AIMS_EMB_INT8`.
  int8 is off by default because it perturbs the vectors the frozen checkpoint
  was trained on.

`max_length` stays at the store's 32,768. Long questions are encoded in full
rather than truncated.

Because `_load` is lazy, a budget checkpoint whose items are all cached never
loads the encoder at all.

`labeling.py` implements reproducible streaming random acquisition with the
standard reserve rule. It is intentionally unrelated to the model score.

Build a ZIP with:

```bash
.venv/bin/python organized/submission/build_submission.py
```

The generated ZIP is only eligible for upload after the organizer ZIP check and
smoke test succeed. This folder has not created a CodaBench submission.

## Other saved ablations

`build_anp_ablations.py` builds three **local-only** archives with the same
random acquisition policy: CA-only ANP (`SA=0`), ANP (`SA=1`), and stacked ANP
(`SA=2`). They use their in-training checkpoints and are not final/refit
models. The prior deterministic CNP width ablations cannot be packaged: those
runs saved metrics but not checkpoint weights.
