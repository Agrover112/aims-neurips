# Submission workspace

This folder builds ZIP files for the online evaluation interface. It does not
upload anything. The `submission` Git branch is where we keep these packages
separate from training and analysis work.

## Start here

There are four benchmark-holdout ANP packages. All use the same architecture,
item embeddings, and random streaming acquisition rule. They differ only in how
training sampled the number of observed labels, or in the optimiser.

| Candidate ZIP | What differs during training | Selected checkpoint | Status |
| --- | --- | --- | --- |
| `benchmark-holdout-anp-w64-filtered.zip` | ordinary episodic training | step 1,200; fixed-validation Brier 0.16749 | built, not uploaded |
| `benchmark-holdout-anp-w64-bhold-budgetladder.zip` | official budgets 0/1/3/7/15/31 with Brier-ALC weights | step 50; fixed-validation Brier 0.21051 | built, not uploaded |
| `benchmark-holdout-anp-w64-sam005.zip` | SAM, rho = 0.05 | step 5,450 | **uploaded, CodaBench 1022033** |
| `benchmark-holdout-anp-w64-denseprefix.zip` | every context prefix 0-31 per episode | step 4,550 | **uploaded, CodaBench 1022030** |

CodaBench 1019401 was a probe (constant 0.8 at budget 0), not a candidate.
Leaderboard scores for 1022033 and 1022030 had not been read back when this
was written; the public leaderboard samples 9 subject-benchmark pairs, so its
standard error is about 0.023 Brier.

### Held-out replay result

On the frozen replay panel `benchmark-holdout-online-v1-s20261018-n15-m80-cap100000`
(test: 45 tasks on algotune, cmmlu, editbench, never trained on; 10 seeds),
macro Brier-ALC under random acquisition:

| model | test ALC |
| --- | --- |
| constant (training mean 0.512) | 0.2472 |
| Jeffreys beta-binomial | 0.2494 |
| ANP w64 SAM | 0.2663 |
| ANP w64 filtered | 0.2725 |
| ANP w64 prefix training | 0.2762 |

Every ANP is significantly worse than the beta-binomial on unseen benchmarks;
the gap is mostly at budgets 0 and 1. The other acquisition policies do not
change this. Full tables: `organized/testing/online/README.md` and
`docs/online-final-results.pdf`.

## Pooled Rasch (`pooled_rasch/`, CodaBench 1024003)

Predicts `p(s, i) = sigmoid(mu_b + theta_s - b_h)` from every revealed label in
the draw, not only the current subject-benchmark pair.

- **Data:** almost none. Only the base rate (training mean) and 3 prior widths
  (picked on validation) come from offline data. Everything else is fit at test
  time on `labeled`, the revealed labels from all pairs in the current draw.
- **Algorithm:** MAP estimation, i.e. logistic regression with Gaussian priors
  (L2). The submission solves it with Newton's method; the replay uses L-BFGS.
- **Online:** it refits from scratch at every checkpoint (0/1/3/7/15/31) on all
  labels so far. There's no carried state, and it's milliseconds to a few
  seconds per fit.

| parameter | meaning | prior |
| --- | --- | --- |
| `mu_b` (benchmark rate) | how easy benchmark b is overall, in logits | N(M0, 2²); M0 = logit(0.512), the training-benchmark mean response |
| `theta_s` (subject ability) | how strong subject s is, relative to the benchmark rate | N(0, 1²) |
| `b_h` (item difficulty) | how hard item h is, relative to its benchmark; items matched by content | N(0, 1²); 0 for an item nobody has labelled |

The widths (2, 1, 1) set how far labels can pull each estimate away from its
prior centre. With no labels at all it predicts sigmoid(M0) = 0.512.

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

Build the BGE-M3 ANP packages (all four benchmark-holdout candidates plus the
older random-subject split ablations; `--only <name>` builds one):

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

All four benchmark-holdout archives were built and passed this checker.

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

## Layout

Everything submission-related lives in this folder.

```text
organized/submission/
  README.md                         this file
  build_bge_anp_submissions.py      builds the four benchmark-holdout ZIPs (+ older ablations)
  aims_predictor/, model.py,        shared ANP predictor, entry point and acquisition rule
    labeling.py, models.txt, ...
  assets/                           checkpoints and metadata.json used by the builders
  modal_*.py                        diagnostics (router, per-budget Brier, latency); upload nothing
  legacy/                           earlier submission templates, kept for reproducibility
    submission/                     original beta-binomial shrinkage predictor (shipped constants)
    submission_mean|budget_shrinkage|neural|online|variants/
    submission_tabfm|tabpfn/        tabular foundation-model packages (not used)
```

`legacy/submission/aims_predictor/predictor.py` holds the shipped beta-binomial
shrinkage constants that `anp_crossattn_blocks.py` copies. The repo's tests
(`tests/`) and `scripts/build_submission.py` / `scripts/build_neural_submission.py`
point at `legacy/`. ZIPs are written to `dist/`, which is gitignored.

CodaBench helpers stay in `scripts/` (`codabench_logs.py`, `watch_codabench.py`,
`audit_codabench_id_overlap.py`).
