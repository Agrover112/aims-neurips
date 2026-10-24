# Result versioning

Status: binding for every ablation number quoted after 2026-10-22.

## v3-postfix (current)

Use this label only when all of the following hold:

- test-faithful 768-d features: item MiniLM over content plus item features,
  subject MiniLM over the eight subject fields;
- MiniLM@512 re-encode, not the earlier 256-token truncation;
- subject block zeroed on the item path (`mode=masked`);
- 104 validation tasks, 10 frozen orders, budgets `0,1,3,7,15,31`;
- five-fold out-of-fold calibration.

Current artifacts: every `r-c-*` JSON whose manifest has
`"arm": "minilm512"` and whose `mode` is `masked`. This covers the r_C ladder,
the loop arms, the AR arms, and the GP arms. Completed tables are in
[`ablation-results.md`](ablation-results.md).

## Additive anchor fix (2026-10-22)

`ContextConditionedIRT(mode="additive_ab")` now computes the ability MAP from
item-only base `a_C,b_C`; context offsets are applied only to target
parameters. This removes the structural circularity. No new validation run
exists. Existing CNP-additive rows are pre-fix and must not be mixed with a
post-fix rerun.

## Calibration v2 (2026-10-22)

`AnchorCalibrator.fit` now:

- fits scale and bias once for the base logistic model, then refits them for
  each fixed anchor blend instead of reusing one unblended fit;
- constrains the scale to be nonnegative by default;
- warns loudly if `nonnegative_scale=False` still selects a negative scale.

The old two-stage fit and negative scales are calibration v1. Every existing
calibration summary and every shipped demo metadata file predates this change.
Do not mix v1 calibration numbers with a calibration-v2 rerun. The submitted
archive is unchanged until it is deliberately rebuilt.

## Cache fingerprints (2026-10-22)

`modal_cnp_ladder.py`, `modal_cnp_disjoint.py`, and
`modal_faithful_ladder.py` now store a signature of the training source,
calibration source, and run config with each cached checkpoint/result. A
mismatch raises instead of silently reusing the cache. Old caches without a
signature must be treated as stale.

## v4.1-faithful-ladder (Stage 1 complete, no promotion)

Use this label only for the corrected-protocol campaign:

- per-visit rotating disjoint acquisition/evaluation split, split seed
  `20261017`; split keys omit the model seed;
- context is a uniform sample from the acquisition half; targets come only from
  the untouched evaluation half;
- seed set before model construction;
- padded, masked batches with 16 distinct tasks; one pair epoch covers 39,894
  task-budget pairs in 2,494 steps (2,493 full steps plus one 6-pair step);
- one pair epoch gives six visits per task, one per budget; a second pair epoch
  is the sensitivity check;
- `brier_alc` applies `6 * w_b` to prediction loss only;
- checkpoint selection on 53 task-level probe tasks spread across 4 train
  benchmarks selected by hash, with only those probe tasks excluded from fit;
- corrected latent loss with `KL / |T|` and no context-posterior NLL; the v4.1
  padded trainer is deterministic-only, so latent controls still use the legacy
  sequential path;
- three seeds and the same 104-task validation grid;
- five-fold out-of-fold calibration, fitted separately for every arm.

The full rung list, objective ladder, decision rules, and commands are in
[`faithful-ablation-ladder.md`](faithful-ablation-ladder.md). Stage 1 results are
in [`faithful-ladder-stage1.md`](faithful-ladder-stage1.md); no rung cleared its
MDE and nothing was promoted. Do not label an old v3 or v4 JSON as v4.1. v4
results, if any, used a fixed split and two-benchmark benchmark-level probe and
are not comparable.

## Retired versions

Do not mix these with v3-postfix or cite them as current:

- `pre-fix` / identity-channel runs with the subject block visible on the item
  path;
- benchmark-aware 1157-d runs that include benchmark descriptions or grading
  criteria unavailable at test time;
- MiniLM@256 truncated runs.

The old numbers can still be reported as a labeled negative control, but not as
the current model comparison.

## Preliminary warning

The current v3-postfix ablations use the preliminary trainer:

- `800` steps, `4` episodes per step, Adam, `lr=3e-4`.

They are version-correct, but not training-converged and not task-coverage
complete. The 3,200 episodes cover less than one pass over 6,702 train tasks.
Do not treat small deltas as final.

## Deferred reruns

The old "rerun v3 with a coverage trainer" plan is superseded by the
corrected-protocol ladder in
[`faithful-ablation-ladder.md`](faithful-ablation-ladder.md). The v3 rows
remain as preliminary evidence only.

The old proposed checklist was:

- [ ] coverage sampler: cycle through all 6,702 train tasks before repeating;
- [ ] batch 16 episodes per step;
- [ ] 5k-10k steps, AdamW, warmup, cosine decay;
- [ ] r_C ladder;
- [ ] loop arms `K=0/1/3`;
- [ ] AR arms with and without carried state;
- [ ] disjoint GP-2PL;
- [ ] metadata-prior GP-2PL;
- [ ] calibrate all arms with the same five-fold OOF protocol.

Never compare an 800-step arm to a coverage-trained arm.
