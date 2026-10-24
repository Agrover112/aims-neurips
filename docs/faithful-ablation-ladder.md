# Corrected-protocol ablation ladder

Status: 2026-10-22. **Stage 1 complete; no rung promoted.** Results are in
[`faithful-ladder-stage1.md`](faithful-ladder-stage1.md). This is not part of
the paper and does not replace the v3-postfix tables.

Purpose: test the GP-2PL/anchor and target-specific ANP correction under a
training protocol that matches the competition, instead of the preliminary
`800`-step `C`-in-`T` trainer.

## Why this ladder exists

The v3 campaign found a promising GP-2PL point estimate, but it is not a fair
test of the context path:

- only 3,200 episodes over 6,702 train tasks, so about 62% of tasks were never
  seen;
- the model is built before the seed is set;
- the latent path adds the full KL to a mean NLL, unlike the official (A)NP
  code, which divides by the number of targets;
- the latent path also uses a context-posterior NLL term that is not part of
  the standard objective;
- CNP-Additive originally changed `a_C,b_C` and then reused them to estimate
  ability, so the context representation was structurally circular. The
  isolated code now uses item-only base parameters for the MAP and applies
  additive offsets to targets only (2026-10-22); the rung remains outside this
  ladder;
- the MAP ability is detached, so the outer loss never trains the context
  encoder through ability.

The ladder fixes the protocol first, then asks which correction survives.

## v4.1 frozen protocol

| item | value |
|---|---|
| train tasks | 6,702 minus 53 held-out probe tasks; expected 6,649, runner records actual |
| validation | 104 tasks, 10 frozen orders |
| test | untouched; never used |
| train split | per-visit rotating disjoint acquisition/evaluation split |
| split seed | `20261017` |
| context | uniform sample from the acquisition half |
| targets | evaluation half only |
| budgets | `0,1,3,7,15,31` |
| ALC weights | `0.1,0.2,0.2,0.2,0.2,0.1` |
| batch | padded, masked; 16 distinct tasks per full batch |
| pair epoch | 39,894 task-budget pairs, 2,493 full steps + one 6-pair step = 2,494 steps |
| visits | one pair epoch gives six visits per task, one per budget |
| sensitivity | a second pair epoch, not the old five-pass framing |
| optimizer | AdamW, lr `3e-4`, weight decay `0.01` |
| schedule | 100-step warmup, cosine to 5% of peak |
| probe | 53 tasks across 4 held-out train benchmarks, every 250 steps |
| checkpoint selection | best held-out train-probe Brier ALC |
| final calibration | five-fold out-of-fold anchor calibration |
| seeds | `20261018`, `20261019`, `20261020` |

The queue covers every `(task, budget)` pair exactly once per pair epoch. Full
batches contain 16 distinct tasks. Budgets are assigned so each task sees all
six budgets and full batches stay near 2-3 tasks per budget. The task-count
remainder is collected into a tail pool; the final batch has 6 pairs and its
loss is divided by the full batch size, not by 6.

The split key depends only on `task_id`, budget, and pair epoch. Every arm and
every model seed sees the same split for the same task-budget visit. The visit
order is driven by the model seed, not by model size or architecture. Context
items are a uniform sample from the acquisition half, not a nested prefix.

The probe is a task-level holdout from train, spread round-robin across 4
train benchmarks selected deterministically by hash. Only the 53 probe tasks are
excluded from fit, which keeps the expected fit pool at 6,649. With the current
corpus the selected benchmarks are `harmbench`, `agi_elo`, `emoji_attack`, and
`afrieval`; the runner records the actual list. The probe is used only for
checkpoint selection and monitoring. The final table uses the 104 validation
tasks.

## Architecture ladder

| rung | item curve | context correction | subject prior | question |
|---|---|---|---|---|
| `2pl` | item-only 2PL + MAP theta | none | no | anchor baseline |
| `2pl_anp` | item-only 2PL + MAP theta | target attention `delta_t(C)` | no | does context help without GP? |
| `gp_2pl` | 2PL + finite-basis GP residual | none | no | does a flexible item curve help? |
| `gp_anp` | 2PL + GP residual | target attention `delta_t(C)` | no | highest-value combined arm |
| `gp_anp_prior` | 2PL + GP residual | target attention | yes | does subject metadata add anything? |
| `latent_np` | latent ability, no anchor | none | no | properly trained latent control |
| `latent_anp` | latent ability, no anchor | target attention | no | properly trained latent attention control |
| `beta` | none | none | none | computed inside every run |

The GP residual is

```text
z_t(theta) = a_t (theta - b_t) + c_t^T phi(theta)
```

with a finite RBF basis and a `c^T K^{-1} c` penalty. It is a MAP correction,
not a full GPIRT posterior.

The ANP correction is

```text
z_t = z_t_base + gamma * delta_t(C)
```

where `delta_t(C)` comes from self-attention over label-conditioned context
tokens and target-to-context cross-attention. It is mean-centered across the
target set, zero-initialized, and L2-shrunk (`1e-3` on the mean squared delta),
so the model starts at the anchor and the correction stays small until the data
support it. The ability MAP is detached; the correction path is trained
directly. That is a deliberate estimator choice, not a claim that the context
encoder was trained through the MAP solve.

The latent rungs use the corrected loss:

```text
L = BCE(targets) + beta * KL(q(z | C union Q) || q(z | C)) / |Q|
```

with one latent sample during training. There is no context-posterior NLL term.
The latent rungs have no MAP anchor (`anchor_scale=0`), so `z` is not redundant
by construction. The v4.1 padded batch trainer covers the deterministic rungs;
the latent rungs still use the legacy sequential path until their batch contract
is implemented.

## What each contrast answers

| contrast | question |
|---|---|
| `2pl_anp - 2pl` | does the target-specific context correction help alone? |
| `gp_2pl - 2pl` | does the GP item curve help alone? |
| `gp_anp - gp_2pl` | does context help on top of the GP curve? |
| `gp_anp - 2pl_anp` | does the GP curve help on top of context? |
| `gp_anp_prior - gp_anp` | does subject metadata help as a prior? |
| `latent_anp - latent_np` | does latent attention help without an anchor? |
| `gp_anp - latent_anp` | anchor vs latent, both with attention |

The 2x2 structure is intentional. A single ladder that adds GP and ANP in one
step cannot tell which component caused the change.

## Objective ladder

Run on the winner of `2pl`, `gp_2pl`, `2pl_anp`, and `gp_anp`:

| objective | loss |
|---|---|
| `bce` | BCE on the evaluation half |
| `bce_brier` | BCE + Brier on the evaluation half |
| `brier_alc` | Brier on the evaluation half, with pair weight `6 * w_b` |

The `brier_alc` queue is uniform over budgets, so the `6 * w_b` prediction
weight makes the epoch expectation equal to ALC. The weight applies only to the
prediction term. The GP penalty and latent KL are not multiplied by it.

Do not replace BCE with Brier without this ablation. Brier is proper, but it can
be unstable and the post-hoc calibrator can absorb scale changes.

## Decision rules

Primary metric: calibrated Brier ALC on 104 validation tasks, five-fold
out-of-fold calibration. Raw Brier ALC is a diagnostic, not the promotion
criterion.

- Paired task delta against the matched control, same tasks and orders.
- Report the 95% task interval over 104 tasks, the seed direction for each of
  the three seeds, and the paired 80%-power MDE.
- A delta smaller than the MDE is `can't tell`, not `no effect`. This includes
  small negative point estimates.
- A rung advances only if the interval excludes zero, the delta clears its MDE,
  and at least two of three seeds move in the same direction.
- Do not transfer a calibrator between arms. Every arm gets its own
  out-of-fold fit.
- If `brier_alc` improves raw Brier but not calibrated Brier, reject it.
- If `gp_anp` does not beat both `gp_2pl` and `2pl_anp`, do not keep the
  combined arm.
- If `gp_2pl` does not beat `2pl`, do not keep the GP residual.
- If `gp_anp_prior` does not beat `gp_anp`, drop the metadata prior.
- If a latent rung does not beat its anchor counterpart, keep it as a negative
  control and do not promote it.
- A point estimate inside the interval is a tie, not a win.

Stop conditions:

- stop the architecture ladder if no rung clears its MDE against `2pl`;
- stop the objective ladder if `bce_brier` and `brier_alc` do not beat `bce`;
- do not run the second pair epoch until at least one rung clears the first
  pair-epoch screen.

## Commands

`--parallel` launches every rung/seed at once and needs enough A100 capacity in
the workspace. Omit it, or run one rung at a time, if the quota is tight.

One-seed smoke first (separate `-smoke` output path):

```bash
modal run modal_faithful_ladder.py \
  --rungs 2pl,gp_2pl,2pl_anp,gp_anp \
  --seeds 20261018 \
  --objective bce \
  --task-limit 64 \
  --validation-limit 8 \
  --steps 24 \
  --probe-count 8 \
  --probe-benchmark-count 4
```

Architecture ladder, three seeds, one full pair epoch:

```bash
modal run modal_faithful_ladder.py \
  --rungs 2pl,gp_2pl,2pl_anp,gp_anp \
  --seeds 20261018,20261019,20261020 \
  --objective bce \
  --pair-epochs 1 \
  --parallel
```

Second pair-epoch sensitivity, only after the first screen:

```bash
modal run modal_faithful_ladder.py \
  --rungs 2pl,gp_2pl,2pl_anp,gp_anp \
  --seeds 20261018,20261019,20261020 \
  --objective bce \
  --pair-epochs 2 \
  --parallel
```

GP-plus-prior follow-up, only if `gp_anp` clears the screen:

```bash
modal run modal_faithful_ladder.py \
  --rungs gp_anp,gp_anp_prior \
  --seeds 20261018,20261019,20261020 \
  --objective bce \
  --pair-epochs 1 \
  --parallel
```

Objective ladder on the winner:

```bash
modal run modal_faithful_ladder.py \
  --rungs gp_anp \
  --seeds 20261018,20261019,20261020 \
  --objective bce,bce_brier,brier_alc \
  --pair-epochs 1 \
  --parallel
```

Latent controls:

```bash
modal run modal_faithful_ladder.py \
  --rungs latent_np,latent_anp \
  --seeds 20261018,20261019,20261020 \
  --objective bce \
  --steps 2100 \
  --parallel
```

Analyze downloaded JSONs:

```bash
.venv/bin/python scripts/analyze_faithful_ladder.py \
  /tmp/opencode/faithful-ladder/*.json
```

## Output contract

Each run writes:

```text
/data/results/faithful-ladder/ladder-v41-{rung}-{objective}-{seed}-{mode}.json
```

A second pair epoch appends `-pe2`; a limited smoke run appends `-smoke` before
`.json`. Smoke, first-epoch, and second-epoch results never share a cache path.

The JSON records:

- `version = v4.1-faithful-ladder`;
- the split seed, budget grid, weights, pair epoch, batch size, queue counts,
  steps, optimizer, schedule, probe settings, and selection rule;
- training history with EMA loss, BCE, Brier, penalty, auxiliary loss, learning
  rate, batch size, pairs seen, and probe Brier ALC;
- selected step and probe score;
- `alc` rows for raw and OOF-calibrated Brier ALC;
- `task_alc` rows for paired analysis.

## Known limitations

- The rotating 50/50 split is a simulation of the competition pools, not the
  real acquisition/evaluation split.
- The probe is a task-level holdout across 4 train benchmarks; it is for
  monitoring, not a second validation set. The final 104-task validation set is
  separate. The probe is fixed across arms and seeds.
- Validation has only five benchmark clusters. Task intervals remain primary;
  benchmark-cluster intervals are a sensitivity check.
- The finite-basis GP is not full GPIRT and does not have a GP marginal
  likelihood. Do not describe it as a posterior over unseen item curves.
- The GP likelihood path is item-specific. Context-item GP coefficients sit
  inside the detached MAP and receive only the prior penalty; context is
  carried by the ANP correction, not by the GP solve.
- The latent rungs use an IRT decoder. They are a latent NP/ANP control for this
  problem, not the original 1-D regression architecture. Their v4.1 batch path
  is deferred.
- `brier_alc` uses every budget once per pair epoch with `6 * w_b` weights. It is
  unbiased for ALC in expectation, but the finite epoch is noisier than a full
  six-budget objective.
- The tail batch has 6 pairs; its prediction loss is divided by 16 so it does
  not get extra per-pair weight.

## Not in this ladder

- AR carried state, looped decoders, and the NR/U-NR residual family. Those are
  deferred until the protocol is fixed.
- Full GPIRT with a marginal likelihood.
- Any test-label tuning or hidden-leaderboard selection.
- Any change to the paper.
