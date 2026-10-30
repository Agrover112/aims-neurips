# AIMS evaluation methods guide

## Scope and current evidence

This is the living description of the current filtered Neural Process work.
It is not a description of the earlier MiniLM/BeaverTails AIRT pilots. Those
remain useful historical controls, but their scores and feature contract must
not be compared directly with the filtered benchmark-holdout runs.

One task is one `(subject, benchmark)` response function. The outcome is binary.
At deployment, a policy sees candidates one at a time, may accept at most 31
labels, and predicts a disjoint evaluation set at budgets
`0, 1, 3, 7, 15, 31`.

The competition metric is macro subject-benchmark Brier ALC:

\[
\operatorname{ALC}=0.1B_0+0.2B_1+0.2B_3+0.2B_7+0.2B_{15}+0.1B_{31},
\]

where lower is better. Brier score, ECE, and ROC-AUC by budget are supporting
diagnostics, not replacements for ALC.

## Data and split

The current NP runs use frozen BGE-M3-long item embeddings. At test time the
available item inputs are text strings only: `item_content`, `item_features`,
`interactors`, and an opaque `benchmark_id`. There are no image, audio,
attachment, benchmark-description, or grading-criterion inputs.

For the current benchmark-holdout experiments, filtering is performed before
the split. The active configuration keeps items with at least 15 responders and
subject-benchmark tasks with at least 80 remaining items. Whole benchmarks are
then assigned to train, validation, or test. Thus every validation/test item is
cold with respect to training benchmarks.

Missing responses remain missing. Repeated responses are aggregated before
episodic sampling. The filtered response matrix, split assignment, and item
embedding store are recorded in the run configuration.

## ANP used in the current runs

The active model is a latent Attentive Neural Process in
`organized/training/anp_crossattn_blocks.py`. It uses a 64-dimensional hidden
state, one context self-attention block, four cross-attention heads, and a
32-dimensional global latent.

For observed context pairs \((x_i,y_i)\), the model forms label-conditioned
values

\[
v_i=h_v(x_i)+h_y(2y_i-1).
\]

Context self-attention contextualizes the values. A target item \(x_t\) then
queries the observed context:

\[
r_t=\operatorname{CrossAttention}\bigl(q(x_t), k(x_C), v_C\bigr).
\]

The latent path mean-pools the same contextualized values and parameterizes
the context prior \(q(z\mid C)\). During training it also forms
\(q(z\mid C\cup T)\). The decoder predicts a Bernoulli logit from
\([x_t,r_t,z]\). The deterministic and latent paths therefore share the
label-conditioned value encoder, but they have different aggregation roles:
target-specific cross-attention for \(r_t\), mean pooling for \(z\).

For the latent model, the loss is

\[
\mathcal L = \operatorname{NLL}(y_T\mid x_T,C,z)
+ \frac{\operatorname{KL}[q(z\mid C\cup T)\|q(z\mid C)]}{|T|}.
\]

Contexts and targets are disjoint in the active trainer. The target labels are
used only to form the training posterior and loss. At inference, predictions
use \(q(z\mid C)\) alone.

## Optimisation variants

The ordinary baseline uses AdamW. The current SAM ablation keeps the model,
data, and inference procedure fixed, but replaces the gradient used for the
AdamW update with the gradient at a locally loss-increasing weight perturbation
of radius \(\rho=0.05\). SAM roughly doubles training work per step and adds
no inference cost.

The prior-anchor ablation changes the decoder parameterization to

\[
\operatorname{logit} p_t = \operatorname{logit}\hat p_{\mathrm{Beta}}(C)
+ f_\theta(x_t,r_t,z),
\]

with the residual output initialized to zero. It starts exactly at the
Beta-Binomial predictor and learns deviations from it. This is distinct from
SAM and must be compared in a factorial ablation.

Context sizes can be sampled either with the competition ALC weights or
uniformly over the scored budgets. These are different training distributions.
Do not call a SAM result a clean SAM-only comparison when the sampler also
changed.

## Streaming acquisition

The replay protocol in `docs/benchmark-holdout-online-replay.md` is the source
of truth for offline online-testing claims. It uses a fixed, disjoint
acquisition/evaluation split and fixed candidate orders. A policy observes only
the current candidate, the labels it has already accepted, and remaining counts.
It cannot score future candidates or read any evaluation labels.

Random prefix and threshold policies are cheap controls. Rank-based policies
are calibrated against train-only probe items and include a reserve rule so they
cannot leave the budget unfilled. Expected Brier Reduction (EBR) estimates

\[
\sum_{y\in\{0,1\}}p(y\mid x,C)
\frac{1}{|R|}\sum_{r\in R}
\left[p(r\mid C\cup\{(x,y)\})-p(r\mid C)\right]^2,
\]

using a train-only reference set \(R\). EBR is streaming-faithful, but it is a
proxy for Brier improvement, not the true unobserved evaluation Brier. Its
reference set must never be built from the task's evaluation items.

## Selection rule

Choose checkpoints using only
`brier_score/unseen_target/validation_fixed_bank`. It is a fixed bank from
unseen validation benchmarks, so its changes reflect the checkpoint rather
than a newly sampled validation episode. Test benchmarks are read only after
the configuration is frozen.

Training loss is useful for detecting numerical failure. It is not the
criterion for choosing between benchmark-holdout models because it is measured
on different benchmarks.

## What remains unproven

The 10k uniform-SAM run has the best fixed-bank value among the recent three
CPU runs, 0.205767 at step 5,450. This is a one-seed checkpoint-selection
result, not a final benchmark claim. The matched ALC-weighted SAM run is in
progress. Neither result establishes a CodaBench gain or a general advantage
of SAM without a paired replay and additional seeds.
