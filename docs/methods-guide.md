# AIMS Evaluation Methods Guide

## 1. Goal and protocol

For subject (s), item (j), and binary outcome (y_{sj}), the system predicts

\[
 p_{sj}=\sigma\{a_j(\theta_s-b_j)\}.
\]

Evaluation is budgeted. At budget (n\), the model sees an acquired context set (C_s(n)) and predicts outcomes for an untouched target set (T_s). Budgets are (0,1,3,7,15,31). The main metric is the weighted Brier area-under-the-learning-curve (ALC), with weights (0.1,0.2,0.2,0.2,0.2,0.1). We also record Brier by budget, ECE, AUC, mean probability, and uncertainty.

## 2. Data preprocessing

The corpus builder reads benchmark observations, item metadata, subject metadata, benchmark metadata, embeddings, and episode assignments. Rows are joined by stable IDs; outcomes are restricted to binary values. Missing outcomes are excluded from training observations and recorded in the manifest. Items are deduplicated and task/fold assignments are persisted.

Each row is represented as

\[
x=[e^{item}_j,\;m_j,\;e^{subject}_s,\;e^{benchmark}_k],
\]

where MiniLM item, subject, and benchmark embeddings are 384-dimensional and (m_j) contains five numeric item features. The resulting pilot vector is 1157-dimensional. Feature slices are recorded in the manifest; subject identity is masked from item-parameter paths so item (a_j,b_j) cannot become subject-conditioned.

Training tasks are meta-training subjects/benchmark episodes. Held-out BeaverTails subjects are used for evaluation. Context and target items are disjoint in the corrected NP episodes. Reveal orders are reshuffled for acquisition-order stability experiments.

## 3. Classical and amortized baselines

**IRT/2PL.** Item discrimination (a_j>0), difficulty (b_j), and subject ability (	heta_s) are learned or fitted by the logistic likelihood.

**Amortized 2PL (AIRT).** Neural networks map item features to (a_j,b_j). At inference, (	heta_s) is fitted from acquired labels by penalized MAP. This is the Online-AIRT track.

**Beta-binomial.** A conjugate control uses

\[
\hat p_n=(2+\sum_{i\le n}y_i)/(4+n).
\]

It provides a stable cold-start prior.

## 4. Neural Process family

The context encoder maps each ((x_i,y_i)) to a representation and aggregates it permutation-invariantly. The latent posterior is

\[
q_\phi(z\mid C)=\mathcal N(\mu_\phi(C),\operatorname{diag}\sigma_\phi^2(C)).
\]

The decoder samples (z), predicts target probabilities, and averages Monte Carlo predictions. Training uses Bernoulli NLL plus a KL term between context-only and context-plus-target posteriors, with corrected disjoint context/target episodes.

**CNP-AIRT.** Deterministic context aggregation; no global latent sampling.

**NP-AIRT.** A stochastic global latent (z) controls the subject-specific ability path.

**ANP-AIRT.** Cross-attention produces a target-specific deterministic representation (r^*), which enters a residual decoder. The residual is mean-centered across targets so it can reorder items without changing the anchor level.

**Stacked-MHSA-ANP.** Adds stacked self-attention blocks before target cross-attention.

**Per-item latent correction.** A latent item network consumes ([x_j,z]) and adjusts (a_j,b_j), making the latent genuinely multi-dimensional rather than only a scalar ability.

**Hybrid anchor model.** Keeps MAP ability as the level-setting anchor and adds the per-item latent correction as a shrinkable deviation.

## 5. Training and meta-training

Models are trained episodically over meta-training tasks. Each episode samples context and target items, with randomized context sizes. Optimizers use gradient clipping and deterministic seeds. Neural Process training logs NLL, KL, auxiliary ability loss, and latent diagnostics. Target labels are not available to the deployed posterior.

## 6. Inference and acquisition

At each budget, only the acquired context labels are supplied. Predictions are generated for the untouched evaluation pool. Acquisition experiments compare fixed reveal orders and multiple reshuffled orders; no retraining is performed during rescoring.

## 7. Calibration

We record reliability/ECE and support post-hoc temperature scaling. Given logits (\ell), calibrated probabilities are (\sigma(\ell/T)). The AnchorCalibrator variant fits a simple temperature/shift using acquired context labels, separately by budget and subject-benchmark task. Calibration changes probability scale but not ranking/AUC.

## 8. Diagnostics and ablations

Diagnostics include latent mean/variance, label-flip sensitivity, per-item logit variance, Jacobian singular values/rank, AUC by budget, Brier curves, ECE curves, and benchmark/fold variance. The main ladder is: Online-AIRT, CNP-AIRT, NP-AIRT, ANP-AIRT, stacked ANP, per-item latent correction, hybrid anchor correction, and calibrated variants.

## 9. Current implementation caveats

The corrected main path uses manifest feature slices, masks subject blocks in item encoders, separates context and target items, and centers ANP residuals. Some auxiliary legacy Modal paths should still be migrated before being used for final claims. Results must be quoted only from the current v3 residual experiment and matching CNP artifact.
