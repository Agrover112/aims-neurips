# Relation-aware attentive amortized IRT: design note

Status: proposed and unimplemented. This note defines experiments; it does not
claim improved performance or novelty.

Notation follows [`notation.md`](notation.md): \(C\) is the acquired context,
\(T\) is the target set, \(r\) is the deterministic ANP path, \(s\) is the
latent encoder path, and \(z\) is the global latent variable.

## The question we are testing

AmortizedIRT supplies item parameters from item content, but a learned subject
ability does not automatically adapt when Codabench reveals outcomes. We test
whether increasingly expressive, label-conditioned inference improves that
content-derived prior under budgets \(0,1,3,7,15,31\).

Let \(s\) identify a subject, \(k\) a benchmark, and \(j\) a target item. For
one subject--benchmark task, let

\[
C=\{(x_i,y_{si})\}_{i=1}^{N_C}
\]

be the acquired outcomes. The model input \(x_i\) contains only information
available before querying \(y_{si}\).

## Common AmortizedIRT base

The item network is shared by every matched comparison:

\[
x_j=E_x(\text{item}_j),\qquad
(b_j,\log a_j)=F_\eta(x_j),\qquad
\theta^0_{sk}=E_s(m_s,m_k),
\]

\[
\ell^0_{skj}=a_j(\theta^0_{sk}-b_j),\qquad
p^0_{skj}=\sigma(\ell^0_{skj}).
\]

If usable metadata are absent, \(E_s\) falls back to a pooled prior. The
two-parameter form is the default; 1PL and 3PL are secondary controls. In all
adaptive variants, \(a_j\) and \(b_j\) remain invariant across subjects,
budgets, and acquired context. This preserves their IRT interpretation.

## Does an NP perform test-time adaptation?

Yes, provided it re-encodes \(C\) whenever labels are acquired. Its learned
weights stay frozen, but its pair-specific representation or posterior changes:

\[
C^{(B)}\longrightarrow q_\phi(z\mid s_C,m_s,m_k)
\longrightarrow p(y_{sj}\mid x_j,r_j^*,z).
\]

The precise description is **amortized test-time inference** or
**conditioning-based adaptation**, not gradient-based fine-tuning. At budget
zero, an explicit gate enforces \(\Delta\theta=0\) and local residual
\(\delta=0\), so every adaptive model returns the identical AIRT prediction.
Without this gate, a learned empty-context prior would define a different
zero-budget predictor.

A deterministic mean-pooled context encoder is a CNP. It is a latent NP only
when it includes a stochastic global variable \(z\) and is trained with the
corresponding conditional variational objective.

## Matched model ladder

The analytic and neural adapters branch from the same AIRT base. A1 is the
non-neural control. The neural branch is cumulative from A2 through A5; A6 is a
negative control rather than another rung.

| ID | Predictor | Scientific question |
|---|---|---|
| A0 | Frozen 2PL AIRT | How good is the zero-label content prior? |
| A1 | AIRT + regularized MAP ability | Is a scalar Bayesian update enough? |
| A2 | CNP-AIRT | How does learned global conditioning compare with A0 and A1? |
| A3 | NP-AIRT | Does a stochastic global task state help beyond A2? |
| A4a | Dot-product ANP-AIRT | Does target-to-context attention help beyond A3? |
| A4b | Multihead ANP-AIRT | Do multiple cross-attention heads learn complementary notions of item relevance? |
| A4c | Stacked Multihead ANP-AIRT | Do context--context interactions help beyond multihead cross-attention? |
| A5 | RAA-IRT | Do explicit item relations help beyond ordinary attention? |
| A6 | RAA-IRT with shuffled relations | Is any A5 gain genuinely relational? |

The essential simple adaptive control is

\[
\hat\theta_{sk}=\arg\max_\theta\left[
-\frac{(\theta-\theta^0_{sk})^2}{2\sigma_0^2}
+\sum_{i\in C_{sk}}\log\operatorname{Bernoulli}
\left(y_{ski};\sigma(a_i(\theta-b_i))\right)
\right].
\]

If a neural adapter cannot beat this update, its additional complexity is not
justified.

### Proposed relation-aware model

The global latent path is

\[
q_\phi(z\mid s_C,m_s,m_k),\qquad
\theta_s=\theta^0_{sk}+\Delta\theta_\psi(z).
\]

ANP-AIRT creates a target-specific context summary

\[
\alpha_{ji}\propto
\exp\!\left(\frac{q(x_j)^\top k(x_i,y_{si})}{\sqrt d}\right),\qquad
r_j^*=\sum_{i\in C}\alpha_{ji}v(x_i,y_{si}).
\]

Following [Kim et al. (2019)](https://arxiv.org/abs/1901.05761), keep three
attention variants distinct. **Dot-product ANP-AIRT** uses a single learned
cross-attention map in the deterministic path. **Multihead ANP-AIRT** replaces
it with multihead cross-attention. **Stacked Multihead ANP-AIRT** additionally
uses two stacked self-attention layers over context representations in both the
deterministic and latent encoder paths before aggregation/cross-attention. The
paper found that stacking more than two self-attention layers brought no clear
gain, so deeper stacks are not part of the primary sweep. Laplace attention is
a parameter-free diagnostic rather than a main model rung.

RAA-IRT adds declared pairwise features, rather than merely giving ANP more
node features:

\[
\rho_{ji}=\left[x_j\odot x_i,\ |x_j-x_i|,\
\cos(x_j,x_i),\ u_j\odot u_i,\ |u_j-u_i|\right],
\]

Here \(u_j\) is a fixed vector made only from the supplied `item_features` and
`interactors` fields and, in the cognitive-feature ablation, annotations
computed offline from permitted item content. Only organizer-permitted
information available before querying \(y_{skj}\) may enter \(\rho_{ji}\); the
target outcome and any derivative of it are forbidden.

\[
\alpha^R_{ji}\propto\exp\!\left[
\frac{q(x_j)^\top k(x_i,y_{si})}{\sqrt d}+h_\rho(\rho_{ji})
\right],
\]

\[
r_j^{R*}=\sum_{i\in C}\alpha^R_{ji}V(x_i,y_{si},\rho_{ji}),
\qquad
\delta_{sj}=\delta_{\max}\tanh d_\omega(x_j,r_j^{R*},z),
\]

\[
P(y_{sj}=1\mid C)=
\mathbb E_{q(z\mid s_C)}\!\left[
\sigma\!\left(a_j(\theta_s-b_j)+\delta_{sj}\right)
\right].
\]

The bounded \(\delta\) is a predictive residual, not a context-dependent item
difficulty. Ablate relation bias in the attention score and relation features
in the value separately. For A6, permute \(\rho_{ji}\) across target--context
pairs within each task and seed while preserving node inputs and the marginal
distribution of every relation feature. Because “Relational Conditional Neural
Process” is already an established name, the proposal is called
**Relation-aware Attentive Amortized IRT (RAA-IRT)**.

## Should GPIRT be a baseline?

Published GPIRT models an item response curve as

\[
f_j\sim\mathcal{GP}(\mu,K),\qquad
y_{sj}\sim\operatorname{Bernoulli}(\sigma(f_j(\theta_s))).
\]

This is scientifically useful but differently scoped: its GP input is ability,
not item content, and it does not directly solve unseen-item cold start.
Therefore it is not a required rung of the main ladder.

An optional matched comparator is a **GP residual adapter for AIRT**:

\[
g_{sk}\sim\mathcal{GP}(0,k(e,e')),\qquad
y_{sj}\sim\operatorname{Bernoulli}
\left(\sigma(\ell^0_{sj}+g_s(x_j))\right).
\]

It conditions on the same acquired outcomes and item embeddings as the neural
adapters. Bernoulli GP inference is non-conjugate, so an implementation must
state its approximation, such as Laplace or variational inference.

## Experimental controls

- Run PAIEC Empirical Mean, PAIEC BLE, PAIEC Empirical Mean + BLE Acquisition,
  beta-binomial, 1PL/2PL, and NCF/calibration baselines separately.
- Give all matched adapters the same AIRT base, item features, training split,
  context prefixes, budgets, tuning budget, and seeds.
- Compare predictors first using identical fixed or random acquisition
  prefixes. Compare random, entropy, Fisher information, and expected
  information acquisition only after fixing the predictor.
- Ablate text embeddings, item metadata, cognitive-scale annotations, and
  subject metadata separately.
- Treat response traces as learning using privileged information: compare the
  same trace-free inference model trained with no trace supervision, auxiliary
  trace targets, and teacher--student latent alignment. Delete the trace branch
  at evaluation and prohibit validation/test target traces.
- Report Brier at every budget, Brier ALC, ECE, paired intervals across the
  same subject--benchmark groups, seed variation, parameter count, runtime,
  memory, and failures by benchmark.
- For a stochastic NP, specify the Monte Carlo estimate, sample count, and
  deterministic seed policy used at evaluation.

## Neural-Process-Family library assessment

[Yann Dubois et al.'s Neural-Process-Family](https://github.com/YannDubs/Neural-Process-Family)
is valuable as a reference implementation. It contains CNP, latent NP,
attentive CNP/NP, and convolutional variants under an MIT license. It should
not be added as a hard dependency yet:

- Its examples and pretrained models target synthetic 1-D and image regression,
  while this project needs a Bernoulli AIRT decoder and episodic
  subject--benchmark contexts.
- Its documented environment pins `skorch==0.8`, which is an avoidable legacy
  constraint for the competition submission.
- Its attention and set-encoding structure can still guide or seed a small,
  native PyTorch implementation. Any reused code must retain the MIT notice and
  cite the corresponding NP/ANP papers and implementation.

The implementation decision should follow a compatibility spike: reproduce a
tiny Bernoulli CNP episode with masks for budgets \(0,1,3,7,15,31\). If adapting
the library requires replacing its likelihood, decoder, batching, and training
wrapper, port only the tested encoder/attention ideas instead.

## Figure sources

- `paper/neurips_2026/figures/irt_model_family.tex`: seven-panel comparison,
  including the essential CNP control and published GPIRT as a separate
  literature comparator.
- `paper/neurips_2026/figures/raa_irt_model.tex`: detailed proposed model.

These are probabilistic-computation diagrams: circles are stochastic nodes,
rounded rectangles are deterministic maps, solid arrows are predictive/model
dependencies, and dashed arrows are inference or test-time conditioning.

The corrected PyTorch-oriented model and training skeleton is recorded in
[`docs/np-airt-implementation.md`](np-airt-implementation.md).
