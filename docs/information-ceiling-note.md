# An information-content ceiling check for the CNP-AIRT ablation ladder

*Working note — not part of the NeurIPS submission.*

## Purpose

Every neural variant in the ablation ladder (`ablation-results.md`,
`uniform-audit.md`) ties or loses to the Beta-Binomial baseline on
calibrated Brier ALC. Before attributing this to undertraining or an
objective bug, check whether the baseline is already near the
**information-theoretic ceiling** for this task: with at most 31 binary
responses, there is a hard limit on how precisely a scalar ability
$\theta$ can be known, and therefore a hard floor on achievable Brier
risk. If the Beta-Binomial baseline already sits at that floor, no
architecture change can beat it, and the ladder rungs are not testing
anything.

## Model

Assume a correctly specified 2PL item response model. Item $i$ has
discrimination $a_i > 0$ and difficulty $b_i$:

$$
p_i(\theta) = \sigma\!\big(a_i(\theta - b_i)\big), \qquad
\sigma(x) = \frac{1}{1+e^{-x}}.
$$

## Step 1: Fisher information from the acquired context

The Fisher information that a single response on item $i$ carries about
$\theta$ is

$$
I_i(\theta) = a_i^2\, p_i(\theta)\big(1 - p_i(\theta)\big).
$$

For a context $C$ of $n$ acquired items, the total information at the
true ability is additive:

$$
I_n(\theta) = \sum_{i \in C} a_i^2\, p_i(\theta)\big(1 - p_i(\theta)\big).
$$

This is exactly the quantity already computed as the negative Hessian in
the existing MAP solver (`faithful_np.py`, `map_theta`):

```python
hessian = -sum(a_C**2 * p_C * (1 - p_C)) - 1 / prior_variance
```

The codebase is already computing $I_n(\theta)$ at every forward pass; it
is simply not being read as a bound.

## Step 2: posterior variance (Laplace approximation)

With a Gaussian prior of variance $\tau^2$ on $\theta$ (the code uses
$\tau^2 = 4$), the Laplace approximation to the posterior at the MAP
estimate $\hat\theta$ gives

$$
\sigma_\theta^2(n) \;\approx\; \frac{1}{I_n(\hat\theta) + 1/\tau^2}.
\tag{1}
$$

By the Cramér–Rao bound, no estimator — linear, Bayesian, or a trained
neural network — can achieve a lower variance than equation (1) if the
2PL model is correctly specified and $(a_i, b_i)$ are known. This is the
hard floor referred to loosely as "~31 bits of information": the precise
statement is that $\sigma_\theta^2(n)$ is bounded below by the
reciprocal of total Fisher information, not literally Shannon bits.

## Step 3: propagating ability uncertainty into item-level Brier risk

For a target item $t$ with parameters $(a_t, b_t)$, the Bayes-optimal
predictor under squared loss is the posterior-predictive probability

$$
\hat p_t = \mathbb{E}_{\theta \sim q(\theta \mid C)}\big[p_t(\theta)\big].
$$

Its achievable risk decomposes exactly into an **estimation** term and an
**irreducible noise** term. For any fixed $\theta$,

$$
\mathbb{E}_{y \sim \mathrm{Bernoulli}(p_t(\theta))}\!\left[(\hat p_t - y)^2 \mid \theta\right]
= (\hat p_t - p_t(\theta))^2 + p_t(\theta)\big(1 - p_t(\theta)\big).
$$

Taking the expectation over $\theta \sim q(\theta \mid C)$ and using
$\hat p_t = \mathbb{E}_\theta[p_t(\theta)]$ for the first term:

$$
R(t) \;=\; \underbrace{\mathrm{Var}_{\theta \sim q}\big[p_t(\theta)\big]}_{\text{residual ability uncertainty}}
\;+\; \underbrace{\mathbb{E}_{\theta \sim q}\big[p_t(\theta)\big(1 - p_t(\theta)\big)\big]}_{\text{irreducible Bernoulli noise}}.
\tag{2}
$$

This is the Bayes risk floor for item $t$: it is what the best possible
predictor achieves, given only the information content of the context,
with the true item parameters known exactly.

### First-order closed form

Linearizing $p_t(\theta)$ around $\hat\theta$,
$p_t(\theta) \approx \hat p_t + a_t\, \hat p_t(1-\hat p_t)(\theta - \hat\theta)$,
and substituting $\sigma_\theta^2(n)$ from equation (1) gives a closed-form
ceiling that needs no simulation:

$$
R(t; n) \;\approx\;
\hat p_t \big(1 - \hat p_t\big)
\;+\;
a_t^2\, \hat p_t^2 \big(1 - \hat p_t\big)^2\, \sigma_\theta^2(n),
\qquad \hat p_t = p_t(\hat\theta).
\tag{3}
$$

## Step 4: the ceiling to compare against

Using the competition's own budget grid and weights,

$$
\mathrm{ALC}_{\min} \;=\; \sum_{b \in \{0,1,3,7,15,31\}} w_b \cdot
\mathbb{E}_{\text{tasks},\, t}\big[R(t;\, n=b)\big],
\tag{4}
$$

where the expectation over target items $t$ and tasks is taken either

- **in closed form**, by plugging the corpus's empirical distribution of
  $(a_i, b_i)$ into equation (3) and averaging, or
- **by Monte Carlo**: simulate $\theta_{\mathrm{true}} \sim
  \mathcal{N}(0, \tau^2)$, draw context and target responses from real or
  fitted $(a_i, b_i)$, run the *existing* MAP + Hessian solver exactly as
  in production, evaluate $R(t)$ by quadrature over the resulting
  $q(\theta)$, and average over many simulated tasks.

## The check

Compare $\mathrm{ALC}_{\min}$ from equation (4) against the measured
Beta-Binomial Brier ALC on validation ($0.222838$, per
`uniform-audit.md`).

- If the measured value is within noise of $\mathrm{ALC}_{\min}$, the
  baseline is already near-optimal given the information content of
  $\le 31$ binary responses. No architecture — CNP, ANP, GP-residual, or
  latent — can beat it without violating the Cramér–Rao bound, and ladder
  rungs 3 through 6 are testing against an unbeatable number rather than
  a weak baseline.
- If the measured value is meaningfully above $\mathrm{ALC}_{\min}$,
  there is real headroom, and the null results documented in
  `ablation-results.md` and `uniform-audit.md` point to a training or
  objective problem (undertraining, KL misweighting, detached MAP
  gradient) rather than a fundamental information limit.

This is a half-day closed-form-plus-Monte-Carlo computation against code
that already exists (`map_theta`), and it should be run before committing
further compute to architecture rungs 3–6 of the ablation ladder.
