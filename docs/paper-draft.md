# What Transfers Is Ability, Not Difficulty: Online Rasch Beats Neural Processes for Unseen LLM Benchmark Evaluation

*Poster hook: the network learned each model's ability, which transfers, and each benchmark's item difficulty, which doesn't; refitting Rasch online fixes it with no training.*

**Keywords:** LLM evaluation, predictive evaluation, test-time adaptation, benchmark shift.

*Draft, 2026-11-01. Numbers come from the runs in this repository; every figure
reference names the PDF and page it lives on. Items marked **[TODO]** are not
done yet.*

---

## Abstract

Evaluating language models across benchmarks is expensive, so a natural goal is to predict how a model will answer each item of a new benchmark after seeing only a few labelled answers. Neural processes offer one approach: trained on many past (model, benchmark) tasks, they infer a new task from a handful of labels. But evaluation rarely stays in distribution. New benchmarks and new models keep arriving, so a predictor trained on past tasks is asked about unseen benchmarks and models. We ask what such a predictor actually transfers. We build an online protocol in which labels are bought one at a time and predictions are scored by Brier ALC, on unseen benchmarks and models. Attentive Neural Processes trained this way fail to beat a Beta-Binomial estimator on the unseen benchmarks. Because item difficulty is specific to each benchmark, we instead estimate it from the new benchmark itself, at test time. In simulations of the online protocol on three unseen benchmarks, a Rasch (one-parameter logistic item response theory) model refit on the labels revealed for all models, with no offline training, improves on the Beta-Binomial baseline (−0.0035 Brier ALC, 95% CI [−0.0046, −0.0026]), and a Gaussian-process prior that ties the difficulty of similar items together improves it further (−0.0049). A model's ability carries over to an unseen benchmark, but item difficulty does not, so it is better estimated online than learned from past benchmarks.

---

## 1. Introduction

Language models are evaluated on hundreds of benchmarks, and the number keeps
growing as new models and new benchmarks appear. Running every model on every
item is expensive in compute, in API cost and, for open-ended items, in human
grading. A predictor that infers a model's per-item results from a handful of
labelled answers would let an evaluator spend labels where they matter, extend a
few observed results across benchmarks and screen new models cheaply. Its
failure mode is costly too: confident, wrong estimates for exactly the cases
that matter most, the new models and the new benchmarks.

This is a few-shot prediction problem over tasks, one task being one *subject*
(a model under given evaluation settings) answering one *benchmark*
(Section 2). Meta-learning is the natural tool: learn from many past tasks how
to predict a new one from a few labels. Neural processes (NPs) [Garnelo et al.,
2018a; 2018b] and their attentive variant (ANP) [Kim et al., 2019] do this by
conditioning on the labelled items in a single forward pass, with no test-time
fitting; amortized item-response models [Truong et al., 2025] do it for item
difficulty from item text. Such methods are mostly validated where test tasks
resemble training tasks. Evaluation rarely does: new benchmarks bring items
unlike any seen before, and nothing in training says how those items differ in
difficulty. Zhang et al. (2025) show the analogous failure for aggregate
benchmark prediction: learned predictors work when a new model resembles the
training models and fail on unfamiliar ones, while an estimator anchored to the
observed labels survives.

We study item-level, streaming prediction under a strict benchmark holdout:
labels are bought one at a time, and the test benchmarks and their models are
never seen in training. We ask three questions.

- **RQ1.** Do neural processes trained on past benchmarks beat a Beta-Binomial
  estimator, which uses no past data, on unseen benchmarks and models?
  (Section 5.1)
- **RQ2.** Does estimating item difficulty at test time, from the new
  benchmark's own labels and with no offline training, do better? (Section 5.2)
- **RQ3.** What explains the gap: which part of a prediction, a subject's
  ability or the difficulty of its items, carries over to a new benchmark?
  (Section 6)

Our answers are: no, yes, and ability but not difficulty. We contribute an
online simulation protocol that mirrors the competition's draws (Section 3), a
negative result and a diagnosis for neural processes (RQ1 and RQ3), and a
test-time Rasch estimator, optionally with a Gaussian-process prior over item
difficulty, that improves on the Beta-Binomial at every budget (RQ2).

---

## 2. Preliminaries

**Tasks.** A *subject* is a language model together with its evaluation settings
(provider, harness, reasoning effort); the same model under different settings is
a different subject. A *benchmark* is a set of *items* (questions or prompts), and
a subject's *response* to an item is binary: $y = 1$ if correct, $0$ otherwise. A
**task is one subject-benchmark pair**: the items that subject answered on that
benchmark, each with its response, that is, a set of pairs $(x_i, y_i)$ where $x_i$
is the embedding of item $i$'s text. Every training episode and every evaluation
in this paper is a task. In our filtered data each subject appears on one
benchmark, so holding out a benchmark also holds out its subjects.

**Prediction from labels.** For a task, a predictor receives a *context* $C$, the
(item, response) pairs revealed so far, and outputs for every *target* item $x$ a
probability $\hat p = p(y = 1 \mid x, C)$. The number of revealed labels
$k = |C|$ is the *budget*; at $k = 0$ only item text and priors are available. A
*neural process* models this conditional. In *meta-training*, each episode
samples a training task, splits its items at random into a context $C$ and targets
$T$, and updates the network to predict the targets' responses from the context
[Garnelo et al., 2018a;b]. For a latent neural process the loss is the negative
evidence lower bound (ELBO),

$$
\mathcal{L}(\phi) = \mathbb{E}_{\tau}\,\mathbb{E}_{C,T}\Big[\,
\mathbb{E}_{z \sim q_\phi(z \mid C \cup T)}\big[-\log p_\phi(y_T \mid x_T, z, C)\big]
+ \tfrac{1}{|T|}\,\mathrm{KL}\big(q_\phi(z \mid C \cup T)\,\|\,q_\phi(z \mid C)\big)\Big],
$$

where $\tau$ ranges over training tasks and $z$ is a latent variable shared by
context and targets. At test time the weights are frozen and a new task's
revealed labels are read in one forward pass. This is what we call *amortized*:
all fitting happens once, before evaluation, across training tasks.

**Metrics.** The Brier score of a set of targets is
$\frac{1}{|T|}\sum_{i \in T} (\hat p_i - y_i)^2$ (lower is better; predicting 0.5
always scores 0.25) [Brier, 1950]. The *learning curve* is the Brier score at each
budget $k \in \{0, 1, 3, 7, 15, 31\}$, and **Brier ALC** is its weighted average,
$\mathrm{ALC} = \sum_k w_k\, \mathrm{Brier}_k$ with
$w = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)$ (the normalized area under the curve on a log
scale), so 70% of the weight is on seven labels or fewer.

**Beta-Binomial baseline.** Treat a subject's answers on a benchmark as
exchangeable Bernoulli draws with one unknown accuracy $\pi$, and put a Beta prior
on it, $\pi \sim \mathrm{Beta}(\alpha, \beta)$. After $n$ revealed labels, $h$ of
them correct, conjugacy gives the posterior $\mathrm{Beta}(\alpha + h,\ \beta + n - h)$.
The probability that the next, unrevealed item is answered correctly is the
posterior mean (the Beta and Gamma functions of the full Beta-Binomial likelihood
cancel):

$$
\hat p = \frac{\alpha + h}{\alpha + \beta + n}.
$$

The *Jeffreys prior* $\alpha = \beta = \tfrac{1}{2}$ [Jeffreys, 1946] is the
standard default prior for a Bernoulli rate: it is defined so that it does not
depend on how the rate is parameterized, so it encodes as little as possible. It
gives

$$
\hat p = \frac{h + 0.5}{n + 1}.
$$

Here $\hat p$ is the same for every item of the task, $h$ counts the correct
revealed answers, $n$ the labels revealed so far, and the prior is worth half a
correct answer ($\alpha$) out of one pseudo-label ($\alpha + \beta$). This is
*shrinkage*: the estimate is pulled from the raw hit rate $h/n$ toward the prior
mean 0.5, strongly when $n$ is small and negligibly when $n$ is large. With no
labels it predicts 0.5. It ignores item content and other tasks; Appendix D.3
tests its exchangeability assumption.

**Item response theory and the Rasch model.** Item response theory (IRT) models
the probability of a correct answer as a function of a latent subject *ability*
$\theta_s$ and properties of the item. In the Rasch (one-parameter logistic) model
[Rasch, 1960; Lord, 1980],

$$
\operatorname{logit} P(y_{si} = 1) = \theta_s - b_i, \qquad
\operatorname{logit}(p) = \log \frac{p}{1 - p},
$$

where $b_i$ is the item's *difficulty*; a subject whose ability equals an item's
difficulty answers it correctly half of the time. Parameters can be fitted by
maximum likelihood. Adding Gaussian priors and maximizing likelihood times prior
gives the *maximum a posteriori* (MAP) estimate, which for Gaussian priors is
ridge-penalized logistic regression.

**In-distribution and out-of-distribution.** A test task is *in-distribution*
(ID) when its benchmark also appears in training, as in a random split of
subjects, and *out-of-distribution* (OOD) when its benchmark never appears, as in
the benchmark holdout, which also holds out the subjects. "Unseen" means OOD.

**Online simulation, single-pair and multi-pair.** We cannot query the
competition's live models, so we evaluate by *online simulation*: the online
procedure, in which labels are revealed one at a time in a frozen candidate
order, is re-run on recorded responses. In a **single-pair** simulation each task
is simulated alone, so the predictor sees only that task's own revealed labels.
In a **multi-pair** simulation several tasks are simulated together in a *draw*,
like the real competition, so the predictor may also use the labels revealed in
the other tasks of the draw.

**Confidence intervals and significance.** Every interval is a 95% bootstrap
interval over tasks. The bootstrap resamples the tasks *with replacement* (each
resample has as many tasks as the original, some appearing several times and
others not at all), recomputes the statistic on each resample, and takes the 2.5th
and 97.5th percentiles over thousands of resamples. Differences between two
models are *paired*: computed per task before resampling. A difference is
*significant* when its 95% CI excludes zero (n.s. = it doesn't).

---

## 3. Setting and evaluation protocol

**Protocol.** At test time the evaluator buys up to 31 labels per task, one
candidate item at a time, deciding to accept or skip each candidate before
seeing the next. After 0, 1, 3, 7, 15 and 31 labels the predictor outputs a
probability for every item in a fixed evaluation pool whose labels are never
revealed, scored by Brier ALC (Section 2). The competition also passes the
predictor every label purchased so far across all sampled tasks, together with
each subject's metadata (model name, provider, release date); Section 5.2
exploits the first.

**Split.** We hold out whole benchmarks: 11 for training, 3 for validation
(ai2d_test, bird_sql, financebench) and 3 for test (algotune, cmmlu,
editbench). No validation or test subject appears in training. Responses are
filtered to items with at least 15 responders and subjects with at least 80
items per benchmark, leaving 6.34 M responses over 123,035 items and 17
benchmarks. Figure A2 contrasts this with a random-subject split, which holds
out subjects but keeps every benchmark in training.

**Online simulation.** Each validation or test task's items are split once,
50/50, into an acquisition pool, from which labels are bought, and the
evaluation pool. Candidate orders over the acquisition pool are frozen (3 orders
per validation task, 10 per test task), so every model sees the same labels in
the same order. We report both protocols of Section 2: single-pair (82
validation and 45 test tasks) and multi-pair (draws of one or two subjects per
benchmark; 60 validation and 150 test draws). Models that ignore other tasks
are placed on the same draws from their single-pair scores, so comparisons are
paired. Unless stated, labels are acquired in random order (Appendix C.4 shows
the acquisition policy does not matter).

---

## 4. Methods

**Attentive Neural Process (ANP).** Each item is a 1024-d BGE-M3 embedding of its
text. A context of (item, label) pairs is encoded by a shared value embedding and
one self-attention block; a target item attends to the context (keys from
embeddings only, values carrying labels) to form a deterministic representation,
and a mean-pooled latent path gives a global $z$ (width 64, 4 heads, 32-d latent).
A decoder maps [target embedding, attention output, $z$] to a logit, and training
minimizes the negative ELBO of Section 2 over episodes from training tasks. The
architecture is drawn next to Kim et al.'s ANP in Figure B1
(`docs/np-architecture-final.pdf`, pp. 1-3). The variants we simulated online
differ in training only (Appendix B.2).

**Prefix training.** Standard training scores each episode at one randomly drawn
context size, whereas online evaluation reveals labels one at a time and scores
every prefix of the same label stream. Prefix training matches that: one random
stream per task, the same held-out targets scored after every prefix of length
$0, 1, \dots, 31$, and the losses averaged before one update (Appendix B.3,
Algorithm 1).

**Sharpness-Aware Minimization (SAM).** One variant adds SAM [Foret et al., 2021]
on top of AdamW. Each update first perturbs the weights by a small step (radius
0.05) in the direction that increases the loss, computes the gradient there, and
applies the AdamW update with that gradient, which favors flat regions of the
loss surface. We tried it because flat minima are expected to generalize better
to unseen benchmarks. It helps a little (test Brier ALC 0.2663 against 0.2725 for
the same ANP without it) and does not change the conclusion: it remains
significantly worse than the Beta-Binomial (Section 5.1).

**Pooled Rasch.** At every checkpoint we fit

$$
\operatorname{logit} p(s, i) = \mu_b + \theta_s - b_i, \qquad
\mu_b \sim \mathcal{N}(m_0, 2^2),\;\; \theta_s \sim \mathcal{N}(0, 1^2),\;\; b_i \sim \mathcal{N}(0, 1^2),
$$

by MAP to *all* labels revealed so far in the draw, across subjects and
benchmarks. Here $\mu_b$ is the base rate of benchmark $b$, and
$m_0 = \operatorname{logit}(0.512)$ is the training-benchmark mean, which is close
to uninformative. Items are matched across subjects by content; an item no one has
labelled gets $b_i = 0$. The three prior widths were chosen on validation. There
is no training: the fit is ridge-penalized logistic regression with about 300
parameters, solved by Newton's method in the submission and L-BFGS in the online
simulation, and refit from scratch at every checkpoint.

**Kernel Rasch.** Answers on text-similar items within a task are correlated
(residual correlation $+0.24$ on test against about $0$ for random item pairs;
Appendix D.3), but pooled Rasch gives each item an independent difficulty prior,
so a label informs only that exact item. Kernel Rasch instead draws the
difficulties of a benchmark's items jointly,

$$
\mathbf{b} \sim \mathcal{N}(\mathbf{0}, K), \qquad
K_{ij} = \tau^2 \Big[ (1 - \rho)\,\delta_{ij} + \rho \exp\!\Big( \frac{\cos(x_i, x_j) - 1}{T} \Big) \Big],
$$

where $x_i$ is the item-text embedding, so items with similar text have similar
difficulty. An item nobody has labelled gets the kriging estimate
$\hat b_i = K_{iL} K_{LL}^{-1} \hat{\mathbf{b}}_L$ from the labelled items $L$. The
settings ($\tau = 1.4$, $\rho = 0.8$, $T = 0.2$) were chosen on validation;
everything else is as in pooled Rasch. Appendix C.8 compares it with a clustering
variant.

**Other baselines.** A constant at the training mean; the raw hit rate $h/n$ (no
shrinkage); a hierarchical Beta-Binomial whose prior mean depends on subject
metadata (Appendix D.5).

---

## 5. Results

### 5.1 RQ1: Neural processes do not beat the Beta-Binomial on unseen benchmarks

**Answer: no.** All three ANP variants simulated on the unseen test benchmarks score significantly worse than the Beta-Binomial, which uses no past data (Table 1, Figure 1).

**Figure 1** (`docs/online-final-results.pdf`, p. 1): single-pair Brier per
budget, validation and test.

**Table 1** (RQ1). Brier ALC on validation and test (single-pair) and the test difference to the Beta-Binomial; negative is better.

| Model (single-pair) | Val ALC | Test ALC | Test minus Beta-Binomial [95% CI] |
| --- | --- | --- | --- |
| Constant (training mean) | 0.2525 | 0.2472 | -0.0022 [-0.0105, +0.0067] |
| Beta-Binomial | 0.1717 | 0.2494 | - |
| ANP | 0.1735 | 0.2725 | +0.0230 [+0.0198, +0.0267] |
| ANP + SAM | 0.1667 | 0.2663 | +0.0168 [+0.0128, +0.0211] |
| ANP, prefix training | 0.1735 | 0.2762 | +0.0267 [+0.0231, +0.0306] |
| Raw hit rate **[POSSIBLE_APPENDIX]** | 0.1892 | 0.2879 | +0.0385 [+0.0355, +0.0413] |

Combining the ANP with the Beta-Binomial, either by falling back to it at 0-1
labels or by averaging the two predictions, removes most of the loss but never
beats it (Appendix C.3).

Every benchmark-holdout ANP is significantly worse than the Beta-Binomial on
test. ANP + SAM beat it on validation (0.1667 against 0.1717), driven by two
validation benchmarks, and the advantage reversed on test: validation wins do
not transfer. The deficit is concentrated at zero and one label (+0.034 and
+0.079 at budgets 0 and 1 for the ANP; Figure 1, Appendix C.1); from three
labels on, the ANP and the Beta-Binomial are within noise. On test the
constant predictor ties the Beta-Binomial: the Beta-Binomial's error at one
label (0.28) is above the constant's, and later budgets only just recover it.

### 5.2 RQ2: Estimating difficulty at test time beats the Beta-Binomial

**Answer: yes, modestly.** A Rasch model refit on every task's revealed labels, with no offline training, improves on the Beta-Binomial at every budget, and a Gaussian-process prior over item text improves it further (Table 2, Figure 2).

**Figure 2** (`docs/online-final-results.pdf`, p. 2): multi-pair Brier per
budget.

**Table 2** (RQ2). Brier ALC on validation and test (multi-pair) and the test difference to the Beta-Binomial; negative is better.

| Model (multi-pair) | Val ALC | Test ALC | Test minus Beta-Binomial [95% CI] |
| --- | --- | --- | --- |
| Constant (training mean) | 0.2527 | 0.2472 | - |
| Beta-Binomial | 0.1685 | 0.2497 | - |
| ANP | 0.1701 | 0.2726 | - |
| **Pooled Rasch** | **0.1674** | **0.2461** | **-0.0035 [-0.0045, -0.0026]** |
| Pooled Rasch without item difficulty | 0.1675 | 0.2477 | -0.0020 [-0.0029, -0.0011] |
| **Kernel Rasch** | 0.1667 | **0.2448** | **-0.0049 [-0.0060, -0.0038]** |

Pooled Rasch and kernel Rasch are the only models below the Beta-Binomial on
test at every budget, and kernel Rasch improves on pooled Rasch by 0.0013 [0.0010,
0.0017] (Appendix C.8). Removing the item term halves the gain, so both sources of shared
evidence contribute: other subjects' labels on the same benchmark sharpen the
benchmark rate and the subject's ability, and their labels on the same item
reveal that item's difficulty. Another subject's label on the same item
predicts the outcome with AUC 0.74 on validation and 0.64 on test; only 34% of
test evaluation items ever receive such a label, which bounds the item term's
contribution.

**Competition-shaped draws.** The competition scores about seven benchmarks with
mostly one subject each, so most benchmarks have no other subject to share with.
On simulated draws with exactly one subject per benchmark the gains shrink:
pooled Rasch, with its priors re-tuned on such draws, improves on the
Beta-Binomial by 0.0032 [0.0029, 0.0035] and kernel Rasch by 0.0024 [0.0021,
0.0027] (Appendix C.9). On the leaderboard, pooled Rasch (CodaBench 1024003) scored
0.2022 against 0.1486 for the Beta-Binomial (1005155), but each submission is
scored on its own nine pairs, so the two scores are unpaired and noisy, and the
online simulation does not reproduce the gap.

**[POSSIBLE_APPENDIX]** Surprisingly, pooled Rasch does not estimate a subject's *ability* better than
the Beta-Binomial (Appendix D.4): its ability prior is tight and shrinks the
estimate. Its advantage comes from the benchmark rate and item difficulty, not
from a sharper ability estimate.

---

## 6. RQ3: What transfers across benchmarks, ability or difficulty?

**Answer: a model's ability, in a limited sense; not the difficulty of items.**

**An ANP's ability component transfers; its difficulty component does not.** We
split each ANP prediction into an *ability component*, its mean over the task's
items (how well the ANP thinks the subject does on this benchmark overall), and a
*difficulty component*, each item's offset from that mean (which items it thinks
are easier or harder than average). Predictions of 0.7, 0.5 and 0.3 for three
items, for example, have ability component 0.5 and difficulty components +0.2, 0
and -0.2. We score each component on its own by pairing it with the
Beta-Binomial's flat prediction for the other (Appendix D.1).

From three labels on, the ability component alone is as good as the
Beta-Binomial on unseen benchmarks for the benchmark-holdout ANPs (within 0.003)
and slightly better for the random-subject-split ANPs (-0.010). Before any label
arrives the benchmark-holdout ANP's ability estimate is miscalibrated (+0.04 to
+0.06), which is the whole deficit of Section 5.1. The difficulty component is
where benchmark-specific structure sits. The random-subject-split ANPs, which
trained on editbench among other benchmarks, learned an ordering that wins on
editbench (about -0.05 to -0.08) and loses on unseen benchmarks (+0.04 to
+0.05). The benchmark-holdout ANPs learned an almost flat ordering, which is
harmless (within 0.003 of the Beta-Binomial). Prefix training is not the cause:
its difficulty component is as harmless as that of the standard ANP. What
"transfers" is therefore an ability estimate that is no better than the
Beta-Binomial's once labels arrive, not a well-calibrated prior over ability.

**Table 3** (RQ3). Test Brier minus the Beta-Binomial's, using the full ANP prediction,
only its ability component, or only its difficulty component (the other replaced
by the Beta-Binomial's flat prediction), for budgets 0-1 and 3-31. Negative is
better; * marks a 95% CI that excludes zero.

| Model | Full, 0-1 | Ability, 0-1 | Difficulty, 0-1 | Full, 3-31 | Ability, 3-31 | Difficulty, 3-31 |
| --- | --- | --- | --- | --- | --- | --- |
| ANP | +0.064* | +0.064* | +0.000 | +0.003 | +0.003 | +0.000 |
| ANP + SAM | +0.040* | +0.037* | +0.003* | +0.003 | +0.000 | +0.002* |
| ANP, prefix training | +0.064* | +0.063* | +0.001 | +0.005* | +0.002 | +0.003* |
| ANP, random-subject split: unseen benchmarks | +0.040* | -0.003 | +0.039* | +0.046* | -0.010* | +0.050* |
| ANP, random-subject split: editbench (seen in training) | -0.073* | -0.025* | -0.053* | -0.081* | -0.011* | -0.073* |

**Text carries within-benchmark, not cross-benchmark, difficulty.** A ridge
probe from BGE-M3 embeddings to item pass rate reaches r = 0.22 to 0.88 when
fitted and tested inside one benchmark, but about 0 when the within-benchmark
ordering is tested on a held-out benchmark (Appendix D.2). Transfer grows
slowly with the number of training benchmarks, from r = 0.007 with one to
0.076 with 14, far from the within-benchmark level. The ANP's budget-0 item
predictions mirror this: r = +0.46 on benchmarks the random-split model
trained on, +0.14 on the rest, and about -0.05 on test for the holdout models
(`docs/item_difficulty-final-results.pdf`, pp. 2 and 10).

**The ANP overfits the training benchmarks.** With the split as the only
difference, the same ANP's validation NLL keeps falling in distribution but
bottoms out near 0.48 at about 3,000 steps and rises to 0.59 on unseen
benchmarks while its training loss keeps falling (Figure 3,
`docs/training-final-results.pdf`, p. 2).

**Subject metadata breaks under model shift.** **[POSSIBLE_APPENDIX]** Test subjects are newer and
stronger than training subjects. A Beta-Binomial prior conditioned on provider
and release date loses on test (+0.013, budget 0 +0.091) because the
release-date term extrapolates; centring pooled Rasch's ability prior on the
provider was chosen on validation and loses on test (Appendix D.5, C.7).

**Gradient-based adaptation is unlikely to help.** **[MEHH_REMOVE]** (untested) MAML-style test-time
fine-tuning [Finn et al., 2017] adapts weights from the same few labels, so it
faces the same 0-7-label regime and cannot recover item difficulty from one
subject's answers. Pooled Rasch is the low-dimensional, closed-form version of
test-time adaptation, and it adapts the parameters that matter.

---

## 7. Related work

**Neural processes.** CNPs and NPs [Garnelo et al., 2018a; 2018b] amortise
inference over functions; ANPs [Kim et al., 2019] add cross-attention to fix
underfitting; incremental transformer NPs [Mortimer et al., 2026] make the
update causal and cacheable. We test them under a benchmark holdout, where the
target functions come from unseen benchmarks.

**Model-based and amortized evaluation.** Item response theory [Rasch, 1960;
Lord, 1980] separates subject ability from item difficulty; Truong et al.
(2025) amortise item parameters from text for efficient LLM evaluation. Our
pooled Rasch is classical IRT refit online; our results suggest the amortized
part is what fails to transfer across benchmarks.

**Item features and Gaussian processes in IRT.** Two lines relax the Rasch
model in different directions. *Explanatory IRT* predicts item difficulty
from item features, classically as a linear function in the linear logistic
test model [Fischer, 1973; De Boeck and Wilson, 2004]. *GPIRT* [Duck-Mayr et
al., 2020] instead places Gaussian-process priors on each item's response
function of ability, relaxing the logistic shape while keeping items
independent. Our kernel Rasch belongs to the first line: a Gaussian-process
prior over item difficulty as a function of the item's text embedding, so
items with similar text share difficulty, refit online at every checkpoint.
It keeps the logistic response function, so it is not GPIRT; the two could
be combined.

**Benchmark prediction from few data.** Zhang et al. (2025) compare 11 methods
on 19 benchmarks for predicting aggregate accuracy and find that learned
predictors fail on unfamiliar models, while an AIPW estimator anchored to the
observed labels holds up. We find the item-level analogue: under benchmark
shift the estimator anchored to observed labels (Beta-Binomial, pooled Rasch)
beats the learned prior. An AIPW-style correction does not carry over to
item-level probabilities from a handful of labels (Appendix C.3).

**Meta-learning.** MAML [Finn et al., 2017] adapts by gradient steps; NPs by
conditioning. Both assume the training tasks resemble the test tasks, which a
benchmark holdout breaks.

---

## 8. Limitations

- **Three held-out benchmarks per role.** Validation and test hold three
  benchmarks each, and benchmark difficulty varies widely (Table A1): test
  benchmarks are mid-to-easy (0.54-0.69) and validation includes the hardest
  (bird_sql, 0.15). Results on this split can turn on which benchmarks are
  held out. **[TODO]** Leave-one-benchmark-out over all 17 benchmarks, with
  the multi-pair draws inside each fold.
- **Training is weighted by subject count.** alignment_faking supplies 4,278
  of 6,013 training subjects (71%). **[TODO]** Weight training losses and
  fitted priors per benchmark.
- **Simulated competition draws.** The multi-pair draws use three test
  benchmarks with one or two subjects each; the competition's hidden draw has
  about seven benchmarks. **[TODO]** Report the CodaBench score of the pooled
  Rasch submission (1024003).
- **One training seed** per ANP configuration.
- **Random acquisition** for the new models; the five acquisition policies were
  compared only for the ANPs and the Beta-Binomial.

---

## 9. Conclusion

On unseen benchmarks with a few revealed labels, the neural processes we trained
do not beat a Beta-Binomial estimator. A model's ability, estimated from its own
labels, carries over to a new benchmark; the difficulty of the benchmark's items
does not, and what the networks learn about it is specific to the benchmarks they
saw. Estimating difficulty at test time from the new benchmark's own labels,
shared across subjects (Rasch) and across similar items (kernel Rasch), improves
on the Beta-Binomial without any offline training. For online evaluation under
benchmark shift, estimation at test time is the first thing to try, before
learning a prior.

---

## References

- **[Brier, 1950]** G. W. Brier. Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1), 1950.
- **[Finn et al., 2017]** C. Finn, P. Abbeel, S. Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML*, 2017.
- **[De Boeck and Wilson, 2004]** P. De Boeck, M. Wilson (eds.). *Explanatory Item Response Models: A Generalized Linear and Nonlinear Approach*. Springer, 2004.
- **[Duck-Mayr et al., 2020]** J. Duck-Mayr, R. Garnett, J. M. Montgomery. GPIRT: A Gaussian Process Model for Item Response Theory. *UAI*, PMLR 124, 2020.
- **[Fischer, 1973]** G. H. Fischer. The linear logistic test model as an instrument in educational research. *Acta Psychologica* 37(6), 1973.
- **[Foret et al., 2021]** P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR*, 2021.
- **[Garnelo et al., 2018a]** M. Garnelo, D. Rosenbaum, C. Maddison, T. Ramalho, D. Saxton, M. Shanahan, Y. W. Teh, D. Rezende, S. M. A. Eslami. Conditional Neural Processes. *ICML*, 2018.
- **[Garnelo et al., 2018b]** M. Garnelo, J. Schwarz, D. Rosenbaum, F. Viola, D. J. Rezende, S. M. A. Eslami, Y. W. Teh. Neural Processes. *ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models*, 2018.
- **[Jeffreys, 1946]** H. Jeffreys. An invariant form for the prior probability in estimation problems. *Proc. Royal Society A* 186, 1946.
- **[Kim et al., 2019]** H. Kim, A. Mnih, J. Schwarz, M. Garnelo, A. Eslami, D. Rosenbaum, O. Vinyals, Y. W. Teh. Attentive Neural Processes. *ICLR*, 2019.
- **[Lord, 1980]** F. M. Lord. *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum, 1980.
- **[Mortimer et al., 2026]** P. Mortimer, C. Diaconu, T. Rochussen, B. K. Mlodozeniec, R. E. Turner. Incremental Transformer Neural Processes. *ICML*, PMLR 306, 2026.
- **[Rasch, 1960]** G. Rasch. *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish Institute for Educational Research, 1960.
- **[Truong et al., 2025]** S. T. Truong, Y. Tu, P. Liang, B. Li, S. Koyejo. Reliable and Efficient Amortized Model-based Evaluation. *ICML*, PMLR 267, 2025.
- **[Zhang et al., 2025]** G. Zhang, F. E. Dorner, M. Hardt. How Benchmark Prediction from Fewer Data Misses the Mark. arXiv:2506.07673, 2025.
- **[TODO]** BGE-M3 embedding model citation; Qwen3 embedding citation.

---

# Appendix

## A. Extended preliminaries and setting

Supports Sections 2 and 3.

**A.1 Benchmark and subject difficulty.** Table A1
(`docs/data-analysis-final-results.pdf`, p. 4) lists every benchmark's subject
accuracy by role.

| Role | Benchmark | Items | Subjects | Mean acc. |
| --- | --- | --- | --- | --- |
| Train | appworld | 462 | 18 | 0.28 |
| Train | alignment_faking | 1,464 | 4,278 | 0.41 |
| Train | cruxeval | 1,245 | 26 | 0.42 |
| Train | bridging_gap | 35,984 | 417 | 0.46 |
| Train | helm_afr | 28,662 | 15 | 0.52 |
| Train | hallusionbench | 948 | 253 | 0.56 |
| Train | harmbench | 801 | 29 | 0.59 |
| Train | afrimedqa | 7,900 | 31 | 0.63 |
| Train | brace | 4,987 | 28 | 0.66 |
| Train | afrieval | 14,141 | 898 | 0.68 |
| Train | agi_elo | 13,393 | 20 | 0.70 |
| Validation | bird_sql | 220 | 120 | 0.15 |
| Validation | financebench | 143 | 16 | 0.50 |
| Validation | ai2d_test | 3,088 | 254 | 0.69 |
| Test | editbench | 464 | 44 | 0.54 |
| Test | cmmlu | 8,994 | 16 | 0.68 |
| Test | algotune | 139 | 18 | 0.69 |

Pooled per subject, mean accuracy is 0.47 (train, 6,013 subjects), 0.52
(validation, 390) and 0.61 (test, 78).

**A.2 How the splits cut the response matrix.** Figure A2
(`docs/data-analysis-final-results.pdf`, p. 2): the random-subject split
removes subjects inside every benchmark; the benchmark holdout removes whole
benchmarks; inside each held-out task the items are split into an acquisition
pool and a scored evaluation pool. Split diagnostics (mean response,
item-pass-rate spread, text shift, retained responses): p. 1.

**A.3 Ability shift.** Figure A3 (`docs/item_difficulty-final-results.pdf`,
p. 13): subject-ability distributions for the training tasks, validation and
test.

## B. Extended methods

Supports Section 4.

**B.1 Architecture.** `docs/np-architecture-final.pdf`: p. 1, NP and ANP after
Kim et al. (2019); p. 2, our ANP; p. 3, its graphical model.

**B.2 Training variants.** All use the benchmark holdout, BGE-M3 embeddings,
width 64, one self-attention block, 4 heads, a 32-d latent, and early stopping
on a fixed validation bank. Overview table: `docs/training-final-results.pdf`,
p. 1.

| Variant | What differs | Steps | Selected step |
| --- | --- | --- | --- |
| ANP | uniform 0-31 context sizes | 100,000 | 1,200 |
| ANP, budget-ladder training | context sizes drawn from the scored budgets, ALC weights | 15,000 | 50 |
| ANP + SAM | sharpness-aware minimisation [Foret et al., 2021], rho = 0.05 | 30,000 | 5,450 |
| ANP + prior anchor | output is the Beta-Binomial logit plus a learned residual | 10,000 | 100 |
| ANP + SAM, budget-ladder training | both | 30,000 | 4,350 |
| ANP, prefix training | every prefix 0-31 of one label stream, same targets (B.3, Algorithm 1) | 10,000 | 4,550 (simulated online; the run's best later moved to 7,800) |
| ANP without item embeddings | all item embeddings set to zero | 15,000 [TODO: finishing] | 400 so far |
| ANP, random-subject split (width 64 / 128) | random-subject split, trains on 7 benchmarks incl. editbench | 100,000 | 79,700 / 97,800 |

**B.3 Prefix training.** *Motivation.* Standard episodic training [Garnelo et al., 2018b] draws one context
size per episode, so each sampled label sequence contributes a loss at a single
size, and context sets of different sizes in a batch come from unrelated subsets.
Online evaluation is different: labels arrive one at a time on a single stream, the
context at budget $k$ is exactly the first $k$ labels of that stream, and every
budget is scored against the same held-out items. The predictive sequence
$p(y_T \mid C_0),\, p(y_T \mid C_1), \dots, p(y_T \mid C_{31})$ should therefore be
learned from nested contexts $C_0 \subset C_1 \subset \cdots$. Prefix training
scores all of them.

*Objective.* For a task with items $I$, draw a random permutation $\varpi$ of $I$.
The stream is $s = (\varpi_1, \dots, \varpi_K)$ with $K = \min(31, |I| - 1)$, and the
targets $T = (\varpi_{K+1}, \dots, \varpi_{K+128})$ are the next up to 128 items,
disjoint from the stream. With $C_k = \{(x_{s_j}, y_{s_j}) : j \le k\}$ the $k$-th
prefix,

$$
\mathcal{L}_{\text{prefix}}(\phi) = \mathbb{E}_{\tau}\,\mathbb{E}_{\varpi}\Big[\frac{1}{K + 1}\sum_{k=0}^{K} \ell_\phi(C_k, T)\Big],
$$

where $\ell_\phi(C, T)$ is the model's usual per-episode loss: the targets' negative
log-likelihood, plus for the latent ANP the KL term of Section 2 (the negative ELBO).

```
Algorithm 1  Prefix training of a neural process
Input:  training tasks D, each a set of (item embedding, binary label) pairs;
        max context K_max = 31; max targets M = 128; tasks per step B = 2;
        steps S; learning rate eta
for step = 1 .. S:
    E <- []                                        # episodes in this step
    for b = 1 .. B:
        draw task t from D                         # shuffled, without replacement per epoch
        pi <- random permutation of t's items
        K  <- min(K_max, |t| - 1)
        stream  <- pi[1 .. K]                      # labels arrive in this order
        targets <- pi[K+1 .. K+M]                  # never in any prefix
        for k = 0 .. K:                            # every prefix of the same stream
            C_k <- stream[1 .. k] with labels
            append (C_k, targets) to E
    loss <- mean over (C, T) in E of l_phi(C, T)   # B * (K+1) episodes, 64 at K = 31
    phi  <- AdamW step on grad loss (gradient clipped to norm 5)
    every 50 steps: score a frozen validation bank the same way, all prefixes
                    of fixed streams, and keep the best checkpoint by target Brier
```

*Cost and settings.* Each step processes B(K + 1) = 64 episodes from two
tasks, against 8 single-size episodes from 8 tasks in the standard recipe, so
a step sees fewer distinct tasks but every context size of each. Validation
uses the same construction on a bank of streams frozen before training, so
checkpoint selection is consistent with the training objective. The run used
10,000 steps; its best validation checkpoint moved from step 4,550 (the one
simulated online and submitted) to 7,800 by the end.

*Effect.* Against uniform-size training, prefix training improves the
one-label Brier on test (-0.004, significant under random and uncertainty
acquisition) but ties on Brier ALC (single-pair test 0.2762 against 0.2725),
and it remains significantly worse than the Beta-Binomial (+0.0267). Matching
the training objective to the evaluation protocol does not fix what fails
under benchmark shift. Code: `_dense_prefix_episodes` in
`organized/training/codex_es_fixed_modal_cnp_train.py`; a step-by-step
walkthrough is in `paper-plan.md`.

## C. Extended results

Supports Section 5 (RQ1 and RQ2). Figures and tables: `docs/online-final-results.pdf`.

**C.1 Where the ANP loses.** `docs/online-final-results.pdf`, p. 14: per-budget
gap to the Beta-Binomial on test; budgets 0-1 carry almost the whole deficit.

**C.2 Training recipe and split** (p. 4). No recipe fixes the benchmark-
holdout ANP. The random-subject-split ANPs tie the Beta-Binomial on test
(-0.0029, +0.0020, n.s.) only because they trained on editbench: they win
there and lose on algotune and cmmlu.

**C.3 Combining the ANP with the Beta-Binomial** (p. 5; test minus Beta-
Binomial).

| Combination | Test |
| --- | --- |
| ANP with Beta-Binomial fallback at 0-1 labels | +0.0025 [+0.0003, +0.0048] |
| ANP-Beta-Binomial ensemble, one weight | +0.0062 [+0.0050, +0.0074] |
| ANP-Beta-Binomial ensemble, per-budget weights | +0.0043 [+0.0027, +0.0062] |
| ANP + residual correction (AIPW-style) | +0.0502 [+0.0465, +0.0537] |
| Beta-Binomial + residual correction | +0.0492 [+0.0452, +0.0529] |

The fallback and the ensemble remove most of the ANP's loss but never beat the
Beta-Binomial. The residual correction, the item-level analogue of AIPW
[Zhang et al., 2025], shifts every prediction by the mean residual of the
labels seen so far; after one label that shift is about +/-0.5, and applied to
the Beta-Binomial it undoes the shrinkage that makes it work (Brier at one
label rises from 0.28 to 0.44).

**C.4 Acquisition policy.** Random, uncertainty, confidence extremes,
Bernoulli variance and expected Brier reduction give near-identical ALC for
the Beta-Binomial and every ANP on both splits (`docs/online-final-results.
pdf`, pp. 8-13).

**C.5 Diverse acquisition.** **[MEHH_REMOVE]** (a null result already covered by C.4) Buying labels so each pair's item clusters stay
evenly covered (streaming, with the reserve rule) changes test ALC by -0.0002
to -0.0006 against random acquisition for the Beta-Binomial and every Rasch
model, none significant. Acquisition order does not matter here either.
Script for C.8 and C.5: `organized/testing/online/kernel_rasch_replay.py`.

**C.6 Item-independent ANP** **[MEHH_REMOVE]** (unfinished). With all item embeddings zeroed,
the ANP can only learn a level. At its step-400 checkpoint it scores 0.1786 on
validation and 0.2714 on test against the Beta-Binomial's 0.1717 and 0.2494,
again almost entirely at zero and one label.

**C.7 What sharing adds (multi-pair)** (p. 7). Pooled Rasch -0.0035; without
item difficulty -0.0020; with ability centred on the provider's training
level +0.0003 (n.s.), worse than pooled Rasch by 0.0039 [0.0011, 0.0070],
despite being best on validation.

**C.8 Sharing difficulty across similar items: kernel Rasch.** Kernel Rasch
(Section 4) correlates difficulties within a benchmark through
$K = \tau^2[(1 - \rho) I + \rho S]$ with $S_{ij} = \exp\big((\cos(x_i, x_j) - 1)/T\big)$
on BGE-M3 embeddings. Clustered Rasch uses $S_{ij} = \mathbb{1}[\text{same k-means cluster}]$
instead. Settings were chosen on the 60 validation draws (kernel: $T = 0.2$,
$\rho = 0.8$, $\tau = 1.4$, at the edge of the grid **[TODO: widen]**; clustered:
$k = 16$, $\rho = 0.5$) and scored on the same 150 test draws as Section 5.2.

| Model (multi-pair) | Val ALC | Val minus Beta-Binomial | Test ALC | Test minus Beta-Binomial | Test minus pooled Rasch |
| --- | --- | --- | --- | --- | --- |
| Beta-Binomial | 0.1685 | - | 0.2497 | - | - |
| Pooled Rasch | 0.1674 | -0.0012 [-0.0034, +0.0009] | 0.2461 | -0.0035 [-0.0046, -0.0026] | - |
| Clustered Rasch | 0.1669 | -0.0017 [-0.0041, +0.0007] | 0.2462 | -0.0034 [-0.0045, -0.0024] | +0.0001 [-0.0002, +0.0004] |
| **Kernel Rasch** | **0.1667** | -0.0019 [-0.0043, +0.0006] | **0.2448** | **-0.0049 [-0.0060, -0.0038]** | **-0.0013 [-0.0017, -0.0010]** |

Validation is in-sample for the clustered and kernel settings, which were
chosen on it. On validation every Rasch variant is ahead of the Beta-Binomial
but none significantly (78 tasks, 60 draws); clustered and kernel Rasch beat
pooled Rasch there by -0.0005 [-0.0008, -0.0002] and -0.0007 [-0.0012,
-0.0002].

Kernel Rasch improves on pooled Rasch at every budget, most after one label
(-0.008 against the Beta-Binomial). The within-task agreement it exploits
rises smoothly with item similarity (residual correlation 0.05 below cosine
0.80, 0.42 above 0.97 on test). On editbench 54% of items have a near-copy
(cosine > 0.97), so much of that benchmark's effect is near-duplicate items;
on the other benchmarks near-copies are 0-4% and agreement at cosines
0.80-0.95 is still 0.08-0.31. **[TODO: promote kernel Rasch to the main
results if it holds under leave-one-benchmark-out.]**

**C.9 Draws shaped like the competition's.** The competition scores about
nine pairs over about seven benchmarks, so most benchmarks have one subject
and nothing to pool with on that benchmark. Re-tuning the prior widths on 120
validation draws with exactly one subject per benchmark keeps a wide
benchmark-rate prior (tau_b = 2; narrower priors lose up to 0.026); kernel
Rasch keeps its earlier settings and pooled Rasch prefers a narrower ability
prior (tau_s = 0.5). On test (150 such draws; Brier ALC minus the
Beta-Binomial): pooled Rasch -0.0011 with the old priors and -0.0032 re-tuned,
kernel Rasch -0.0024, all significant; on the earlier draws re-tuned pooled
Rasch reaches -0.0051. On validation in this shape, kernel Rasch (0.1735)
ties the Beta-Binomial (0.1738) and pooled Rasch (0.1742) does not beat it.

On the leaderboard, pooled Rasch (1024003) scored 0.2022 against 0.1486 for
the Beta-Binomial (1005155), with Brier 0.298 against 0.181 after one label.
Each submission is scored on its own nine pairs (these two share one pair), so
the comparison is unpaired and noisy; the online simulation does not reproduce the
one-label gap. **[TODO: kernel Rasch 1026512 score; a paired comparison would
need the same pairs.]** Script: `organized/testing/online/codabench_shape_replay.py`.

## D. Extended analysis

Supports Section 6 (RQ3).

**D.1 Ability and difficulty components.** The extended version of Table 3, with 95% CIs and
the width-128 model (`docs/online-final-results.pdf`, p. 15). Entries are the test Brier
minus the Beta-Binomial's; * marks a CI that excludes zero. "Ability component only" pairs
the ANP's mean prediction with a flat difficulty component, and "difficulty component
only" pairs the ANP's item offsets with the Beta-Binomial's ability estimate.

*Budgets 0-1.*

| Model | Full ANP | Ability component only | Difficulty component only |
| --- | --- | --- | --- |
| ANP | +0.064* [+0.056, +0.072] | +0.064* [+0.056, +0.072] | +0.000 [-0.001, +0.001] |
| ANP + SAM | +0.040* [+0.029, +0.051] | +0.037* [+0.026, +0.048] | +0.003* [+0.001, +0.005] |
| ANP, prefix training | +0.064* [+0.056, +0.073] | +0.063* [+0.055, +0.072] | +0.001 [-0.000, +0.002] |
| ANP, random-subject split: unseen | +0.040* [+0.029, +0.052] | -0.003 [-0.014, +0.008] | +0.039* [+0.032, +0.046] |
| ANP, random-subject split: editbench (seen) | -0.073* [-0.081, -0.065] | -0.025* [-0.031, -0.020] | -0.053* [-0.059, -0.046] |
| ANP, random-subject split (width 128): unseen | +0.038* [+0.027, +0.048] | -0.006 [-0.017, +0.003] | +0.036* [+0.031, +0.041] |
| ANP, random-subject split (width 128): editbench (seen) | -0.052* [-0.062, -0.040] | -0.011* [-0.019, -0.004] | -0.045* [-0.052, -0.038] |

*Budgets 3-31.*

| Model | Full ANP | Ability component only | Difficulty component only |
| --- | --- | --- | --- |
| ANP | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.007] | +0.000 [-0.003, +0.003] |
| ANP + SAM | +0.003 [-0.001, +0.006] | +0.000 [-0.003, +0.004] | +0.002* [+0.000, +0.004] |
| ANP, prefix training | +0.005* [+0.002, +0.010] | +0.002 [-0.001, +0.006] | +0.003* [+0.001, +0.005] |
| ANP, random-subject split: unseen | +0.046* [+0.036, +0.055] | -0.010* [-0.013, -0.005] | +0.050* [+0.041, +0.059] |
| ANP, random-subject split: editbench (seen) | -0.081* [-0.089, -0.073] | -0.011* [-0.014, -0.009] | -0.073* [-0.080, -0.064] |
| ANP, random-subject split (width 128): unseen | +0.050* [+0.044, +0.057] | -0.009* [-0.013, -0.005] | +0.052* [+0.045, +0.058] |
| ANP, random-subject split (width 128): editbench (seen) | -0.079* [-0.090, -0.068] | -0.009* [-0.012, -0.007] | -0.072* [-0.082, -0.063] |

**D.2 Text embeddings and item difficulty.** Ridge probe from BGE-M3
embeddings to item pass rate (within-benchmark-centred targets).

| Fit on | Score on | r |
| --- | --- | --- |
| 80% of a benchmark's items | its other 20% | 0.22 (cmmlu) to 0.88 (editbench) |
| 11 training benchmarks | held-out benchmarks, within each | about 0 |
| 1 / 4 / 8 / 11 / 14 non-test benchmarks | test benchmarks, within each | 0.007 / 0.020 / 0.037 / 0.037 / 0.076 |

`docs/item_difficulty-final-results.pdf`: budget-0 predictions against observed
item difficulty for every ANP (pp. 1-12; seen against unseen benchmarks for
the random-split models on p. 10). Data-analysis summary:
`docs/data-analysis-final-results.pdf`, p. 3.

**D.3 Are answers IID within a task?** The Beta-Binomial assumes a subject's
answers on one benchmark are IID given one rate. They are not: after removing
each task's mean, a subject's answers on two text-similar items (BGE-M3
nearest neighbours, mean cosine 0.84) correlate +0.24 [0.18, 0.30] on test and
+0.19 [0.17, 0.22] on validation, against about 0 for random item pairs
(editbench +0.48). A group-weighted Beta-Binomial that lets labels on similar
items count more, $\hat p_i = \dfrac{\kappa g + \sum_j w_{ij} y_j}{\kappa + \sum_j w_{ij}}$, with $g$ the Beta-Binomial estimate\nand similarity weights $w_{ij}$ ($\kappa = 1$, temperature 0.1, chosen on validation), gains little: -0.0006 [-0.0009, -0.0003] on validation and
-0.0002 [-0.0005, +0.0001] on test. It helps at 15-31 labels (-0.0035 at 31 on
test) and slightly hurts at 1-3: with a handful of labels a target rarely has a
similar labelled item from the same subject. The structure is more likely to
pay off through other subjects' labels on similar items, a kernel or GP version
of pooled Rasch **[TODO]**. Script: `organized/testing/online/noniid_beta_replay.py`.

**D.4 Ability estimates.** `docs/item_difficulty-final-results.pdf`, pp. 14-15.
RMSE of each model's per-task ability estimate on test:

| Model | 1 label | 7 labels | 31 labels |
| --- | --- | --- | --- |
| Beta-Binomial | 0.13 | 0.07 | 0.05 |
| Pooled Rasch | 0.18 | 0.11 | 0.07 |
| ANP | 0.15 | 0.09 | 0.09 |
| ANP, random-subject split | 0.12 | 0.07 | 0.06 |

The ANP's loss does not grow with subject ability (Spearman rank correlation about 0 on
test; `docs/online-final-results.pdf`, p. 16).

**D.5 Subject metadata (hierarchical Beta-Binomial)** (p. 6). A Beta-Binomial
whose prior mean depends on CodaBench's subject metadata, fitted on training
benchmarks, L2 chosen on validation.

| Prior mean from | Test minus Beta-Binomial |
| --- | --- |
| nothing (learned constant) | -0.0006 [-0.0027, +0.0017] |
| provider | -0.0043 [-0.0078, +0.0001] |
| model name, provider | -0.0049 [-0.0090, +0.0001] |
| provider, release date | +0.0131 [+0.0046, +0.0220] |
| model name, provider, release date | -0.0039 [-0.0084, +0.0015] |

A leave-one-benchmark-out check over the 14 non-test benchmarks did not
confirm the small gains (provider: -0.0002 [-0.0046, +0.0054], better on 10 of
14; model name and provider: -0.0020 [-0.0054, +0.0014], 9 of 14). Test models
extend to 2026-03 against 2025-06 in training, so the release-date term
extrapolates; only 27% of test model names occur in training.

**D.6 Loss curves.** `docs/training-final-results.pdf`: p. 2 (Figure 3,
in-distribution against out-of-distribution on one axis); pp. 3-12 (ELBO, NLL,
KL, fixed-bank Brier, latent statistics, split-half and response-holdout
generalisation, each in- and out-of-distribution).

**D.7 Predictions as labels arrive.** **[MEHH_REMOVE]** `docs/np-context-final.pdf` (also
`docs/training-final-results.pdf`, pp. 13-16): on bird_sql, a benchmark the
random-split ANP trained on, its predicted curve follows the subject's
difficulty pattern and beats the Beta-Binomial after one label (0.131 against
0.151); on algotune, never trained on, the curve stays flat and loses at every
budget (0.213 against 0.206 at 31 labels).

## E. Reproducibility

| What | Where |
| --- | --- |
| Online simulation, single-pair | `organized/testing/online/modal_benchmark_holdout_replay.py`, `run_sweep.py` |
| Online simulation, multi-pair and pooled Rasch | `organized/testing/online/modal_pooled_rasch.py` |
| Hierarchical Beta-Binomial | `organized/testing/online/modal_hier_beta.py` |
| ANP training | `organized/training/codex_es_fixed_modal_cnp_train.py`, `modal_anp_crossattn_train.py`, `local_anp_train.py` |
| Figures | `organized/utils/plot_*_final*.py`, `plot_online_main.py`, `plot_online_posthoc.py`, `plot_ability_final.py`; display names in `model_names.py` |
| Pooled Rasch submission | `organized/submission/pooled_rasch/` |
| CodaBench submissions (leaderboard Brier ALC, each on its own 9 pairs) | 1005155 Beta-Binomial 0.1486; 1019401 probe 0.2026; 1022033 ANP + SAM 0.1808; 1022030 ANP, prefix training 0.1918; 1024003 pooled Rasch 0.2022; 1026512 kernel Rasch [TODO] |

## F. Open items

- [TODO] Leave-one-benchmark-out with multi-pair draws inside each fold.
- [TODO] Per-benchmark weighting of training losses and fitted priors.
- [TODO] CodaBench scores for 1022033, 1022030 and 1024003.
- [TODO] Final item-independent ANP (training to 15,000 steps).
- [TODO] Wider or fitted priors for pooled Rasch (ability width, item-difficulty
  width about 1.4 from data, Student-t), and the amortized-difficulty prior mean
  `b_i ~ N(f(x_i), tau)`.
- [TODO] Additional training seeds.
