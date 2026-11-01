# Simple Estimators Beat Meta-Learners for Online LLM Evaluation on Unseen Benchmarks

*Draft, 2026-11-01. Numbers come from the runs in this repository; every figure
reference names the PDF and page it lives on. Items marked **[TODO]** are not
done yet.*

---

## Abstract

Evaluating a language model on a new benchmark is expensive, so a natural goal
is to predict its answers from a handful of labelled items. Neural processes
and other meta-learners promise exactly this: trained on many
(subject, benchmark) tasks, they should infer a new subject's behaviour from a
few labels. We test this promise in the AIMS online-evaluation setting, where
a predictor sees labels one at a time and is scored by Brier score at budgets
of 0 to 31 labels, on benchmarks held out from training. Across eight
Attentive Neural Process (ANP) variants, none beats a two-line beta-binomial
estimator on unseen benchmarks; the best is worse by 0.017 Brier-ALC. The
deficit sits almost entirely at zero and one label. Diagnostics show why: the
ANP's estimate of a subject's overall level transfers, but its learned item
ordering is benchmark-specific and actively hurts on new benchmarks, while
text embeddings predict item difficulty within a benchmark but not across
benchmarks. Mixing the ANP with the beta-binomial limits the damage but never
turns it into a gain, and conditioning a beta-binomial prior on subject
metadata breaks under the shift to newer models. The only method that beats
the beta-binomial is a Rasch model refit at test time on every subject's
revealed labels, which needs no training at all: it improves Brier-ALC by
0.0035 (95% CI 0.0026 to 0.0045) on simulated competition draws. In this
regime, sharing evidence across subjects at test time matters more than
anything a meta-learner learns offline.

---

## 1. Introduction

Language models are evaluated on hundreds of benchmarks, and running every
model on every item is costly. Predicting a model's per-item correctness from
a few observed answers would let an evaluator spend labels where they matter.
This is a few-shot prediction problem over tasks, one task being one
*subject* (a model under a given harness) answering one *benchmark*, and
meta-learning is the textbook tool for it: learn from many training tasks how
to adapt to a new one from a few labels.

Neural processes (NPs) [Garnelo et al., 2018a; 2018b] and their attentive
variant (ANP) [Kim et al., 2019] are a natural fit. They condition on a context
set of labelled items and predict the rest in one forward pass, with no
test-time gradient steps. Amortised item-response models [Truong et al., 2025]
follow the same logic for item difficulty, predicting it from item text.

The difficulty is transfer. A held-out *subject* on a benchmark seen in
training is easy; a held-out *benchmark* is not, because nothing in training
says how that benchmark's items differ in difficulty. Zhang et al. (2025) show
the analogous failure for aggregate benchmark prediction: learned predictors
work when a new model resembles the training models and fail on unfamiliar
ones, while an estimator anchored to the observed labels survives.

We study this question for calibrated, item-level, streaming prediction under
a strict benchmark holdout. Our contributions:

1. **A benchmark-held-out streaming replay** of the AIMS online protocol, with
   frozen candidate orders, disjoint acquisition and evaluation pools, and a
   multi-pair variant that mirrors the competition's draws (Section 2).
2. **A negative result for meta-learning.** Eight ANP variants, plus
   combinations with the beta-binomial, never beat a beta-binomial on unseen
   benchmarks (Section 4, Appendix C).
3. **A diagnosis.** The ANP's level estimate transfers and its item ordering
   does not; text embeddings carry within-benchmark but not cross-benchmark
   difficulty; subject metadata breaks under model shift (Section 5,
   Appendix D).
4. **A simple estimator that wins.** A Rasch model refit online on every
   pair's revealed labels beats the beta-binomial at every budget, with no
   training (Section 4.2).

---

## 2. Setting and evaluation protocol

**Task.** A subject *s* answers the items of a benchmark *b*; responses are
binary. At test time the evaluator buys up to 31 labels, one candidate item at
a time, deciding to accept or skip each candidate before seeing the next. After
0, 1, 3, 7, 15 and 31 labels the predictor outputs a probability for every
item in a fixed, never-revealed evaluation pool, scored by Brier score
[Brier, 1950]. The primary metric is Brier-ALC, a weighted average over the six
budgets with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1 (normalised trapezoidal area
on log2(1 + budget)), so 70% of the score comes from seven labels or fewer.

**What the predictor sees.** The competition passes the predictor every label
purchased so far *across all sampled subject-benchmark pairs*, together with
each subject's metadata (model name, provider, release date). Section 4.2
exploits the first.

**Split.** We hold out whole benchmarks: 11 for training, 3 for validation
(ai2d_test, bird_sql, financebench) and 3 for test (algotune, cmmlu,
editbench). No validation or test subject appears in training. Responses are
filtered to items with at least 15 responders and subjects with at least 80
items per benchmark, leaving 6.34 M responses over 123,035 items and 17
benchmarks. Figure A2 contrasts this with a random-subject split, which holds
out subjects but keeps every benchmark in training.

**Replay.** Each validation or test task's items are split once, 50/50, into
an acquisition pool and an evaluation pool. Candidate orders over the
acquisition pool are frozen (3 orders per validation task, 10 per test task),
so every model sees exactly the same labels in the same order. We report two
protocols:

- **Single-pair:** each task is replayed alone, on its own labels; 82
  validation and 45 test tasks.
- **Multi-pair:** competition-like draws of one or two subjects per benchmark,
  where the predictor may use every pair's labels; 60 validation and 150 test
  draws. Models that ignore other pairs are placed on the same draws from
  their single-pair scores, so comparisons are paired.

All intervals are 95% bootstrap intervals over tasks. Unless stated, labels
are acquired in random order (Appendix C.5 shows the acquisition policy does
not matter).

---

## 3. Methods

**Beta-binomial.** The predicted probability for every item is the posterior
mean under a Jeffreys prior Beta(0.5, 0.5) [Jeffreys, 1946]:
`p = (hits + 0.5) / (labels + 1)`, where *hits* counts correct answers among
the revealed labels of the current pair. It ignores item content and other
pairs, and predicts 0.5 with no labels.

**Attentive Neural Process.** Each item is a 1024-d BGE-M3 embedding of its
text. A context set of (item, label) pairs is encoded by a shared value
embedding and one self-attention block; a target item attends to the context
(keys from embeddings only, values carrying labels) to form a deterministic
representation, and a mean-pooled latent path gives a global *z* (width 64,
4 heads, 32-d latent). A decoder maps [target embedding, attention output,
*z*] to a logit. Training maximises the ELBO over episodes sampled from
training tasks. Figure B1 shows the architecture next to Kim et al.'s ANP
(`docs/np-architecture-final.pdf`, pp. 1-3). Variants differ in training
recipe only (Appendix B.2).

**Pooled (shared-evidence) Rasch.** At every checkpoint we fit

    logit p(s, i) = mu_b + theta_s - b_i

to *all* labels revealed so far in the draw, across subjects and benchmarks,
by MAP estimation under Gaussian priors `mu_b ~ N(M0, 2^2)`,
`theta_s ~ N(0, 1^2)`, `b_i ~ N(0, 1^2)`. Items are matched across subjects
by content; an item no one has labelled gets `b_i = 0`. `M0 = logit(0.512)`,
the training-benchmark mean, which is close to uninformative; the three prior
widths were chosen on validation. There is no training: the fit is ridge-
penalised logistic regression with about 300 parameters, solved by Newton's
method in the submission and L-BFGS in the replay, and refit from scratch at
every checkpoint.

**Other baselines.** A constant at the training mean; the raw hit rate
(no shrinkage); a hierarchical beta-binomial whose prior mean depends on
subject metadata (Appendix C.3).

---

## 4. Results

### 4.1 Meta-learners do not beat the beta-binomial on unseen benchmarks

**Figure 1** (`docs/online-final-results.pdf`, p. 1): single-pair Brier per
budget, validation and test.

| Model (single-pair) | Val ALC | Test ALC | Test minus beta-binomial [95% CI] |
| --- | --- | --- | --- |
| Constant (training mean) | 0.2525 | 0.2472 | -0.0022 [-0.0105, +0.0067] |
| Beta-binomial | 0.1717 | 0.2494 | - |
| ANP | 0.1735 | 0.2725 | +0.0230 [+0.0198, +0.0267] |
| ANP + SAM | 0.1667 | 0.2663 | +0.0168 [+0.0128, +0.0211] |
| ANP, dense-prefix training | 0.1735 | 0.2762 | +0.0267 [+0.0231, +0.0306] |
| Raw hit rate | 0.1892 | 0.2879 | +0.0385 [+0.0355, +0.0413] |

Every benchmark-holdout ANP is significantly worse than the beta-binomial on
test. ANP + SAM beat it on validation (0.1667 against 0.1717), driven by two
validation benchmarks, and the advantage reversed on test: validation wins do
not transfer. The deficit is concentrated at zero and one label (+0.034 and
+0.079 at budgets 0 and 1 for the ANP; Figure 1, Appendix D.1); from three
labels on, the ANP and the beta-binomial are within noise. On test the
constant predictor ties the beta-binomial: the beta-binomial's error at one
label (0.28) is above the constant's, and later budgets only just recover it.

### 4.2 Sharing labels across subjects at test time does beat it

**Figure 2** (`docs/online-final-results.pdf`, p. 2): multi-pair Brier per
budget.

| Model (multi-pair) | Val ALC | Test ALC | Test minus beta-binomial [95% CI] |
| --- | --- | --- | --- |
| Constant (training mean) | 0.2527 | 0.2472 | - |
| Beta-binomial | 0.1685 | 0.2497 | - |
| ANP | 0.1701 | 0.2726 | - |
| **Pooled Rasch** | **0.1674** | **0.2461** | **-0.0035 [-0.0045, -0.0026]** |
| Pooled Rasch without item difficulty | 0.1675 | 0.2477 | -0.0020 [-0.0029, -0.0011] |

Pooled Rasch is the only model below the beta-binomial on test at every
budget. Removing the item term halves the gain, so both sources of shared
evidence contribute: other subjects' labels on the same benchmark sharpen the
benchmark rate and the subject's ability, and their labels on the same item
reveal that item's difficulty. Another subject's label on the same item
predicts the outcome with AUC 0.74 on validation and 0.64 on test; only 34% of
test evaluation items ever receive such a label, which bounds the item term's
contribution.

Surprisingly, pooled Rasch does not estimate a subject's *ability* better than
the beta-binomial (Appendix D.4): its ability prior is tight and shrinks the
estimate. Its advantage comes from the benchmark rate and item difficulty, not
from a sharper ability estimate.

---

## 5. Why the meta-learners fail

**Level transfers; item ordering does not.** We split each ANP prediction
into a *level* (its mean over the task's items) and an *ordering* (each item's
offset from that mean), and score each part against the beta-binomial
(Appendix D.2, `docs/online-final-results.pdf`, p. 15). For random-subject-
split ANPs, which trained on editbench, the ordering is what wins on editbench
(about -0.07 Brier against the beta-binomial) and what loses on the unseen
benchmarks (+0.04 to +0.05). Their level alone is slightly better than the
beta-binomial on unseen benchmarks from three labels on (-0.010). The
benchmark-holdout ANPs learned an almost flat ordering, which is harmless;
their entire loss is a miscalibrated level at zero and one label.

**Text carries within-benchmark, not cross-benchmark, difficulty.** A ridge
probe from BGE-M3 embeddings to item pass rate reaches r = 0.22 to 0.88 when
fitted and tested inside one benchmark, but about 0 when the within-benchmark
ordering is tested on a held-out benchmark (Appendix D.3). Transfer grows
slowly with the number of training benchmarks, from r = 0.007 with one to
0.076 with 14, far from the within-benchmark level. The ANP's budget-0 item
predictions mirror this: r = +0.46 on benchmarks the random-split model
trained on, +0.14 on the rest, and about -0.05 on test for the holdout models
(`docs/item_difficulty-final-results.pdf`, pp. 2 and 10).

**The models overfit the training benchmarks.** With the split as the only
difference, the same ANP's validation NLL keeps falling in distribution but
bottoms out near 0.48 at about 3,000 steps and rises to 0.59 on unseen
benchmarks while its training loss keeps falling (Figure 3,
`docs/training-final-results.pdf`, p. 2).

**Subject metadata breaks under model shift.** Test subjects are newer and
stronger than training subjects. A beta-binomial prior conditioned on provider
and release date loses on test (+0.013, budget 0 +0.091) because the
release-date term extrapolates; centring pooled Rasch's ability prior on the
provider was chosen on validation and loses on test (Appendix C.3, C.4).

**Gradient-based adaptation is unlikely to help.** MAML-style test-time
fine-tuning [Finn et al., 2017] adapts weights from the same few labels, so it
faces the same 0-7-label regime and cannot recover item difficulty from one
subject's answers. Pooled Rasch is the low-dimensional, closed-form version of
test-time adaptation, and it adapts the parameters that matter.

---

## 6. Related work

**Neural processes.** CNPs and NPs [Garnelo et al., 2018a; 2018b] amortise
inference over functions; ANPs [Kim et al., 2019] add cross-attention to fix
underfitting; incremental transformer NPs [Mortimer et al., 2026] make the
update causal and cacheable. We test them under a benchmark holdout, where the
target functions come from unseen benchmarks.

**Model-based and amortised evaluation.** Item response theory [Rasch, 1960;
Lord, 1980] separates subject ability from item difficulty; Truong et al.
(2025) amortise item parameters from text for efficient LLM evaluation. Our
pooled Rasch is classical IRT refit online; our results suggest the amortised
part is what fails to transfer across benchmarks.

**Benchmark prediction from few data.** Zhang et al. (2025) compare 11 methods
on 19 benchmarks for predicting aggregate accuracy and find that learned
predictors fail on unfamiliar models, while an AIPW estimator anchored to the
observed labels holds up. We find the item-level analogue: under benchmark
shift the estimator anchored to observed labels (beta-binomial, pooled Rasch)
beats the learned prior. An AIPW-style correction does not carry over to
item-level probabilities from a handful of labels (Appendix C.2).

**Meta-learning.** MAML [Finn et al., 2017] adapts by gradient steps; NPs by
conditioning. Both assume the training tasks resemble the test tasks, which a
benchmark holdout breaks.

---

## 7. Limitations

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
  compared only for the ANPs and the beta-binomial.

---

## 8. Conclusion

On unseen benchmarks with a few labels, the meta-learners we trained do not
beat a beta-binomial estimator, and the reason is identifiable: what they learn
about item difficulty is specific to the benchmarks they saw. What does help is
not learned offline at all: refitting a Rasch model at test time on every
subject's revealed labels. For online evaluation under benchmark shift, simple
estimators that share evidence across subjects are a stronger starting point
than meta-learned priors.

---

## References

- **[Brier, 1950]** G. W. Brier. Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1), 1950.
- **[Finn et al., 2017]** C. Finn, P. Abbeel, S. Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML*, 2017.
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

## A. Data and split

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

## B. Neural process details

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
| ANP + prior anchor | output is the beta-binomial logit plus a learned residual | 10,000 | 100 |
| ANP + SAM, budget-ladder training | both | 30,000 | 4,350 |
| ANP, dense-prefix training | every context prefix 0-31 per episode | 10,000 | 4,550 (replayed; the run's best later moved to 7,800) |
| ANP without item embeddings | all item embeddings set to zero | 15,000 [TODO: finishing] | 400 so far |
| ANP, random-subject split (width 64 / 128) | random-subject split, trains on 7 benchmarks incl. editbench | 100,000 | 79,700 / 97,800 |

**B.3 Loss curves.** `docs/training-final-results.pdf`: p. 2 (Figure 3,
in-distribution against out-of-distribution on one axis); pp. 3-12 (ELBO, NLL,
KL, fixed-bank Brier, latent statistics, split-half and response-holdout
generalisation, each in- and out-of-distribution).

**B.4 What in-context learning looks like.** `docs/np-context-final.pdf` (also
`docs/training-final-results.pdf`, pp. 13-16): on bird_sql, a benchmark the
random-split ANP trained on, its predicted curve follows the subject's
difficulty pattern and beats the beta-binomial after one label (0.131 against
0.151); on algotune, never trained on, the curve stays flat and loses at every
budget (0.213 against 0.206 at 31 labels).

## C. Ablations

All ablations: `docs/online-final-results.pdf`, table on p. 3, curves on pp.
4-7.

**C.1 Training recipe and split** (p. 4). No recipe fixes the benchmark-
holdout ANP. The random-subject-split ANPs tie the beta-binomial on test
(-0.0029, +0.0020, n.s.) only because they trained on editbench: they win
there and lose on algotune and cmmlu.

**C.2 Combining the ANP with the beta-binomial** (p. 5; test minus beta-
binomial).

| Combination | Test |
| --- | --- |
| ANP, beta-binomial below 2 labels (gate) | +0.0025 [+0.0003, +0.0048] |
| ANP/beta-binomial blend, one weight | +0.0062 [+0.0050, +0.0074] |
| ANP/beta-binomial blend, per budget | +0.0043 [+0.0027, +0.0062] |
| ANP + residual correction (AIPW-style) | +0.0502 [+0.0465, +0.0537] |
| Beta-binomial + residual correction | +0.0492 [+0.0452, +0.0529] |

Gating and blending remove most of the ANP's loss but never beat the
beta-binomial. The residual correction, the item-level analogue of AIPW
[Zhang et al., 2025], shifts every prediction by the mean residual of the
labels seen so far; after one label that shift is about +/-0.5, and applied to
the beta-binomial it undoes the shrinkage that makes it work (Brier at one
label rises from 0.28 to 0.44).

**C.3 Subject metadata (hierarchical beta-binomial)** (p. 6). A beta-binomial
whose prior mean depends on CodaBench's subject metadata, fitted on training
benchmarks, L2 chosen on validation.

| Prior mean from | Test minus beta-binomial |
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

**C.4 What sharing adds (multi-pair)** (p. 7). Pooled Rasch -0.0035; without
item difficulty -0.0020; with ability centred on the provider's training
level +0.0003 (n.s.), worse than pooled Rasch by 0.0039 [0.0011, 0.0070],
despite being best on validation.

**C.5 Acquisition policy.** Random, uncertainty, confidence extremes,
Bernoulli variance and expected Brier reduction give near-identical ALC for
the beta-binomial and every ANP on both splits (`docs/online-final-results.
pdf`, pp. 8-13).

**C.6 Item-independent ANP** [TODO: final]. With all item embeddings zeroed,
the ANP can only learn a level. At its step-400 checkpoint it scores 0.1786 on
validation and 0.2714 on test against the beta-binomial's 0.1717 and 0.2494,
again almost entirely at zero and one label.

**C.7 Are answers IID within a task?** The beta-binomial assumes a subject's
answers on one benchmark are IID given one rate. They are not: after removing
each task's mean, a subject's answers on two text-similar items (BGE-M3
nearest neighbours, mean cosine 0.84) correlate +0.24 [0.18, 0.30] on test and
+0.19 [0.17, 0.22] on validation, against about 0 for random item pairs
(editbench +0.48). A group-weighted beta-binomial that lets labels on similar
items count more, `p_i = (kappa g + sum_j w_ij y_j) / (kappa + sum_j w_ij)`
with similarity weights `w_ij` (kappa = 1, temperature 0.1, chosen on
validation), gains little: -0.0006 [-0.0009, -0.0003] on validation and
-0.0002 [-0.0005, +0.0001] on test. It helps at 15-31 labels (-0.0035 at 31 on
test) and slightly hurts at 1-3: with a handful of labels a target rarely has a
similar labelled item from the same subject. The structure is more likely to
pay off through other subjects' labels on similar items, a kernel or GP version
of pooled Rasch **[TODO]**. Script: `organized/testing/online/noniid_beta_replay.py`.

## D. Diagnostics

**D.1 Where the ANP loses.** `docs/online-final-results.pdf`, p. 14: per-budget
gap to the beta-binomial on test; budgets 0-1 carry almost the whole deficit.

**D.2 Level against item ordering.** p. 15. See Section 5.

**D.3 Text embeddings and item difficulty.** Ridge probe from BGE-M3
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

**D.4 Ability estimates.** `docs/item_difficulty-final-results.pdf`, pp. 14-15.
RMSE of each model's per-task ability estimate on test:

| Model | 1 label | 7 labels | 31 labels |
| --- | --- | --- | --- |
| Beta-binomial | 0.13 | 0.07 | 0.05 |
| Pooled Rasch | 0.18 | 0.11 | 0.07 |
| ANP | 0.15 | 0.09 | 0.09 |
| ANP, random-subject split | 0.12 | 0.07 | 0.06 |

The ANP's loss does not grow with subject ability (Spearman rho about 0 on
test; `docs/online-final-results.pdf`, p. 16).

## E. Reproducibility

| What | Where |
| --- | --- |
| Replay, single-pair | `organized/testing/online/modal_benchmark_holdout_replay.py`, `run_sweep.py` |
| Replay, multi-pair and pooled Rasch | `organized/testing/online/modal_pooled_rasch.py` |
| Hierarchical beta-binomial | `organized/testing/online/modal_hier_beta.py` |
| ANP training | `organized/training/codex_es_fixed_modal_cnp_train.py`, `modal_anp_crossattn_train.py`, `local_anp_train.py` |
| Figures | `organized/utils/plot_*_final*.py`, `plot_online_main.py`, `plot_online_posthoc.py`, `plot_ability_final.py`; display names in `model_names.py` |
| Pooled Rasch submission | `organized/submission/pooled_rasch/` |
| CodaBench submissions | 1019401 (probe), 1022033 (ANP + SAM), 1022030 (ANP, dense-prefix), 1024003 (pooled Rasch) [TODO: scores] |

## F. Open items

- [TODO] Leave-one-benchmark-out with multi-pair draws inside each fold.
- [TODO] Per-benchmark weighting of training losses and fitted priors.
- [TODO] CodaBench scores for 1022033, 1022030 and 1024003.
- [TODO] Final item-independent ANP (training to 15,000 steps).
- [TODO] Wider or fitted priors for pooled Rasch (ability width, item-difficulty
  width about 1.4 from data, Student-t), and the amortised-difficulty prior mean
  `b_i ~ N(f(x_i), tau)`.
- [TODO] Additional training seeds.
