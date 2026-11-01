# Paper plan: online LLM evaluation under benchmark shift

## Working title

**Do Learned Priors Transfer? Online LLM Evaluation Across Unseen Benchmarks**

Alternatives:

- **Online LLM Evaluation Under Benchmark Shift**
- **Testing Psychometric and Neural Priors for Sequential LLM Benchmarking**

## Central questions and claim boundary

The paper is organised around three questions. A task is a
subject--benchmark pair. In the filtered corpus used here, each retained
subject occurs in one benchmark, so benchmark holdout also makes subjects
disjoint across train, validation, and test.

### Q1. Held-out benchmarks: does meta-training transfer?

Train on some benchmarks, then evaluate on benchmarks absent from training.
This is the primary question: does the Neural Process learn a conditional
prior that transfers to genuinely new benchmark response functions, rather
than exploiting benchmark-specific response structure?

### Q2. Extrapolation: how does it behave outside the training ability range?

Stratify held-out tasks by subject response rate or inferred ability relative
to the training distribution. This asks whether failures concentrate among
unusually strong or weak subjects. It is a diagnostic, not a new split or a
causal explanation.

### Q3. Small-budget generalization: does context help quickly?

Measure Brier at budgets 0, 1, 3, 7, 15, and 31. The key region is 1--7
labels: does each additional acquired response improve predictions
consistently, or does the model remain near a constant predictor?

The contribution is a controlled benchmark-held-out study of a context-only
Beta-Binomial baseline and neural-process predictors [Garnelo et al., 2018a,b]. IRT results may appear
only when replayed on the same frozen panel; older benchmark-held-out grids
remain historical context, not a matched main-text comparison. The report must
not claim that a neural model beats the baseline unless a paired comparison on
the frozen panel supports it.

The completed untouched test result is negative but informative. On 45 tasks
from three test-only benchmarks, the constant training-mean predictor
(0.24725) and the context-only Beta-Binomial reference (0.24943) are tied
under the paired comparison: constant minus Beta is -0.00218 with 95% interval
[-0.01099, +0.00664]. The three clean ANP variants tested with the same random
streaming policy are worse: filtered ANP 0.27247, dense-prefix ANP 0.27617,
and SAM ANP 0.26627. The paired intervals exclude zero for all three ANP--Beta
differences. Thus, for the tested frozen embeddings and ANP configurations,
text-derived item representations did not transfer reliably under benchmark
shift. This does **not** establish that Beta-Binomial is uniquely better than
every simple predictor. The exact results live in the
[frozen replay table](organized/testing/online/README.md#results---test-final-3-untouched-benchmarks).

The report should answer these questions in that order. Any experiment that
does not clarify transfer, ability-range extrapolation, or low-budget
adaptation is supporting evidence only and belongs in the appendix.

## Terminology and framing

Use **benchmark-held-out out-of-distribution generalization** for the main
setting. The ANP provides **amortized few-shot adaptation**: it maps the
observed labelled context to predictions without updating its weights at test
time [Garnelo et al., 2018a,b; Kim et al., 2019]. The online task is a form of
**active evaluation** under a strict streaming constraint, because future
candidates are hidden [Li et al., 2025].

Use **sequential probabilistic prediction** when describing updates from newly
acquired labels, and **calibration under distribution shift** when discussing
Brier and ECE. The psychometric component can be framed as **measurement
validity under sparse response matrices**. These terms describe what was
actually evaluated; avoid calling the method continual learning, reinforcement
learning, or parameter-level test-time adaptation.

## Competition technical-report and release contract

This deliverable is a **non-anonymous NeurIPS technical report**, not a
long-form research paper. It must fit approximately **4--8 pages of main text**
in the NeurIPS template; references and appendices are unrestricted. Its job is
to let a reviewer answer five questions without reading the repository:

1. What data was used, what was filtered, and what information is available at
   prediction time?
2. What exact predictor and streaming acquisition rule were submitted?
3. How was the method implemented, selected, and evaluated?
4. What did the final results establish, including negative results?
5. Can another team rerun the submitted method from the released materials?

The OpenReview upload must link to a public release containing the code used to
develop **and run** the submission, plus any permitted training data that are
not part of `measurement-db`. Do not redistribute supplied `measurement-db`
content unless its licence explicitly permits it. For that data, release the
reader/preprocessing code, field schema, stable IDs or hashes where allowed,
and exact filtering/split manifests. Release any extra data only after checking
that it is permitted; otherwise provide a script that regenerates it from its
authorised source.

### Six-page main-text budget

Aim for six pages, leaving room to expand or contract within the 4--8 page
limit without losing the required story.

| Main-text pages | Section | Required content |
| --- | --- | --- |
| 1 | **Problem and contribution** | The three questions: held-out-benchmark transfer, ability-range extrapolation, and small-budget adaptation. Explain why benchmark-mixed masks and full-pool acquisition answer easier questions. |
| 2 | **Data and protocol** | Binary response schema, cleaning/support filter, benchmark roles, ability-range stratification, allowed information, nested budgets, and Figure 1 plus one compact panel table. |
| 3 | **Methods** | Context-only Beta-Binomial, shared-encoder latent ANP, and the dense-prefix training objective. State exactly what adapts at test time: context, not weights. |
| 4 | **Experimental setup** | Frozen validation/test roles, random streaming policy, checkpoint rule, macro Brier ALC, paired aggregation, ability strata, and no-leakage controls. |
| 5 | **Results** | Three subsections, one per question: transfer, ability extrapolation, then Brier curves at 1--7 labels. Include the constant sanity check and three clean ANPs. |
| 6 | **Limitations and conclusion** | One neural seed, three held-out benchmarks, incomplete ANP-policy grid, panel-manifest reconciliation, release pointer, and narrow conclusions for each question. |

Put non-benchmark-holdout results, full filter counts, hyperparameters,
architecture diagrams, all development cells, all training/KL diagnostics,
policy formulae, exact commands, graph/spectral analyses, and historical
comparisons in the appendix. This keeps the main text evaluable without
turning it into an experiment dump.

### Release manifest to prepare before OpenReview submission

| Release item | What must be present | Evidence / location in this repository |
| --- | --- | --- |
| **Versioned source** | Public repository URL, fixed commit hash or release tag, licence, and no credentials or private paths. | `submission` branch; final tag still required. |
| **Submitted method** | Exact submission package, its routing/prediction entrypoint, model/checkpoint provenance, and a smoke-test command. | `organized/submission/`; final chosen package and hash still required. |
| **Environment** | Pinned Python/package environment, CPU/GPU expectations, and deterministic seeds where possible. | Add a minimal environment file and reproduction README; do not rely on a personal Modal profile. |
| **Data contract** | `measurement-db` reader, expected schema, filtering code, split seed, hashes/manifests, and an explicit statement of what is not redistributed. | `organized/training/benchmark_holdout_filters.py`, [protocol](docs/benchmark-holdout-online-replay.md), filter/audit artifacts. |
| **Permitted extra data** | Only releasable external/derived training data, with source, licence, checksum, and regeneration command. | Decide per artifact; never silently include cached benchmark content. |
| **Reproduction commands** | One path for preprocessing, one for training or checkpoint retrieval, one for evaluation/replay, and one for building/checking the submitted archive. | [online replay README](organized/testing/online/README.md), submission README; consolidate into one top-level reproduction guide. |
| **Result ledger** | Frozen panel ID/manifest, checkpoint selection record, final per-budget and ALC scores, and paired intervals. | [online results](organized/testing/online/README.md); resolve manifest inconsistencies first. |

The report should link to this release once the commit/tag is frozen. Do not
link a mutable branch as the sole reproducibility reference.

## Main-text report outline

### 1. Introduction: transfer is the difficult part

State the setting in plain language. A subject is evaluated on a benchmark that
was excluded from model training. Candidates arrive one at a time, and the
policy must accept or reject the current candidate before seeing the future
stream. With few labels, the predictor must adapt to the observed response
history while carrying a transferable belief about item difficulty.

Explain why common shortcuts make this setting easier: random masked cells can
share benchmark structure, and full-pool acquisition can inspect candidates
that a streaming evaluator has not yet revealed.

Place the result next to aggregate benchmark prediction. Zhang et al. (2025)
compare 11 methods on 19 benchmarks for predicting a model's overall accuracy
from a few evaluated items. Learned predictors work when the new model
resembles models already seen, but fail to beat a random-sample average on
newer, more capable models; only an augmented inverse propensity weighting
(AIPW) estimator, which corrects a learned prediction with the observed
labels, stays consistently ahead, and modestly. Our setting differs in three
ways: the shift is to unseen *benchmarks* rather than stronger *subjects*,
predictions are item-level probabilities scored by Brier rather than one
aggregate accuracy, and acquisition is streaming. The conclusion agrees: a
learned prior helps only when the test resembles training, and the
context-only estimator is hard to beat under shift.

Contributions:

1. A benchmark-held-out streaming replay protocol with disjoint acquisition
   and scoring pools.
2. A data-construction procedure that handles duplicate questions, sparse
   responses, and cross-split textual overlap.
3. Matched development and untouched-test comparisons of a context-only
   Beta-Binomial predictor and clean benchmark-held-out ANP checkpoints.
4. A streaming-policy analysis that distinguishes completed paired controls
   from policies that have not yet been tested for a given model.

### 2. Evaluation protocol: benchmark-held-out streaming replay

Define a **task** as a `(subject, benchmark)` pair, an **item** as a question
within that benchmark, and an outcome as a binary correct/incorrect response.
The main text evaluates only benchmark-held-out streaming replay. Benchmark
roles are assigned before fitting; exact duplicate text is removed across
roles, and bipartite response support is filtered within each role. Thus every
validation/test benchmark and item is absent from neural-model training. In
the retained corpus, subjects are also disjoint across roles.

The split is deliberately stricter than a random masked-cell or random-subject
split. Those benchmark-mixed evaluations remain useful diagnostics of
response-matrix recovery, but they do not establish transfer to a new
benchmark. Their results and the in-distribution-to-benchmark-held-out
generalization-gap discussion belong in Appendix B.

#### 2.1 Frozen validation and untouched test

The final evaluation is a streaming replay over frozen benchmark-held-out
panels. Validation contains 82 tasks from `ai2d_test`, `bird_sql`, and
`financebench`; untouched test contains 45 tasks from `algotune`, `cmmlu`, and
`editbench`. These labels must be checked against the final panel manifest
before writing the report because the current replay README has stale
`cap19000`/`cap100000` header metadata.

At each arrival, the policy sees the current item representation and the
previously acquired labels. It must accept or reject before future candidates,
rejected labels, and the disjoint scoring pool are visible. Evaluate the nested
budgets `0, 1, 3, 7, 15, 31` with macro-task Brier score [Brier, 1950],
per-budget ECE [Guo et al., 2017], and

\[
\operatorname{ALC}=0.1B_0+0.2B_1+0.2B_3+0.2B_7+0.2B_{15}+0.1B_{31},
\]

where lower is better. Validation selects/checks models and policies; the test
panel is reported once with its frozen checkpoint and policy. This strict
interface makes full-pool Fisher argmax an exploratory analysis, not a valid
deployment policy.

### 3. Constructing a clean benchmark-held-out panel

Describe the preprocessing as part of the scientific method:

- Canonicalise repeated answer rows and retain one binary response per
  `(subject, item, benchmark)`.
- Collapse exact duplicate question text within each benchmark.
- Keep missing answers masked rather than imputing them.
- Apply per-benchmark bipartite support filtering after splitting: enough item
  responders, enough items per subject, and no degenerate rows or columns.
- Remove exact content-hash duplicates that cross train, validation, and test
  roles, then re-run the support filter. This is an exact-overlap control, not
  a claim of semantic decontamination.

These controls follow the measurement motivation in Truong et al. (2025), but
our held-out benchmark roles and streaming replay are a separate protocol.

Include one compact before/after support table in the main text. Put the
random-subject comparison, response-matrix visualisations, and split-shift
diagnostics in Appendix B: they explain the setting but are not direct
evidence for the primary claim.

### 4. Methodology: Neural Processes and latent task adaptation

All neural-process variants receive a frozen content embedding `e_i` for an
item and only acquire a response label when the online policy accepts it. They
do not update network weights at test time. Their test-time adaptation is
amortized: the observed context changes the prediction for the current task
[Garnelo et al., 2018a,b].

#### 4.0 What a Neural Process function means here

For both split protocols, one function realisation is one task
\(\tau=(s,b)\): the responses of subject \(s\) on benchmark \(b\). Its latent
response surface is

\[
f_{s,b}(e_i)=\Pr(Y_{s,b,i}=1\mid e_i,s,b).
\]

The NP observes only a finite labelled subset and estimates this surface as
\(p_\theta(Y_{s,b,i}=1\mid e_i,\mathcal C_{s,b})\), where \(e_i\) is an
item-content embedding and \(\mathcal C_{s,b}\) is the set of labels acquired
so far for that task. Meta-training learns parameters shared across many such
response surfaces. At test time, a new context set changes the conditional
prediction, not the weights. This is the relevant in-context-learning property
of NPs: the model adapts to a new response function by conditioning on examples
rather than by performing a new gradient fit [Garnelo et al., 2018a,b].

The task definition is the same in the two protocols, but the function
distribution is not. In non-benchmark holdout, a test task may belong to a
benchmark represented among training tasks, so it asks whether the model can
adapt to a new subject or item within familiar benchmark structure. In
benchmark holdout, every validation/test function belongs to a benchmark that
is absent from training. It asks whether the learned conditional prior itself
transfers to a new benchmark. The same in-context update rule is used in both
cases; only the distribution of functions changes.

#### 4.1 Conditional Neural Process and latent Neural Process

For a context set \(C=\{(e_i,y_i)\}\), first encode each observed response and
pool the resulting set representation, following the CNP set encoder
[Garnelo et al., 2018a]:

\[
h_i=h_\phi(e_i,2y_i-1),\qquad r_C=\operatorname{Agg}_{i\in C} h_i.
\]

The deterministic CNP predicts a target response with

\[
p_\theta(y_t=1\mid e_t,C)=\sigma\!\left(g_\theta(e_t,r_C)\right).
\]

The latent NP adds a global task variable \(z\), with a context prior and a
training-only target-informed posterior [Garnelo et al., 2018b],

\[
p_\phi(z\mid C)=\mathcal N(\mu_C,\operatorname{diag}\sigma_C^2),\qquad
q_\phi(z\mid C,T)=\mathcal N(\mu_{C\cup T},
\operatorname{diag}\sigma_{C\cup T}^2),
\]

and minimises target binary cross-entropy plus the per-target KL term,

\[
\mathcal L_{\mathrm{NP}}=
\operatorname{BCE}(y_T,\hat y_T)+
\frac{1}{|T|}\operatorname{KL}\!\left[q_\phi(z\mid C,T)\,\|\,p_\phi(z\mid C)\right].
\]

At evaluation, the target labels are unavailable, so prediction uses only
\(p_\phi(z\mid C)\), with the prior mean for the reported deterministic
probability. This is the precise sense in which the latent path represents
task-level uncertainty rather than online gradient adaptation.

#### 4.2 Attentive Neural Process

ANP replaces one shared context summary for all targets with a target-specific
deterministic representation [Kim et al., 2019]. In the evaluated
implementation, the labelled-context value encoder and optional context
self-attention are shared by the deterministic and latent paths; target queries
and context keys are separate:

\[
v_i=\operatorname{SA}\!\left(v_\phi(e_i,2y_i-1)\right),\qquad
r_t=\operatorname{MHA}\!\left(q_\phi(e_t),\{k_\phi(e_i)\}_{i\in C},
\{v_i\}_{i\in C}\right),
\]

\[
p(y_t=1\mid e_t,C)=\sigma\!\left(g_\theta(e_t,r_t,z)\right),\qquad
(\mu_C,\log\sigma_C^2)=\operatorname{LatentHead}
\!\left(\operatorname{mean}_{i\in C}v_i\right).
\]

We made this shared-encoder choice because both paths condition on the same
labelled context, while the benchmark-held-out training data are limited. The
deterministic path reads target-specific information from the shared values;
the latent head pools those values into one task-level distribution. Queries,
keys, the latent projection, and decoder remain separate. A fully separate
latent encoder is a valid ablation, but it was not the evaluated architecture
and must not be implied by the diagram. This is set-based attention, not an
autoregressive state model: when a label arrives, the normal ANP recomputes its
context representation from the current set.

#### 4.3 Dense-prefix ANP training

Ordinary training samples one context budget per task episode. Dense-prefix
training samples a randomly ordered context stream \((a_1,\ldots,a_K)\) and a
fixed disjoint target set \(T\). It trains all nested contexts
\(C_k=\{a_1,\ldots,a_k\}\), including \(C_0=\varnothing\), in the same
episode:

\[
\mathcal L_{\mathrm{dense}}=
\frac{1}{K+1}\sum_{k=0}^{K}
\mathcal L_{\mathrm{NP}}(C_k,T),\qquad K\leq31.
\]

The intent is to make predictions coherent across the exact nested online
budgets. It neither carries a cached hidden state from prefix \(k\) to
\(k+1\) nor trains an autoregressive response generator; each prefix is still
encoded as a set. Incremental Transformer Neural Processes instead use causal
masking, key--value caching, and autoregressive training [Mortimer et al.,
2026]. Our objective borrows the nested-prefix motivation, not that causal
architecture. The paper should present it as a tested training objective, not
as a solved causal online architecture.

The main text needs no more implementation detail than this. The full
episodic sampler, weighted/uniform/dense context schemes, and the disjoint
context--target convention are Appendix C material.

#### Appendix C.1 Algorithm: episodic NP training under each split

**Algorithm 1: Construct the task distribution and train an NP/ANP**

```text
Inputs: observed binary responses O, item embeddings e, split protocol P,
        context-target sampler q, model parameters theta

1. Form one task D_(s,b) = {(e_i, y_(s,b,i))} for every subject-benchmark pair.
   Treat D_(s,b) as one sampled response function f_(s,b).

2. If P = non-benchmark holdout:
      a. Apply the global filtered matrix construction (n >= 30 support filter).
      b. Randomly split tasks into train, validation, and test task sets.
      c. Draw a global held-out-item set. Remove those item responses from
         train/validation tasks; retain them only for the cold-item diagnostic.
      d. Let Q_train be the remaining train tasks.

   If P = benchmark holdout:
      a. Canonicalise repeated responses and within-benchmark duplicate text.
      b. Filter support within each benchmark, assign benchmark roles, remove
         exact duplicate text that crosses roles, then re-run the core filter.
      c. Let Q_train contain only tasks from train benchmarks. Validation and
         test tasks come from their own benchmark roles and are never sampled
         for an optimiser update.

3. Repeat for each optimiser step:
      a. Draw distinct tasks tau from Q_train using shuffled task order.
      b. For each tau, call q(D_tau) to produce context C and targets T.
      c. Compute target BCE conditional on C for a deterministic NP, or the
         negative ELBO for a latent NP/ANP. Average episode losses and update
         theta once.

4. Select the checkpoint by Brier score on a fixed bank of validation episodes.
   The bank is disjoint from the training tasks under either protocol.
```

The non-benchmark protocol has two static test axes in addition to its normal
held-out tasks: split-half adaptation on warm items and zero-response cold-item
prediction. The benchmark-held-out protocol does not add a random item split:
every evaluation item is already cold with respect to its whole benchmark.
Neither protocol imputes a missing response.

#### Appendix C.2 Algorithm: context and target schemes

**Algorithm 2: Build one training episode from a task \(D_\tau\)**

```text
Inputs: task D_tau with n observed responses, maximum context K, sampler mode q
Draw a random permutation pi of the n observed responses.

Weighted budget ladder:
    Sample k from {0, 1, 3, 7, 15, 31} intersect {0, ..., min(K, n - 1)}
    with probabilities proportional to the Brier-ALC weights
    (0.1, 0.2, 0.2, 0.2, 0.2, 0.1).
    Set C = pi[:k] and sample disjoint T from later entries of pi.

Uniform budget ladder:
    Sample k uniformly from the same eligible ladder rungs.
    Set C = pi[:k] and sample disjoint T from later entries of pi.

Dense-prefix context:
    Let K' = min(K, n - 1), stream = pi[:K'], and fix a disjoint target set
    T from pi[K':]. Return every episode (C_k, T), where C_k = stream[:k]
    for k = 0, ..., K'. Average their losses before the optimiser update.
```

Weighted and uniform schemes change the distribution of sampled context sizes;
they do not reweight an episode loss after it is drawn. "Uniform" means equal
probability over eligible *ladder rungs*, not over every integer from zero to
31. Dense-prefix training uses all integer prefixes of one candidate order and
the same withheld targets at every prefix. It therefore trains nested updates
directly, but still recomputes each set representation rather than caching a
causal state.

#### Appendix C.3 Context-target overlap is a modelling choice

The original CNP construction can reconstruct observed context points, so its
target set may overlap the context set \(C\subseteq T\) [Garnelo et al.,
2018a]. In the active-evaluation convention, context and scored targets are
disjoint, \(C\cap T=\varnothing\) [Li et al., 2025]. Both choices are useful,
but they answer different questions:

- **Non-disjoint targets** reward reconstruction of labels the model has
  already received. The historical CNP-AIRT pipeline used this as its original
  training convention, and a separate post-hoc control retrained it with
  disjoint targets.
- **Disjoint targets** require prediction of unseen labels from acquired ones.
  The current filtered NP/ANP trainer, including all reported dense-prefix
  runs, uses this convention. Dense-prefix targets are disjoint from every
  prefix by construction.

The main benchmark-held-out results must therefore be described as disjoint
context-to-unseen-target prediction. Do not use the historical overlap control
as a matched result on the frozen current panel.

#### Appendix D.1 Model families and matched online result ledger

| Family / checkpoint | Transfer signal and online update | Validation Brier ALC | Untouched-test Brier ALC | Interpretation |
| --- | --- | ---: | ---: | --- |
| Constant training mean | Fixed probability estimated on training data | 0.25254 | 0.24725 | Tied with Beta on this panel; sanity check, not an adaptive predictor. |
| Beta-Binomial | \((s+0.5)/(n+1)\); current acquired labels only | 0.17175 | 0.24943 | Reference; no item history or cold-item fallback. |
| Latent ANP, filtered w64 | BGE-M3-long item embeddings + deterministic and latent context paths | 0.17346 | 0.27247 | Worse than the reference on test. |
| Latent ANP, dense-prefix w64 | Same ANP; all nested prefixes trained jointly | 0.17345 | 0.27617 | Worse than the reference on test. |
| Latent ANP, SAM w64 | Same ANP; sharpness-aware training objective | 0.16665 | 0.26627 | Best tested ANP, still worse than the reference on test. |

All rows use the same frozen benchmark-held-out replay and random streaming
policy. Constant minus Beta is `-0.00218` (95% CI `[-0.01099,+0.00664]`), so
the two simple predictors are tied. The ANP differences from Beta-Binomial are
respectively `+0.02305`, `+0.02675`, and `+0.01684`; their paired 95%
intervals exclude zero. These are one-seed neural results, so they establish
the direction of this tested comparison, not a broad claim that attention or
latent NPs cannot transfer.

The Beta-Binomial row is the posterior mean of a Binomial likelihood under the
Jeffreys \(\operatorname{Beta}(1/2,1/2)\) prior [Jeffreys, 1946]. It is a
conjugate sequential predictor, not an item-difficulty model.

Derivation, for the methods section. A subject's answers on one benchmark are
treated as exchangeable Bernoulli draws with accuracy \(p\), and
\(p \sim \operatorname{Beta}(\alpha,\beta)\). After \(n\) revealed labels with
\(h\) correct, the posterior is \(\operatorname{Beta}(\alpha+h,\,\beta+n-h)\).
The predictive probability that the next item is answered correctly is the
posterior mean; the Beta and Gamma functions of the beta-binomial likelihood
cancel, leaving

\[
P(y_{\text{next}}=1 \mid h, n) = \frac{\alpha + h}{\alpha + \beta + n}
\;\overset{\alpha=\beta=1/2}{=}\; \frac{h + 0.5}{n + 1}.
\]

Here \(h\) counts correct answers among the labels revealed for the current
subject-benchmark pair, \(n\) is the number revealed (0 to 31), and the 0.5 and
1 are the prior's half correct answer out of one pseudo-label. The prediction
is the same for every item of the task, is 0.5 with no labels, and approaches
the raw hit rate as \(n\) grows. Exchangeability is tested in the draft's
Appendix C.7: answers on text-similar items are correlated, so it does not
hold exactly.

### 5. Online testing: streaming acquisition policies

Every policy scores only the *current* candidate and reserves enough later
arrivals to fill the remaining budget. The score may use the predictor and
acquired context, but never a future candidate, a rejected label, or the
hidden scoring pool. This differs from standard pool-based active evaluation,
where a selector can reason over a larger candidate collection [Li et al.,
2025]. The main text names random prefix, uncertainty, confidence extremes,
Bernoulli variance, and Expected Brier Reduction (EBR), but puts their exact
decision rules and the EBR equation in Appendix E.

The completed untouched-test controls are Beta-Binomial only: random `0.24943`,
uncertainty `0.24942`, confidence extremes `0.24784`, Bernoulli variance
`0.24964`, and EBR `0.24739` macro Brier ALC. Relative to random, the paired
intervals for confidence (`-0.00158`, 95% CI `[-0.00557,+0.00240]`),
uncertainty (`-0.00001`, `[-0.00251,+0.00250]`), variance (`+0.00022`,
`[-0.00385,+0.00429]`), and EBR (`-0.00203`, `[-0.00520,+0.00113]`) all span
zero. This compact policy control can be a main-text table if space permits;
otherwise it belongs in Appendix E. Do not extrapolate this null result to
ANP-policy combinations that were not run.

### 6. Experiments and results

Make this section answer three questions rather than list runs.

**What did development validation establish?**
Use the frozen 82-task benchmark-held-out *validation* panel only for model and
policy development. On random-prefix acquisition, Beta-Binomial and the
filtered w64 ANP were tied (0.17175 versus 0.17346; paired difference +0.00171,
95% CI [-0.00172, +0.00514]). The SAM variant looked better on this development
panel (0.16665), but that observation was not replicated on the untouched test
panel. The complete development table is in the
[validation replay results](organized/testing/online/README.md#results---validation-unseen-benchmarks).

**What did ability-range extrapolation establish?**
For each task, compute its observed response rate using evaluation labels only
after prediction is complete. Define training ability bands from the training
tasks, then report test Brier separately for in-range, low-response-rate, and
high-response-rate tasks. The band assignment is post-hoc analysis, never an
input to the model or acquisition policy. If the final panel is too small for
stable strata, report the task-level scatter and confidence intervals in the
appendix and state that ability extrapolation remains unresolved.

**What did the untouched test establish?**
Fix random-prefix acquisition and compare only clean benchmark-held-out
checkpoints on the 45-task, three-benchmark test panel. The constant
training-mean baseline (0.24725) is tied with Beta-Binomial (0.24943): paired
difference -0.00218, 95% CI [-0.01099, +0.00664]. Filtered ANP is worse than
Beta by +0.02305 (95% CI [+0.01969, +0.02641]), dense-prefix ANP by +0.02675
([+0.02286, +0.03063]), and SAM ANP by +0.01684 ([+0.01265, +0.02104]). This
is evidence against the tested learned text-conditioned prior on this panel,
not evidence that adaptive shrinkage is useless or that ANPs fail generally.
Random-subject-split ANPs belong only in an explicitly labelled diagnostic
appendix: they saw the evaluation benchmarks and are not evidence of transfer.

**What does the completed acquisition evidence establish?**
All five implemented policies are streaming-valid: random prefix, uncertainty,
confidence extremes, Bernoulli variance, and Expected Brier Reduction score
only the current candidate and use a reserve rule. On the untouched test, every
completed Beta-Binomial control is tied with random prefix: confidence extremes
differs by `-0.00158` (95% CI `[-0.00557,+0.00240]`), uncertainty by
`-0.00001` (`[-0.00251,+0.00250]`), variance by `+0.00022`
(`[-0.00385,+0.00429]`), and EBR by `-0.00203`
(`[-0.00520,+0.00113]`). This supports a limited claim: these completed
streaming acquisition rules did not improve the context-only estimator on this
panel. It does not generalise to unrun ANP-policy cells.

**What do the latent diagnostics establish?** The selected benchmark-held-out
checkpoints have non-zero validation KL, posterior/prior mean shifts, and some
active latent dimensions. They are therefore not a simple complete-posterior-
collapse failure. They do not show that the latent path *caused* a prediction
improvement: there is no matched deterministic-path ablation on the frozen
test. Put curves and per-dimension diagnostics in Appendix D, and keep this
one-sentence qualification in the main text only if space remains.

## Concrete figure and table plan

Use a small main report. Every quantitative display below has a source result
linked here; do not use historical CNP-AIRT grids, random-subject-split scores,
or the superseded five-task replay as main evidence.

### Main report

| Item | What it should show | Exact evidence / data source | Status and rule |
| --- | --- | --- | --- |
| **Figure 1 — faithful streaming benchmark-holdout protocol** | A schematic: train benchmarks → unseen validation/test benchmarks; for one task, current candidate → accept/reject → label only if accepted → predict disjoint scoring items at nested budgets. Explicitly cross out future-pool access. | [Protocol](docs/benchmark-holdout-online-replay.md); [panel definition](organized/testing/online/README.md#panel) | New schematic, not a numerical result. Keep it simple enough that it explains why full-pool Fisher selection is invalid here. |
| **Table 1 — benchmark-held-out panel audit** | Retained responses, tasks, items, response rate, support thresholds, exact-text overlap rule, and train/validation/test benchmark roles. Include only the benchmark-held-out corpus. | [Benchmark-holdout summary](docs/benchmark-holdout-split-distribution-summary.csv); [protocol](docs/benchmark-holdout-online-replay.md) | Main-text protocol table. Do not put a random-split comparison beside it: that belongs in Appendix B. |
| **Figure 2 — Brier learning curves on untouched test** | Macro-task Brier at budgets 0, 1, 3, 7, 15, 31 for constant training mean, Beta-Binomial, filtered ANP, dense-prefix ANP, and SAM ANP, all with random-prefix acquisition. Lower is better. | [Test per-budget table](organized/testing/online/README.md#brier-by-number-of-labels-acquired) | Primary result. Five lines only; do not include contaminated ANPs. Shade paired bootstrap intervals if per-task files are exported, otherwise use the table for intervals. |
| **Table 2 — final paired test comparison** | Macro Brier ALC, difference from Beta-Binomial, 95% paired CI, and verdict for constant, Beta-Binomial, and the three clean ANPs. | [Test ALC and paired differences](organized/testing/online/README.md#results---test-final-3-untouched-benchmarks) | Primary result. It must show the constant/Beta tie and the three neural losses. |
| **Table 3 — acquisition controls on untouched test** | Beta-Binomial random prefix, uncertainty, confidence extremes, Bernoulli variance, and EBR: ALC and paired difference versus random. | [Test policy rows and intervals](organized/testing/online/README.md#results---test-final-3-untouched-benchmarks) | Include only if space permits; otherwise Appendix E. Do not place empty ANP-policy cells in the table. |
| **Figure 3 — ability-range extrapolation** | Test Brier by post-hoc subject response-rate/ability stratum, with the training-range boundaries shown explicitly. | Final per-task replay outputs plus the training-task response-rate distribution | Required for Q2 if the 45-task panel supports stable intervals; otherwise appendix-only with a limitation statement. Never use test labels to select a model or policy. |

### Appendix

| Item | What it should show | Exact evidence / data source | Status and rule |
| --- | --- | --- | --- |
| **Figure A1 — benchmark-shift diagnostic** | Two small bars for validation and test: label-rate shift and BGE centroid similarity to training. Optionally add the corresponding random-subject-split values in a muted colour. | [Benchmark-holdout summary](docs/benchmark-holdout-split-distribution-summary.csv); [random-split summary](docs/non-benchmark-holdout-split-distribution-summary.csv) | Data ready. This explains the split; it is not a causal explanation of model failure. |
| **Figure A1a — before/after response matrix and retained support** | Matrix sparsity/degree and retained rows/columns before versus after cleaning. | [Final data analysis](docs/final-data-analysis.pdf); [data preparation record](docs/data-preparation.md) | Appendix only. It substantiates data hygiene, not model quality. |
| **Table A1 — complete development sweep** | All validation models × policies, per-budget Brier, ALC, and paired intervals. Separate clean and contaminated models with an unmissable label. | [Validation replay results](organized/testing/online/README.md#results---validation-unseen-benchmarks) | Data ready. This records the SAM development improvement without turning it into a final claim. |
| **Table A1a — offline non-benchmark-holdout results** | Cell, subject split-half, item split-half, and zero-response cold-item AUC/Brier for classic IRT, MIRT, and amortized IRT. | [Static result ledger](README.md#everything-run-so-far-side-by-side-2026-10-27) | Diagnostic appendix only. Mark shared-benchmark structure and do not compare its AUC directly with online Brier ALC. |
| **Figure A2 — item-pass-rate diagnostic** | A restrained bar/scatter diagnostic: observed item pass-rate SD versus predicted SD, plus overall and within-benchmark correlation, for the 297-item diagnostic set. | [Held-out ANP diagnostic](docs/np-online-passrate-figure11-summary.csv); [non-holdout reference diagnostic](docs/non-benchmark-anp128-qwen-item-passrate-summary.csv) | Existing summaries support this, but label the item subset and construction precisely. Keep it appendix-only: it motivates the difficulty-transfer hypothesis; it does not prove the cause. |
| **Table A2 — data cleaning and retained support** | Exact duplicate/missing-response handling, support thresholds, number retained after each filter, and exact content-overlap audit. | [Replay protocol](docs/benchmark-holdout-online-replay.md); filtering/export artifacts | Needs an explicit generated audit table before submission. Do not reconstruct counts from memory. |
| **Figure A3 — ANP architecture** | Latent and deterministic paths, contextual/self-attention and target cross-attention, inputs/outputs, and what is shared. | [Architecture design note](docs/np-design.pdf) | Appendix-only. It should document the evaluated configuration, not imply that attention was the main contribution. |
| **Figure A4 — training and latent diagnostics** | Train/validation objective components, KL, posterior/prior variance and mean shift, active dimensions, and checkpoint selection. | [Training results](docs/training-final-results.pdf) | Appendix only. It rules out complete latent collapse in the selected runs; it is not a causal explanation of test performance. |
| **Table A3 — checkpoint/protocol ledger** | Checkpoint name, training split, embedding arm, selection panel, candidate-order seeds, and replay role. | [Online replay result ledger](organized/testing/online/README.md) | Needed for reproducibility. Exclude random-split checkpoints from any transfer row. |

### Displays to leave out

- Do **not** make a headline figure from random-subject-split ANPs: benchmark
  overlap makes those scores a diagnostic only.
- Do **not** put language, kNN-graph, spectral/Laplacian, response-function, or
  item-pass-rate plots in the main text. They are exploratory data diagnostics,
  not direct evidence for the benchmark-transfer claim.
- Do **not** use [`online-replay-acquisition-validation.pdf`](docs/online-replay-acquisition-validation.pdf)
  as final evidence: it is the earlier, low-resolution five-task replay.
- Do **not** plot a policy as successful merely because it was best on
  validation; the SAM validation result reversed on test.

### 7. Discussion and conclusion

Interpret the results at the right level. In our setting, the tested frozen item
embeddings and filtered ANP configuration did not yield a reliable improvement
over a context-only Beta-Binomial predictor under benchmark shift. This replay
does not test an item-history empirical prior: every prediction from the
baseline depends only on acquired labels in the current task.

State that sequential policies need coherent updates as the context grows.
Dense-prefix training was evaluated, but it was worse than Beta-Binomial on the
untouched test panel; it is therefore evidence against this particular training
change, not a solved direction. The present evidence is consistent with
difficulty transfer being a bottleneck, but does not isolate it causally from
model calibration or online adaptation.

Relate this to Zhang et al. (2025): their learned benchmark predictors also
fail under extrapolation (to stronger models), and the estimator that survives
is the one anchored to observed labels (AIPW). The item-level analogue here is
a predictor that starts from the context-only Beta-Binomial and lets the
network move it only when that helps. We tested four such combinations
post hoc on the saved random-acquisition replays (same panel, candidate
orders and seeds; macro Brier-ALC difference to Beta-Binomial on the
untouched test panel, 45 tasks, paired bootstrap 95% CI):

| combination | w64 filtered | w64 SAM | w64 dense-prefix |
| --- | --- | --- | --- |
| ANP alone | +0.023 | +0.017 | +0.027 |
| gate: Beta-Binomial below 2 labels, ANP after | +0.003 | +0.004 | +0.007 |
| blend $w\,p_{\text{ANP}} + (1-w)\,p_{\text{BB}}$, $w$ fitted on validation | +0.006 | +0.008 | +0.007 |
| per-budget blend, $w_k$ fitted on validation | +0.004 | +0.008 | +0.008 |
| AIPW-style shift (below) | +0.050 | +0.053 | +0.056 |

Every row is significantly worse than Beta-Binomial on test. Gating and
blending remove most of the ANP's loss, which confirms that most of it sits at
budgets 0 and 1, but no amount of ANP helps on unseen benchmarks.

The AIPW-style item-level analogue was the worst, roughly twice as bad as the
ANP alone, on validation too. It shifts every item prediction by the mean
residual of the labels seen so far, $p_i + \frac{1}{k}\sum_{j \le k}(y_j -
\hat p_j)$, where $\hat p_j$ is the prediction made before label $j$ was
revealed. After one label that shift is about $\pm 0.5$, which is far too
noisy. The same correction applied to Beta-Binomial itself was also worse
(test ALC 0.2986 vs 0.2494; Brier at budget 1 rose from 0.28 to 0.44),
because it undoes the shrinkage that makes Beta-Binomial work at small $k$.
AIPW succeeds in Zhang et al. because they estimate one aggregate accuracy,
where residual noise averages out over the benchmark; it does not carry over
to calibrated per-item probabilities from a handful of labels.

The ANP results above are offline recombinations of saved predictions, not
new replays; they are exact for random acquisition, whose label order does
not depend on the predictor.

**Future work.** A causal-masked ANP could encode acquired responses in their
arrival order, so each new label extends a prefix representation rather than
re-encoding the whole context set. With cached attention states, this could
also reduce sequential inference cost, following the incremental mechanism of
Mortimer et al. (2026). We did not complete or evaluate this variant, so it is
a proposed direction rather than a result.

## Implementation walkthrough: dense-prefix training

This is the concrete training loop behind the phrase "dense-prefix training."
It is written in the same order as the discussion that motivated it.

**What is a task?** One task is one `(subject, benchmark)` pair. It contains
the items that this subject answered on that benchmark and each recorded binary
response. In Neural Process language, this full response pattern is the
function that an episode samples from.

**Step 0.0: start iteration `t`.** Sample a batch of distinct training
`(subject, benchmark)` tasks.

**Step 0.1: make a fake stream for each task.** Select a context-candidate
subset of the task's answered items and randomly order it. Each entry is an
item embedding with a recorded response, but the response is treated as hidden
until that entry joins a prefix. This simulates labels arriving one at a time.

**Step 0.2: choose targets.** Sample a fixed, disjoint target set from the
same task. These are the items whose responses the model must predict. The
target set stays fixed while the prefixes for this task change. The training
loop does not necessarily use every answered item from the pair: it uses a
bounded context stream and a capped target subset.

**Step 0.3: use nested prefixes.** For a stream `A, B, C, D, ...`, a prefix
of size zero reveals nothing, a prefix of size one reveals the response for
`A`, and a prefix of size three reveals the responses for `A, B, C`. The
planned budget ladder is `0, 1, 3, 7, 15, 31`.

**Step 0.4: predict targets at every prefix.** At prefix zero, ANP receives no
observed labels and predicts the fixed targets from their item embeddings and
its prior. At later prefixes, it receives the revealed context item embeddings
and binary labels, plus the same target-item embeddings. It outputs one
probability for each target response.

**Step 0.5: collect loss.** Compute binary cross-entropy between each target
probability and its hidden recorded response. Average over targets to obtain
one loss per prefix, then average across prefixes to obtain one loss for this
task. A latent ANP also adds its KL term according to the model's existing
objective.

**Step 0.6: finish the batch.** Repeat the same process for every task in the
batch. Average the task losses into one batch loss, backpropagate once, and
update the model weights once. The next iteration samples a new batch and can
sample new context and target subsets.

The distinction from online evaluation is simple: training can inspect the
recorded outcomes to construct losses, while the model itself receives only the
prefix labels. At test time, the stream is real: the next candidate arrives,
the policy decides whether to acquire its label, and future candidates remain
unknown.

## References

This plan uses author--year citation marks; render them with the final NeurIPS
template's selected citation style. The references below distinguish scholarly
foundations from repository artifacts that establish the exact data, protocol,
and results. The latter should be released with a permanent commit or archive
link, not cited as substitutes for the scientific literature.

### Scholarly references

- **[Brier, 1950]** Glenn W. Brier. "Verification of forecasts expressed in
  terms of probability." *Monthly Weather Review*, 78(1):1--3, 1950.
  [DOI](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2).
- **[Garnelo et al., 2018a]** Marta Garnelo, Dan Rosenbaum, Christopher
  Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo
  Rezende, and S. M. Ali Eslami. "Conditional Neural Processes." *ICML*,
  PMLR 80:1704--1713, 2018. [PMLR](https://proceedings.mlr.press/v80/garnelo18a.html);
  [local PDF](literature/Conditional%20Neural%20Processes.pdf).
- **[Garnelo et al., 2018b]** Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum,
  Fabio Viola, Danilo Rezende, S. M. Ali Eslami, and Yee Whye Teh. "Neural
  Processes." arXiv:1807.01622, 2018. [arXiv](https://arxiv.org/abs/1807.01622);
  [local PDF](literature/Neural%20Processes.pdf).
- **[Guo et al., 2017]** Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q.
  Weinberger. "On Calibration of Modern Neural Networks." *ICML*, PMLR
  70:1321--1330, 2017. [PMLR](https://proceedings.mlr.press/v70/guo17a.html).
- **[Jeffreys, 1946]** Harold Jeffreys. "An invariant form for the prior
  probability in estimation problems." *Proceedings of the Royal Society A*,
  186(1007):453--461, 1946. This is the source for the
  \(\operatorname{Beta}(1/2,1/2)\) prior used by the Beta-Binomial reference.
  [DOI](https://doi.org/10.1098/rspa.1946.0056).
- **[Kim et al., 2019]** Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta
  Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh.
  "Attentive Neural Processes." *ICLR*, 2019.
  [OpenReview](https://openreview.net/forum?id=SkE6PjC9KX);
  [local PDF](literature/Attentive%20Neural%20Processes.pdf).
- **[Li et al., 2025]** Yang Li, Jie Ma, Miguel Ballesteros, Yassine Benajiba,
  and Graham Horwood. "Active Evaluation Acquisition for Efficient LLM
  Benchmarking." *ICML*, PMLR 267:35581--35602, 2025.
  [PMLR](https://proceedings.mlr.press/v267/li25bp.html);
  [local PDF](literature/Active%20Evaluation%20Acquisition%20for%20Efficient%20LLM%20Based%20Benchmarking.pdf).
- **[Lord, 1980]** Frederic M. Lord. *Applications of Item Response Theory to
  Practical Testing Problems*. Lawrence Erlbaum Associates, 1980.
  [Publisher record](https://www.routledge.com/Applications-of-Item-Response-Theory-to-Practical-Testing-Problems/Lord/p/book/9780898590067).
- **[Mortimer et al., 2026]** Philip Mortimer, Cristiana Diaconu, Tommy
  Rochussen, Bruno Kacper Mlodozeniec, and Richard E. Turner. "Incremental
  Transformer Neural Processes." *ICML*, PMLR 306:90425--90469, 2026.
  [PMLR](https://proceedings.mlr.press/v306/mortimer26a.html);
  [local PDF](literature/Incremental%20Transformer%20Neural%20Processes.pdf).
- **[Truong et al., 2025]** Sang T. Truong, Yuheng Tu, Percy Liang, Bo Li, and
  Sanmi Koyejo. "Reliable and Efficient Amortized Model-based Evaluation."
  *ICML*, PMLR 267:60238--60265, 2025.
  [PMLR](https://proceedings.mlr.press/v267/truong25c.html);
  [local PDF](literature/Reliable%20and%20Amortized%20Model-based%20Evaluation.pdf).
- **[Zhang et al., 2025]** Guanhua Zhang, Florian E. Dorner, and Moritz Hardt.
  "How Benchmark Prediction from Fewer Data Misses the Mark."
  arXiv:2506.07673, 2025 (v2 2026).
  [arXiv](https://arxiv.org/abs/2506.07673).

### Competition and repository sources

- **[AIMS task materials, 2026]** Competition task background and allowed
  prediction setting: [local research brief](literature/NeurIPS%20AIMS%20Competition%20Research.pdf).
  Before submission, replace this convenience copy with the official challenge
  URL and version/date.
- **[AIMS NeurIPS team, 2026a]** [Online testing notes](<Online Testing.md>)
  and [benchmark-held-out replay protocol](docs/benchmark-holdout-online-replay.md).
  These specify the local faithful streaming replay, metrics, and no-leakage
  checks.
- **[AIMS NeurIPS team, 2026b]** [Online replay result ledger](organized/testing/online/README.md),
  [data-preparation record](docs/data-preparation.md), and
  [final data audit](docs/final-data-analysis.pdf). These are the traceable
  sources for the reported panels, preprocessing, and numerical results.

## Appendix

- Full data audit, filtering thresholds, exact-overlap counts, and
  per-benchmark retention counts.
- Architecture diagrams, hyperparameters, checkpoint-selection rules, and the
  relationship between the checkpoint and its replay result.
- Per-budget Brier, ECE, macro and micro metrics, with paired uncertainty where
  the result is used for a claim.
- Equations and implementation details for valid streaming policies.
- Candidate-order seeds, split identifiers, and reproduction commands.
- Release manifest: immutable source revision, submission-package checksum,
  dependency environment, data permissions, and any non-`measurement-db`
  training-data provenance.
- Earlier CNP-AIRT, calibration, and full-pool Fisher experiments, clearly
  labelled as historical protocols that are not comparable to the frozen
  benchmark-held-out panel.

## Evidence rules before drafting results

1. Use only completed, reproducible results in the main report. The current
   final clean evidence is the 45-task untouched test replay; the 82-task
   validation replay is for development analysis.
2. Never combine the historical 104-task CNP-AIRT grid with the current
   82-task panel; do not use contaminated random-subject-split ANPs as transfer
   evidence.
3. Identify the exact frozen panel and source-item cap from its manifest. The
   current README contains conflicting `cap19000` and `cap100000` labels, and
   its panel summary and test paragraph disagree about the number of candidate
   order seeds. Reconcile both against the manifest before any result is quoted.
4. Report every main comparison with data split, panel, checkpoint, candidate
   orders, budgets, and paired uncertainty interval. The current clean ANP
   evidence uses one documented training seed; state that limitation.
5. Do not claim a neural or acquisition win from an incomplete run, an
   unpaired comparison, or a confidence interval that includes zero.

## Split representativeness: TODO (raised 2026-10-31)

The fixed benchmark-holdout split (11 train, 3 validation, 3 test benchmarks)
gives only three held-out benchmarks per role, so which benchmarks land in
test decides much of the result. Benchmark difficulty differs a lot:

| role | benchmarks (mean subject accuracy) |
| --- | --- |
| validation | bird_sql 0.15, financebench 0.50, ai2d_test 0.69 |
| test | editbench 0.54, cmmlu 0.68, algotune 0.69 |
| train | appworld 0.28 ... agi_elo 0.70; alignment_faking holds 4,278 of the 6,013 training subjects (71%) at mean 0.41 |

Test subjects average 0.61 against 0.47 in training. Much of that gap is
benchmark difficulty and alignment_faking's weight, not subjects being
stronger. Two changes before the final report:

- **Weighting:** weight the training loss, and any fitted prior, so each
  benchmark counts equally instead of each subject. Then alignment_faking's
  4,278 subjects count as one benchmark, not 71% of the data.
- **Leave-one-benchmark-out:** train on all benchmarks except one, test on the
  one left out, and rotate through all 17. That gives 17 held-out results
  instead of 3. Only the hierarchical beta-binomial check (14 folds) has used
  this so far.

The online curves and the multi-pair draws both still use the one fixed split
(60 validation and 150 test draws from the same 3 + 3 benchmarks). Multiple
draws do not replace leave-one-benchmark-out; they answer different questions:

- Multiple draws reshuffle which subjects are tested together, within the same
  three test benchmarks.
- Leave-one-benchmark-out changes which benchmarks are held out.

They combine: run the draws inside each leave-one-benchmark-out fold.

## Evidence still required before a final-results report

- The panel manifest and exact-overlap/support audit archived with the paper
  artifacts, including a subject-overlap report across split roles.
- Additional clean ANP training seeds and either all ten candidate orders or a
  manifest-backed stated rationale for the order count actually used.
- Paired ANP policy-versus-random intervals and per-budget intervals. The
  current untouched-test paired policy evidence is sufficient only for the
  completed Beta controls.
- A same-panel IRT replay if IRT is retained as a main-report comparator.
