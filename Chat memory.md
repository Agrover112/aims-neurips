# Chat memory

## Project and deadline

- Project: AIMS/PAIEC predictive evaluation competition, targeting a NeurIPS
  2026 poster.
- Submission deadline: 30 October 2026, 23:59 Anywhere on Earth.
- Deliverables: Codabench submission, reproducible code, OpenReview upload, and
  a roughly 4–8 page NeurIPS-format technical report.
- Selection priorities: technical soundness, clarity, reproducibility,
  experimental analysis, predictive performance, and scientific rigor. Novelty
  is secondary. A clean ablation study is more valuable than an unverifiable
  complex model.

## Competition prediction task

- The predictor estimates `p(y=1 | subject, item, acquired labels)`.
- A subject is a model plus its evaluation configuration (provider, harness,
  reasoning effort, version, and related metadata).
- The runtime supplies one subject–item input and a list of acquired binary
  labels. The predictor must be stateless and work without network access.
- Official budgets are `0, 1, 3, 7, 15, 31` acquired labels.
- Primary metric is Brier ALC, with weights
  `0.1, 0.2, 0.2, 0.2, 0.2, 0.1`, respectively. Lower is better.
- Report Brier at every budget, ALC, calibration/ECE, benchmark failures, seed
  variance, runtime, memory, and uncertainty intervals.

## Data preprocessing

Raw Measurement Data Bank tables contain benchmarks, subjects, items, responses,
and (separately) response traces.

Filtering and validation:

- Keep binary, item-level benchmarks.
- Drop rows whose outcome is missing.
- Reject observed non-binary outcomes rather than silently coercing them.
- Reject orphan response-to-subject or response-to-item links.
- Keep raw files unchanged and record dropped rows/skipped benchmarks in the
  manifest.
- `chartmuseum` has only missing response outcomes in the current snapshot, so
  it remains in raw metadata but contributes no supervised observations.

Canonicalization:

- `item_uid = benchmark_id + item_id`.
- `subject_uid` is a stable signature of the model and evaluation settings.
- `task_id = subject_uid + benchmark_id`.
- Preserve repeated trials; repeated responses for the same source item remain
  together.
- Join response rows to canonical subjects and items.

Outer splits are by complete benchmark, not random response rows:

- train benchmarks: 80%
- validation benchmarks: 10%
- test benchmarks: 10%

This prevents benchmark-style leakage. The temporary Modal `full` snapshot
uses a benchmark-level `n_items >= 80` prefilter for fast competition-shaped
experiments; this is not a statement that smaller benchmarks are forbidden from
training. A future all-benchmark snapshot can add them.

Within each eligible subject–benchmark task:

- Require at least 80 distinct source items for official-style evaluation.
- Split distinct items into a persistent 50/50 acquisition pool and untouched
  evaluation pool; odd counts give the extra item to acquisition.
- Acquisition reveals one recorded outcome per acquired item.
- Every response row of an evaluation item is scored; repeated trials stay in
  the same pool.
- Contexts are nested across budgets: budget 3 includes the budget-1 labels.
- The offline split is deterministic for reproducibility, but does not claim to
  reproduce Codabench’s private persistent random split.

## Traces

- 54 available trace files were downloaded to Modal under
  `/data/datasets/full/traces` (~21.66 GiB); only
  `agentic_review_perturbation` had no published trace file.
- Traces contain response-level reasoning/behavior text and can reveal the
  observed outcome. They are privileged training-only information.
- During offline training, a trace encoder may provide auxiliary behavior or
  failure-mode targets, teacher representations, or latent distillation.
- During Codabench inference, traces must not be required and target traces must
  never be used. The deployed model must run with the trace directory absent.
- Compare the same trace-free inference model trained without traces versus with
  trace supervision.

## Baselines and model ladder

Organizer/reference baselines:

1. Constant/random baseline.
2. Empirical Mean: mean of acquired labels, using 0.5 at budget zero.
3. BLE acquisition baseline.
4. Empirical Mean with BLE Acquisition.

Our beta-binomial smoothed empirical mean is an additional auditable baseline.

Planned model ladder:

### GP-AIRT

An Amortized IRT base plus a Gaussian-process item residual:

`ell_sj = a_j * (theta_s - b_j) + g_s(e_j)` and
`y_sj ~ Bernoulli(sigmoid(ell_sj))`.

Use a sparse variational or Laplace approximation because Bernoulli GP inference
is non-conjugate. Treat this as a matched comparator, not necessarily the core
submission.

### Deterministic CNP-AIRT

Encode each context pair and pool permutation-invariantly:

`r_C = Pool({ h(x_i, y_i) : i in C })`.

Decode directly with an AIRT/2PL head. No latent random variable is used; the
same context gives the same prediction.

### Latent NP-AIRT

Infer a latent posterior:

`q(z | C) = Normal(mu_C, diag(sigma_C^2))`.

then average decoder predictions over samples from `q(z | C)`. Offline
training uses prediction loss plus posterior matching/KL-style regularization.

### ANP-AIRT

Add target-specific attention:

`r_C,j = Attention(x_j, { (x_i, y_i) : i in C })`.

so each target item receives a different context representation. The context
set encoder remains permutation-invariant; attention is permutation-equivariant
with respect to the context items.

ANP variants from Kim et al. (2019), arXiv:1901.05761:

- Dot-product ANP-AIRT: single learned cross-attention in the deterministic
  path.
- Multihead ANP-AIRT: multihead target-to-context cross-attention.
- Stacked Multihead ANP-AIRT: multihead cross-attention plus two stacked
  self-attention layers over contexts in both deterministic and latent paths.
- Laplace-attention ANP is a parameter-free diagnostic. The paper also reports
  that stacking beyond two self-attention layers did not show clear gains.

All models adapt online by recomputing their state from the supplied acquired
labels. They do not perform gradient updates at test time.

## Offline evaluation protocol

1. Fit encoders, scalers, calibration, and model parameters only on outer-train
   benchmarks.
2. Select hyperparameters on validation benchmarks.
3. For each held-out test task, use the fixed acquisition/evaluation item pools.
4. Recompute the model state at each nested label budget.
5. Predict all untouched evaluation rows, average within each task, then average
   tasks equally.
6. Use identical episodes, seeds, acquisition prefixes, and target pools for
   every baseline and ablation.

The online process is conditioning on newly revealed labels, not changing the
offline weights. At budget zero, the model uses its prior/metadata pathway.

## Ablations for the report

- organizer baselines and beta-binomial;
- 1PL, 2PL, and AmortizedIRT;
- GP-AIRT;
- deterministic CNP-AIRT;
- latent NP-AIRT;
- ANP-AIRT;
- remove item text embeddings, item metadata, cognitive scales, and subject
  metadata separately;
- no trace supervision versus auxiliary trace supervision versus latent
  teacher–student alignment;
- attention/relational additions only if the simpler model justifies them.

The report should include Brier learning curves, calibration plots, per-
benchmark failures, seed variance, and negative results. Disclose training data
sources and use of LLM coding assistants.

## Repository notes

- Main data/evaluation code: `src/aims_starter/data.py` and
  `src/aims_starter/evaluation.py`.
- Modal ingestion and storage: `modal_ingest.py` and `modal_app.py`.
- Starter kit: `reference/starting_kit/`.
- Organizer baselines: `reference/paiec_baseline/`.
- Design and training notes: `docs/np-airt-implementation.md` and
  `docs/raa_irt_design.md`.
- The image shown in chat visualizes this preprocessing pipeline.

## Immediate next implementation step

Create and freeze one shared competition-shaped episode index, then run the
organizer baselines and beta-binomial on it. Every later model must consume the
same episode assignments before adding GP, CNP, NP, or attention complexity.

## Step 0: canonical mathematical notation

The research implementation now follows one shared convention. NP/ANP code
uses `x_C`, `y_C`, `x_T`, `y_T`, latent-path encodings `s_i` and `s_C`,
deterministic representations `r_i` and `r_star`, distributions `q_z_C` and
`q_z_T`, and samples `z`. IRT code uses subject ability `theta` and item
parameters `a`, `b`, and `c`. Amortized item parameters are functions of `x`.
The authoritative glossary is `docs/notation.md`; future equations, plots,
logs, and reports must use the same symbols.
