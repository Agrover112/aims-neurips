# Predicting Task Difficulty Without Rollouts

**Paper:** Krsteski, S., & Meyer, C. (2026). *Predicting Task Difficulty Without Rollouts.* COLM 2026 Workshop on Agent Behavior.
**arXiv:** [2608.05797v3](https://arxiv.org/pdf/2608.05797) [cs.LG], 15 Oct 2026.
**Affiliations:** Andromede AI, EPFL.

---

## 0. One-paragraph summary

The paper asks whether the difficulty of an agentic task can be estimated **before running the agent on it**. The authors fit a 1PL IRT model to a partially observed agent-task response matrix, use the fitted task difficulty \(\beta_t\) as the target, standardize it within each benchmark to \(z(\beta_t)\), and train a ridge regression to predict it from task-side features. The best single feature family is not a static text embedding. It is the **token-level entropy of a generated reasoning trace**. Entropy helps in-distribution and weakly out-of-distribution. Embeddings help in-distribution and collapse out-of-distribution. The paper's own conclusion is that cross-benchmark transfer remains open.

The paper is useful for our project for three reasons:

1. It independently shows that **AUC can be high while item-level difficulty ranking is zero**.
2. It shows that **static text embeddings do not transfer across benchmarks**, matching our item-difficulty result.
3. It introduces **token entropy over a generated reasoning trace** as a weak but real OOD difficulty signal.

It does **not** show that text-only difficulty prediction is solved. It does **not** predict a specific subject's response. It predicts a population-level task difficulty.

---

## 1. What they predict

They use IRT with:

- agents \(a_1,\dots,a_N\),
- tasks \(t_1,\dots,t_M\),
- a partially observed binary response matrix \(Y_{a,t}\),
- agent ability \(\theta_a\),
- task difficulty \(\beta_t\).

The 1PL model is:

\[
P(Y_{a,t}=1 \mid \theta_a,\beta_t)=\sigma(\theta_a-\beta_t).
\]

The fitted \(\beta_t\) is a scalar summary of apparent task difficulty. It can absorb task-side covariates such as feasibility, scoring strictness, familiarity, contamination, and noise. Likewise, \(\theta_a\) can absorb agent-side covariates such as scaffold quality, tool access, prompt format, and noise.

The main prediction target is the **within-benchmark standardized difficulty**:

\[
z(\beta_t).
\]

This removes the benchmark-level average and forces the predictor to rank tasks **inside** a benchmark, not just guess which benchmark is harder.

### 1.1 Jargon table

| Term | Plain meaning |
|---|---|
| **Ex ante difficulty** | Difficulty guessed **before** running the agent |
| **Rollout** | Actually running the agent on the task |
| **\(\beta\)** | Generic task difficulty score |
| **\(\beta_t\)** | Fitted difficulty of task \(t\) from real outcomes. Higher means harder |
| **\(z(\beta_t)\)** | \(\beta_t\) rescaled within its benchmark. \(0\) is average, \(+1\) is harder than average |
| **\(\hat{z}_t\)** | Difficulty predicted from task features before rollouts |
| **Residual \(r_t\)** | \(r_t = z(\beta_t) - \hat{z}_t\). The part the prediction missed |

### 1.2 Reading the residual

| Residual | Interpretation |
|---|---|
| **\(r_t > 0\)** | Task was **harder than predicted**. Possible broken task, infeasible task, missing information, or strict grader |
| **\(r_t < 0\)** | Task was **easier than predicted**. Possible contamination, familiarity, or leakage |

The residual is a **diagnostic**, not a pure pre-rollout prediction. It requires the empirical difficulty \(z(\beta_t)\) from rollouts.

### 1.3 Residual examples

| Task | Observed \(z(\beta_t)\) | Predicted \(\hat{z}_t\) | Residual \(r_t\) | Success rate | Raw \(\beta_t\) | Reading |
|---|---:|---:|---:|---:|---:|---|
| `django-14725` | 1.46 | 0.05 | **+1.41** | 0.090 | 2.58 | Much harder than expected. Flagged for missing prompt information |
| `django-11451` | -0.51 | 0.05 | **-0.56** | 0.718 | -2.35 | Easier than expected. Discussed as a contamination example |

---

## 2. What features they use

Let \(C_t\) be the full task-side context:

- environment description,
- action-space description,
- task description.

Every feature must be cheaper to compute than running rollouts. Features use \(C_t\) directly or through a generated reasoning trace \(R_t\).

| Feature set | What it is |
|---|---|
| **Random control** | \(x_t \sim \mathcal{N}(0,1)\), independent of the task. Should predict nothing |
| **Length baseline** | \(x_t^{\text{len}} = \log(1+|C_t|)\). Controls for task text length |
| **Embedding** | \(e_t = \phi(R_t)\), where \(\phi\) is `bge-base-en-v1.5`. Reduced with fold-aware PCA to avoid overfitting |
| **Entropy** | Shannon entropy of next-token distributions from Qwen3-8B-Base over \(R_t\) |
| **Full** | Entropy summaries from a five-model scorer panel, cross-scorer disagreement, benchmark metadata, structural properties of \(C_t\), and embeddings |

### 2.1 The reasoning trace

Claude Sonnet 4.6 generates \(R_t\) from \(C_t\), with a **4k-token cap**.

Claude is **not** the evaluated agent. It is a **task explainer**. The trace is a richer description of what solving the task involves: steps, constraints, ambiguities, and likely pitfalls.

### 2.2 How entropy is computed

Given the trace

\[
R_t = (r_1,\dots,r_L),
\]

an open-weight scorer defines

\[
p_\ell(v) = p(v \mid C_t, r_{<\ell}),
\]

where \(v\) is a token from the scorer's vocabulary.

The entropy at position \(\ell\) is

\[
H_\ell = -\sum_v p_\ell(v)\log p_\ell(v),
\]

using the **renormalized top-100 next-token distribution** for speed.

The primary configuration uses **Qwen3-8B-Base**. The full configuration uses a five-model scorer panel.

Entropy is computed over the **reasoning-trace tokens**, not over the benchmark item tokens.

At each trace token, the scorer is asked: "given the task and the trace so far, how unsure am I about the next token?" High entropy means the scorer is surprised. They treat that as a difficulty signal.

### 2.3 What entropy features summarize

They do not use only the mean entropy. They summarize the entropy curve:

- mean,
- standard deviation,
- quantiles,
- top-k means,
- slope,
- first- and last-decile means,
- total variation,
- pairwise gaps between scorer summaries.

The paper argues that the **shape** of the entropy trajectory carries information that global averages miss.

### 2.4 Why Claude and Qwen

| Model | Role |
|---|---|
| **Claude Sonnet 4.6** | Writes the reasoning trace. Feature generator, not evaluated |
| **Qwen3-8B-Base** | Reads the trace and reports next-token entropy. Surprise meter, not evaluated |
| **Agent population** | The 497 agent configurations whose outcomes define \(\beta_t\) |
| **Ridge regression** | Predicts \(z(\beta_t)\) from the features |

The paper is **not** saying "we can predict how hard a task will be for Claude." It is saying "we can predict the task's IRT difficulty for the agent population from task-side features."

---

## 3. Main results

### 3.1 Difficulty prediction

Target: \(z(\beta_t)\). Metric: Spearman \(\rho\). KF is in-distribution. LOBO is leave-one-benchmark-out.

| Feature set | KF \(\rho\) | LOBO \(\rho\) |
|---|---:|---:|
| Random control | near 0 | near 0 |
| Length baseline | 0.086 | 0.101 |
| Embedding | 0.107 | **0.017** |
| Entropy | **0.193** | **0.137** |
| Full | **0.399** | **0.225** |

Key reading:

- **Embeddings collapse out of distribution.** This matches our ANP item correlation: \(+0.186\) train, \(-0.050\) test.
- **Entropy survives OOD better than embeddings**, but the gain is modest.
- **Full features are best**, but LOBO is still only 0.225.
- The paper's own conclusion: cross-benchmark transfer remains open.

### 3.2 Estimator choice does not remove the gap

| Estimator | KF \(\rho\) | KF pAcc | LOBO \(\rho\) | LOBO pAcc |
|---|---:|---:|---:|---:|
| Linear | 0.451 | 0.676 | 0.295 | 0.602 |
| Ridge | 0.454 | 0.678 | 0.277 | 0.597 |
| Random forest | 0.475 | 0.683 | 0.130 | 0.545 |
| XGBoost | 0.453 | 0.678 | 0.190 | 0.563 |

The held-out-benchmark gap is not an estimator bug. It is a transfer problem.

### 3.3 AUC can mislead

| Estimator | AUC | \(\rho_\beta\) | \(\rho_{z(\beta)}\) | pAcc |
|---|---:|---:|---:|---:|
| Oracle IRT | 0.937 | 1.000 | 1.000 | 1.000 |
| Benchmark average | 0.806 | 0.519 | not defined | **0.000** |
| Constant difficulty | 0.715 | not defined | not defined | **0.000** |
| Shuffled | 0.610 | -0.011 | -0.005 | 0.500 |

A constant predictor gets **AUC 0.715**. A benchmark-average predictor gets **AUC 0.806** and still has **zero within-benchmark ranking accuracy**.

This is exactly the failure mode our models show. High cell AUC can be driven by subject ability and benchmark averages, not item difficulty.

---

## 4. Smoothing, shrinkage, and regularization

The paper does **not** use the words "smoothing" or "shrinkage." The closest ideas are:

| Mechanism | What it does |
|---|---|
| **Ridge regression** | L2 regularization |
| **Fold-aware PCA** | Compresses embeddings without leaking across folds |
| **Zero-centered Gaussian priors** on \(\theta_a\) and \(\beta_t\), with learned global scales | Effectively partial pooling / shrinkage, but not named that |
| **Variational posterior** | Keeps uncertainty in each ability and difficulty |
| **Within-benchmark standardization** | Removes benchmark-level shifts before ranking |
| **Ability marginalization** | Avoids fragile joint maximum likelihood |

So the fitting is genuinely variational. The paper's IRT parameters are not pure point estimates. The priors and posterior scales do regularize the fit, even though the authors do not frame it as empirical Bayes or shrinkage.

---

## 5. Their setting vs our setting

### 5.1 What is known at test time

| | Their paper | Our competition |
|---|---|---|
| **Agents/subjects** | Known pool, used to fit the IRT target | Hidden, new subjects |
| **Tasks/benchmarks** | New tasks. LOBO holds out benchmarks | Hidden, new benchmarks |
| **Response matrix** | Partially observed, used to fit \(\beta_t\) | Hidden evaluation labels |
| **Known at test** | Task text, agent pool | Subject metadata, item text, benchmark ID, acquired labels |
| **Unknown at test** | Task difficulty \(\beta_t\) | Subject ability, item difficulty, evaluation labels |
| **Prediction unit** | One task, population difficulty | One subject-item pair, individual outcome |
| **Target** | \(z(\beta_t)\) | \(y_{s,i}\), then Brier score |

Their agents are known in the sense that their outcomes define the IRT target. The predictor itself does not use agent identity. It predicts a **population-level task difficulty**.

Our hidden test has new subjects **and** new benchmarks. We must predict a specific subject's outcome on a specific item. That is a harder identification problem.

### 5.2 Conceptual comparison

| Aspect | Their paper | Our IRT/MIRT |
|---|---|---|
| **Model** | 1PL: scalar \((\theta_a,\beta_t)\) | 1/2/3PL, including multidimensional MIRT |
| **Uncertainty** | Variational Gaussian posterior \((q(\theta),q(\beta))\) | Mostly point estimates; no variational posterior |
| **Objective** | ELBO with SVI/ClippedAdam | BCE/MAP-style objective with AdamW |
| **Output** | Posterior means of \(\beta_t\), then predict difficulty from metadata | Direct response predictions; no downstream difficulty predictor |
| **Holdout** | K-fold and leave-one-benchmark-out | Cell, subject/item splits; benchmark-holdout ANP work is separate |

Ours is conceptually similar in estimating ability and difficulty, but their fitting is genuinely variational. Ours is primarily point-estimate/MAP optimization.

---

## 6. Traces in the competition

The paper generates its own reasoning traces with Claude Sonnet 4.6. The competition does **not** provide reasoning traces.

During online testing, a submission receives:

- `labeled`: outcomes purchased so far, across all sampled pairs,
- the current `[subject, item]` input,
- acquisition context counts.

It does **not** receive:

- evaluation-pool outcomes,
- future acquisition candidates,
- rejected candidates' labels,
- other submissions' traces,
- the full hidden response matrix.

So the paper's trace feature is not directly available in the competition. A submission could generate its own traces if the environment allows an LLM call, but that is expensive and the expected gain is small.

---

## 7. What this means for our project

1. **Stop using AUC as evidence of item difficulty.** Use within-benchmark Spearman, pairwise accuracy, or item pass-rate correlation. The paper shows AUC 0.806 with zero within-benchmark ranking.

2. **Static text embeddings are not the rescue.** The paper's LOBO embedding correlation is 0.017. Our ANP test item correlation is -0.05. The two results agree.

3. **Token entropy is the one text signal that survives OOD**, but it is weak at 0.137. It is a prior, not a solution. It may be worth a small probe on our public data, but it will not beat cross-pair labels.

4. **The paper supports our negative result.** Learned text priors do not transfer across benchmarks. The missing signal is collaborative: other subjects' responses to the same items.

5. **Do not claim a text-only SOTA win.** The honest claim is: text priors fail OOD; entropy helps a little; collaborative labels and proper regularization are the stronger path.

---

## 8. Candidate next experiment

If we want to test the paper's idea on our data:

1. Generate a reasoning trace for each item or task with a small open model.
2. Compute token-entropy summaries with a separate scorer.
3. Add them to our item-difficulty probe.
4. Evaluate with **within-benchmark Spearman** and **pairwise accuracy**, not AUC.
5. Compare against a cross-pair label model.

Expected outcome from the paper: entropy will beat static embeddings OOD, but the gain will be small. If it does not beat the cross-pair label model, cite it as related work and keep the negative-result framing.

---

## 9. Citation

```bibtex
@inproceedings{krsteski2026predicting,
  title={Predicting Task Difficulty Without Rollouts},
  author={Krsteski, Stefan and Meyer, Charlotte},
  booktitle={COLM 2026 Workshop on Agent Behavior},
  year={2026},
  url={https://arxiv.org/pdf/2608.05797}
}
```
