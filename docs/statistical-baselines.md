# Statistical baselines on the frozen benchmark split

Entry point: `organized/training/modal_statistical_baselines.py`. All five models run on CPUs in `arshiaswe` and use the existing `wandb` secret. They do not change or stop the TabPFN job.

## Exact data and evaluation reuse

Inputs are the checksummed prepared TabPFN artifacts in `/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1`: the canonical 11/3/3 benchmark split, filtering, 6,481 tasks, 123,035 items, frozen BGE-M3 embeddings projected to 64 features by training-only PCA. The existing filtering removes constant-outcome items; these experiments measure that filtered population.

The evaluator imports the **same `make_bank` function** used by TabPFN. Validation uses seed `20261018+2000` and the same stratified 64-task bank with identical 50/50 pools, candidate orders, nested label prefixes, target items and population weights. Final full validation and test use seed `20261018+3000`, matching TabPFN. Test labels are not used for fitting, diagnostic curves or checkpoint selection.

## Models

1. `beta_binomial`: response-only empirical-Bayes Beta-Binomial. Alpha and beta are estimated from training subject–benchmark response totals. A fixed Beta(2,2) reference is recorded separately for compatibility with the previous baselines.
2. `rasch`: hierarchical response-only Rasch with unit discrimination. Training subject abilities and training item difficulties use Gaussian-shrunk coordinate Newton estimates; hyperparameter standard deviations use empirical point-estimate updates bounded to [.2,3]. At inference, ability uncertainty is integrated by 81-point Gauss-Hermite quadrature. Unseen item difficulties are integrated under the learned Gaussian prior (21-point quadrature). No embeddings are used. Unknown evaluation items therefore receive identical predictions within a task. Training diagnostic items can use training-fitted item difficulties.
3. `feature_rasch`: explanatory Rasch. Item easiness is a regularized linear function of standardized PCA features; discrimination is fixed at one. Training subject effects have a N(0,1) prior. At inference only the current pair's revealed responses update the quadrature posterior for ability. No per-test-item parameters are fitted using hidden labels.
4. `linear_regression`: ordinary least squares linear probability regression, using standardized item features plus a smoothed subject context success rate. The training context-rate feature uses all other observed fitting responses in the same task, **excluding the target row's own label**. At inference it uses only acquired responses. Predictions are clipped to [0,1]. There is no global ridge penalty; rank-deficient normal equations use least squares with a numerical rank cutoff.
5. `bayesian_rbf`: approximate RBF Gaussian process **regression**, using 128 fixed random Fourier features and Bayesian Gaussian weights. The feature scale is fitted only to training items; length scale is sqrt(feature dimension). Gaussian observation variance .25 and new-subject intercept variance .25 are fixed before validation. Training subject effects are integrated using a Schur complement. At inference a local posterior conditions on the current task's acquired responses without mutating the global posterior or sharing evidence across held-out pairs. The Gaussian posterior mean is clipped to [0,1]. This is not exact GP classification and does not use a Bernoulli likelihood. Weight covariance, precision and feature-map parameters are saved.

Rasch is itself a statistical model. The first two models ignore embedding values; the other three use features. The response-only cold-start constraint is intentional, not worked around by calibrating held-out item difficulties from evaluation labels.

## Forge curves

Project: `aims-predictive-evaluation` under the W&B key's default entity, on `forge.coreweave.com`. Each model has a separate run, grouped by the run label.

Exact keys include:

```text
diagnostic/heldout_subjects/same_subject_split_half/brier_score
diagnostic/training_subjects/same_subject_split_half/brier_score
diagnostic/training_subjects/same_subject_response_holdout/roc_auc
diagnostic/training_subjects/same_subject_split_half/roc_auc
```

These are **data-efficiency curves**, with `training_fraction` as the W&B x-axis: 5%, 10%, 25%, 50%, 100% of fitting response labels. Every stage uses the same deterministic per-task permutation, so subsets are nested and identical across models. All training tasks are represented. Fixed preprocessing may use all unlabeled training item features. Each statistical fit starts fresh at its stage; these points are not fabricated gradient epochs.

Diagnostic tasks/pools remain fixed across all stages and models. `heldout_subjects` means **validation-benchmark tasks**, not the final test tasks. Split-half diagnostics reveal the full acquisition half, and response-holdout diagnostics use an 80/20 context/target carve. These diagnostic contexts can exceed 31 labels and are separate from competition-budget scoring. Training diagnostics are in-sample: global parameters may have been fitted using their targets. They are not estimates of generalization.

Brier includes every scored task. AUC is averaged only over tasks with both target classes, and eligible task counts are logged separately. Main validation logs also include Brier at 0/1/3/7/15/31 and aggregate Brier ALC with weights .1/.2/.2/.2/.2/.1. The selected data-fraction checkpoint minimizes fixed-bank validation ALC. Selected models receive one full final validation/test evaluation with per-task scores and benchmark-macro summaries.

## Persistence and execution

Each model writes `/data/results/statistical-baselines/<run-label>/<model>/`:

- `last.pkl`: completed-stage resume state, model parameters, transforms, code/data identity, next stage, history, fixed banks and W&B identity.
- `best.pkl` and `best-validation.json`: selected parameters and validation results.
- `curves.json`, `final-validation.json`, `final-test.json`, `completed.json`.

Writes use temporary files plus atomic rename, followed by a volume commit. Interrupted **stages** restart deterministically; hidden L-BFGS or Newton iterates are not claimed to be checkpointed. Re-running the command resumes from the last completed fraction. Completed runs and incompatible configurations are rejected/treated as immutable. Pickles are trusted artifacts created by this trainer.

```powershell
# All five real-data smoke tests: 1% fitting labels, short numerical fits; no test evaluation.
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_statistical_baselines.py --smoke

# Five learning-curve stages per model, followed by selected-model evaluation.
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_statistical_baselines.py
```

Jobs request 2 CPUs and 8 GiB each, no GPUs. A small workflow coordinates the five independent jobs and reports failures without immediately cancelling unaffected models.

Seven focused tests cover nested training-only response subsets, train-only feature mapping, all five inference paths, exchangeable unknown-item Rasch predictions, exact Gaussian-system equivalence, ordinary least-squares equivalence, diagnostic metric names and single-class AUC handling.

## Completed comparison

All five full runs completed. Every run has five curve points and all four requested diagnostic keys. Local JSONs are in `reports/statistical-benchmark-20261029-v1/`. Fetch/re-verify with `scripts/fetch_statistical_results.py`.

All models scored the same 78 test tasks from algotune, cmmlu and editbench. The TabPFN cohort and fixed Beta(2,2) reference also match (reference agreement tolerance 1e-6), verifying common evaluation episodes rather than only common benchmark names.

| Model | Selected fitting-label fraction | Test Brier ALC |
|---|---:|---:|
| Empirical-Bayes Beta-Binomial | 5% | 0.243423 |
| Hierarchical Rasch | 10% | 0.244100 |
| Feature Rasch | 5% | 0.235079 |
| Ordinary linear regression + context rate | 5% | 0.235597 |
| Bayesian RBF GP regression approximation | 100% | 0.257082 |
| TabPFN LoRA reference | selected update 900 | 0.236410 |
| Fixed Beta(2,2) reference | no fitting | 0.237026 |

Fractions/checkpoints were selected using validation, not test results. These are point estimates on a single three-benchmark test split; small differences are not significance claims. Both the comparison app and the TabPFN app were confirmed stopped after completion.

Full comparison app: https://modal.com/apps/arshiaswe/main/ap-QYtEXYWGvfWfB11lbs54N4
