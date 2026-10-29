# Bayesian adapter and full acquisition experiments

Launch group `bayesian-adapters-20261029-v1` evaluates two new models on all 390 validation tasks, with the canonical benchmark-level split and filtering. Test outcomes are not accessed. The frozen ANP is SAM's step-700 checkpoint. All fits/evaluation use nested random acquisition contexts at budgets 0, 1, 3, 7, 15, 31 and fixed query items, using the established seed+3000 full-validation bank.

* ANP correction: zero-context item logits are fixed. An independent Normal offset and Normal slope (centered on one) posterior is integrated on a 21-by-21 Gauss-Hermite grid. Six prior-scale pairs are compared using disjoint support/query episodes from 66 training tasks, stratified over 11 training benchmarks. Select training-query Brier ALC, not validation. No acquired outcome is supplied to the ANP when obtaining its corresponding likelihood input.
* Bernoulli GP classification: fixed text-feature Rasch easiness is the mean function. Covariance is an RBF on standardized training-fitted 64-dimensional BGE PCA features plus a constant random-intercept kernel. The context posterior uses the logistic Bernoulli likelihood, Newton/Laplace inference, and 24-node quadrature of predictive latent uncertainty. Nine amplitude/length-scale combinations are compared on the same training episode bank. Intercept prior SD is one. This is an exact context kernel, not the old random-Fourier Gaussian regression baseline.

The GP's reused feature-Rasch mean was fitted on training responses but its training fraction was previously selected using validation. This is an extension of that selected model; the entire pipeline is not independent of validation. Item-head uncertainty is not integrated. New kernel hyperparameters and ANP prior settings use only training episode outcomes. Training query episodes can share observations with the original mean-model fitting, so training-query scores are fitting criteria rather than unbiased estimates of transfer. Final validation provides the transfer check.

Each model saves source/data/code identities, `last.json` (completed parameter trials), `best.json`, `inference-state.npz` (fixed item logits and GP features), and `completed.json` with every task/budget and selected settings. Restart resumes between deterministic parameter trials. There is no gradient optimizer or scheduler to restore. A smoke run for each model must finish before either full run starts. Forge logs fitting scores, final validation budgets, and the familiar subject diagnostic keys. Diagnostic context is capped at 64 and targets at 512 to match the recent ANP diagnostic convention. These are final diagnostic points, not optimizer learning curves.

Expanded acquisition group `anp-sam-acquisition-full-20261029-v2` compares random, uncertainty and approximate expected Brier reduction on all 390 validation tasks with three candidate permutations each (1,170 trajectories per arm). It preserves the pilot's frozen model, training-only 32-item reference/probe banks, streaming score/rank pacing and 31 purchases. Split seed is unchanged; expanding the task bank changes RNG consumption inside `make_bank`, so this is a new matched comparison across arms, not a literal superset of the pilot's item splits. All arms within the expanded comparison use identical item splits and streams. No future candidate features or hidden evaluation outcomes enter acquisition decisions.

All functions use arshiaswe, two CPU cores, 8 GiB, a six-hour execution cap, and Modal secret `wandb` with required `WANDB_API_KEY`; logs go to CoreWeave Forge `aims-predictive-evaluation`. Checkpoints are on volume `aims-neurips-data`, under `/results/bayesian-adapters/<group>/<kind>/` and `/results/anp-acquisition/<group>/<policy>/`. Full acquisition results should report task-weighted and benchmark-macro ALC, each benchmark's paired difference, budgets, runtime and forced purchases. With only three validation benchmarks, uncertainty about transfer remains substantial even with more tasks.

Commands from the worktree:

```powershell
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_anp_acquisition.py
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_bayesian_adapters.py
```

Eight tests cover quadrature symmetry and recovery, label updates, GP locality, duplicate-context numerical stability, acquisition calculations and causal label access.
