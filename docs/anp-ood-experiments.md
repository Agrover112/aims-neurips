# Four matched ANP OOD experiments

Entry point: `organized/training/modal_anp_ood.py`. Group: `anp-ood-bge1024-20261029-v1`. Workspace: `arshiaswe`. All four full runs were launched after four real-data two-update checkpoint/resume checks passed.

## Shared setup

- Existing canonical 11/3/3 benchmark split and filtering; identical task artifacts and validation episode banks used by the earlier experiments.
- Frozen **raw 1,024-dimensional BGE-M3** item vectors. A cached reordered matrix aligns these vectors with the prepared task item indices. No PCA is applied to these ANPs.
- Original `AttentiveNeuralProcess`: width 64, one context self-attention block, four attention heads, latent dimension 32, dropout 0. Fresh initialization with seed 20261018 in every arm.
- 3,000 optimizer updates. One uniformly selected subject–benchmark task from **each of the 11 training benchmarks per update**. All arms use the same NumPy episode stream.
- Disjoint context and target items. Context size uniform from 0 to min(64,n-1); up to 128 targets per episode.
- Original latent ANP negative-ELBO-per-target objective. AdamW, learning rate 3e-4, weight decay .01, gradient clipping 5. No learning-rate scheduler.
- CPU jobs: 2 CPUs / 8 GiB each. No GPU jobs are requested. Detached orchestration survives local network disconnects.

## The four arms

1. **balanced_erm**: arithmetic mean of the 11 benchmark episode losses.
2. **groupdro**: multiplicative benchmark weights using detached exponentially smoothed group losses. EMA .9, weight-update learning rate .01. The q weights sum to one and are checkpointed. Groups are training benchmarks only; validation labels do not update q. The shared AdamW regularization and validation-selected checkpoints remain in effect. This minimizes weighted training ELBO, not directly worst-domain Brier ALC.
3. **swa**: the balanced ERM online trajectory, with uniform parameter averaging from update 600 every 10 updates. No BatchNorm refresh is needed: the ANP uses LayerNorm. Validation uses averaged parameters once averaging starts. Averaged checkpoints become eligible after at least five samples; the first scheduled eligible validation is update 700. Earlier diagnostic points use the online network and cannot become the selected SWA checkpoint. This is SWA, not a claim to implement the full SWAD algorithm.
4. **sam**: balanced objective with a two-pass SAM perturbation of radius .05 and AdamW as the base optimizer. The exact same episode tensors and latent random samples are reused in both passes. Perturbations are restored before the optimizer update and in a `finally` block on second-pass failure.

## Evaluation and Forge

W&B project: `aims-predictive-evaluation` in `arshiaswe-kth-royal-institute-of-technology`, hosted on Forge CoreWeave. The existing `wandb` secret injects `WANDB_API_KEY` into each training job.

Checkpoints are selected by the same fixed 64-task validation bank as the earlier models: 50/50 acquisition/evaluation pools, one random acquisition trajectory, budgets 0/1/3/7/15/31, fixed targets, population-weighted subject–benchmark Brier ALC (.1/.2/.2/.2/.2/.1). The latent prior mean is used for deterministic prediction. Full validation is scored after restoring the selected checkpoint. **The previously inspected test benchmarks are not evaluated in these development runs.**

All requested diagnostic keys are logged every 100 optimizer updates:

```text
diagnostic/heldout_subjects/same_subject_split_half/brier_score
diagnostic/training_subjects/same_subject_split_half/brier_score
diagnostic/training_subjects/same_subject_response_holdout/roc_auc
diagnostic/training_subjects/same_subject_split_half/roc_auc
```

Here `heldout_subjects` means validation-benchmark subjects. Training diagnostic tasks are sampled once (32 tasks stratified across the 11 training benchmarks); validation diagnostic tasks use the fixed validation bank. Diagnostic contexts are capped at **64**, with at most **512** targets to bound self-attention cost. This differs from the statistical baselines' uncapped diagnostic contexts. Compare the four ANP arms' diagnostic curves with each other; the main six-budget ALC protocol remains shared across all model families. Diagnostic contexts can exceed competition label budgets and are not the competition score.

Forge also records per-benchmark episode counts and GroupDRO weights, SWA averaging counts, training objective and SAM perturbed objective. The x-axis is optimizer step, not the training-fraction axis used by the closed-form statistical comparisons.

## Checkpoints

Directory: `/data/results/anp-ood/anp-ood-bge1024-20261029-v1/<variant>/`.

- `last.pt`: atomic resume checkpoint at initialization and every validation/update-100 boundary. Stores online weights, AdamW moments, RNG states (Python, NumPy generator/global, Torch), next update, histories, episode banks, task/target draw counts, GroupDRO q/EMA, SWA state/count, W&B identity and configuration/data/code fingerprints.
- `best.pt`: validation-selected inference weights plus resume metadata. For SWA, `state_dict` contains the **averaged** weights and `online_state_dict` retains the optimizer trajectory separately.
- `curves.json`, `best-validation.json`, `final-validation.json`, `completed.json`.

Re-running an unfinished command resumes the last saved update. Completed or incompatible configurations are rejected/reused as immutable. Pickles are trusted artifacts created by this trainer. No private credentials are written into checkpoints.

```powershell
# Includes all four smoke/resume checks, then starts all four full runs.
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_anp_ood.py
```

Five local tests passed: balanced/disjoint/reproducible sampling, GroupDRO normalization and hard-group weighting, exact averaging, SAM noise reuse and restoration on failure, and finite evaluation/diagnostic probabilities. All four real-data stop/resume checks passed, and all four full jobs completed their first optimizer update.

Modal app: https://modal.com/apps/arshiaswe/main/ap-KjBFCrNS2dn4LUDG7n9oE5

Forge runs:

- Balanced ERM: https://forge.coreweave.com/wandb/arshiaswe-kth-royal-institute-of-technology/aims-predictive-evaluation/runs/ood-f0187bb10c9a9b1c16
- GroupDRO: https://forge.coreweave.com/wandb/arshiaswe-kth-royal-institute-of-technology/aims-predictive-evaluation/runs/ood-3bb2fd1afc8bb43d6f
- SWA: https://forge.coreweave.com/wandb/arshiaswe-kth-royal-institute-of-technology/aims-predictive-evaluation/runs/ood-0ac1b3e05ec2229179
- SAM: https://forge.coreweave.com/wandb/arshiaswe-kth-royal-institute-of-technology/aims-predictive-evaluation/runs/ood-44b11b330043f36e55
