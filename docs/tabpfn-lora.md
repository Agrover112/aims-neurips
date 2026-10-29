# Episodic TabPFN-v2 LoRA

This trainer fine-tunes one pretrained TabPFN-v2 classifier across subject–benchmark tasks. It never pools labels from different subjects into a single TabPFN context. It uses the existing `benchmark_holdout_filters.py` unchanged, from repository commit `7b2be08`, and asserts the requested 11/3/3 train/validation/test benchmark counts. That canonical filter removes constant-outcome items; conclusions apply to that filtered population.

The ANP app `ap-CQDNejLoEE6t8UlZlGAwr3` was confirmed stopped before this work. This implementation is in an isolated worktree on branch `codex/tabpfn-lora-benchmark`; earlier ANP edits are preserved in the original checkout.

## Features and model

- Existing frozen `bge_m3_long` item embeddings, already copied into `arshiaswe:aims-neurips-data`.
- PCA to 64 features, fitted only on up to 50,000 unique training items with a fixed seed. Projection, fitted item IDs and explained variance are saved. No validation/test items enter PCA fitting.
- Package `tabpfn==9.2.0`, explicitly using architecture v2 and `Prior-Labs/TabPFN-v2-clf/tabpfn-v2-classifier-v2_default.ckpt`. The resolved model revision and file hash are saved.
- Custom PyTorch weight parametrizations implement LoRA on attention Q/K/V/output linear projections; rank 8, alpha 16, zero-initialized B, frozen base weights. This is not full-weight fine-tuning or a call to the single-dataset sklearn fine-tuning wrapper.
- AdamW, learning rate 1e-4, 50-step warmup, weight decay .01, gradient clip 1, four episodes per optimizer step, 32 disjoint query items per training episode. Context budgets sampled from 3/7/15/31; contexts with only one observed class are skipped for gradient updates. Frozen base weights are checked in unit tests.
- Zero-label and single-class contexts use an explicit Beta(2,2) fallback during both base and adapted evaluation; in particular budget 0 predicts .5 and budget 1 does not invoke TabPFN. No fake labeled rows are added.

## Selection and evaluation

Each evaluation episode has fixed 50/50 acquisition/evaluation pools, one random acquisition trajectory and nested evidence at budgets 0/1/3/7/15/31. All evaluation items are scored in chunks of 256. Brier ALC uses weights .1/.2/.2/.2/.2/.1.

Checkpoint selection uses a fixed 64-task validation bank stratified across the three validation benchmarks. Each sampled task is weighted by benchmark task count / sampled task count, estimating equal subject–benchmark task weighting. The unadapted base model is a candidate best checkpoint, so fine-tuning cannot force selection of a worse validation checkpoint. After restoring the selected adapters, full validation and test results are saved once. Full result files include per-task/per-budget scores, benchmark-macro ALC and a Beta-Binomial baseline.

The canonical loader aggregates repeated observed outcomes and thresholds them to binary labels. This is an offline replay on those canonical cells, not execution by the competition runner.

## Checkpoints and resume

Outputs: `/data/results/tabpfn-lora/<run-label>/`.

- `last.pt`: atomic full resume checkpoint every 100 optimizer steps and on the final step. Includes adapters, AdamW moments, warmup scheduler, next optimizer step, epoch order, Python/NumPy/Torch/CUDA RNG states, validation bank, history, configuration fingerprint, preprocessing, split audit and W&B identity. Float32 training has no AMP scaler (`grad_scaler=None`).
- `best.pt`: selected adapter checkpoint with the same metadata; frozen base weights are referenced by pinned revision and SHA-256 instead of being duplicated into every checkpoint.
- `baseline-validation.json`, `best-validation.json` (if adaptation improves), `final-validation.json`, `final-test.json`, and `completed.json`.

The same command resumes an unfinished run. Incompatible data/configuration/core code fingerprints are rejected. Completed runs are immutable; use a fresh label for a new experiment. Files written by this trainer are trusted local artifacts; resume uses `torch.load(..., weights_only=False)` to restore NumPy/Python RNG objects.

## Commands

From this checkout in PowerShell:

```powershell
# CPU preparation + two L4 optimizer steps, intentionally stopping and resuming after step 1.
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_tabpfn_lora.py --smoke

# Full 3,000-step run. W&B uses the existing wandb / WANDB_API_KEY secret.
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_tabpfn_lora.py
```

Preparation runs on 8 CPUs / 32 GiB; training requests one L4 with A10G as a capacity fallback, plus 4 host CPUs / 32 GiB. Preparation is cached and checksummed. The GPU smoke test does not read the test split. W&B project: `aims-tabpfn-lora` in the authenticated account's default entity.

A small CPU workflow function coordinates preparation and training inside Modal, so a local DNS disconnect cannot interrupt the pipeline. Always use `--detach`. The upstream model loader can cache built models; the trainer deep-copies the frozen loaded base before registering LoRA, keeping later calls and resumes independent.

The real preparation produced 6,481 tasks and 123,035 unique items with the canonical 11/3/3 split. PCA was fitted on 50,000 unique training items. All preparation artifacts and the public base checkpoint are cached in `arshiaswe`.

## Verified execution

Five local tests passed, including frozen-weight/adapter behavior, atomic checkpoint and RNG restoration, fallback behavior, nested disjoint pools, and the canonical 11/3/3 split. The real GPU smoke test ran on an NVIDIA A10G with 96 adapter targets and 294,912 trainable parameters. It saved after optimizer update 1, restored the full state in a subsequent call, and completed update 2. It did not read the test benchmarks. Its tiny validation bank is for software verification, not a performance claim.

- Smoke app: https://modal.com/apps/arshiaswe/main/ap-uiFXBQ3y1g8km958ekPsG3
- Smoke W&B: https://forge.coreweave.com/wandb/arshiaswe-kth-royal-institute-of-technology/aims-tabpfn-lora/runs/lora-f37f03fd5bf956a46a
- Full 3,000-update app: https://modal.com/apps/arshiaswe/main/ap-qeat53IvgLiLCQScx1YNnc
- Full W&B: https://forge.coreweave.com/wandb/arshiaswe-kth-royal-institute-of-technology/aims-tabpfn-lora/runs/lora-94bef21cf13f3bf0ac

The full run started on an NVIDIA A10, completed its first optimizer update, and reached update 100. Fixed-bank validation ALC changed from the base value 0.191205 to 0.188016. These are initial validation results, not held-out-test results or evidence of statistical significance. Test evaluation remains scheduled after final checkpoint selection.

Final update: all 3,000 optimizer updates completed. The selected checkpoint was update 900 (fixed-bank validation ALC 0.187652462). Final test ALC was 0.236409618 on 78 tasks / 3 held-out benchmarks. The app is stopped. The statistical-baseline comparison verified the same test cohort and the same fixed Beta(2,2) reference episodes.

References: [official TabPFN source](https://github.com/PriorLabs/TabPFN), [base model](https://huggingface.co/Prior-Labs/TabPFN-v2-clf).
