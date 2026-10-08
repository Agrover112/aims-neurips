# Troubleshooting

## A submission ZIP is rejected

Check that `model.py` is at the ZIP root, not inside a `submission/` directory. Build with `python scripts/build_submission.py`, then run both the static checker and smoke test. Do not add undeclared Hugging Face downloads or dependencies that require compilation at evaluation time.

## Predictions fail validation

`predict()` must return one finite probability in `[0, 1]`. Check for `NaN`, infinity, arrays, and accidental logits. Run `pytest` and the official smoke test before packaging.

## Local imports work but packaged imports fail

Submission code cannot rely on the repository's `src/` directory. Every runtime dependency must be in `submission/`, installed through an allowed wheel, or part of the standard library. The packaging script tests the copied directory in an isolated temporary location.

## Adaptation does not improve Brier ALC

Inspect every budget, not only budget 31. Early budgets carry most of the curve's total weight. Verify that evidence matching uses the full subject identity and stable anonymous `benchmark_id`. Compare uncertainty acquisition with random acquisition because uncertain items can be unrepresentative.

## Results vary between runs

Record the configuration, seed, source revision, package versions, and data snapshot. Random splits should be grouped by subject-item pair. API predictors can remain nondeterministic even with a fixed seed, so preserve raw predictions and report repeated runs.

## WandB does not start

Install the experiment extra, authenticate locally, and set `wandb_mode = "online"` in a personal config. Keep the default disabled mode in committed configs so tests and collaborators do not make network calls.

## Modal cannot access WandB

Create a Modal secret named `aims-wandb` containing `WANDB_API_KEY`. Never place the key in TOML, source code, shell history, or a committed `.env` file.

