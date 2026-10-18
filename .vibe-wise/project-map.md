# Project Map

## Purpose
Support reproducible research and competition submissions for the AIMS Predictive AI Evaluation Challenge, with GitHub collaboration between the repository owner and a friend.

## Requirements
The detailed scientific and engineering requirements are recorded in `BEGIN.md`. The submission must follow the official root-level predictor interface, reproduce the published Brier ALC calculation, support test-time adaptation, and keep experiments reproducible. Collaboration must not expose competition data or service credentials in Git.

## Components
- `submission/`: self-contained competition code; `model.py` calls a beta-binomial predictor and `labeling.py` implements adaptive acquisition.
- `src/aims_starter/`: offline Brier, calibration, ALC, and result I/O utilities.
- `scripts/`: local evaluation, plotting, and ZIP packaging entry points.
- `configs/`: shared TOML experiment and tracking configuration.
- `modal_app.py`: remote Modal experiment entry point.
- `tests/` and `.github/workflows/ci.yml`: local and GitHub validation.
- `reference/`, `literature/`, and `paper/`: supplied competition code, research leads, and report template.

## Main Flow
`[subject, item] + revealed labels` -> `submission/model.py` -> beta-binomial probability -> official evaluator

`truth + per-budget probabilities` -> `scripts/run_experiment.py` -> Brier/ECE by budget -> Brier ALC -> local JSON and optional WandB run

`experiment inputs + TOML config` -> `modal_app.py` -> Modal remote function -> Brier ALC result

`submission/` -> packaging script -> root-level submission ZIP -> official static checker and smoke test

## Data and Trust Boundaries
- GitHub stores reviewed source, configuration examples, documentation, and CI history. Generated outputs and credentials are ignored.
- Modal executes remote jobs and charges the active `agrover112` workspace. That workspace is an older personal workspace and does not accept members; shared-token access remains a security and billing decision.
- WandB stores metrics and artifacts in the `agrgnn-kth-royal-institute-of-technology/aims-predictive-evaluation` project.
- Hidden evaluation labels and private competition feedback must not enter the repository.

## Build and Deployment
- Install: `uv venv .venv` then `uv pip install --python .venv/bin/python -e '.[dev,experiment,modal]'`.
- Test: `.venv/bin/python -m pytest`.
- Package and validate: `python scripts/build_submission.py`, then the two official checks documented in `README.md`.
- Remote compute: `modal run modal_app.py --predictions examples/predictions.json`.
- CI: pushes and pull requests to `main` run tests and official submission checks.

## Unknowns
Programming experience, stack familiarity, learning goal, and workflow preferences. The learner is new to this codebase and wants depth only in components being changed. The long-term shared Modal access method and safeguards remain unresolved.
