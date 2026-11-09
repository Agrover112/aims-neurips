# Checkpoint setup and first replay

This setup keeps the checked-in historical demo separate from the newer
filtered-data experiments. The two `.pt` files in `demo/forecaster/assets/`
are CNP-AIRT and Online-AIRT weights for the older MiniLM-based demo. They
are sufficient to verify real checkpoint loading and episodic inference.
They are not the current filtered CNP/ANP or amortized-IRT checkpoints.

## Install on CPU

From the repository root, with Python 3.11 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev,experiment,modal]' -r demo/forecaster/requirements.txt 'transformers==4.57.6' 'scikit-learn>=1.5,<2'
```

In Windows PowerShell, replace `.venv/bin/python` with
`.\.venv\Scripts\python.exe`. The same installation commands work.

## Replay the committed checkpoints

```bash
.venv/bin/python scripts/run_checkpoint_baseline.py
```

Windows:

```powershell
& .\.venv\Scripts\python.exe scripts/run_checkpoint_baseline.py
```

The first run downloads MiniLM. Later runs can pass `--offline` to verify
cached-encoder inference. Use a fresh `--output` path for each run; previous
results are never overwritten. The replay uses the exact frozen acquisition
order exported in `tasks.json`, all six official budgets, the same 60
evaluation items per task, the shared Brier/ALC helpers, and equal task weights.
It saves asset hashes, source revision, actual encoder revision, per-task
predictions, and per-budget scores. It does not train, calibrate, select a
model, rank the candidate pool, or read official test outcomes. Five already
explored demo tasks are an installation check, not an independent performance
claim or the full 104-task validation result.

## Newer checkpoints require the correct Modal workspace

The Git checkout contains only the two historical demo checkpoints. Current
filtered artifacts live on a volume named `aims-neurips-data`, with separate
copies in `agrover112` and `aims-neurips`. They are different volumes.

The current filtered CNP trainer saves best-validation checkpoints under:

```text
/results/cnp/checkpoints/<architecture>-<run-label>-<arm>-seed<seed>.pt
```

Filtered amortized-IRT weights are saved under:

```text
/results/data-analysis/amortized-irt-<arm>-weights-seed<seed>.pt
```

Use your own authenticated Modal account and inspect both relevant profiles:

```bash
.venv/bin/modal profile list
.venv/bin/modal volume ls aims-neurips-data /results/cnp/checkpoints --profile agrover112
.venv/bin/modal volume ls aims-neurips-data /results/data-analysis --profile aims-neurips
```

Select the intended completed run explicitly. Never choose a model from test
scores or silently fall back to a demo checkpoint. Follow the root README's
current trainer commands when starting a new run. Do not copy a teammate's
credentials into this repository or a submission.

## Streaming acquisition work

A competition policy gets the current subject/item, acquired labels and
budget context, then returns accept/skip. The demo's `next_item()` instead
ranks its whole remaining pool; it is not the competition streaming policy.
Use the supplied starting kit as the interface reference. Before comparing
Bayesian information-value policies, freeze a compatible predictor and replay
identical one-at-a-time candidate streams on disjoint acquisition/evaluation
items. Full-history matrix-completion scores are not budget-zero or ALC scores.

## Test import repairs

At source commit `2af8c8aa166254422967d0e6eff4561037b79e5b`,
`tests/test_mirt_acquisition.py` imports `modal_mirt_acquisition`, which was
deleted in commit `4646a91f913e5a2e13dc05aaad7807c966ddc110`.
That prevented full pytest collection. The two numerical helpers are now
retained unchanged in `src/aims_starter/mirt_acquisition.py`, and the tests
import them from there. The retired Modal experiment is not restored.
The Bayesian-MIRT test also imports its moved module from
`organized.training`, so an obsolete root import no longer silently skips
tests in an environment where Torch and Modal are installed. CI now installs
the declared development dependencies before running the suite; the former
base-only install failed on NumPy imports in a clean runner. Optional Torch
and Modal tests run locally when those dependencies are installed.
