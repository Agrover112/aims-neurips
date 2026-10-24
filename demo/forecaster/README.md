# AI Evaluation Forecaster

A real-checkpoint demo for budgeted AI evaluation. Pick a held-out
subject–benchmark task, buy a few labels, and watch three forecasters update on
the same 60 evaluation items:

- **Beta–Binomial** — Laplace-smoothed base rate; the transparent baseline.
- **Online-AIRT** — 2PL item network with a MAP ability update.
- **CNP-AIRT** — neural process with an IRT head and prefit full-BCE calibration.

This is a custom FastAPI + vanilla web UI, not Gradio. The app loads frozen
weights and runs inference only. It never trains or fine-tunes.

![Forecaster benchmark view](screenshot.png)

## Run

Use the repository's Python environment (torch and transformers are already
installed there), or install the demo requirements:

```bash
cd /home/agrov/aims-neurips
uv pip install --python .venv/bin/python -r demo/forecaster/requirements.txt
.venv/bin/python -m uvicorn server:app --app-dir demo/forecaster \
  --host 127.0.0.1 --port 8787
```

Open <http://127.0.0.1:8787>.

The first model load takes roughly 15 seconds on CPU because MiniLM and the two
checkpoints are loaded once at startup. Subsequent forecasts take tens of
milliseconds.

## What is in the box

```
demo/forecaster/
  server.py                 FastAPI routes + in-memory run sessions
  engine.py                 frozen MiniLM + CNP-AIRT + Online-AIRT inference
  assets/
    tasks.json              5 validation tasks: 31 acquisition, 60 evaluation
    cnp_airt.pt             selected CNP-AIRT checkpoint
    online_airt_2pl.pt      selected Online-AIRT checkpoint
    metadata_cnp.json       calibration + checkpoint metadata
    metadata_airt.json      calibration + checkpoint metadata
  static/                   the UI (HTML, CSS, vanilla JS)
  screenshot*.png
```

The data are validation tasks from the benchmark-held-out protocol. The five
tasks cover AfriMed-QA, AppWorld, CMMLU, CRUXEval, and DevBench. The acquisition
pool has 31 real labels; the evaluation set has 60 real labels that the UI does
not reveal until **Score held-out run** is clicked. The demo uses one frozen
acquisition order per task; the 10-order grid is evaluated offline. The
diagnostic Brier calibrator is not used. The forecast curve at every
budget is computed from the actual checkpoints on those 60 items, not from a
synthetic curve.

## UI behavior

- **Task cards** switch between the five subject–benchmark pairs.
- **Budget chips** (0, 1, 3, 7, 15, 31) reveal labels up to the selected
  acquisition budget. **Reveal outcome** buys one label at a time.
- **Policy selector** switches between uncertainty, hardest, and random
  acquisition. The next item is chosen by the loaded CNP-AIRT forecast.
- **Forecast chart** shows the mean predicted probability over the fixed
  60-item evaluation set at every revealed-label prefix.
- **Score held-out run** reveals the actual evaluation labels and reports
  success rate plus Brier scores for all three forecasters.
- **Custom probe** accepts free text and returns a label-free prediction with a
  persistent out-of-distribution warning. The released checkpoint masks subject
  features, so custom predictions are driven by item content and the prior.

## Scope and limits

- Validation-only evidence. The hidden leaderboard is never used.
- No benchmark-ID embedding; benchmark is treated as a level/random effect in
  the paper, not as an input feature.
- Subject features are masked in both demo checkpoints. This is deliberate and
  matches the selected validation arm.
- The custom probe is not a validated evaluation. The support number is a
  nearest-neighbour cosine check against the 455 demo items, not a calibrated
  OOD detector.
- The demo stores sessions in memory; restarting the server clears runs.

## Regenerating the task export

The selected tasks and labels come from the Modal volume. The exporter is at
`scripts/export_forecaster_tasks.py`; it writes
`/data/results/demo/tasks.json`, which can be downloaded with:

```bash
.venv/bin/modal volume get aims-neurips-data /results/demo/tasks.json \
  demo/forecaster/assets/tasks.json
```

The asset file contains evaluation labels so the local scorer can work. Keep it
off static hosting; `server.py` only exposes item text and labels after an
explicit reveal or score request.
