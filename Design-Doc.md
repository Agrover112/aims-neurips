# AIMS Starter Kit — Design Document

**Purpose**: This document explains the codebase structure, core components, and how they fit together for the AIMS Predictive AI Evaluation Challenge (PAIEC). It is written for software engineers to share with engineering managers and technical stakeholders.

---

## 1. Overview

The repository is a collaboration-ready research workspace for the **AIMS Predictive AI Evaluation Challenge (PAIEC)**. It preserves the official submission interface, provides an auditable adaptive baseline, reproduces the published Brier ALC calculation, and offers hooks for repeatable local, Modal, and Weights & Biases experiments.

**Primary offline evaluation** uses benchmark-held-out episodic evaluation with six cumulative label budgets: `0, 1, 3, 7, 15, 31`. The aggregated metric is **Brier ALC** (weighted combination of per-budget Brier scores).

---

## 2. Directory Structure

```
src/aims_starter/       — Core Python package (metrics, data, models, evaluation)
scripts/                — Packaging, scoring, and plotting commands
configs/                — Committed experiment configurations (TOML)
docs/                   — Assumptions, troubleshooting, concept dictionary
reference/              — Official starting kit and public baselines
tests/                  — Unit and interface tests
```

---

## 3. Core Components

### 3.1 Data Pipeline (`data.py`)

**Responsibility**: Preprocess raw Measurement Data Bank tables into canonical Parquet format.

- Discovers benchmark directories containing `benchmarks.{parquet,csv}`, `subjects.{parquet,csv}`, `items.{parquet,csv}`, `responses.{parquet,csv}`
- Renames columns to canonical names (`label` → `response`)
- Generates stable subject signatures (`subject_signature`) using hash of normalized fields
- Creates deterministic group splits: 80% train, 10% validation, 10% test (via SHA-256 group hashing)
- Constructs `input_id` = `task_id::item_uid::condition_uid` and `observation_id` = hash of input + trial
- Persists episode index (`episode_items.parquet`, `episode_tasks.parquet`) with acquisition/evaluation pool assignments
- Outputs `manifest.json` with format version, seed, eligible benchmarks, split policy, and row counts

**Key functions**:
- `prepare_measurement_data()` — Main entry point; converts raw tables → Parquet
- `build_episode_index()` — Creates deterministic 50/50 acquisition/evaluation split per task
- `iter_episodes()` — Yields nested episodes for given budgets

### 3.2 Evaluation & Metrics (`evaluation.py`, `metrics.py`)

**Metrics** (`metrics.py`):

- `brier_score()` — Mean squared probability error for binary outcomes
- `calibration_ece()` — Count-weighted Expected Calibration Error using 10 fixed equal-width bins
- `brier_alc()` — Combines per-budget Brier scores using official normalized log-area weights: `(0.1, 0.2, 0.2, 0.2, 0.2, 0.1)`
- `DEFAULT_BUDGETS = (0, 1, 3, 7, 15, 31)`

**Evaluation simulation** (`evaluation.py`):

- `simulate_beta_binomial()` — Benchmark-held-out episodic evaluation with a smoothed mean baseline
- Trains on outer-training benchmarks, evaluates on held-out benchmarks
- Reports per-budget Brier scores, ECE, and Brier ALC
- Supports nested validation with different benchmark splits (train/validation/test)

### 3.3 Neural Models (`neural_pilot.py`, `faithful_np.py`)

**Online-AIRT** (`neural_pilot.py`):

- `_OnlineAIRT` — 2PL item response theory model
  - Item network: `Linear → Tanh → Linear(2)` predicts discrimination & difficulty from item content
  - Subject abilities (`theta`) updated online via MAP adaptation
  - Trained offline on all but one subject-task; ability updated as labels arrive

**Neural Process variants** (`faithful_np.py`):

- `FaithfulNPAIRT` — NP/ANP latent model with IRT-anchored decoder
  - Variants: `np` (vanilla), `anp` (attentive), `stacked_mhsa_anp` (2-layer MHSA)
  - Item projection → item parameters (discrimination, difficulty) via 2PL
  - Latent variable inference: `q(z | C ∪ T)` with KL-regularized posterior
  - Decoder: anchor term (`discrimination × (θ - difficulty)`) + residual network
  - Supports Monte Carlo prediction with variance estimation

**Amortized 2PL** (`faithful_np.py`):

- `Amortized2PL` — Learns ability mapping from item features offline
  - Task abilities are amortized parameters; updated online via few-gradient steps

**TfNP-D-AIRT** (`faithful_np.py`):

- `TNPDAIRT` (TfNP-D-AIRT) — Transformer Neural Process with exact Bernoulli head and IRT-anchored logit
  - Uses transformer encoder with causal masking over context + target tokens
  - IRT anchor: `discrimination × (θ - difficulty)` added to residual head

**Training & scoring** scripts in both files train models for `steps=800` (amortized) or `steps=800-900` (NP variants) with Adam optimizer and cosine LR scheduling.

### 3.4 Smoke Tests (`smoke.py`)

**`run_benchmark_smoke()`** — Validates benchmark episode integrity before model training:

- Ensures exactly one benchmark in input
- Checks balanced acquisition/evaluation pools (consecutive ranks, ≥80 items)
- Verifies nested context budgets are properly nested
- Confirms no acquisition/evaluation overlap
- Validates item/subsject feature joins are complete
- Runs 200-step optimization of tiny 2PL model; checks loss reduction
- Returns check dictionary summarizing validation pass/fail status

### 3.5 Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `build_submission.py` | Package model.py + optional labeling.py into submission ZIP |
| `run_experiment.py` | Run experiment with predictions.json + config |
| `plot_results.py` | Plot Brier/ALC results from metrics.json |
| `plot_neural_pilot.py` | Plot neural pilot curve results |
| `plot_paper_figure_gallery.py` | Generate paper figure gallery |
| `plot_paper_preview.py` | Generate paper preview figures |
| `plot_irt_family_curves.py` | Plot IRT family learning curves |
| `download_measurement_tables.py` | Download Measurement Data Bank tables |
| `prepare_measurement_data.py` | Prepare measurement data parquets |
| `log_neural_pilot_wandb.py` | Log neural pilot results to WandB |

### 3.6 Reference Baselines (`reference/paiec_baseline/`)

Organizer-provided reference models:

- `empirical_mean` — Mean acquired outcome per subject-benchmark pair
- `ble` — Bayesian Linguistic Evaluator (LLM evidence retrieval + belief revision)
- `empirical_mean_ble_acquisition` — Empirical mean with BLE choosing acquisition items

Additional ablations listed in `docs/raa_irt_design.md` table span from simple beta-binomial through AmortizedIRT + AnchorCalibrator + Fisher Acquisition.

### 3.7 Configuration (`configs/`)

- `default.toml` — Default experiment configuration
- `wandb-online.example.toml` — WandB online experiment configuration

### 3.8 Documentation (`docs/`)

- `concept-dictionary.md` — Competition terminology and flow
- `assumptions.md` — Assumptions and limitations
- `data-preparation.md` — Data preparation pipeline documentation
- `np-airt-implementation.md` — Implementation skeleton for NP-AIRT
- `raa_irt_design.md` — RAA-IRT design note with feature/calibration matrix
- `collaboration-setup.md` — GitHub/Modal/WandB access setup
- `troubleshooting.md` — Common issues and resolutions

---

## 4. Evaluation Protocol (Summary)

The **primary competition evaluation** follows these steps:

1. **Group splits**: Outer split by complete benchmark (never random response rows). Fixed held-out fold acceptable for development.

2. **Outer training**: Fit model, preprocessing, item representations, calibrators using only outer-training benchmarks. Use separate held-out benchmarks for hyperparameter selection.

3. **Label acquisition**: For each eligible subject–benchmark task in the test fold, divide distinct `input_id` values into label-acquisition pool and disjoint evaluation pool. Repeated inputs stay together.

4. **Budget reveal**: At official cumulative budgets `0, 1, 3, 7, 15, 31`, reveal outcomes. Use same nested acquisition prefix at every budget (budget 3 contains labels from budgets 1 and 3).

5. **Prediction**: Predict still-hidden outcomes in the untouched evaluation pool. Average rows within each subject–benchmark pair, then average pairs equally to match leaderboard aggregation.

6. **Metrics**: Report Brier score at every budget. Calculate Brier ALC using weights `(0.1, 0.2, 0.2, 0.2, 0.2, 0.1)`.

7. **Reporting**: Also report calibration plots/ECE, benchmark-level failures, runtime, and paired uncertainty intervals clustered by benchmark.

---

## 5. Model Ladder (from raa_irt_design.md)

The proposed model ladder (oldest → newest), each building on the previous:

| Model | New component |
|---|---|
| PAIEC Empirical Mean | Organizer baseline |
| PAIEC BLE | Organizer baseline |
| Beta-Binomial | Simple adaptive control |
| AmortizedIRT | Item-content embeddings predict IRT parameters |
| AmortizedIRT + Item Metadata | Add `item_features` and `interactors` |
| AmortizedIRT + Cognitive Scales | LLM-labelled reasoning and skill features |
| AmortizedIRT + Subject Metadata | Initialize ability from subject metadata |
| AmortizedIRT + Online Ability Update | Update subject-benchmark ability from acquired outcomes |
| AmortizedIRT + Multidimensional Ability | Replace scalar ability with cognitive skill vector |
| AmortizedIRT + AnchorCalibrator | Recalibrate logits using acquired anchors |
| AmortizedIRT + AnchorCalibrator + Fisher Acquisition | Select labels using expected Fisher information |
| NCF | Raw subject-item neural predictor |
| NCF + Platt Recalibration | Online scaler |
| NCF + AnchorCalibrator | Grouped shrinkage calibration |

*Note: Values in illustrative results table are fabricated; show intended reporting format.*

---

## 6. How to Read This Codebase

**For engineers**: Start with `src/aims_starter/data.py` to understand data flow, then `metrics.py` for evaluation metrics. Model code is in `neural_pilot.py` (Online-AIRT + NP variants) and `faithful_np.py` (paper-faithful NP-AIRT/ANP-AIRT/TfNP-D-AIRT with IRT decoder). Smoke tests in `smoke.py` validate episode integrity. Reference baselines in `reference/paiec_baseline/` for organizer-provided comparisons.

**For managers**: The codebase implements the AIMS PAIEC challenge submission pipeline. Key deliverables are:
- Development code + submission ZIP
- Offline evaluation results (Brier ALC at budgets 0/1/3/7/15/31)
- Calibration analysis (ECE plots)
- Reproducible experiment configuration
- Documentation of all training sources and LLM assistant use

**Typical workflow**:
1. `python scripts/prepare_measurement_data.py` → canonical Parquet files
2. `python -m unittest discover -s tests` → run interface tests
3. `python scripts/run_experiment.py predictions.json` → evaluate model
4. `python scripts/plot_results.py results/latest/metrics.json` → visualize results

---

## 7. Key Design Decisions

- **Deterministic episode assignments**: SHA-256 hash-based grouping ensures reproducibility; same split across all models for fair comparison.
- **Nested budget protocol**: Labels accumulate (`budget 3` contains labels from budgets 1 and 3), matching the official competition format.
- **Subject–benchmark weighting**: Results averaged equally across subject–benchmark pairs, not by number of responses (matches leaderboard aggregation).
- **Brier ALC weights**: `(0.1, 0.2, 0.2, 0.2, 0.2, 0.1)` emphasizes intermediate budgets (3, 7, 15) where most label acquisition happens.
- **Stateless predictors**: Codabench workers may restart between budgets; predictors must rebuild state from complete supplied `labeled` list on every call.
- **Grouped splits**: Repeated subject–item outcomes never cross train/val/test boundaries.

---

## 8. Output Artifacts

Per experiment run, the following files are generated:

- `results/latest/metrics.json` — Per-budget Brier scores, ECE, Brier ALC, per-task details
- `results/latest/plots/` — Visualizations of learning curves, calibration plots
- `dist/submission.zip` — Packaged submission (model.py + labeling.py)
- `checkpoints/` — Trained model state dicts (AIRT, NP, ANP, stacked-MHSA)
- `manifest.json` — Auditable protocol metadata (seed, eligible benchmarks, split policy)
- `logs/` — Training history, loss diagnostics

---

## 9. Getting Started (for new contributors)

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,experiment]'

# 2. Verify tests pass
python -m unittest discover -s tests

# 3. Prepare data (requires Measurement Data Bank tables)
python scripts/prepare_measurement_data.py /path/to/raw/data /path/to/output

# 4. Run a quick neural pilot (one benchmark, fast)
python -c "
from aims_starter import run_neural_pilot
# ... requires prepared Parquet data
"

# 5. Full experiment
python scripts/run_experiment.py examples/predictions.json --config configs/default.toml
python scripts/plot_results.py results/latest/metrics.json
```

---
*Document generated from codebase analysis. For the latest state, consult the repository's `docs/` directory and `CONTRIBUTING.md`.*