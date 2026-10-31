# AIMS predictive evaluation starter kit

> **Read this before doing anything else — for any agent opening this repo (Claude Code, Codex, OpenCode, or a human).**
> Multiple agents work on this repository in parallel across separate terminals/sessions, on the same Modal volume (`aims-neurips-data`, workspace `agrover112`) and the same git remote. Before starting new work:
> 1. **Run `git log --oneline -10` and `git status` first.** Other agents commit directly; what you remember from an earlier session may already be stale or superseded.
> 2. **Check `docs/report/README.md`** for the standing results summary (what has and hasn't worked, with statistical caveats) before re-deriving a number that may already be measured.
> 3. **Check the "Data filtering changes everything" entry in the Decision log** (below) before training another model on the unfiltered response matrix — it is very likely the wrong baseline now.
> 4. **GPU jobs:** `modal_r_c_ablation.py` and friends default to `gpu="H100"` via `os.environ.get("R_C_GPU", "L4")`-style overrides in some files but hard H100 in others — check the decorator before launching, and prefer `--detach` plus one Modal app per job (detach only keeps the *last* spawned function alive otherwise).
> 5. Competition deadline: **2026-10-30, 23:59 AoE** for the final codebase + report (see `reference/starting_kit/README.md`). As of this writing no submission zip (`model.py` + `labeling.py`) has been built from the current best model.

> **⚠️ IMPORTANT (2026-10-26): FORGET ABOUT TASK TRACKING** — TaskCreate/TaskUpdate have no persistent state. All coordination happens via `git log`, `git status`, and this README. Current focus: **data filtering + offline IRT/MIRT training with weight-saving** (see sections below). Every result gets added to a table here.

This repository is a collaboration-ready research workspace for the AIMS Predictive AI Evaluation Challenge (PAIEC). It preserves the official submission interface, adds an auditable adaptive baseline, reproduces the published Brier ALC calculation, and provides hooks for repeatable local, Modal, and Weights & Biases experiments.

For the public training corpus, follow the reusable
[data-preparation pipeline](docs/data-preparation.md). It converts the gated
Measurement Data Bank tables into canonical subjects, items, observations,
aggregated inputs, task indices, and leakage-safe deterministic splits shared
by every model.

The corrected latent NP-AIRT model and episodic training pseudocode are saved
in [the implementation skeleton](docs/np-airt-implementation.md). All model
code and reporting use the shared [NP/ANP and IRT notation](docs/notation.md).

The current predictor is intentionally modest. It estimates a subject-benchmark success probability with beta-binomial shrinkage, using only labels revealed by the evaluator. Treat it as a control condition for stronger content-based, embedding, IRT, or Bayesian models.

## Repository map

```text
organized/submission/    competition submissions (current ANP builds; older ones in legacy/)
organized/               filtering-era experiments: training/, testing/, utils/
organized/utils/         plot scripts + plotstyle.py (see its README before plotting)
src/aims_starter/        offline metrics and experiment utilities
scripts/                 packaging, scoring, and plotting commands
configs/                 committed experiment configurations
tests/                   unit and interface tests
docs/                    assumptions and troubleshooting
reference/starting_kit/  unpacked official kit supplied with this project
reference/paiec_baseline/ unpacked public baseline supplied with this project
literature/              preliminary references, not design constraints
```

The source ZIP files remain local and are ignored by Git. Unpacked reference code is included for convenient review. Check its upstream terms before redistributing or modifying it outside a private collaboration repository.

## Current Neural Process code

There are two unrelated Neural Process branches in this repository. Do not
mix their checkpoints, data splits, or scores.

| Code | Status | Purpose |
|---|---|---|
| `organized/training/codex_es_fixed_modal_cnp_train.py` | **Current** | Filtered response-matrix CNP/NP trainer. Uses Qwen item embeddings, shuffled task epochs, disjoint context and target sets, fixed-bank validation, and CPU-only Modal jobs. |
| `organized/training/cnp_blocks.py` | **Current** | Reusable CNP/NP architecture used by the trainer above. It contains the set encoder, pooling, decoder, deterministic CNP path, and latent NP path. |
| `organized/training/modal_latent_cnp_train.py` | **Current latent-CNP entry point** | Matched pooled-latent baseline for ANP: linear context encoder, mean pooling, and global Gaussian latent. |
| `organized/training/modal_anp_crossattn_train.py` | **Current ANP entry point** | Target-to-context ANP launcher. It reuses the filtered trainer and replaces deterministic pooling with target-specific cross-attention. |
| `organized/training/anp_crossattn_blocks.py` | **Current ANP architecture** | Target-to-context cross-attention model. Context self-attention is optional; the latent path remains global and mean-pooled. |
| `organized/training/modal_aea_np_train.py` | **Current AEA-NP entry point** | Set-Transformer latent NP matching the neural-process architecture in the Active Evaluation Acquisition paper. It is a separate model from the ANP implementation. |
| `organized/training/aea_np_blocks.py` | **Current AEA-NP architecture** | Induced Set Transformer context encoder, learned attention pooling for the Gaussian latent, target-to-context cross-attention, and Set Transformer target decoder. |
| `organized/training/modal_cnp_train.py` | Superseded | Earlier version of the filtered trainer. It selects checkpoints from resampled validation episodes and does not save a volume checkpoint. Do not launch it. |
| `src/aims_starter/faithful_np.py` | Historical submission code | CNP-AIRT/ANP-AIRT code used by the earlier full-protocol submission experiments. It is not the current filtered standalone CNP. |
| `organized/training/modal_cnp_ladder.py`, `modal_cnp_disjoint.py`, `modal_np_collapse.py` | Historical ablations | Results-specific controls for the earlier CNP-AIRT/BeaverTails branch. Keep them for reproduction, not as a starting point for new filtered CNP runs. |

### Run the current filtered CNP

Use the `agrover112` Modal profile. The trainer requests 8 CPU cores and 32
GiB RAM. It does not request a GPU.

Run one named configuration directly. A unique `run_label` prevents W&B and
volume-output collisions across reruns.

```bash
.venv/bin/modal run --detach --profile agrover112 \
  --name cnp-width64-YYYYMMDD \
  organized/training/codex_es_fixed_modal_cnp_train.py::run \
  --seed 20261018 --width 64 --steps 101825 \
  --run-label width64-YYYYMMDD
```

The main architecture controls are:

- `width`: hidden and pooled representation size `d`.
- `n_blocks`: encoder and decoder depth.
- `encoder`: `linear`, `mhsa`, `set_transformer`, or `sgformer`.
- `pooling`: `mean` or `mhsa`.
- `use_latent`: switches the deterministic CNP to the latent NP ELBO path.
- `residual_proj`: enables learned residual projections around width changes.

The current deterministic width sweep uses `linear` encoder, `mean` pooling,
two blocks, Qwen3-8B embeddings truncated to 1,024 dimensions, seed
`20261018`, and 101,825 optimizer steps. Eight subject-benchmark functions
form one optimizer batch. That schedule gives 200 shuffled passes over the
4,073 training functions.

The trainer saves the best fixed-bank checkpoint whenever held-out unseen-target
Brier improves. Checkpoints contain the `state_dict`, full run configuration,
selection metric, Brier value, and optimizer step:

```text
/data/results/cnp/checkpoints/<architecture>-<run-label>-<arm>-seed<seed>.pt
```

It also writes an atomic resumable checkpoint every 500 steps at the same path
with a `.resume.pt` suffix. This contains model and AdamW states, NumPy and
Torch RNG states, the remaining shuffled-task queue, best-model state,
coverage counters, fixed validation bank, curves, and the W&B run ID. Reusing
the same run label resumes automatically after a preemption. A different
architecture, optimiser, split, or embedding arm is rejected; increasing the
step budget is allowed.

These paths are inside the Modal Volume of the workspace used to launch the
job. The path is identical across workspaces, but the stored weights are not:
`aims-neurips-data:/data/results/cnp/checkpoints/` in `aims-neurips` is
separate from the volume with that name in `agrover112`.

It also saves result rows, training curves, and exact target-coverage Parquet
files under `/data/results/cnp/`. W&B run names include architecture, width,
run label, embedding arm, and seed. Its paired panels show training versus
validation losses, diagnostic train versus held-out scores, and final test
scores. The true test set is evaluated once after checkpoint selection.

Run the local architecture tests before changing `cnp_blocks.py`:

```bash
.venv/bin/python -m unittest tests.test_cnp_blocks
```

### Run the target-to-context ANP

This is the ANP mechanism from the paper: every target item queries the labelled
context items. It is not attention pooling. The trainer reuses the CNP's
filtered split, fixed validation bank, coverage accounting, W&B panels, and
checkpointing.

```bash
.venv/bin/modal run --detach --profile agrover112 \
  --name anp-crossattn-YYYYMMDD \
  organized/training/modal_anp_crossattn_train.py::main \
  --seed 20261018 --width 64 --self-attention-blocks 2 \
  --cross-attention-heads 4 --use-latent --latent-dim 32 \
  --steps 3000 --run-label anp-crossattn-YYYYMMDD
```

It carries over `seed`, `arm`, `width`, `use_latent`, `latent_dim`,
`residual_proj`, `dropout`, `steps`, `wandb_mode`, and `run_label`. It drops
the pooled-CNP `encoder` and `pooling` flags. The ANP-specific controls are
`self_attention_blocks` and `cross_attention_heads`. Width must divide evenly
across the number of cross-attention heads.

**Path note.** ANP has two representations: a target-specific deterministic
path $r_t$ from target-to-context cross-attention, and an optional global
Gaussian latent path $z$. In the current implementation, the context
self-attention blocks are **shared** by these paths: their output supplies both
the cross-attention values for $r_t$ and the mean-pooled input to $z$. Figure 8
of Attentive Neural Processes depicts separate deterministic and latent
encoders, so this is a deliberate implementation difference to keep in mind
when interpreting the ablation.

Run both architecture test files after changing either implementation:

```bash
.venv/bin/python -m unittest tests.test_cnp_blocks tests.test_anp_crossattn_blocks
```

### Run the AEA-NP Set Transformer model

AEA-NP is the paper's latent Neural Process structure, not the paper's PPO
acquisition policy. It keeps a Gaussian latent, encodes context pairs with
induced Set Transformer blocks, infers the latent with learned attention
pooling, cross-attends from targets to the labelled context, then applies
induced Set Transformer decoder blocks over all targets.

```bash
.venv/bin/modal run --detach --profile agrover112 \
  --name aea-np-YYYYMMDD \
  organized/training/modal_aea_np_train.py::main \
  --seed 20261018 --width 32 --item-dim 16 --label-dim 16 \
  --encoder-blocks 2 --decoder-blocks 2 --attention-heads 8 \
  --n-inducing 16 --latent-dim 32 --steps 3000 \
  --run-label aea-np-YYYYMMDD
```

The first three dimensions, 16, 16, and 32, match the paper's small-model
setting. `n_inducing` controls the Set Transformer bottleneck. AEA-NP always
uses its latent, so it deliberately has no `use_latent`, `encoder`, or
`pooling` launcher flags. The shared trainer supplies the same data split,
validation bank, checkpoint policy, W&B panels, and coverage diagnostics as
the CNP and ANP runs.

Both ANP and AEA-NP launchers accept `--gpu T4` for an explicit GPU run. GPU
runs request 4 CPU cores and 16 GiB host RAM. W&B records the requested
accelerator and actual device, and groups CNP, ANP, and AEA-NP runs with the
same embedding arm and seed under `filtered-np-<embedding>-seed<seed>`.

```bash
.venv/bin/python -m unittest \
  tests.test_cnp_blocks tests.test_anp_crossattn_blocks tests.test_aea_np_blocks
```

## Set up

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,experiment]'
PYTHONPATH=src:submission python -m unittest discover -s tests
```

The submission itself has no third-party dependency.

## Interactive forecaster demo

`demo/forecaster/` is a real-checkpoint demo of budgeted evaluation. It loads
the selected CNP-AIRT and Online-AIRT checkpoints, freezes MiniLM, and compares
their forecasts with a Beta baseline on five validation tasks. It is a custom
FastAPI + vanilla web UI, not Gradio.

```bash
uv pip install --python .venv/bin/python -r demo/forecaster/requirements.txt
.venv/bin/python -m uvicorn server:app --app-dir demo/forecaster \
  --host 127.0.0.1 --port 8787
```

Open <http://127.0.0.1:8787>. The demo never trains; it reveals 31 acquisition
labels, forecasts 60 held-out items, and scores the run only when asked. See
[`demo/forecaster/README.md`](demo/forecaster/README.md) for the scope and the
custom out-of-distribution probe.

## Build and check a submission

```bash
python scripts/build_submission.py
python reference/starting_kit/tools/check_submission_zip.py dist/submission.zip
python reference/starting_kit/tools/run_smoke_test.py organized/submission/legacy/submission/

# Neural submission: selected checkpoint plus the fitted calibrator.
modal volume get aims-neurips-data \
  /results/allbench-deterministic-2seed/seed-20261018-masked-checkpoints/cnp_airt.pt \
  /tmp/cnp_airt.pt
python scripts/build_neural_submission.py --checkpoint /tmp/cnp_airt.pt --mode masked \
  --calibration /tmp/calibration-20261018-masked.json --output dist/submission-neural-final.zip
```

The generated archive has `model.py` and optional `labeling.py` at its root, as required by the official kit. `dist/submission.zip` is the final Beta-Binomial entry (Codabench `1005155`); `dist/submission-neural-final.zip` is the pre-specified CNP-AIRT entry (`1005187`). Generated ZIPs, results, credentials, and local data are excluded from Git. The calibration JSON listed above is produced by `modal run modal_calibration.py`.

### Submit to Codabench

The organizer's API client at `reference/paiec_baseline/submit_api.py` targets competition 17828, phase 29785. Authenticate once; the client stores a token at `~/.config/paiec/codabench-token` with mode 600 and never stores the password:

```bash
.venv/bin/python reference/paiec_baseline/submit_api.py --login
```

Submit the selected archive, list submissions, or read a status:

```bash
.venv/bin/python reference/paiec_baseline/submit_api.py dist/submission-neural-final.zip
.venv/bin/python reference/paiec_baseline/submit_api.py --list
.venv/bin/python reference/paiec_baseline/submit_api.py --status SUBMISSION_ID
```

Watch a run and read its logs:

```bash
.venv/bin/python scripts/watch_codabench.py SUBMISSION_ID
.venv/bin/python scripts/codabench_logs.py SUBMISSION_ID
.venv/bin/python scripts/codabench_logs.py SUBMISSION_ID --follow
```

The client checks competition approval and submission limits, creates a private dataset record, uploads the ZIP to a signed storage URL, marks the upload complete, and creates the submission. The phase ends 2026-10-31 11:59 UTC (2026-10-30 23:59 AoE) and allows 50 submissions per day, 1000 per person. The phase hides score output, so the first run confirms that the pipeline finishes; the score may remain hidden until the organizers release it. `CODABENCH_TOKEN` overrides the saved token, and `CODABENCH_TOKEN_FILE` moves it.

## Run an offline experiment

Prepare a JSON file containing one truth array and one probability array per official label budget. A runnable file is included at `examples/predictions.json`:

```json
{
  "truth": [0, 1],
  "probabilities": {
    "0": [0.5, 0.5],
    "1": [0.4, 0.6],
    "3": [0.3, 0.7],
    "7": [0.2, 0.8],
    "15": [0.1, 0.9],
    "31": [0.1, 0.9]
  }
}
```

Then run:

```bash
python scripts/run_experiment.py predictions.json
python scripts/plot_results.py results/latest/metrics.json
```

WandB is disabled in the shared configuration. After `wandb login`, use the included online example or copy it for a real experiment:

```bash
WANDB_PROJECT=aims-predictive-evaluation python scripts/run_experiment.py \
  examples/predictions.json --config configs/wandb-online.example.toml
```

Use `.env.example` as a list of environment variables, not as a place to store a key in Git.

Modal is the shared execution environment. Source and configuration stay in
GitHub; the `aims-neurips-data` Volume holds immutable raw snapshots, processed
Parquet tables, checkpoints, and results. Hugging Face credentials live only in
a Modal Secret named `huggingface-secret` with the key `HF_TOKEN`.

Install the Modal extra, authenticate, and verify the remote image and Volume:

```bash
python -m pip install -e '.[modal]'
modal profile current
modal run modal_app.py --action health
modal run modal_app.py --action inspect
```

The profile must point to the shared `agrover112` workspace. Create the secret
through the link printed by Modal or at the workspace Secrets page; never put
the token in Git, a command argument, or the Volume. Then run the two-benchmark
vertical slice:

```bash
modal run modal_ingest.py --dataset-name pilot --limit 2
modal run modal_app.py --action inspect
modal run modal_app.py --action evaluate-pilot
```

Once that succeeds, ingest the complete eligible table corpus and evaluate the
held-out-benchmark baseline:

```bash
modal run modal_ingest.py --dataset-name full --limit 0
modal run modal_app.py --action evaluate-full
```

Response traces are stored separately because they are much larger than the
canonical tables and are not available for hidden test outcomes. Preserve the
public traces as offline training evidence with the resumable command below:

```bash
modal run modal_ingest.py --action traces --dataset-name full
```

The trace corpus can be used to learn public-subject priors, behavior/failure
features, or auxiliary representations. Any useful signal must be distilled
into model parameters or features available from the competition input;
Codabench does not reveal hidden target traces at prediction time.

The current `full` Modal snapshot uses a temporary benchmark-level
`n_items >= 80` prefilter for fast competition-shaped experiments. This is not
a restriction on permitted training data: the official 80-item rule applies
to evaluated subject--benchmark pairs. If more training diversity is needed,
we will create a separate snapshot containing all binary item-level benchmarks,
including smaller benchmarks, while keeping them out of official-style Brier
ALC evaluation unless the individual pair has at least 80 observed items. See
[data preparation](docs/data-preparation.md#current-modal-snapshot-17-october-2026).

Ingestion writes to ephemeral SSD first and publishes to the Volume only after
both downloading and preprocessing succeed. Existing complete snapshots are
reused; incomplete directories cause a hard failure instead of silent
overwriting. Use `modal volume get aims-neurips-data ...` only when a local copy
of a result or checkpoint is actually needed.

## Scientific workflow

Use grouped splits so repeated outcomes for the same subject-item pair never cross train and validation sets. Compare every adaptive method with random acquisition and a predictor that ignores labels. Report Brier score and calibration by budget, aggregate them with the published ALC weights, and retain uncertainty across seeds or folds.

Read the [concept dictionary](docs/concept-dictionary.md) for the competition's terminology and flow. Read [assumptions and limitations](docs/assumptions.md) before interpreting results. The preliminary literature review is input to hypothesis generation only. It does not limit candidate methods, and its claims require independent verification.

## Evaluation protocol

### What the Amortized IRT paper evaluates

[Truong et al. (2025)](https://proceedings.mlr.press/v267/truong25c.html)
use several related but distinct evaluations:

1. **Response prediction.** Randomly mask 20% of the observed entries in the
   subject--item response matrix, fit on the remaining 80%, and evaluate the
   hidden binary responses using ROC AUC. Results are averaged through
   10-fold cross-validation.
2. **Unseen-question amortization.** Hold out whole questions and predict their
   IRT difficulty from their text embeddings. This tests whether the learned
   content-to-difficulty mapping generalizes to new questions rather than only
   reconstructing fitted question parameters.
3. **Subset generalization.** Estimate a test taker's ability from one set of
   50 questions and use that estimate to predict responses on a separate set
   of 50 questions.
4. **Adaptive-testing simulation.** Draw 200 synthetic test takers with
   `theta ~ Normal(0, 1)`, compare random question selection with Fisher
   information selection, allow up to 400 questions, repeat five times, and
   measure empirical reliability of the resulting ability estimates.

The paper's random 80/20 entry masking is a useful interpolation diagnostic,
but it is not our primary competition evaluation. It can expose the model to
other responses from the same benchmarks and therefore does not reproduce the
competition's hidden-benchmark cold start. The paper also focuses on ROC AUC
and ability reliability, whereas PAIEC scores probabilistic outcome prediction
with Brier ALC.

### Our primary offline evaluation

Use **benchmark-held-out episodic evaluation**:

1. Make the outer split by complete benchmark, never by random response rows.
   Use grouped benchmark folds for the final comparison; a fixed held-out fold
   is acceptable for fast development.
2. Fit the model, preprocessing, item representations, and calibrators using
   only the outer training benchmarks. Use separate held-out benchmarks for
   hyperparameter selection or nested validation.
3. For each eligible subject--benchmark task in the test fold, divide distinct
   `input_id` values into a label-acquisition pool and a disjoint evaluation
   pool. Repeated trials for one input must remain together.
4. Reveal acquisition outcomes at the official cumulative budgets
   `0, 1, 3, 7, 15, 31`. Use the same nested acquisition prefix at every
   budget: the three-label context contains the label used at budget one, for
   example.
5. At every budget, predict the still-hidden outcomes in the untouched
   evaluation pool. Average rows within each eligible subject--benchmark pair,
   then average pairs equally to match the leaderboard aggregation.
6. Report Brier score separately at every budget and calculate:

   ```text
   Brier ALC = 0.1*B_0 + 0.2*B_1 + 0.2*B_3
             + 0.2*B_7 + 0.2*B_15 + 0.1*B_31
   ```

7. Reuse identical folds, target pools, acquisition prefixes, and random seeds
   for every baseline and ablation. Also report calibration plots/ECE,
   benchmark-level failures, runtime, and paired uncertainty intervals
   clustered by benchmark.

NP-family models are trained episodically on training benchmarks: sample a
context set at one of the official budgets and score a disjoint target set.
Validation and early stopping use held-out benchmarks. No response from a
final test benchmark may influence fitting or model selection.

### Simulating the online label-budget process

For a single held-out subject--benchmark task, reconstruct the prediction from
the complete context supplied at that budget:

```text
budget 0:  C = {}                       -> predict evaluation targets
budget 1:  C = {(x_1, y_1)}             -> infer state again -> predict
budget 3:  C = {(x_1, y_1), ..., (x_3, y_3)} -> infer again -> predict
budgets 7, 15, 31: continue with the same nested context prefix
```

For NP-AIRT, "infer state" means recomputing `q(theta | C)` or its more general
latent representation. For MAP IRT or GP-IRT it means recomputing the ability
estimate or posterior from `C`. The subject's real ability and the offline
network weights do not change. Online adaptation initially means conditioning
on the newly revealed labels, not performing gradient updates.

Codabench workers may restart between budgets, so the predictor must be
stateless: it rebuilds its state from the complete supplied `labeled` list and
bundled artifacts on every call. Begin experiments with a seeded random or
fixed acquisition order to isolate prediction quality. Only after that result
is stable should acquisition policies such as Fisher information, uncertainty,
or expected information gain be compared while holding the predictor fixed.

After model selection is complete, refit the chosen configuration on all
eligible public benchmarks, freeze the weights, and package that checkpoint
for submission. Public benchmark-held-out results remain the evidence reported
in the paper; the refitted all-data model is used only for the final Codabench
submission.

## Decision log (paper-relevant)

Points below are settled unless tagged otherwise. Update this list when a decision changes.

### Data filtering changes everything (2026-10-26, in progress — not yet applied to any submitted model)

**Finding:** every model trained so far (CNP/ANP/GELU variants, Bayesian CNP-IRT, all MIRT/IRT rungs) was trained on the raw, unfiltered response data. Measured within-benchmark correlation between predicted and observed item pass rate on **validation** items: **0.02–0.03** for every one of them (see `docs/model-results-visualization.pdf`, pages 8–9). That is not "architecture doesn't matter" noise — it means none of these models have learned item difficulty at all; they are predicting benchmark-level averages.

**Cause, confirmed by measurement, not by guessing:** 38% of items have a single response (pass rate is literally 0 or 1), and the per-task item cap (see `train-items-per-task-cap` in Claude's own memory / the per-task weighting discussion) throws away most of what little repetition exists. See `docs/item-passrate-distribution.pdf` pages 6–8 ("Cap Audit", built by Codex) for the exact mechanism: the 128-item training cap retains only 7.9% of eligible rows, and large benchmarks like `bbq` (58,414 eligible items) keep well under 1% of their rows, destroying the cross-subject overlap item-difficulty estimation needs.

**The fix being tested — classic IRT (no text/embeddings) before vs after filtering**, following the cleaning recipe in the "Reliable and Amortized Model-based Evaluation" paper (`literature/`), §5:
1. drop duplicate item content (`content_hash`),
2. drop zero-variance items (everyone right or everyone wrong — no discriminative power),
3. drop items answered by fewer than `n` distinct subjects,
4. drop subjects answering fewer than 80 items in a benchmark (already the competition's own pair-eligibility rule).

Retention (raw → after steps 1–4, `n=15`): **54 → 17 benchmarks**, 1,338,722 → 122,973 items (9.2%). At `n=30` (the paper's own threshold): only **7 benchmarks** survive, 56,308 items (4.2%) — stricter filtering fits much better but covers much less (see below), so `n` is a real quality-vs-coverage knob, not a free lunch.

**Result, plain 1PL/2PL/3PL (every item/subject parameter a free scalar, jointly MLE-fit by Adam, no text, no cold start — this is a sanity check, not a submission candidate), 3 seeds, random 20% cell holdout:**

| Model | AUC before | AUC after (n≥15) | AUC after (n≥30) | Brier before | Brier after (n≥15) | Brier after (n≥30) |
|---|---:|---:|---:|---:|---:|---:|
| 1PL | 0.769 | 0.843 | 0.861 | 0.192 | 0.173 | 0.168 |
| 2PL | 0.756 | 0.851 | 0.872 | 0.195 | 0.149 | 0.136 |
| 3PL | 0.750 | 0.844 | 0.873 | 0.199 | 0.153 | 0.136 |

And the direct answer to "can item difficulty be learned from this data at all": a plain 2PL fit on the `n≥15` filtered matrix, scored with no held-out split (every cell used, predicted item pass rate = mean over that benchmark's fitted subjects), gets **within-benchmark correlation 0.98** with the true pass rate (122,973 items). The neural models get 0.02–0.03 on the same quantity. **The task is learnable; our models have not been trained on data that lets them learn it.**

Full plots, reproducibility commands in each script's docstring: `docs/model-results-visualization.pdf` pages 6–13 (`modal_response_matrix.py`, `modal_prediction_dump.py`, `modal_classic_irt_filter_test.py`, `modal_filter_retention.py`, `scripts/plot_model_vs_data.py`, `scripts/plot_filter_recovery.py`). Raw item pass-rate distributions and the cap audit: `docs/item-passrate-distribution.pdf` (`modal_item_passrate.py`, `modal_item_answerer.py`, `scripts/plot_item_passrate.py`).

**Row/column (subject/item) generalization, `n≥30` filter, split-half protocol (paper §5.1, extended symmetrically to items — the paper itself only tests subject generalization this way; item generalization in the paper uses a separate text-based model, out of scope for this no-text check):** per benchmark, hold out 20% of subjects and 20% of items entirely from the core fit; for each held-out subject, split its responses on training items into two random halves, fit theta on half 1 (item params frozen), score on half 2; symmetrically for held-out items (fit `a,b` on half 1, subject thetas frozen, score on half 2). 3 seeds × 7 benchmarks (`n≥30` filter):

| axis | model | AUC | Brier | n held-out (pooled) |
|---|---|---:|---:|---:|
| subject | 1PL | 0.854 ± 0.094 | 0.160 ± 0.028 | 3,414 |
| subject | 2PL | 0.854 ± 0.102 | 0.106 ± 0.046 | 3,413 |
| subject | 3PL | 0.855 ± 0.104 | 0.105 ± 0.045 | 3,411 |
| item | 1PL | 0.612 ± 0.247 | 0.200 ± 0.049 | 30,993 |
| item | 2PL | 0.599 ± 0.314 | 0.171 ± 0.092 | 31,021 |
| item | 3PL | 0.600 ± 0.315 | 0.169 ± 0.091 | 31,023 |

Subject generalization is strong (AUC ~0.85, matches the paper). Item generalization is weak and high-variance (AUC ~0.60, sd up to 0.31) — a held-out item only gets split across whichever training subjects happened to answer it, so with `n≥30` each half can be ~15 points, too little to pin down a free `(a,b)`. Script: `modal_classic_irt_generalization.py` (Adam/MLE). An L-BFGS variant (matches the paper's own optimizer) is `modal_classic_irt_generalization_lbfgs.py` — same protocol, same filters. **Result: optimizer choice doesn't matter.** L-BFGS, 3 seeds: subject AUC 0.854/0.855/0.854 (1/2/3PL), item AUC 0.612/0.610/0.609 — matches Adam to within noise (0.854/0.854/0.855 and 0.612/0.599/0.600). Both converge to the same fit; Adam is kept for the rest of the investigation.

**Not yet decided / open:**
- What `n` to actually use for the real model (15 vs 30 vs something else — coverage/quality tradeoff above).
- Whether to also change the train/val/test **split** (whole-benchmark holdout) — this is a separate axis from filtering; filtering fixes signal quality, the split controls transfer. Current within-benchmark-on-unseen-items failure is primarily a split/transfer problem that filtering alone will not fix, per the same investigation.
- No neural model (CNP, MIRT, Bayesian CNP-IRT) has yet been retrained on the filtered data. Do that next, and recheck the within-benchmark correlation on the filtered split before concluding filtering alone solves it end to end.
- `labeled` / `item_features` / cold-start generalization to never-seen items is unaffected by this filtering question and still needs the text/embedding-based difficulty predictor; the filtered classic-IRT result establishes an upper bound (0.98 within-benchmark, in-sample), not a deployable solution.
- Adam vs L-BFGS for classic IRT fitting: not yet compared head-to-head.

### Modal workspaces: `agrover112` vs `aims-neurips` — what runs where (2026-10-27)

There are **two separate Modal workspaces**, each with its own volume that happens to share the name `aims-neurips-data`. They are **different volumes**. A file written in one is not visible from the other. Check which profile is active before launching anything:

```bash
modal profile list                 # the • marks the active one
MODAL_PROFILE=aims-neurips modal run --detach <script>    # per-command override (preferred)
```

| | `agrover112` | `aims-neurips` |
|---|---|---|
| role | personal account, shared with unrelated projects | project workspace |
| other volumes present | yes — `airfoil-metrics`, `tdl-*`, `gram-*`, `gnn-fairness-vol` | none, only `aims-neurips-data` |
| `datasets/full/processed` | ✅ | ✅ |
| v2 embedding stores (me5_instruct, bge_m3_long, qwen3_4b, qwen3_8b) | ✅ all 4 | ✅ all 4 |
| v1 embedding stores (stale, 64% coverage) | ✅ | ✅ |
| `results/data-analysis` (IRT/MIRT reruns) | ✅ **all current results live here** | ⚠️ only the Bayesian rerun |

**Input data is identical in both.** Verified: the Bayesian job launched in `aims-neurips` reports `filtered (n>=30): 5,150,869 rows, 7 benchmarks`, matching the `agrover112` runs exactly.

**Where things currently are:**

| job | workspace | status |
|---|---|---|
| classic IRT (filter-test, generalization, lbfgs) | `agrover112` | ✅ done, 3/3 seeds + weights |
| filter-retention | `agrover112` | ✅ done, 3/3 seeds + weights |
| MIRT (filtered, generalization) | `agrover112` | ✅ done, 3/3 seeds + weights |
| Bayesian MIRT (old, prior bug) | `agrover112` | ❌ void — do not use |
| **Bayesian MIRT (reruns, prior fixed)** | **`aims-neurips`** | 🔄 running |
| amortized IRT | not yet run | — |

**Consequence to keep in mind:** results are now **split across two workspaces**. Any aggregation step has to pull from both, or everything should be consolidated into one workspace first. Don't assume a single `modal volume ls` shows you the whole picture.

**Hardware rule:** everything under `organized/testing/offline/` is now **CPU only** (`cpu=16`, CPU torch wheel). No offline job needs a GPU — see the budget note below. Scripts under `organized/training/` still contain `gpu="L4"` and one `gpu="H100"`; those are older neural-training jobs and must not be launched casually.

### Standing rule: filtered data, shared code under `organized/` (2026-10-27)

**Everything from here on runs on the n≥30 filtered data and lives under `organized/`.** This applies to every model — classic IRT, MIRT, Bayesian MIRT, Amortized IRT, and any neural retrain — and to every agent working in this repo (Claude Code, Codex, OpenCode alike).

- **Data:** n≥30 filtered matrix only. Not raw, not n≥15. Raw data gives 0.02–0.03 within-benchmark correlation (item difficulty is unlearnable from it); n≥30 filtered gives 0.98.
- **Code location:** `organized/training/`, `organized/testing/offline/`, `organized/testing/online/`, `organized/utils/`. Do not add new experiment scripts elsewhere.
- **Metrics:** `organized/testing/offline/modal_offline_metrics.py` (AUC, Brier, ECE, LogLoss). Do not write a fresh ECE/Brier helper — the repo already drifted into several disagreeing copies of the same quantity.
- **Figures:** `organized/utils/plotstyle.py` and `organized/utils/cnp_figures.py`. No private palettes or local rcParams. See `organized/utils/README.md`.
- **Orchestration:** `organized/run_evaluation_pipeline.py`, then `organized/aggregate_results.py`.

If something needed is missing from `organized/`, add it there and import it rather than inlining a one-off copy.

### MIRT/3PL bugs found and fixed (2026-10-27) — reruns required, Amortized IRT planned

**Every MIRT, Bayesian-MIRT, and 3PL number produced before 2026-10-27 is void.** Two modelling bugs in `organized/testing/offline/`, found while answering "which n-PL is MIRT actually using?":

1. **MIRT was not multidimensional.** The kernel was `a[i] * (theta[s].sum(dim=1) - b[i])` — a *scalar* discrimination times the **sum** of θ's components. Σθ_d is one number, so D=2 and D=4 were algebraically identical to D=1, differing only in the implied prior variance. Fixed to compensatory MIRT: `(a[i] * theta[s]).sum(dim=1) - b[i]` with `a` of shape `(n_items, D)`. Verified on synthetic 3-dimensional data: final training loss now actually moves with D (1PL: 0.619 → 0.589 → 0.573 for D=1/2/4; before the fix these were identical).
2. **3PL had no guessing parameter.** The branch was `kind in ("2pl", "3pl")` with no `c` anywhere, so every "3PL" result was a duplicate 2PL differing only by init seed. Fixed with a `logit_c` parameter capped at 0.5. Verified: on data generated with true guessing `c=0.25`, the fitted mean ĉ is 0.262; 1PL/2PL correctly stay at their untrained init since `c` isn't in their parameter list.

Three more crash bugs surfaced while fixing the above — these jobs produced **nothing**, not just wrong numbers: `modal_mirt_d2_d4_filtered.py`, `modal_bayesian_mirt_d1_d2_d4_filtered.py`, and `modal_classic_irt_filter_test.py` all called an undefined `_fit_core()` (`NameError`); `modal_filter_retention.py` used `np.array()` without importing numpy. All fixed in the same pass; see `f4f9fc3`.

**Repeated IRT runs** (same protocols as above, same filters, now with correct math — must be rerun before any result from these jobs is trusted):

| script | protocol | models affected | bug(s) fixed | status |
|---|---|---|---|---|
| `modal_classic_irt_filter_test.py` | before/after filter, n=15 & n=30, 20% cell holdout | 1PL/2PL/3PL | `_fit_core` NameError (job never ran) | rerun pending |
| `modal_classic_irt_generalization.py` | subject/item split-half, n≥30 | 1PL/2PL/3PL | 3PL had no guessing param | rerun pending |
| `modal_classic_irt_generalization_lbfgs.py` | subject/item split-half, n≥30, L-BFGS | 1PL/2PL/3PL | 3PL had no guessing param | rerun pending |
| `modal_filter_retention.py` | retention counts + 2PL item pass-rate recovery | 2PL | missing `numpy` import (job never ran) | rerun pending |
| `modal_mirt_d2_d4_filtered.py` | 20% cell holdout, n≥30 | MIRT D=2,4 × 1PL/2PL/3PL | `_fit_core` NameError (job never ran) + scalar-`a` bug + no 3PL guessing | rerun pending |
| `modal_mirt_d2_d4_generalization.py` | subject/item split-half, n≥30 | MIRT D=2,4 × 1PL/2PL/3PL | scalar-`a` bug + no 3PL guessing | rerun pending |
| `modal_bayesian_mirt_d1_d2_d4_filtered.py` | 20% cell holdout, n≥30, Gaussian priors | Bayesian MIRT D=1,2,4 × 1PL/2PL/3PL | `_fit_core` NameError (job never ran) + scalar-`a` bug + no 3PL guessing | rerun pending |
| `modal_bayesian_mirt_d1_d2_d4_generalization.py` | subject/item split-half, n≥30, Gaussian priors | Bayesian MIRT D=1,2,4 × 1PL/2PL/3PL | scalar-`a` bug + no 3PL guessing | rerun pending |

3 seeds (20261018/19/20) each, as before. Old weights/results/plots for these models get replaced by the reruns, not kept alongside them — but only once the reruns land, not before.

#### Rerun status and results (2026-10-27)

| script | status | results + weights |
|---|---|---|
| `modal_classic_irt_filter_test.py` | ✅ complete | 3/3 seeds, n=15 and n=30 |
| `modal_classic_irt_generalization.py` | ✅ complete | 3/3 seeds + weights |
| `modal_classic_irt_generalization_lbfgs.py` | ✅ complete | 3/3 seeds + weights |
| `modal_filter_retention.py` | ✅ complete | 3/3 seeds + weights |
| `modal_mirt_d2_d4_filtered.py` | ✅ complete | 3/3 seeds + weights |
| `modal_mirt_d2_d4_generalization.py` | ✅ complete | 3/3 seeds + weights |
| `modal_bayesian_mirt_*` | ❌ void, rerunning | third bug found — see below |

**Classic IRT, subject/item split-half generalization, n≥30, 3 seeds (Adam):**

| axis | model | AUC | Brier | n pooled |
|---|---|---:|---:|---:|
| subject | 1PL | 0.855 ± 0.003 | 0.160 ± 0.000 | 3,393 |
| subject | 2PL | 0.856 ± 0.001 | 0.105 ± 0.002 | 3,400 |
| subject | 3PL | 0.857 ± 0.001 | 0.105 ± 0.001 | 3,398 |
| item | 1PL | 0.611 ± 0.006 | 0.200 ± 0.002 | 30,757 |
| item | 2PL | 0.599 ± 0.008 | 0.170 ± 0.002 | 30,710 |
| item | 3PL | 0.588 ± 0.005 | 0.169 ± 0.003 | 30,758 |

L-BFGS gives the same picture (subject 0.853/0.853/0.854, item 0.611/0.610/0.610), confirming the earlier "optimizer choice doesn't matter" finding still holds post-fix.

**3PL is now a distinct model.** Before the fix it was a byte-identical clone of 2PL. Classic item AUC now separates: 2PL 0.599 vs 3PL 0.588.

**MIRT, subject/item split-half generalization, n≥30, 3 seeds — the headline result:**

| axis | model | AUC | Brier | n pooled |
|---|---|---:|---:|---:|
| subject | 1PL D=2 | 0.851 ± 0.002 | 0.142 ± 0.002 | 3,414 |
| subject | 2PL D=2 | 0.851 ± 0.004 | 0.106 ± 0.001 | 3,411 |
| subject | 2PL D=4 | 0.852 ± 0.001 | 0.103 ± 0.002 | 3,414 |
| subject | 3PL D=4 | 0.855 ± 0.001 | 0.102 ± 0.002 | 3,406 |
| item | 1PL D=2 | 0.611 ± 0.003 | 0.196 ± 0.003 | 30,920 |
| **item** | **2PL D=2** | **0.759 ± 0.010** | **0.151 ± 0.004** | 30,915 |
| **item** | **2PL D=4** | **0.760 ± 0.004** | 0.164 ± 0.003 | 30,938 |
| item | 3PL D=2 | 0.730 ± 0.014 | 0.157 ± 0.003 | 30,908 |
| item | 3PL D=4 | 0.749 ± 0.003 | 0.168 ± 0.003 | 30,950 |

**Multidimensional MIRT transforms item generalization: 0.599 → 0.759 AUC** (classic 2PL vs MIRT 2PL D=2), while subject generalization is unchanged at ~0.85. This was completely invisible before the fix, because the scalar-`a` bug made D=2 and D=4 algebraically identical to D=1. A held-out item's response pattern is explained by a *direction* in ability space rather than one scalar. The held-out item's `a_j ∈ R^D` is fit on half the subjects and scored on the other half, so it is honestly held out — no leakage.

**D=2 is enough.** Testing spread against seed sd (not an arbitrary threshold): on the generalization protocol, 5 of 6 D=2-vs-D=4 comparisons fall within 2 sd, i.e. noise. D only separates reliably on the easier cell-holdout protocol. D=4 costs more and buys nothing on the axis that matters.

**MIRT, random 20% cell holdout, n≥30, 3 seeds:**

| model | AUC | Brier |
|---|---:|---:|
| 1PL D=2 | 0.860 ± 0.001 | 0.162 ± 0.000 |
| 2PL D=2 | 0.884 ± 0.000 | 0.131 ± 0.000 |
| 2PL D=4 | 0.891 ± 0.002 | 0.127 ± 0.001 |
| 3PL D=4 | 0.887 ± 0.002 | 0.130 ± 0.001 |

#### Third bug: Bayesian MIRT prior scaling (found 2026-10-27, fixed, rerunning)

The Bayesian jobs mixed a `.mean()` likelihood with a `.sum()` log-prior:

```python
loss = BCE.mean() + (theta².sum() + b².sum() + 0.5*log_a².sum()) / (2*PRIOR_SCALE²)
```

With thousands of items the summed prior outweighs the mean likelihood by roughly the observation count. It crushed `log_a → 0`, so `a = 1`, so **every 2PL silently collapsed into a 1PL**. Evidence: in the completed runs Bayesian 1PL and 2PL agree to **8e-6 AUC**, while the non-Bayesian MIRT differs by **0.063** on the same data. Brier was also inflated (0.22 vs 0.13).

Fix: divide the summed prior by the observation count, making it a correct MAP objective. Verified on synthetic data with genuine discrimination spread (true `sd(log a) = 0.6`): `sd(log_a)` recovers **0.0 → 0.64** and 2PL separates from 1PL by 0.0089 AUC. **All pre-fix Bayesian MIRT numbers are void.**

**Rerun complete (2026-10-27), 3/3 seeds, cell holdout, n≥30:**

| model | D=1 | D=2 | D=4 |
|---|---:|---:|---:|
| 1PL | 0.865 ± 0.001 | 0.865 ± 0.001 | 0.865 ± 0.001 |
| 2PL | 0.875 ± 0.001 | 0.891 ± 0.001 | 0.896 ± 0.002 |
| 3PL | 0.876 ± 0.000 | 0.890 ± 0.001 | 0.893 ± 0.001 |

Sanity check built into the result: 1PL has no discrimination parameter, so it is correctly **flat across D** (0.865 regardless) while 2PL/3PL scale with it — the dimensionality is entering exactly where it should and nowhere else. 2PL/3PL now separate cleanly from 1PL everywhere, confirming the fix on real data (not just the synthetic check above). Matches the non-Bayesian MIRT's D=2 finding: D=2 captures most of the gain, D=4 adds little.

### Amortized IRT (2026-10-27)

Item difficulty **and discrimination** predicted from question-text embeddings via a GELU MLP (`organized/testing/offline/modal_amortized_irt.py`), trained through the IRT likelihood — see the design note above. Scope, set for time/budget: **2PL only**, **D=1 and D=2**, three embedding arms (`me5_instruct`, `bge_m3_long`, `qwen3_8b` — Qwen3-4B skipped), 3 seeds, n≥30 filtered data. Paper-faithful difficulty form `σ(a·(θ−b))`, not the slope-intercept form the MIRT jobs use — verified `p=0.5` exactly when `θ=b`. Ran on `cpu=4` in the `aims-neurips` workspace (~$1.86, ~1h wall; see the workspace section above) after a throughput gate confirmed ~0.26 s/step against the ~17 s/step the earlier GPU misconfiguration produced.

Three axes: `cell` (random 20% holdout), `subject` (split-half on held-out subjects), `item_coldstart` (held-out items scored from the embedding alone — **zero responses contribute to the fit**, the entire point of amortizing).

**AUC by axis, D, embedding arm (3 seeds pooled):**

| axis | D | me5_instruct | bge_m3_long | qwen3_8b |
|---|---|---:|---:|---:|
| cell | 1 | 0.798 ± 0.001 | 0.824 ± 0.002 | 0.825 ± 0.000 |
| cell | 2 | 0.799 ± 0.002 | 0.826 ± 0.001 | 0.826 ± 0.001 |
| subject | 1 | 0.760 ± 0.003 | 0.790 ± 0.002 | 0.793 ± 0.002 |
| subject | 2 | 0.759 ± 0.002 | 0.788 ± 0.000 | 0.795 ± 0.001 |
| **item_coldstart** | **1** | 0.618 ± 0.004 | 0.614 ± 0.003 | 0.618 ± 0.002 |
| **item_coldstart** | **2** | 0.620 ± 0.002 | 0.624 ± 0.004 | 0.623 ± 0.003 |

**The headline number is item_coldstart ≈ 0.62 AUC with zero responses to the held-out item.** Classic IRT cannot score an unseen item at all (no free parameter exists for it); amortized IRT gets meaningfully-better-than-chance AUC from the embedding alone. It is well below the warm classic-IRT item-generalization number (0.60 with a 50/50 response split, vs this at 0/100) and far below MIRT's 0.76 (which also uses responses) — the right comparison is chance (0.5), which it clears, not the response-using numbers.

**D=1 vs D=2, averaged across arms:** cell +0.001 (real but tiny, sd≈0.0006), subject −0.0003 (noise), item_coldstart **+0.006 (real, sd≈0.0014)**. Dimensionality helps a little on cold-start, nowhere else — a smaller effect than in MIRT, consistent with the amortized model's capacity sitting mostly in the embedding heads rather than in θ.

**Embedding arm comparison (D=2, averaged across axes):** bge_m3_long 0.640 ± 0.004, qwen3_8b 0.640 ± 0.003, me5_instruct 0.633 ± 0.002. bge_m3_long and qwen3_8b are statistically tied and both edge out me5_instruct by ~0.007 AUC — a real but small gap, not a reason to drop any arm.

### Everything run so far, side by side (2026-10-27)

AUC, mean ± 1 std.dev. over 3 seeds (20261018/19/20), n≥30 filtered data. `cell` = random 20% cell holdout, `subject`/`item` = split-half generalization (the held-out unit still contributes half its responses to scoring), `item_coldstart` = **zero** responses to the held-out item.

| family | cell | subject | item (50% responses) | item_coldstart (0 responses) |
|---|---:|---:|---:|---:|
| Classic 1PL | — | 0.855 ± 0.003 | 0.611 ± 0.006 | not possible |
| Classic 2PL | — | 0.856 ± 0.001 | 0.599 ± 0.008 | not possible |
| Classic 3PL | — | 0.857 ± 0.001 | 0.588 ± 0.005 | not possible |
| MIRT 2PL D=2 | 0.884 ± 0.000 | 0.851 ± 0.004 | 0.759 ± 0.010 | not possible |
| MIRT 2PL D=4 | 0.891 ± 0.002 | 0.852 ± 0.001 | 0.760 ± 0.004 | not possible |
| Bayes MIRT 2PL D=2 | 0.891 ± 0.001 | — | — | not possible |
| Bayes MIRT 2PL D=4 | 0.896 ± 0.002 | — | — | not possible |
| Amortized 2PL D=2, me5_instruct | 0.799 ± 0.002 | 0.759 ± 0.002 | — | 0.620 ± 0.002 |
| Amortized 2PL D=2, bge_m3_long | 0.826 ± 0.001 | 0.788 ± 0.000 | — | **0.624 ± 0.004** |
| Amortized 2PL D=2, qwen3_8b | 0.826 ± 0.001 | 0.795 ± 0.001 | — | 0.623 ± 0.003 |

"not possible" is the point, not a gap in testing: classic/MIRT/Bayesian MIRT all fit a free parameter per item from its responses, so an item with 0 responses has no parameter to predict with. Only the amortized model takes text as input, so it is the only family that can be asked about an item before any response exists.

**Best setting per family, against what the competition needs (streaming, scoring items as they arrive with few or no labels):**

| family | best setting | AUC | usable on an unseen item? |
|---|---|---:|---|
| Classic IRT | 1PL | 0.611 ± 0.006 (item) | ❌ |
| MIRT | 2PL D=4 | 0.760 ± 0.004 (item) | ❌ |
| Bayesian MIRT | 2PL D=4 | 0.896 ± 0.002 (cell) | ❌ |
| **Amortized IRT** | **2PL D=2, qwen3_8b** | **0.623 ± 0.003** (item_coldstart) | **✅ only one** |

Bayesian MIRT 2PL D=4 is the strongest fit overall (0.896) but only once an item already has responses in the pool. For a brand-new item arriving in a streaming setting, amortized IRT is the only family with a usable prediction at all — 0.623 looks lower only because it is being compared against chance (0.5), not against the response-using numbers above.

**qwen3_8b over bge_m3_long, decided on `subject`, not `item_coldstart`:**

| | cell | subject | item_coldstart |
|---|---:|---:|---:|
| bge_m3_long | 0.826 ± 0.001 | 0.788 ± 0.000 | 0.624 ± 0.004 |
| qwen3_8b | 0.826 ± 0.001 | **0.795 ± 0.001** | 0.623 ± 0.003 |

`item_coldstart` is a dead heat (0.001 gap, inside both sds). `subject` is where they actually separate: qwen3_8b leads by 0.007 AUC, outside both sds — a real gap. Recommendation: **ship 2PL, D=2, `qwen3_8b`**.

**Embedding coverage check (2026-10-27):** before planning Amortized IRT, checked whether the existing embedding stores cover the n≥30 filtered item universe (56,576 items, 7 benchmarks) or need re-encoding — `organized/utils/modal_embedding_coverage.py`. Result: **100% coverage, nothing missing**, for the four v2 arms built over the full 1.36M-item corpus:

| embedding arm | dim | coverage |
|---|---:|---:|
| `me5_instruct-v2` | 1024 | 100% (0 missing) |
| `bge_m3_long-v2` | 1024 | 100% (0 missing) |
| `qwen3_4b-v2` | 1024 | 100% (0 missing) |
| `qwen3_8b-v2` | 1024 | 100% (0 missing) |
| v1 arms (minilm, bge_base, llama31_8b) | 384–4096 | 64.5% — stale, built over an older 218k-item universe; do not use |

**Planned Amortized IRT ablations** (after *Reliable and Efficient Amortized Model-based Evaluation* §4.1 — the paper only ran this for Rasch/1PL with a linear head; 2PL/3PL amortized is new here). Design settled in conversation on 2026-10-27:

- **Unidimensional only** — no multidimensional θ or `a`; that axis is MIRT's job, not this model's.
- `b_j = f_φ(e_j)`, GELU MLP (not ReLU) on the question embedding.
- `d_j = g_φ(e_j)` is **also amortized** (separate head, same embedding) — one 2PL/3PL arm, not a free-scalar-`d` vs amortized-`d` comparison (that comparison was considered and explicitly dropped).
- Trained **through the IRT likelihood** (E/M-style, like the paper), not by regressing on pre-fit difficulties — that two-stage approach already exists as a diagnostic in `modal_item_difficulty_probe.py` and is not what this is.

| model | b | d | guessing c | dims | status |
|---|---|---|---|---|---|
| Amortized 1PL | `f_φ(e_j)`, GELU MLP | fixed at 1 | — | 1 | planned |
| Amortized 2PL | `f_φ(e_j)`, GELU MLP | `g_φ(e_j)`, GELU MLP | — | 1 | planned |
| Amortized 3PL | `f_φ(e_j)`, GELU MLP | `g_φ(e_j)`, GELU MLP | free scalar per item | 1 | planned |

Each to be tested against 4 embedding arms (me5_instruct, bge_m3_long, qwen3_4b, qwen3_8b) for the paper's own robustness-to-embedding-choice check (their Fig. 10), with the same subject/item split-half generalization protocol as the classic-IRT reruns above, offline metrics (AUC/Brier/ECE/LogLoss) via `organized/testing/offline/modal_offline_metrics.py`, and figures via `organized/utils/plotstyle.py` / `cnp_figures.py` — the shared code under `organized/`, not a bespoke copy (see `organized/utils/README.md`).

### How to reproduce the filtering/splitting/classic-IRT investigation

All scripts are CPU-only Modal jobs (`cpu=16, memory=131_072`), read `observations.parquet` + `items.parquet` from the `aims-neurips-data` volume, 3 seeds each, `--detach` required (local entrypoint dies otherwise kills the job). Run in this order:

1. **Filter effect (before/after, random 20% cell holdout):**
   ```
   modal run --detach modal_classic_irt_filter_test.py::main                    # n>=15 (default)
   modal run --detach modal_classic_irt_filter_test.py::main --min-subjects 30  # n>=30
   ```
   Output: `/results/data-analysis/classic-irt-filter-test-seed{seed}[-n30].parquet`.

2. **Retention counts + full-data 2PL item recovery (no holdout):**
   ```
   modal run --detach modal_filter_retention.py::main
   ```
   Output: `filter-retention-seed{seed}.parquet`, `filter-2pl-items-seed{seed}.parquet`.

3. **Row/column (subject/item) generalization, split-half, n>=30:**
   ```
   modal run --detach modal_classic_irt_generalization.py::main          # Adam
   modal run --detach modal_classic_irt_generalization_lbfgs.py::main    # L-BFGS
   ```
   Output: `classic-irt-generalization[-lbfgs]-seed{seed}.parquet`, column `axis` is `"subject"` or `"item"`.

4. **Pull results from the volume** (paths are volume-relative, **not** the `/data/...` mount path):
   ```
   .venv/bin/modal volume get aims-neurips-data /results/data-analysis/<file>.parquet <local_dir>/
   ```

5. **Rebuild the PDFs** (each script's docstring has its exact arg list):
   ```
   python scripts/plot_filter_recovery.py <classic_irt_dir> <items_avg.parquet> <base.pdf> <out.pdf>
   python scripts/plot_model_vs_data.py ...   # see docstring
   python scripts/plot_item_passrate.py ...   # see docstring
   ```
   Both target PDFs (`docs/model-results-visualization.pdf`, `docs/item-passrate-distribution.pdf`) are append-only across agents — always extract the current page count with `pypdf` and re-append, never regenerate from page 1, or you will duplicate another agent's pages.

### Naming

- Canonical display names: NP-AIRT, ANP-AIRT, Stacked-MHSA-ANP-AIRT, Online-AIRT, TfNP-D-AIRT, CNP-AIRT, Beta-Binomial, TabPFN-v2.
- `src/aims_starter/model_names.py` is the single source of truth. Stored metric keys stay stable; only display names change.
- Paper wording for the taxonomy: "deterministic corrections vs latent-mediated corrections". Amortized-deterministic: CNP-AIRT, TfNP-D-AIRT. Amortized-latent: NP-AIRT, ANP-AIRT, Stacked-MHSA-ANP-AIRT.

### Protocol

- Primary evaluation: benchmark-held-out episodic evaluation, budgets 0, 1, 3, 7, 15, 31, Brier ALC weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1.
- Current ablation evidence uses the uniform grid: 104 validation tasks, 10 frozen orders, the Beta-Binomial baseline, and five-fold out-of-fold calibration. Test is read only for the final selected model and the baseline. The binding contract is `docs/ablation-protocol.md`; the audit is `docs/uniform-audit.md`.
- The acquisition order is frozen at build time (seed 20261017). `beaver_episode_items.parquet` is reused by every seed and fold, so seeds cannot disagree on the label stream. The acquisition ablation uses 10 fixed candidate streams, not the architecture/calibration order grid.
- The b1 to b3 spike is fold-1 driven. Seeds agree and no code bug was found.
- Two full-scale seeds (`20261018`, `20261019`) are run. A third seed remains deferred until the uniform harness and report are fully audited; do not describe the two-seed spread as a stable seed variance.

### Model design

- 2PL only. 3PL is skipped: there is no guessing mechanism on free-form harm labels, and $(a, b, c)$ is not identifiable at 31 labels.
- $\theta_s$ is never a network output. It is the analytic MAP (Newton, prior $N(0,4)$, clamp $\pm 4$). Networks learn item-side $a(x)$, $b(x)$ and correction terms only.
- The official (A)NP code averages the target log-likelihood and divides the KL by the number of target points. The old v2 path used a mean NLL plus the full KL, which over-weighted KL by roughly $|T|$; that path is retired from the corrected ladder. The corrected latent control uses target-posterior reconstruction and $\mathrm{KL}(q(z|s_T) \,\|\, q(z|s_C)) / |T|$, with no context-posterior NLL. Cite the official objective, not a local derivation.
- $T \supseteq C$ in training comes from the image experiments in NP and CNP. Their regression setups use disjoint sets, and ANP states disjoint targets in the appendix. Our training now uses disjoint $C$ and $T$.
- The 1PL row moved up in priority. Truong et al. amortize difficulty only (Rasch), so 1PL isolates what the 2PL discriminations add. For the policy section: under 1PL, entropy-greedy and Fisher-greedy rank items identically; in 2PL they diverge only through $a(x)^2$.

### Latent analysis (settled)

- With the old decoder, $z$ entered only through a scalar $\theta$ via `theta_from_z`. Effective latent rank is 1, and the KL parks the other 63 directions at the prior. Measured $\mathrm{KL}(q_T \| q_C) \approx 2 \times 10^{-7}$, $\max|\mu_T - \mu_C| \approx 1.6 \times 10^{-4}$.
- Structural claim for the paper: within an episode the only global degree of freedom is scalar ability, and the MAP anchor owns it. A benchmark random effect is absorbed by $\theta$, because every item in an episode shares one benchmark. Latent redundancy is expected, not an optimization failure. The falsifier: $z$ matters if $\theta$ is not explicitly parameterized.
- Fix 1 (per-item latent decoder, $z$ enters the item parameter net) and Fix 2 (disjoint $C/T$) were both applied. One-fold diagnostics after the fixes: $d_z=64$ gives Brier ALC 0.3712, AUC 0.632-0.634, Jacobian effective rank 28. $d_z=1$ gives 0.3843, AUC 0.625, rank 1.
- Reading: extra latent directions exist but buy almost nothing. The gap between $d_z=64$ and $d_z=1$ is 0.013, inside the seed spread. The latent track is a negative control, currently worse than a constant 0.5 (0.25).
- AUC above 0.5 with Brier above 0.25 means the probability scale is wrong, not that predictions are inverted.
- Hybrid (MAP anchor plus per-item latent correction): Brier ALC 0.3181, AUC 0.647-0.724, Jacobian rank 33, $\mathrm{KL}(q_T \| q_C) = 0.032$. Temperature scaling at $T = 4$ (upper grid edge) lowers Brier to 0.2551, still above the constant-predictor score 0.25. Ranking improves over latent-only, but the corrected model has no Brier skill beyond a constant, and temperature scaling is monotone so it cannot create resolution. Keep this arm as the negative control.

### Feature separation (settled)

- The item vector was $1157 = $ item MiniLM (384) + item numeric (5) + subject MiniLM (384) + benchmark MiniLM (384), concatenated flat in `modal_faithful.py::_x_for_rows`.
- Old behavior: `irt_parameter_net` and `map_theta` consumed the subject and benchmark blocks, so $a(x)$ and $b(x)$ were subject-conditioned and the MAP anchor was fitted with item parameters that already encoded the subject. No label entered that path. The correct term is double counting or non-identifiability, not leakage or circularity.
- Budget-0 miscalibration cannot come from MAP, because `map_theta` returns 0 with no context. It came from subject-conditioned target item parameters. In the latent variants, budget 0 also had a prior-sampled `theta_from_z(z)`.
- Fix: `subject_feature_slice` zeroes the subject block on the item path (`encode_item_x`), and `map_theta` receives item-only encodings. Subject metadata may still inform the prior over $\theta$ through the context encoder. Keep one subject channel; $\theta$ itself remains MAP.
- All neural metrics recorded before this fix are stale and must not be cited.

### Result versioning (2026-10-22)

`v3-postfix` is the current label: 768-d test-faithful features, MiniLM@512,
subject block zeroed on the item path (`mode=masked`), 104 validation tasks,
10 orders, and five-fold OOF calibration. Every `r-c-*` JSON with manifest
`"arm": "minilm512"` and `mode: masked` is v3-postfix, including the r_C
ladder, loops, AR, and GP arms. Pre-fix, benchmark-aware, visible-subject, and
MiniLM@256 rows are retired and must not be mixed with v3-postfix. The binding
registry is [`docs/result-versioning.md`](docs/result-versioning.md).
Completed current-campaign tables are in
[`docs/ablation-results.md`](docs/ablation-results.md). The corrected-protocol
ladder is specified in
[`docs/faithful-ablation-ladder.md`](docs/faithful-ablation-ladder.md). Stage 1
ran; see [`docs/faithful-ladder-stage1.md`](docs/faithful-ladder-stage1.md). No
rung was promoted, and these results are not part of the v3 tables.

The current v3-postfix ablations are **preliminary**: `800` steps x `4`
episodes is only 3,200 episodes for 6,702 train tasks. The replacement is the
v4.1 corrected-protocol ladder (`docs/faithful-ablation-ladder.md`), not another
v3 patch. Do not compare v3 and v4.1 trainers.

### Test-time input contract (2026-10-19)

- The Codabench `predict()` input provides only the item fields `item_content`, `item_features`, `interactors`, and an opaque `benchmark_id` alias, plus the subject fields `normalized_name`, `provider`, `release_date`, `access_date`, `harness`, `reasoning_effort`, `harness_version`, and `subject_features_extra`. Benchmark names, descriptions, grading criteria, and asset manifests are not provided.
- The v2 pilot vector is 1157 dims and includes a 384-dim benchmark-description block, a grading-criterion-derived item text, and five item numeric columns. Three of those inputs do not exist at test time. The v2 pilot numbers are therefore benchmark-aware offline numbers, not a submission configuration.
- The test-faithful vector is 768 dims: item MiniLM over content plus item features, and subject MiniLM over the eight subject fields. The benchmark alias is only a grouping key; the benchmark level must come from $\theta$ and the acquired labels.
- Consequence for the subject channel: subject metadata is available at test time, so the pre-fix subject-conditioned item path is legitimate as an input contract. What the feature-separation decision forbids is structural double counting, not the metadata itself. The v3 pilot rebuilds the features under the test contract and compares CNP-AIRT and Online-AIRT with the subject block masked versus visible.

### Test-faithful pilot: the subject channel (2026-10-19)

Same 100 meta tasks and four BeaverTails folds as the post-fix pilot, but at 768 dims under the test contract (item MiniLM over content plus features, subject MiniLM over the eight subject fields).

| model | masked | visible |
|---|---|---|
| CNP-AIRT | 0.1964 (0.0045) | 0.1593 (0.0080) |
| Online-AIRT | 0.1911 (0.0032) | 0.1603 (0.0043) |
| Beta-Binomial | 0.1942 | 0.1942 |

- The visible channel recovers the pre-fix level (0.159) without the benchmark block. The earlier 0.160 came from subject metadata, not from benchmark descriptions or grading criteria.
- The per-budget delta (CNP-AIRT, masked to visible) is $-0.051/-0.045/-0.056/-0.032/-0.021/-0.012$ at budgets $0/1/3/7/15/31$. The gain is largest at low budgets and decays as labels take over, which is the signature of a metadata prior on the level rather than of target leakage.
- Per held-out subject, the gain concentrates where the benchmark pool misrepresents the subject. For the two subjects with base rate about 0.53, whose three training co-subjects sit at 0.94-0.99, CNP-AIRT moves 0.318 to 0.223 and 0.386 to 0.259. The two near-ceiling subjects are unchanged (0.059 to 0.060, 0.075 to 0.096).
- Reading: the subject block is a test-available level channel for subjects that their benchmark pool does not represent. The masked model keeps the structural claim that metadata may only inform $\theta$, but the prior ablation showed that channel is empty, and masking costs about 0.037 ALC here. Both modes run on the benchmark-held-out protocol; the submission uses the mode that wins there.

### BeaverTails pilot, post-fix (2026-10-19)

Clean-feature run, 3 seeds, four leave-one-subject-out folds, budgets 0/1/3/7/15/31, Brier ALC:

| model | Brier ALC | seed std |
|---|---|---|
| Beta-Binomial | 0.1942 | 0 |
| Online-AIRT | 0.1944 | 0.0062 |
| CNP-AIRT | 0.2017 | 0.0134 |
| ANP-AIRT | 0.2062 | 0.0151 |
| Stacked-MHSA-ANP-AIRT | 0.2082 | 0.0126 |
| TfNP-D-AIRT | 0.2111 | 0.0051 |
| NP-AIRT | 0.2145 | 0.0058 |

- No neural model beats Beta-Binomial after the subject channel is removed. Online-AIRT ties it and carries the best ranking by a wide margin: AUC 0.766-0.784 across budgets against 0.689-0.719 for the NP family and 0.681-0.700 for CNP-AIRT.
- Online-AIRT per budget: 0.273/0.216/0.246/0.178/0.136/0.120. It wins budgets 15 and 31 (0.136/0.120 versus 0.171/0.165 for Beta-Binomial) and loses budgets 0-3, which carry half the ALC weight. The deficit is the missing subject level at low budgets, not the ranking.
- The four BeaverTails subjects have base rates 0.986/0.940/0.526/0.533. Per-subject constant-predictor floors are 0.014/0.056/0.249/0.249, mean 0.142. Every model sits above its subject's floor, and on the two informative subjects the neural rows lose to Beta-Binomial:

| subject base rate | floor | Online-AIRT | CNP-AIRT | Beta-Binomial |
|---|---|---|---|---|
| 0.986 | 0.014 | 0.050 | 0.040 | 0.094 |
| 0.940 | 0.056 | 0.073 | 0.063 | 0.117 |
| 0.526 | 0.249 | 0.285 | 0.318 | 0.262 |
| 0.533 | 0.249 | 0.370 | 0.386 | 0.304 |

- The pre-fix rows (0.159-0.198) were identity-channel numbers. For CNP-AIRT the tell is the flat budget curve, 0.158-0.166 at every budget: the subject embedding supplied the subject's base rate while context labels added about 0.008. Post-fix the same model moves with labels (0.220 to 0.175) but starts from a worse level.
- Decision: the post-fix rows are the primary table; the pre-fix numbers appear only as the identity-channel ablation.
- Calibration: at budget 0 every row is miscalibrated (ECE 0.23-0.34). Online-AIRT's ECE falls from 0.310 to 0.105 by budget 31, against 0.083 for Beta-Binomial, so the remaining loss at low budgets is a level/scale problem rather than a ranking problem.

### Ability prior on theta (tested, rejected; 2026-10-19)

The prior mean is an adaptation-time quantity, so it rescored saved checkpoints instead of retraining. Three arms share the trained item banks: `none` ($\mu=0$), `benchmark` (mean MAP ability of the training tasks in the evaluated benchmark, three BeaverTails subjects), and `subject` (benchmark mean plus a ridge deviation on the 384-dim subject metadata embedding, penalty chosen by leave-one-out).

| model | none | benchmark | subject |
|---|---|---|---|
| Online-AIRT | 0.1944 | 0.1947 | 0.1947 |
| NP-AIRT | 0.2145 | 0.2145 | 0.2144 |
| ANP-AIRT | 0.2062 | 0.2219 | 0.2219 |
| Stacked-MHSA-ANP-AIRT | 0.2082 | 0.2100 | 0.2100 |
| TfNP-D-AIRT | 0.2111 | 0.2132 | 0.2132 |
| CNP-AIRT | 0.2017 | 0.2124 | 0.2124 |
| Beta-Binomial | 0.1942 | 0.1942 | 0.1942 |

- The `none` arm reproduces both frozen artifacts exactly at every budget: maximum absolute Brier difference $0.00$ against `metrics-calibration.json` and against the CNP aggregate. The rescore path is numerically identical to training-time scoring.
- The subject ridge collapses in every model and fold: $\alpha=1000$ (the grid ceiling) and a subject deviation that moves the prior mean by about $0.001$. A label-proxy pre-check agrees. On 96 benchmark-adjusted training tasks, ridge $\alpha$ pins at the ceiling, and 3/5/10/20-nearest-neighbour deviations are worse than predicting zero deviation (LOO MAE 1.32/1.17/1.08/1.02 versus 0.93).
- The benchmark arm is not a free win either. It leaves Online-AIRT and NP-AIRT unchanged, costs CNP-AIRT 0.011 ALC (worse in all three seeds) and ANP-AIRT 0.016, and is inside seed noise for Stacked and TfNP-D. At budget 0 it improves Online-AIRT (0.273 to 0.267) and NP-AIRT (0.260 to 0.246) but hurts ANP-AIRT (0.243 to 0.256) and CNP-AIRT (0.220 to 0.250). At higher budgets the prior mean anchors the MAP away from the labels; CNP-AIRT at budget 31 (0.175 to 0.163) is the one consistent gain.
- Reading: the item network already encodes the benchmark-level base rate through $b(x)$, so an explicit benchmark prior double-counts it at budget 0 and fights the likelihood later. Subject metadata carries no usable leniency signal on this corpus, so the sanctioned metadata-to-prior channel is empty here. Keep `prior_mean=0` as the default; `metrics-prior.json` stays as the falsification test.

### Current uniform ablation (2026-10-20)

This is the current evidence source. Every ablation uses the same 104 validation
tasks, ten frozen orders, budgets `0,1,3,7,15,31`, Brier ALC weights
`0.1,0.2,0.2,0.2,0.2,0.1`, the Beta-Binomial baseline, and five-fold
out-of-fold calibration. Test is read only for the final selected model and the
baseline. The consolidated output is
`/tmp/opencode/uniform-v1/analysis/summary.json`; the audit is
[`docs/uniform-audit.md`](docs/uniform-audit.md).

Selected configuration: seed `20261018`, masked. Deltas are paired task
differences, arm minus Beta, after out-of-fold calibration. Negative is better.

| arm | raw ALC | OOF cal ALC | delta vs Beta | 95% task CI |
|---|---:|---:|---:|---|
| Beta-Binomial | 0.2228 | 0.2228 | 0.0000 | fixed |
| CNP full | 0.2349 | 0.2226 | -0.0003 | [-0.0015,+0.0009] |
| CNP zero correction | 0.2324 | 0.2225 | -0.0003 | [-0.0015,+0.0009] |
| CNP retrained, no correction | 0.2355 | 0.2228 | +0.0000 | [-0.0011,+0.0011] |
| 1PL | 0.2364 | 0.2235 | +0.0007 | [-0.0003,+0.0018] |
| 2PL | 0.2678 | 0.2233 | +0.0005 | [-0.0006,+0.0016] |

The four-configuration mean is not better: CNP full `0.2231` against Beta
`0.2228`, delta `+0.0003` with interval `[-0.0007,+0.0013]`. The learned
ability correction is neutral after calibration (`+0.000014`,
`[-0.000029,+0.000056]`) and hurts raw ALC (`+0.002553` selected,
`+0.000999` mean).

Calibration: full BCE is the best submitted-family variant. Context-only
(`0.2245`), anchor-only (`0.2246`), and Platt-only (`0.2276`) are worse than
full BCE (`0.2226`). Task-balanced full BCE is `0.2224`, a difference of
`-0.000123` with interval `[-0.000567,+0.000295]`, so task balancing is not a
reliable effect. A Brier-loss calibrator reaches `0.2210`, `-0.001579` with
task interval `[-0.002581,-0.000477]` and benchmark interval
`[-0.002637,-0.000410]`. It is not adopted: fitted scales are negative at
budgets 0, 1, 3, and 7 (`-3.03`, `-3.47`, `-6.00`, `-3.81`), the anchor weight
reaches `0.96`, and the gain comes from inverting a validation ranking that is
near chance (`AUC 0.48`). Test raw AUC is `0.51` to `0.56`, so the inversion does
not transfer.

Disjoint-episode control: selected calibrated `0.2227` vs original `0.2226`,
delta `+0.000130`, interval `[-0.000002,+0.000261]`, a near tie. Raw improves in
the selected masked cell (`-0.001482`) but worsens in the other three; the mean
raw delta is `+0.001508`. The four-config calibrated mean delta is `-0.000113`
with benchmark interval `[-0.000306,+0.000045]`. No consistent win.

Acquisition: no active ranking policy beats random CNP. Seed `20261018`:
uncertainty `+0.0010`, Fisher `+0.0074` (`[+0.0027,+0.0128]`), disagreement
`+0.0016`. Seed `20261019`: uncertainty `+0.0011`, Fisher `+0.0043`,
disagreement `+0.0011`. The seed mean for Fisher is `+0.0058` with task interval
`[+0.0014,+0.0111]`. The Beta predictor with random order is indistinguishable
from CNP with random order.

Budget shrinkage: out-of-fold kappa is `4.0` at every budget at or above 1, so
the tuned arm is exactly the flat Beta prior. Flat and tuned both score
`0.222837959`, delta zero. This is a null by construction.

Final test: Beta `0.2182`; selected CNP `0.2211` raw and `0.2199` calibrated,
delta `+0.001754` with task interval `[+0.000401,+0.002998]` and six-benchmark
cluster interval `[-0.000716,+0.002914]`. Two-seed masked mean `0.2265` raw and
`0.2197` calibrated, delta `+0.001500` with task interval
`[+0.000385,+0.002605]`. Test is not an ablation grid.

Commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_uniform_ablations.py
PYTHONPATH=src .venv/bin/python paper/neurips_2026/plot_uniform.py
cd paper/neurips_2026 && /tmp/opencode/tectonic main.tex
```

The report is now 20 pages: main text and references through page 10,
appendices A-I on pages 11-13, and the filled NeurIPS Paper Checklist on pages
14-20. The report uses three uniform figures: `figures/uniform-brier.png`,
`uniform-ladder.png`, and `uniform-acquisition.png`. The plotting script also
writes `uniform-benchmark-order.png` as a diagnostic, but that figure is not in
the report.

The results notes below, through the pre-report material, are historical notes
from earlier mixed validation/test runs. They are kept for provenance and are
not evidence for any current table. In particular, the old order-0 validation
numbers, the test-only calibration decomposition, the threshold/reserve
acquisition table, and the "context-only wins" claim are superseded by the
uniform grid above. The Report draft section after them describes the current
20-page report.

### Full-protocol corpus and pre-registered selection (2026-10-19)

The benchmark-held-out protocol is now built end to end. The announced split is 54 benchmarks, but two train benchmarks (`chi_bench`, `haiid`) have no eligible tasks, so the effective split is 41 train / 5 validation / 6 test benchmarks and 6702 / 104 / 22 tasks. The feature store `allbench-features-v1` holds test-faithful 768-dim features for 1,356,312 items and 8,060 subjects; evaluation items are capped at 500 per task. Validation is scored on acquisition order 0 and test on orders 0-9.

Integrity checks on the built corpus: train and evaluation benchmark sets are disjoint (intersection 0); the `dataset=` tag carried inside `item_features` never overlaps between train and evaluation items; benchmark names that appear inside item text are input-available at test time and cannot carry information across disjoint benchmarks. The splits, counts, and orders match `evaluation-protocol-v2` exactly.

The submission choice is frozen before any test number is read:

1. **Feature channel.** Pick the mode (masked or visible subject block) with the lower validation calibrated Brier ALC, averaged over the two seeds. Ties break on validation raw ALC, then on masked (the more conservative channel).
2. **Model.** Submit CNP-AIRT, the deterministic single-forward model. Online-AIRT needs a per-call 200-step MAP fit and stays a comparison row.
3. **Seed.** Among the two seeds of the chosen mode, pick the lower validation calibrated ALC; ties break on the lower seed id.
4. **Calibration.** The 18 anchor-calibrator parameters are fitted on validation benchmarks only and applied unchanged to test. Validation calibration numbers are in-sample and are never quoted as test evidence.
5. **Reporting.** Once the rule is applied, test is scored once for the selected configuration. Masked/visible, both models, and both seeds remain in the report as the comparison table.

### First full-protocol pass (2026-10-19)

Run 1 of the benchmark-held-out evaluation finished on all four configurations (2 seeds x masked/visible). Two defects showed up and were fixed before any number was promoted:

- The aggregate summariser grouped `by_budget` and `brier_alc` without `mode`, averaging the masked and visible arms. The artifact now carries the mode.
- Online-AIRT's scoring-time MAP used Adam (lr 0.05, 200 steps) on the one-parameter context likelihood and overshot at budget 1: a single label saturated the sigmoid, giving validation Brier 0.38 at budget 1, worse than 0.28 at budget 0. Replaced by a capped Newton solve of the same convex objective (`map_theta_2pl`). Run 1's Online numbers are void.

Interim findings from the CNP-AIRT and Beta-Binomial arms (CNP's scoring path is unchanged; run 2 recomputes it and doubles as a determinism check):

- Validation ALC (macro over 104 tasks, order 0): Beta-Binomial 0.2249, CNP-AIRT masked 0.2348, CNP-AIRT visible 0.2347.
- Test ALC (macro over 22 tasks, 10 orders): Beta-Binomial 0.2182, CNP-AIRT masked 0.2265, CNP-AIRT visible 0.2332.
- The BeaverTails pilot's large visible-subject gain (0.196 to 0.159) did not replicate across the 52-benchmark protocol. Per-benchmark deltas are mixed: visible helps `afrimedqa`, `alpha_sql`, `cmmlu`, `devbench` and hurts `appworld`, `disco`, `cruxeval`, `dave`; the aggregate difference is inside seed noise.
- The pilot's high item AUC (CNP-AIRT 0.72-0.76 at every budget) came from item overlap, not from text generalization. Its four BeaverTails tasks share one item bank: all 420 items of every held-out task are labelled under the other three tasks. The model could learn per-item difficulty from other subjects and reuse it. The full protocol holds out benchmarks, so no evaluation item is ever seen in training, and AUC falls to 0.52-0.55. The pilot measured subject-held-out transfer on shared items; the full protocol measures benchmark-held-out transfer on unseen items. These are different questions, and only the second matches the leaderboard.
- CNP-AIRT has weak item discrimination on unseen benchmarks: test AUC rises 0.52 to 0.53 across budgets; the median per-task AUC at budget 31 is 0.54 and 45% of test tasks sit above 0.55.
- On this protocol the simple Beta-Binomial baseline beats every neural configuration of run 1.

Rule amendment, decided on validation only: the candidate set for the submission now includes the Beta-Binomial baseline. If no neural configuration beats the Beta-Binomial on validation calibrated ALC, the submission is the Beta-Binomial. Everything else in the frozen rule stands, and test is still reported once for the selected configuration with the full comparison table kept in the report.

### Calibrated full-protocol results and submission decision (2026-10-19)

Run 2 recomputed all four configurations with the exact Newton MAP and mode-aware aggregation. CNP-AIRT's numbers are identical to run 1 at every budget and both seeds, so the pipeline is deterministic across retrains. The Online-AIRT budget-1 spike survives the solver fix (validation Brier 0.38 at budget 1): the cause is transfer overconfidence on unseen benchmarks, not optimizer overshoot, and it is calibration's job to absorb it.

The AnchorCalibrator was then fitted per model and mode on validation benchmarks only (104 tasks, all 10 acquisition orders, 1M-sample cap) and applied unchanged to test. The logit-level rescore reproduces the stored probabilities to a maximum absolute Brier difference of 3.6e-8.

ALC, macro over tasks, mean over 2 seeds:

| model | mode | val raw | val calibrated | test raw | test calibrated |
|---|---|---|---|---|---|
| Beta-Binomial | - | 0.2228 | - | 0.2182 | - |
| CNP-AIRT | masked | 0.2328 | 0.2225 | 0.2265 | 0.2197 |
| CNP-AIRT | visible | 0.2335 | 0.2229 | 0.2332 | 0.2196 |
| Online-AIRT | masked | 0.2706 | 0.2230 | 0.2557 | 0.2194 |
| Online-AIRT | visible | 0.2654 | 0.2231 | 0.2521 | 0.2197 |

Validation rows are computed on the calibrator's ten-order collection (all 10 acquisition orders, the quantity the frozen rule compares). The runner's primary readout scores validation at order 0 only: Beta-Binomial 0.2249, CNP-AIRT masked raw 0.2348. The ten-order Beta-Binomial figure 0.2228 was produced by `scripts/beta_validation_orders.py` (added 2026-10-19) so the baseline and the calibrated models are compared on the same collection; the calibrated CNP's margin (0.0003-0.0006) is inside seed noise.

- Calibration compresses every configuration onto the same curve (validation 0.2225-0.2231, test 0.2194-0.2197) and recovers 0.047 ALC for raw Online-AIRT. The fitted anchors run 0.55 at budget 1 up to 0.90-0.95 at budget 31: at high budgets the calibrated forecast is mostly the context base rate, and the model's own ranking contributes the remainder.
- At budget 0 the fitted scale of the selected model is negative (seed 20261018 masked: scale -1.55, bias -0.06, anchor 0.00). The item-only ranking is faintly anti-correlated with outcomes on validation. It helps there (validation Brier 0.237) and hurts on test (0.260 against the baseline's 0.250): validation-fitted budget-0 structure does not transfer. This is reported as found.
- Frozen-rule selection: CNP-AIRT masked has the lowest validation calibrated ALC, and seed 20261018 is the better of its two seeds (validation calibrated 0.2222 versus 0.2227). The selected configuration's single test report is **0.2199 calibrated ALC** (raw 0.2211), against the Beta-Binomial's 0.2182. The calibrated neural model and the smoothed group mean are statistically indistinguishable on this protocol; the neural ranking adds little beyond the context rate on unseen benchmarks.
- The submission is the selected checkpoint with the 18 fitted calibration parameters: `dist/submission-neural-final.zip`. The organizer's ZIP checker passes, and the reference smoke test passes at the default and with the worker pool (`--concurrency 4`, MiniLM pre-downloaded, hub offline; the two sample rows use two workers), with `brier=0.269932` on the sample data. The predictor is stateless, filters evidence by subject and benchmark, embeds item text and subject fields with `sentence-transformers/all-MiniLM-L6-v2`, and applies the calibrator per revealed-label budget.
- The Beta-Binomial row also stays as the control that the calibrated neural model ties. If the leaderboard run contradicts the validation ordering, the fallback archive is `dist/submission.zip` (the Beta-Binomial submission), already checked.

### Calibration decomposition (2026-10-20)

`modal_calibration_ablation.py` refits the calibrator on validation and evaluates on the disjoint test tasks, for both models and all four seed/mode configurations. It uses ten frozen orders, the same task-level ALC as the main report, and adds the controls the user asked for: component decomposition, BCE versus Brier fitting, row-weighted versus task-balanced fitting, per-budget versus one shared fit, and a fixed 0.5 at budget 0. The refits are post hoc; nothing is fitted on test and the submitted archive is unchanged.

Selected configuration (CNP-AIRT, masked, seed 20261018), test Brier ALC:

| variant | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.2588 | 0.2442 | 0.2238 | 0.2112 | 0.2000 | 0.1935 | 0.2211 |
| Platt only | 0.2595 | 0.2526 | 0.2331 | 0.2164 | 0.2027 | 0.1945 | 0.2264 |
| Anchor only | 0.2500 | 0.2387 | 0.2224 | 0.2100 | 0.1999 | 0.1939 | 0.2186 |
| Full | 0.2595 | 0.2397 | 0.2227 | 0.2107 | 0.2000 | 0.1937 | 0.2199 |
| Context only | 0.2500 | 0.2387 | 0.2232 | 0.2109 | 0.2006 | 0.1941 | 0.2191 |
| Full, budget 0 = 0.5 | 0.2500 | 0.2397 | 0.2227 | 0.2107 | 0.2000 | 0.1937 | 0.2190 |
| Full, Brier fit | 0.2596 | 0.2387 | 0.2234 | 0.2113 | 0.2009 | 0.1942 | 0.2202 |
| Full, task-balanced | 0.2604 | 0.2383 | 0.2220 | 0.2103 | 0.2001 | 0.1939 | 0.2196 |
| Full, shared | 0.2521 | 0.2392 | 0.2221 | 0.2104 | 0.2003 | 0.1938 | 0.2190 |

Mean test Brier ALC over the four seed/mode configurations:

| variant | CNP-AIRT | Online-AIRT |
|---|---:|---:|
| raw | 0.2299 | 0.2539 |
| Platt only | 0.2266 | 0.2233 |
| Anchor only | 0.2195 | 0.2191 |
| Full | 0.2196 | 0.2196 |
| Context only | 0.2191 | 0.2191 |
| Full, Brier fit | 0.2196 | 0.2200 |
| Full, task-balanced | 0.2191 | 0.2188 |

- The context anchor does the work. Context-only beats the full calibrator by 0.00053 in all four configurations for both models, and anchor-only is within 0.0014 of full in every cell (mean gap 0.00015 for CNP-AIRT and 0.00053 for Online-AIRT). Platt-only is 0.004-0.007 worse than full on average and, on the selected seed, worse than raw at every budget.
- The full form wins on validation (selected seed 0.2222 versus 0.2246 for anchor-only) but not on test. The selected seed's full calibrator has a negative budget-0 scale ($-1.54$, bias $-0.06$); its test B0 is 0.2595 against 0.2500 for anchor-only. Forcing budget 0 to 0.5 raises full ALC from 0.2199 to 0.2190.
- The fixed-0.5 control is not uniformly better. On the second masked seed the full fit's budget 0 is 0.2467, better than the 0.2500 constant, so the zero-half variant is 0.0003 worse there. The fixed control helps 3/4 CNP and 4/4 Online configurations, but it is a robustness guard, not a free gain.
- Task-balanced fitting is the one modification that helps both models in all four configurations: mean ALC falls by 0.00056 for CNP-AIRT and 0.00087 for Online-AIRT relative to row weighting. The Brier objective is mixed: 0.00006 better on average for CNP-AIRT, 0.00032 worse for Online-AIRT, and it wins on only 2/4 CNP and 1/4 Online cells. One shared calibrator across budgets is 0.00043 (CNP) and 0.00049 (Online) better than per-budget on average, but wins on 3/4 and 4/4 cells respectively, so the evidence is small.
- All mean context/full/anchor differences are below 0.001 ALC, inside the seed spread; the selected seed's anchor-full gap is 0.0013. The robust conclusion is that the context-rate anchor, not the learned logistic term or the neural ranking, carries the transferable calibration gain. The selected CNP-AIRT configuration's anchor-only advantage over full is not stable across seeds, so the report states that caveat rather than claiming a uniform win.

Decision: keep the submitted archive and the pre-registered calibrator. The decomposition is reported as a post-hoc diagnostic in Appendix C, with the selected-seed table and the four-configuration means. Run `modal run modal_calibration_ablation.py --seed 20261018 --mode masked --models cnp_airt,online_airt_2pl --orders 10`, then `.venv/bin/python scripts/analyze_calibration_ablation.py <json files> --output <summary.md>`. Artifacts: `/data/results/allbench-calibration-ablation/calibration-ablation-{seed}-{mode}-cnp_airt-online_airt_2pl.json`; local summary `/tmp/opencode/calibration-ablation/summary-all.md`.

### Encoder contribution ladder (2026-10-20)

`modal_cnp_ladder.py` adds the five-rung architecture ladder requested for the report: Beta-Binomial; 1PL item difficulty plus MAP ability; 2PL item difficulty and discrimination plus MAP ability; CNP-AIRT with the set-encoder correction zeroed at inference; and CNP-AIRT with the learned correction. It also trains a sixth robustness arm, a CNP item head retrained with `use_theta_delta=False`. The zeroed arm reuses the full CNP checkpoint with `delta_scale=0`, so the decisive comparison isolates the exact inference path. Every non-Beta arm gets its own validation-only `AnchorCalibrator`; raw and calibrated test ALC are both reported.

Selected configuration (CNP-AIRT, masked, seed 20261018), test Brier ALC:

| arm | raw | calibrated |
|---|---:|---:|
| Beta-Binomial | 0.2182 | -- |
| 1PL + MAP | 0.2295 | 0.2195 |
| 2PL + MAP | 0.2573 | 0.2194 |
| CNP item head + MAP ($\Delta\theta=0$) | 0.2201 | 0.2197 |
| CNP-AIRT ($\Delta\theta(C)$) | 0.2211 | 0.2199 |
| CNP item head, retrained without $\Delta\theta$ | 0.2202 | 0.2191 |

Mean over the four seed/mode configurations:

| arm | raw | calibrated |
|---|---:|---:|
| Beta-Binomial | 0.2182 | -- |
| 1PL + MAP | 0.2280 | 0.2193 |
| 2PL + MAP | 0.2539 | 0.2196 |
| CNP item head + MAP ($\Delta\theta=0$) | 0.2290 | 0.2196 |
| CNP-AIRT ($\Delta\theta(C)$) | 0.2299 | 0.2196 |
| CNP item head, retrained without $\Delta\theta$ | 0.2306 | 0.2194 |

- The encoder correction is empirically redundant. On the selected configuration, full minus zero is $+0.00091$ raw (paired 95% interval $[-0.00011,+0.00193]$) and $+0.00022$ calibrated ($[+0.00002,+0.00042]$); zeroing improves raw ALC in all four configuration means. The calibrated difference is one fifth of the seed spread and not operationally meaningful.
- The correction is not zero. On the selected test tasks, its mean is $-0.086$ at budget 1 and $-0.077$ at budget 31; SD is $0.092$ to $0.110$ (variance $0.008$ to $0.012$), mean absolute value is $0.10$ to $0.11$, and the mean absolute probability shift is $0.011$ to $0.015$. The sign flips across seeds and modes (the four-configuration mean is $+0.008$ to $+0.014$), so the correction has no stable direction.
- Full minus zero Brier is positive at every test budget on the selected configuration: $+0.00169$ at budget 1, $+0.00118$ at 3, $+0.00097$ at 7, $+0.00050$ at 15, and $+0.00047$ at 31. The correction does not improve the forecast.
- The 1PL control exposes a second negative result: 2PL discrimination is worse than 1PL on raw ALC in all four configurations (mean $0.2539$ against $0.2280$), but the same calibration erases the gap ($0.2196$ against $0.2193$). The selected paired 2PL minus 1PL raw difference is $+0.02780$ ($[+0.01959,+0.03601]$); calibrated it is $-0.00008$ ($[-0.00132,+0.00117]$).
- The retrained no-correction arm is mixed. In the selected configuration it is better raw ($0.2202$ against $0.2211$) and calibrated ($0.2191$ against $0.2199$), but the full model is better raw in three of four configurations and the calibrated mean gap is $+0.0002$ in the full model's favor. This does not establish a training-time encoder benefit; it also does not rescue the direct correction.
- Verdict: the Neural Process set encoder's learned ability update is empirically redundant on this corpus. The item head plus MAP and a 1PL control both match the full model after calibration. No submitted archive changes; Beta-Binomial remains the final entry.

The selected run is `20261018-masked`; the full set is `20261018-masked`, `20261018-visible`, `20261019-masked`, and `20261019-visible`, all with ten orders and all tasks. Commands:

```bash
.venv/bin/modal run modal_cnp_ladder.py --seed 20261018 --mode masked --orders 10
.venv/bin/python scripts/analyze_cnp_ladder.py \
  /tmp/opencode/cnp-ladder/cnp-ladder-20261018-masked.json \
  /tmp/opencode/cnp-ladder/cnp-ladder-20261018-visible.json \
  /tmp/opencode/cnp-ladder/cnp-ladder-20261019-masked.json \
  /tmp/opencode/cnp-ladder/cnp-ladder-20261019-visible.json \
  --output /tmp/opencode/cnp-ladder/summary-all.md
```

Artifacts: `/data/results/allbench-cnp-ladder/cnp-ladder-{seed}-{mode}.json` and `.../seed-{seed}-{mode}-checkpoints/`; local copies in `/tmp/opencode/cnp-ladder/`. The selected run matches the stored CNP calibration run to a maximum absolute Brier difference of $1.2\times10^{-8}$ over 1{,}944 rows. The ladder is post hoc and does not change the pre-specified selection.

### Online-AIRT submission (2026-10-20)

The Online-AIRT package uses the masked checkpoint from seed 20261018 (`seed-20261018-masked-checkpoints/online_airt_2pl.pt`), the lowest validation calibrated Online-AIRT cell (`0.2229`). The source folder is `submission_online/`; the build script vendors the same `faithful_np.py`, `metrics.py`, and `calibration.py` used by the CNP package. The runtime solves the one-dimensional MAP ability from the revealed context labels, applies the 2PL item head, and then applies the six fitted calibration entries for `online_airt_2pl`.

The archive includes `labeling.py` byte-identical to the policy in the first `dist/submission.zip` (Sha256 `19ea92f03cde63c7b7766c05e36fc44acf9274a12825e580dcf49f0ca9d9d903`). It purchases labels whose prediction is within 0.15 of 0.5, and otherwise reserves enough candidates to fill the budget. This makes the Online-AIRT run an active-acquisition run, unlike the random-acquisition CNP-AIRT run in `1005187`.

A second active-policy archive packages the CNP-AIRT checkpoint in `dist/submission-neural-active.zip` with the same `labeling.py` and the same six CNP calibration entries. It uses the same builder with `--template submission_neural --model cnp_airt --labeling submission/labeling.py`. Its organizer checker passes, and the smoke test passes at concurrency 4 with `brier=0.269932` on the two sample rows. Codabench submission `1005225` was created from it; its score was pending at the time of writing.

Build and check:

```bash
.venv/bin/python scripts/build_neural_submission.py \
  --checkpoint /tmp/opencode/online_airt_2pl-20261018-masked.pt \
  --mode masked \
  --calibration /tmp/opencode/allbench2/calibration-20261018-masked.json \
  --model online_airt_2pl \
  --template organized/submission/legacy/submission_online \
  --checkpoint-name online_airt_2pl.pt \
  --output dist/submission-online-airt.zip
python reference/starting_kit/tools/check_submission_zip.py dist/submission-online-airt.zip
python reference/starting_kit/tools/run_smoke_test.py /tmp/opencode/online-submission-smoke --concurrency 4
```

The organizer checker passes. The smoke test passes at concurrency 1 and 4 with `brier=0.273092` on the two sample rows; concurrency 4 used two isolated workers. Codabench submission `1005210` was created from `dist/submission-online-airt.zip`; its score was pending at the time of writing.

### Active-acquisition ablation (validation; 2026-10-20)

The hidden active CNP run (`1005225`) scored worse than random CNP (`1005187`), but no offline experiment had compared acquisition rules on the same episodes. `modal_active_ablation.py` adds that comparison on the 104 validation tasks. It uses ten random candidate streams per task, the same streams for every arm, one acquisition trajectory to 31 labels, and nested prefixes at budgets 0, 1, 3, 7, 15, and 31. Evaluation targets are the disjoint evaluation pool. The CNP arms use the selected masked checkpoint (seed 20261018) and its validation-fitted calibrator. The Beta arms use the report's `(2 + successes) / (4 + n)` predictor. The task is the independent unit: streams are averaged within task, then the table reports the mean and standard error across 104 tasks. Paired deltas use the same task-level averages.

Notation:

- `P_β`: Beta-Binomial predictor, `p = (2 + successes) / (4 + n)`.
- `P_C`: CNP-AIRT predictor, masked, seed 20261018, calibrated.
- `A_R`: random stream order.
- `A_U(P)`: buy when `|p_P - 0.5| <= 0.15`.
- `A_V(β)`: buy when the Beta posterior variance is at least 0.01.
- `A_D`: buy when `|p_C - p_β| >= 0.15`.
- Entropy acquisition is not a separate arm. At a fixed threshold, `H(p)` is a monotone transform of `|p - 0.5|`, so it makes the same decisions as `A_U`. `P_β + A_U(C)` is identical to `P_β + A_U(β)`, and `P_β + A_D` is identically zero, because the Beta predictor has no item-level signal.

| arm | study | predictor | acquisition | Brier ALC | ECE ALC |
|---|---|---|---|---:|---:|
| `C-D` | Priority-2 | `P_C` | disagreement | 0.22067 (0.00438) | 0.12385 (0.00335) |
| `C-Uβ` | Priority-2 | `P_C` | Beta-mean | 0.22088 (0.00433) | 0.12441 (0.00330) |
| `C-UC` | Priority-1 | `P_C` | CNP uncertainty | 0.22088 (0.00431) | 0.12469 (0.00329) |
| `C-R` | Priority-1 | `P_C` | random | 0.22089 (0.00433) | 0.12517 (0.00324) |
| `C-Vβ` | Priority-2 | `P_C` | Beta variance | 0.22091 (0.00434) | 0.12512 (0.00328) |
| `β-R` | Priority-1 | `P_β` | random | 0.22164 (0.00392) | 0.12795 (0.00418) |
| `β-Uβ` | Priority-1 | `P_β` | Beta-mean | 0.22166 (0.00391) | 0.12777 (0.00419) |
| `β-Vβ` | Priority-1 | `P_β` | Beta variance | 0.22167 (0.00392) | 0.12803 (0.00422) |

The acquisition rules do nothing. Against its own random baseline, `C-D` gains 0.00021 Brier (paired standard error 0.00068; wins 52 of 104 tasks), `C-Uβ` gains 0.00001 (0.00032; 57 wins), and `C-UC` is flat (0.00000; 0.00039). `β-Uβ` is 0.00003 worse than `β-R` (0.00029; 54 wins). `C-D` is 0.00099 better than `β-Uβ` (0.00085; 52 wins), which is the predictor and acquisition combination, not a policy gain. Every paired interval spans zero.

The reason is mechanical. On a random stream, the uncertainty rule rarely fires: the 0.15 threshold is not matched to the calibrated probability scale, so the reserve rule decides almost every purchase. At budget 31, `C-D` selects at mean stream position 0.96, `C-Uβ` at 0.77, and `C-UC` at 0.54, against 0.017 for random. The policies are late-stream reserve rules, not item-information acquisition. The Beta posterior-variance arm is also a stopping rule, because the Beta predictor is constant across items. With item discrimination near chance (test AUC 0.52--0.56), no acquisition rule can create signal that the predictor does not have.

Decision: no acquisition variant improves offline. Keep Codabench `1005155` (Beta-Binomial with the first active policy) as the final entry because it is simpler and the paired offline evidence is at least a tie. The hidden archives received different subject-benchmark groups (9, 9, 8, and 8 groups; `1005155` and `1005187` share no exact group), so the 0.038 Brier gap is unpaired and does not establish a predictor or acquisition advantage. The report discloses the post-hoc switch from the pre-specified CNP-AIRT selection. Run `modal run modal_active_ablation.py --streams 10`, then `.venv/bin/python scripts/analyze_active_ablation.py <rows.parquet> --output <analysis.json>`. Artifacts: `/data/results/allbench-active-ablation/ablation.json` and `ablation_rows.parquet`. The competition-variant policies are in `submission_variants/`.

### Disjoint-episode control for CNP-AIRT (tested, null; 2026-10-19)

`_episode` defaults to `disjoint_context_target=False`, so training episodes place context items in the target set (at budget 31 roughly half the training targets are already revealed), while evaluation targets are disjoint from the acquisition pool. The open question was whether this mismatch explained the raw gap to the baseline. `modal_cnp_disjoint.py` retrained CNP-AIRT with the flag on (2 seeds x masked/visible, same architecture, steps, and episode sampler), scored the identical protocol (validation order 0, test orders 0-9), and refit the AnchorCalibrator per configuration on the validation ten-order collection. Results root: `/data/results/allbench-disjoint-cnp`; the logit rescore matches the stored rows to a maximum absolute Brier difference of 2.1e-8 (1,944 matched rows per run).

| model | mode | val raw | val calibrated | test raw | test calibrated |
|---|---|---|---|---|---|
| CNP-AIRT (disjoint) | masked | 0.2330 | 0.2224 | 0.2291 | 0.2196 |
| CNP-AIRT (disjoint) | visible | 0.2363 | 0.2229 | 0.2362 | 0.2194 |

The calibrated endpoint is unchanged (test 0.2194-0.2196 against 0.2196-0.2197 before; validation 0.2224-0.2229 against 0.2225-0.2231). Raw ALC is slightly worse in all four cells (0.0002-0.0030), inside the raw seed spread (masked test sd 0.0098). The budget-0 fitted scale stays negative in all four runs (-0.6 to -1.8), so the budget-0 inversion is a property of the item head on unseen benchmarks, not of the training mismatch. Verdict: the mismatch is not the bottleneck; the submission stays the pre-registered CNP-AIRT masked seed 20261018, and the control is reported as a post-hoc ablation in Appendix H. `train_cnp` gained the `disjoint_context_target` keyword (default `False`, backward compatible).

### Report draft (2026-10-20)

The technical report for the competition lives in `paper/neurips_2026/main.tex` (NeurIPS 2026 template, `[preprint]` option, 20 pages total: main text and references through page 10, appendices A-I on pages 11-13, and the filled NeurIPS Paper Checklist on pages 14-20). It is built around the uniform validation grid above and keeps test in a separate final table. The appendices cover the corpus audit, implementation, calibration parameters, per-budget and per-order validation metrics, per-benchmark ALC, per-configuration disjoint differences, acquisition intervals, and the reproducibility inventory. The checklist is `paper/neurips_2026/checklist.tex` and is included with `\input{checklist}`. The report uses three uniform figures, `uniform-brier.png`, `uniform-ladder.png`, and `uniform-acquisition.png`, generated by `paper/neurips_2026/plot_uniform.py`. The script also writes `uniform-benchmark-order.png` as a diagnostic, but the report does not include it. Built PDF: `paper/neurips_2026/main.pdf`. The old 25-page report with mixed validation and test tables is backed up outside the repo at `/tmp/opencode/main-old-20261020.tex` and is not evidence.

Build (no system LaTeX; static tectonic binary):

```bash
cd paper/neurips_2026
/tmp/opencode/tectonic main.tex
```

The main ablation tables use the selected seed `20261018` masked. Four-configuration means appear only in the robustness rows and are labeled. Test is reported once for the final selected model and the baseline; it is not an ablation source. The prior-mean and metadata-ridge negative results remain historical notes below and are not evidence for the current report.

### Results bookkeeping (settled)

- Frozen stale root: `results/faithful-np-beavertails-3seed-v2-fixed` (pre-feature-fix, ANP residual route off). Post-fix roots: `faithful-np-beavertails-3seed-v3-residual` for the NP family and `cnp-airt-beavertails-3seed-v2-residual` for CNP. Never quote metrics from the frozen root.
- `run_seed` and `run_cnp_seed` compare a `training_signature` (hash of the training code plus protocol constants) before reusing cached metrics, and raise instead of silently returning stale files.
- Training and every rescore pass build ANP rows through `_variant_model_kwargs`, so the checkpoint `decoder_residual` weights always have a module to load into. A drift here broke the first calibration pass with unexpected-key errors.
- The ability-prior work added eval-only `prior_mean` arguments to `faithful_np.py`, so the training signature no longer matches the cached runs. The checkpoints and seed metrics remain valid because the `none` arm reproduces the calibration rescore exactly; rerunning training requires a new result root.
- `metrics-prior.json` holds the three prior arms. `prior.json` per seed holds the per-fold prior means, ridge penalty, and leave-one-out error. `modal run modal_faithful.py --prior-only` rebuilds both from saved checkpoints without retraining.

### Encoder re-selection: multilinguality and item text (2026-10-24)

Every embedding arm run so far (`minilm`, `minilm512`, `bge_base`, `bge_m3`,
`llama31_8b`, `bge_base_2pladapt`) shares two properties that a corpus audit
shows are wrong for this data: the encoded text concatenates `item_features`
onto `item_content`, and four of the six encoders are English-only. Sources for
every number below: `modal_field_audit.py`, `modal_feature_diagnostics.py`, and
the language and content-length audits whose figures are in
[`docs/language-distribution.pdf`](docs/language-distribution.pdf).

**The corpus is 9% non-English, measured.** Across all 1{,}356{,}312 protocol
items, 122{,}931 are non-English, spanning 24 languages. Language is read from
`lang=`-style tags in `item_features` where present and from the dominant
Unicode script otherwise.

| Language | Items | Benchmarks (of 52) | Items cut at 512 tokens |
|---|---:|---:|---:|
| Chinese | 39{,}743 | **9** | 1.7% |
| Basque | 24{,}817 | 1 | **47.6%** |
| Amharic | 18{,}054 | 3 | 15.0% |
| Russian | 6{,}096 | 3 | 5.1% |
| 20 African languages | ~33{,}000 | 1 | ~0% |

Chinese is the one that is structurally hard to avoid: it appears in nine
benchmarks (`cmmlu`, `ceval`, `harmbench`, `editbench`, `emoji_attack`,
`clasheval`, `braveguard`, `disco`, `agentic_review_perturbation`), not just the
two Chinese-named ones. Every other non-English language sits in one to three
benchmarks. `afrieval` alone carries 20 African languages, ten of which
(Nigerian Pidgin, Luganda, Moore, Bambara, Fon, Twi, Ewe, Tswana, Ghomala, Luo)
appear in no published language list of any encoder considered; those 11{,}133
items are byte-level noise under every candidate and are a floor, not a choice.

The train/validation/test benchmark split is **our** construction and is not
used to weight any of this. The competition draws its own subject-benchmark
pairs, so a language's importance is its corpus-wide presence, not which of our
splits it landed in.

**Why the current encoders fail.** `all-MiniLM-L6-v2` and `bge-base-en-v1.5` are
English-only. `Llama-3.1-8B-Instruct` officially supports eight languages
(en, fr, de, hi, it, pt, es, th), which intersects the list above at **zero** --
the 4096-dim `llama31_8b` arm cannot represent Chinese, Basque, Amharic,
Russian, or any African language in the corpus.

**Selection evidence.** Vendor-reported MTEB means were not used. Two
independent benchmarks were:

| Model | MMTEB (ICLR 2025) | AfriMTEB-Lite (EACL 2026) | MTEB(LLM) (COLM 2026) |
|---|---:|---:|---:|
| multilingual-e5-large-instruct | **63.2** (1st of 12) | **62.0** | 69.7 (19th of 26) |
| AfriE5-large-instruct | -- | **63.7** | -- |
| bge-m3 | 59.6 | 55.0 | 66.5 |
| Qwen3-Embedding-8B | 70.6 | 49.7 | 77.0 |
| Qwen3-Embedding-4B | 69.5 | 46.0 | **75.9** |
| Qwen3-Embedding-0.6B | 64.3 | **40.7** | 72.5 |
| LaBSE | 52.1 | -- | -- |
| EmbeddingGemma-300m | 61.2 | **29.1** | 72.2 |
| GritLM-7B | 60.9 | 47.3 | 63.1 |
| e5-mistral-7b-instruct | 60.3 | 45.8 | 64.4 |
| all-MiniLM-L6-v2 (current) | **41.4** (last) | -- | -- |

Three findings drive the choice. First, advertised language counts are
uninformative: `mmBERT-base` claims 1{,}800 languages and scores 28.9 on
AfriMTEB; `EmbeddingGemma-300m` claims 100+ and scores 29.1. Second, parameter
count does not buy multilinguality -- MMTEB's stated conclusion is that
Mistral-based 7B embedders lose to a 560M XLM-R model because "Mistral [is]
predominantly pre-trained on English, while XLM-R targets 100 languages". Third,
general-purpose quality and multilingual coverage are in tension: mE5 ranks 1st
on MMTEB and 19th of 26 on MTEB(LLM), and Qwen3 is the reverse. No single model
wins both, so the arms are chosen to span the trade-off rather than to pick a
presumed winner.

Two further models were checked and rejected. `jina-embeddings-v5-text-small` is
built on the Qwen3-0.6B backbone that scores 40.7 on AfriMTEB, is trained on 32
languages with the list unpublished, and is CC-BY-NC. `EmbeddingGemma 2` moves
MTEB Multilingual from 61.15 to 61.36 over a predecessor that scored 29.1 on
African languages, and has no AfriMTEB result.

**Item text changes to `item_content` alone.** The current template is
`f"item: {content}\nmetadata: {item_features}"`. `item_features` has no shared
schema across benchmarks; keys are per-harness. Some keys are genuine difficulty
signal (`prompt_condition=cot` vs `nocot` in `cmmlu`, `lang`, `prompt_variant`),
but others are per-item fingerprints (`appworld_task_id`, `source_task_id`,
`datetime`) and at least one is a benchmark label in disguise: every `devbench`
item carries an identical 180-character `input_scope=` sentence. This is the
mechanism behind the measured result that item-difficulty $R^2$ is 0.78--0.82
overall, approximately 0 within a benchmark, and negative on a held-out
benchmark: the embeddings learn benchmark identity. The new stores encode
`item_content` only. Parsed `item_features` keys that survive a "present in at
least two benchmarks" filter are a separate structured channel, not text.

**Planned arms.** Four stores, all reduced to 1024 dimensions so the comparison
varies encoder and context length but not dimensionality. Native dimensions are
read from each model's `config.json`.

| Arm | Model | Native dim | Context | 1024-d via |
|---|---|---:|---:|---|
| `me5_instruct` | `intfloat/multilingual-e5-large-instruct` | 1024 | 512 (hard) | native |
| `bge_m3_long` | `BAAI/bge-m3` | 1024 | 8192 | native |
| `qwen3_4b` | `Qwen/Qwen3-Embedding-4B` | 2560 | 32768 | MRL truncate + renormalize |
| `qwen3_8b` | `Qwen/Qwen3-Embedding-8B` | 4096 | 32768 | MRL truncate + renormalize |

`me5_instruct` against `bge_m3_long` is the controlled context-length test: same
XLM-R backbone, same 1024 dimensions, 512 against 8192. `qwen3_4b` tests whether
Chinese strength outweighs African weakness. `qwen3_8b` is contingent on `4b`
winning. MRL truncation is self-reported by Qwen and unverified at 1024; if the
truncated arm underperforms, the full 2560-dim arm separates "MRL lost
information" from "Qwen is wrong for this corpus".

The 512-token cap is the one thing `me5_instruct` cannot change, since it is
XLM-R's position-embedding limit. Corpus-wide that cap cuts 10.5% of items and
20.1% of tokens. The binding case is `bertaqa`, whose median item is **511**
tokens against the 512 cap, so 47.6% of Basque items are truncated. An earlier
note in this log that truncation "does not hurt" was measured on the aggregate,
English-dominated corpus and does not cover that case.

**Encoding plan.** 43.2% of item contents are exact duplicates: 769{,}888
unique texts among 1{,}356{,}312 items, carrying 59.3% of the characters.
Encoding unique texts and broadcasting by content hash is exact. With that plus
length-sorted dynamic batching, a 4096-token cap (99.5% of items whole), bf16,
and vLLM for the Qwen decoders, the four stores cost roughly three GPU-hours in
total, about one hour forty wall-clock when run as parallel Modal containers.

| Arm | GPU | Naive | Optimized |
|---|---|---:|---:|
| `me5_instruct` | A10G | ~55 min | ~12 min |
| `bge_m3_long` | A10G | ~70 min | ~15 min |
| `qwen3_4b` | H100 | ~5 h | ~50 min |
| `qwen3_8b` | H100 | ~9 h | ~1 h 40 |

Existing stores are kept. `cnp_gelu` is the best selected arm (validation
calibrated ALC 0.219412; `cnp_ln_gelu` is 0.000059 lower by mean but not
reliable); `bge_base_2pladapt` is the previous best at 0.222584. The residual
2PL adapter should still be refitted on the winning store; GELU is now the only
architecture change that has moved the number.

### me5 encoder and v4.1 sampler results (2026-10-25)

Full tables and caveats: [`docs/me5-encoder-and-v41-sampler.md`](docs/me5-encoder-and-v41-sampler.md).

Validation, 104 tasks, out-of-fold calibration, Beta-Binomial reference
`0.222838` (identical to six decimals against `docs/ablation-results.md`, so
these rows are comparable to the published r_C ladder).

| Change | Effect | Replicates? |
|---|---:|---|
| `item_content`-only text + multilingual-e5-large-instruct | **-0.0017 to -0.0023** | **yes** -- 3 seeds, 6 architectures |
| v4.1 coverage sampler, `cnp` / `anp_attn` | **-0.0012 to -0.0015** | 1 seed, paired CI excludes zero |
| v4.1 coverage sampler, IRT-anchored rungs | `+0.00003` | no effect |
| bge-m3 in place of me5 | **+0.0005 (worse)** | 5 arms, all above Beta |
| Multidimensional ability (D=4) | sign flips by seed | **no -- withdrawn** |

Best selected arm: `cnp_gelu` under the v4.1 sampler, **`0.219412`** (3-seed
mean), `-0.003426` against Beta and `-0.0032` against the previous best arm
ever run (`bge_base_2pladapt`, `0.222584`). The paired t-interval barely
excludes zero (`[-0.006770, -0.000081]`), but the task bootstrap interval
includes zero (`[-0.006362, +0.000243]`) and the uncorrected sign-flip p-value
is 0.030. With ~20 arms tested, this is not a reliable Beta win. The GELU item
projector is still a reliable improvement over `cnp` (see below).

#### Full 3-seed ranking

All rows use multilingual-e5-large-instruct. Lower ALC is better;
Beta-Binomial is `0.222838`. Bold `cnp_gelu` is the selected arm;
`cnp_ln_gelu` is #1 by mean but is a control (not reliably better than
`cnp_gelu`).

| # | Rung | Stack | Seeds | Episode | Acq. | ALC | Δβ |
|---:|---|---|---|---|---|---:|---:|
| 1 | `cnp_ln_gelu` | CNP+input LN+GELU projector | 18/19/20 | v4.1 | R | 0.219353 | -0.003485 |
| 2 | **`cnp_gelu`** | CNP+GELU projector | 18/19/20 | v4.1 | R | **0.219412** | -0.003426 |
| 3 | `bcirt_d4_gelu_all` | B-CIRT+full GELU | 18/19/20 | v4.1 | R | 0.219653 | -0.003185 |
| 4 | `cnp` | CNP | 18/19/20 | v4.1 | R | 0.219721 | -0.003117 |
| 5 | `anp_attn` | ANP | 18/19/20 | v4.1 | R | 0.219784 | -0.003054 |
| 6 | `anp_gelu` | ANP+GELU projector | 18/19/20 | v4.1 | R | 0.219795 | -0.003043 |
| 7 | `anp_gelu_all` | ANP+full GELU | 18/19/20 | v4.1 | R | 0.219798 | -0.003040 |
| 8 | `cnp_ln` | CNP+input LN | 18/19/20 | v4.1 | R | 0.219822 | -0.003016 |
| 9 | `bcirt_d4_ln_gelu` | B-CIRT+input LN+GELU projector | 18/19/20 | v4.1 | R | 0.219967 | -0.002871 |
| 10 | `cnp_gelu_all` | CNP+full GELU | 18/19/20 | v4.1 | R | 0.220050 | -0.002788 |
| 11 | `mirt_scalar_d4` | IRT+MD+CNP | 18/19/20 | v4.1 | R | 0.220077 | -0.002761 |
| 12 | `bcirt_d4` | B-CIRT | 18/19/20 | v4.1 | R | 0.220151 | -0.002687 |
| 13 | `mdcnp_d4` | MD-CNP | 18/19/20 | v4.1 | R | 0.220186 | -0.002652 |
| 14 | `direct_loop0` | IRT+CNP+direct+loop0 | 18/19/20 | random | R | 0.220245 | -0.002593 |
| 15 | `mirt_direct_loop0_d4` | IRT+MD+CNP+direct+loop0 | 18/19/20 | random | R | 0.220296 | -0.002542 |
| 16 | `mirt_scalar_d4` | IRT+MD+CNP | 18/19/20 | random | R | 0.220366 | -0.002472 |
| 17 | `scalar` | IRT+CNP | 18/19/20 | random | R | 0.220494 | -0.002344 |
| 18 | `mdanp_d4` | MD-ANP | 18/19/20 | v4.1 | R | 0.220539 | -0.002299 |
| 19 | `additive_loop3` | IRT+CNP+additive+loop3 | 18/19/20 | random | R | 0.220970 | -0.001868 |
| 20 | `mirt_additive_loop3_d4` | IRT+MD+CNP+additive+loop3 | 18/19/20 | random | R | 0.221093 | -0.001745 |

Legend:

- **CNP** = mean-pooled context to Bernoulli head; no IRT, no latent `z`.
- **CNP+GELU projector** = CNP with GELU only in the item encoder; context encoder and decoder stay ReLU.
- **+input LN** = LayerNorm on the item input before the projector.
- **+full GELU** = GELU in the item projector, context encoder, and decoder.
- **ANP** = context self-attention plus target cross-attention to Bernoulli head; no IRT, no latent `z`.
- **IRT+CNP** = 2PL + MAP theta + pooled context.
- **IRT+MD+CNP** = MIRT D=4 + MAP theta + pooled context.
- **MD-CNP** = diagonal Gaussian `q(theta | C)` + item-only MIRT head, probit marginalization; item parameters are point estimates.
- **MD-ANP** = same as MD-CNP, with context self-attention and attention pooling.
- **B-CIRT** = joint diagonal Gaussian `q(theta | C)` and `q(log a_i, d_i | x_i)`, item-only MIRT head, probit marginalization.
- **+direct** = context directly sets item parameters `(a,b)` or `(a,d)`.
- **+loop0** = no-refinement control.
- **+additive+loop3** = additive item offsets + 3 shared transformer loops.
- **Seeds:** `18` = `20261018` only; `18/19/20` = 3-seed mean.
- **Episode:** `v4.1` = full coverage, disjoint rotating splits; `random` = ~7.7% coverage, overlapping.
- **Acq.:** `R` = random fixed hash order; the same 10 frozen orders are used for every row.
- **Δβ** = ALC minus Beta-Binomial.

#### GELU and normalization controls

All rows use e5-instruct and the v4.1 sampler. `cnp_gelu` changes only the
item projector; `*_all` also changes the context encoder and decoder.

| Rung | Stack | Seeds | Episode | Acq. | ALC | Δβ | Δ vs `cnp_gelu` [95% CI] |
|---|---|---|---|---:|---:|---:|---:|
| `cnp_gelu` | CNP+GELU projector | 18/19/20 | v4.1 | R | **0.219412** | -0.003426 | reference |
| `cnp_ln` | CNP+input LN | 18/19/20 | v4.1 | R | 0.219822 | -0.003016 | +0.000409 [+0.000244, +0.000575] |
| `cnp_ln_gelu` | CNP+input LN+GELU projector | 18/19/20 | v4.1 | R | 0.219353 | -0.003485 | -0.000059 [-0.000264, +0.000146] |
| `cnp_gelu_all` | CNP+full GELU | 18/19/20 | v4.1 | R | 0.220050 | -0.002788 | +0.000637 [+0.000333, +0.000942] |
| `anp_gelu` | ANP+GELU projector | 18/19/20 | v4.1 | R | 0.219795 | -0.003043 | +0.000383 [+0.000132, +0.000634] |
| `anp_gelu_all` | ANP+full GELU | 18/19/20 | v4.1 | R | 0.219798 | -0.003040 | +0.000385 [+0.000006, +0.000765] |
| `bcirt_d4_ln_gelu` | B-CIRT+input LN+GELU projector | 18/19/20 | v4.1 | R | 0.219967 | -0.002871 | +0.000555 [-0.000144, +0.001253] |
| `bcirt_d4_gelu_all` | B-CIRT+full GELU | 18/19/20 | v4.1 | R | 0.219653 | -0.003185 | +0.000240 [-0.000711, +0.001192] |

All rows are 3-seed means. `cnp_gelu` and `cnp_ln_gelu` have paired t-intervals
against Beta that exclude zero, but both are marginal: the task bootstrap
interval for `cnp_gelu` includes zero, and `cnp_ln_gelu` is not reliably better
than `cnp_gelu` (paired p=0.59) and has 4x the seed spread. Neither is a
reliable Beta win; `cnp_gelu` remains the selected arm. The MD-CNP/ANP
rows are reliably worse than `cnp_gelu` (paired Δ +0.00077/+0.00113), so a
posterior over theta alone does not help. The joint item+theta posterior does
not recover that gap: `bcirt_d4` is 0.00074 worse than `cnp_gelu` (paired CI
[+0.000035, +0.001441]); the best BCIRT variant, `bcirt_d4_gelu_all`, is
still 0.00024 worse (CI [-0.000711, +0.001192]). `bcirt_d4` per-seed ALCs are
0.220522, 0.219709, and 0.220220 (spread 0.000813).

`cnp_gelu` is lower than Beta at every budget; BCIRT's gain is concentrated at
small budgets and its B31 is slightly worse:

| Arm | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---:|---:|---:|---:|---:|---:|
| `cnp_gelu` | **0.23772** | **0.23366** | **0.22332** | **0.21364** | **0.20662** | **0.20193** |
| `bcirt_d4` | 0.24314 | 0.23479 | 0.22287 | 0.21370 | 0.20678 | 0.20210 |
| Beta-Binomial | 0.25000 | 0.23957 | 0.22662 | 0.21493 | 0.20707 | 0.20200 |

The BCIRT task-level interval still crosses zero and the method remains behind
`cnp_gelu`.

Three readings worth carrying forward:

- **The encoder is the only large effect.** The GELU item projector is a
  smaller but reliable architecture effect: -0.000308 vs `cnp`, bootstrap CI
  [-0.000422, -0.000195], and lower Brier at every budget. It does not produce
  a reliable win over Beta-Binomial. Full-GELU and input LayerNorm are worse,
  so the gain is specific to the item projector.
- **Coverage matters exactly where ability is learned.** `cnp` and `anp_attn`
  must learn adaptation from `r_C` and were training on 7.7% of
  `(task, budget)` pairs; the IRT rungs solve for ability at inference and are
  unaffected. Under the fixed sampler `anp_attn` now beats `scalar`, reversing
  the earlier reading that the IRT anchor was worth about `0.0005`.
- **Context length was not the bottleneck.** bge-m3 truncates 1,838 items at
  8192 tokens against me5's 172,297 at 512 -- 94x fewer -- and still loses to
  it on all five arms.

The remaining gap splits almost exactly in half: `0.0595` from not knowing
`theta` and `0.0590` from not knowing `(a_i, b_i)`. Work so far has addressed
only the second.

### Theta prior, Bayesian MIRT, and Fisher acquisition (2026-10-25)

These ablations use 104 validation tasks, ten frozen acquisition orders, and
five-fold out-of-fold calibration. Lower Brier ALC is better. The prior,
Bayesian MIRT, and acquisition tables use different fitted models, so compare
rows within a table rather than across tables.

#### Gaussian prior variance

| IRT model | $\theta$ prior | calibrated ALC | delta vs Beta-Binomial | 95% interval |
|---|---|---:|---:|---:|
| 2PL | $N(0,1)$ | **0.220335** | -0.002503 | [-0.005398, 0.000994] |
| 2PL | $N(0,4)$ | 0.220922 | -0.001916 | [-0.004786, 0.001471] |
| 2PL | $N(0,9)$ | 0.221125 | -0.001713 | [-0.004546, 0.001609] |
| 2PL | $N(0,16)$ | 0.220762 | -0.002076 | [-0.004966, 0.001389] |
| Rasch | $N(0,1)$ | 0.220990 | -0.001848 | [-0.004403, 0.001131] |
| Rasch | $N(0,4)$ | 0.221883 | -0.000955 | [-0.002891, 0.001167] |
| Rasch | $N(0,9)$ | 0.222363 | -0.000475 | [-0.002253, 0.001451] |
| Rasch | $N(0,16)$ | 0.222577 | -0.000261 | [-0.001998, 0.001613] |
| Beta-Binomial | -- | 0.222838 | -- | -- |

The current evidence supports a zero-centered Gaussian prior for $\theta$,
with $N(0,1)$ as the working default. In the paired 2PL comparison,
$N(0,1)$ improves on $N(0,4)$ by 0.000588 ALC, with a 95% interval of
[-0.000889, -0.000278]. The intervals against Beta-Binomial still cross zero,
so the result supports the tighter variance, not a conclusive overall win.

The more elaborate FLOPs prior does not improve this result. FLOPs are known
for only 40 of 104 validation tasks, and their within-benchmark correlation
with fitted difficulty is 0.187 ($R^2=0.035$). At variance 4, the FLOPs-prior
ALC is 0.221069 against 0.220922 for the zero-mean prior. Keep the Gaussian
prior unless better subject-level features provide a stronger prior signal.

#### Joint online ability and difficulty updates

`mirt_scalar_d4_jointb4_tau1` keeps the multilingual E5 item head fixed and
updates two latent states from each acquired label: the four-dimensional
ability $\theta$ and a four-dimensional benchmark-specific difficulty
correction $z_B$. Its logits are

```text
a_i^T theta - d_i - u_i^T z_B
```

The update is a joint Newton MAP solve with independent $N(0,1)$ priors. The
difficulty loadings $u_i$ are centered over the acquired context, so a constant
benchmark offset cannot duplicate ability. With one acquired label the
centered loading is zero, and only $\theta$ updates.

The first run used the masked multilingual E5 feature store, the v4.1 sampler,
104 validation tasks, ten frozen orders, and random acquisition. Lower ALC is
better.

| Seed | `mirt_scalar_d4`, $\theta\sim N(0,4)$ | joint $\theta,z_B\sim N(0,1)$ | delta |
|---:|---:|---:|---:|
| 20261018 | 0.219945 | 0.219947 | +0.000002 |
| 20261019 | 0.219734 | 0.220100 | +0.000366 |
| 20261020 | 0.220553 | **0.218769** | **-0.001785** |
| Mean | 0.220077 | **0.219605** | **-0.000472** |

The joint configuration beats Beta-Binomial by 0.003233 ALC. Averaging the
three seed differences within each task, it beats scalar MIRT on 76 of 104
tasks; the task-bootstrap interval for the mean delta is
[-0.000645, -0.000294]. The seed result is uneven: seed 18 is tied, seed 19 is
worse, and seed 20 supplies the gain.

This is not yet a clean estimate of difficulty adaptation. The comparison
changes both the online update and the ability prior from $N(0,4)$ to
$N(0,1)$. The required control is `mirt_scalar_d4_tau1`; until it is run, the
0.000472 improvement belongs to the full configuration, not to $z_B$ alone.
The joint Fisher replay is implemented but was not part of this run.

#### Bayesian MIRT encoder and dimension

| Encoder | MIRT dimension | raw ALC | calibrated ALC | delta vs Beta-Binomial |
|---|---:|---:|---:|---:|
| Qwen3-Embedding-8B | 4 | 0.230728 | **0.221604** | -0.001234 |
| Qwen3-Embedding-8B | 2 | 0.233895 | 0.222210 | -0.000628 |
| multilingual-e5-large-instruct | 2 | 0.283388 | 0.222348 | -0.000489 |
| bge-m3 | 2 | 0.367518 | 0.222818 | -0.000020 |
| Qwen3-Embedding-4B | 4 | 0.236008 | 0.222859 | +0.000021 |
| bge-m3 | 4 | 0.345347 | 0.223045 | +0.000208 |
| Qwen3-Embedding-4B | 2 | 0.233758 | 0.223125 | +0.000287 |
| multilingual-e5-large-instruct | 4 | 0.363887 | 0.223217 | +0.000379 |

Qwen3-Embedding-8B with four-dimensional ability is the best Bayesian MIRT
arm. Its interval against Beta-Binomial crosses zero, but it improves on its
two-dimensional counterpart by 0.000605, with a paired 95% interval of
[-0.001059, -0.000098]. Multilingual E5 with two dimensions is the next useful
reference. The working encoder set for Bayesian MIRT is therefore Qwen3-8B
plus multilingual E5; this is not yet evidence that Bayesian MIRT as a whole
beats Beta-Binomial. The large raw-to-calibrated gaps also make calibration a
required part of this comparison.

Dimensions 8 and 16 do not improve on D=4. The best D=8 and D=16 rows are
multilingual E5 at 0.222531 and Qwen3-4B at 0.222759, both worse than
Qwen3-8B D=4 at 0.221604, and every interval against Beta-Binomial crosses
zero. D=16 beats D=8 only for bge-m3 (-0.000425, CI [-0.000640, -0.000210])
and Qwen3-4B (-0.000531, CI [-0.000766, -0.000297]); the other two encoders
move the other way inside zero. Convergence is poor at both dimensions:
R-hat medians are 1.97-2.25 and maxima 5.55-8.17. These are not usable
posterior estimates.

| Encoder | D | calibrated ALC | delta vs Beta | 95% interval | R-hat median / max |
|---|---:|---:|---:|---:|---:|
| multilingual-e5-large-instruct | 8 | **0.222531** | -0.000307 | [-0.002585, +0.001972] | 2.251 / 7.315 |
| Qwen3-Embedding-8B | 8 | 0.222749 | -0.000089 | [-0.001726, +0.001549] | 1.990 / 8.168 |
| bge-m3 | 8 | 0.223243 | +0.000405 | [-0.001576, +0.002385] | 2.143 / 7.057 |
| Qwen3-Embedding-4B | 8 | 0.223290 | +0.000452 | [-0.000985, +0.001889] | 2.003 / 6.048 |
| Qwen3-Embedding-4B | 16 | **0.222759** | -0.000079 | [-0.001441, +0.001283] | 1.975 / 6.261 |
| multilingual-e5-large-instruct | 16 | 0.222813 | -0.000025 | [-0.001901, +0.001852] | 1.999 / 5.553 |
| bge-m3 | 16 | 0.222818 | -0.000020 | [-0.002085, +0.002044] | 2.237 / 7.980 |
| Qwen3-Embedding-8B | 16 | 0.222992 | +0.000154 | [-0.001271, +0.001580] | 1.989 / 5.783 |

#### MIRT acquisition

| Acquisition rule | calibrated ALC | delta vs random | 95% interval |
|---|---:|---:|---:|
| Fisher-D | **0.217602** | **-0.002319** | **[-0.003993, -0.000673]** |
| Fisher-A | 0.217614 | **-0.002307** | **[-0.003998, -0.000647]** |
| Fisher-E | 0.219145 | -0.000776 | [-0.002153, 0.000646] |
| Random | 0.219921 | -- | -- |
| Max-min | 0.219930 | +0.000010 | [-0.000746, 0.000771] |
| Uncertainty | 0.220448 | +0.000527 | [-0.000902, 0.001964] |

Fisher-D and Fisher-A are tied to practical precision, and both beat random.
Fisher-D is the working default because it has the lowest mean ALC. This result
uses one trained checkpoint seed and needs replication. It does not overturn
the earlier null result for CNP threshold policies: the predictor and policy
families are different, and the MIRT policy ranks candidates by an explicit
information criterion.

Current working interpretation:

- Keep a zero-centered Normal prior over $\theta$, provisionally $N(0,1)$.
- Carry Qwen3-Embedding-8B and multilingual E5 forward for Bayesian MIRT.
- Keep D=4 as the working Bayesian MIRT dimension; D=8 and D=16 do not
  improve the calibrated ALC and are not converged.
- Use Fisher-D as the default MIRT acquisition rule, with Fisher-A as the
  closest control.

### Open items

- Shared `x_encoder` for the masked and unmasked paths; the slice `(389, 773)` is hard-coded in ten places in `modal_faithful.py`; the 5 numeric columns are near-dead and unstandardized; the benchmark block is constant within an episode.
- ANP-AIRT and Stacked-MHSA-ANP-AIRT route $r^*$ into the logits through the decoder residual head at scale 0.1. $r^*$ is computed from subject-masked encodings, and the residual is a post-anchor correction; $r^*$ must not enter the MAP fit because context items would attend to their own labels.
- Pilot runs use 32-dim hashed item features and the main runs use the 1157-dim mixed features. Their numbers are not comparable.
- Order of work: deterministic track rerun on all benchmarks (Online-AIRT, CNP-AIRT, Beta-Binomial, 2 seeds), AnchorCalibrator on existing artifacts, organizer baselines plus a Codabench smoke submission, then one latent-track run.
- Encoder re-selection is complete for multilingual E5, bge-m3, Qwen3-4B, and Qwen3-8B. Bayesian MIRT currently favors Qwen3-8B with four-dimensional ability, and the acquisition ablation favors Fisher-D. Both findings need additional trained-model seeds. Not yet done: the achievable-ceiling ladder (refit the oracle 2PL with `b_i` constrained to a function of item text, within-benchmark and then held-out-benchmark, which separates the text-information limit from the transfer limit and decides whether further item-encoder work pays), posterior marginalization at inference (`marginalize_logits` exists in `r_c_attention.py` but no `forward` calls it and `map_theta` still discards the Hessian), within-benchmark centering of `b`, a 3PL lower asymptote, and v4.1 runs at more than one seed.
- The metadata-to-prior channel is closed by the ability-prior ablation; do not reopen it without new subject features. The budget 0-3 level gap remains, but it is not a metadata problem: the benchmark level is already represented by the item net, and the pilot's four subjects make the leave-one-out mean anti-correlated with the held-out subject's base rate.

## Presentation-selection strategy

Our default research choice is the simplest version of the proposed method that
we can validate convincingly. The organizers do not explicitly favor simple
models, but their published criteria put technical soundness, clarity,
reproducibility, and experimental analysis ahead of methodological novelty.
Predictive performance and scientific rigor also count, although the page does
not specify their relative weights or the number of poster slots. It lists five
oral presentations and states that there will be no formal peer review. We
therefore cannot infer or promise that any particular score or report will be
accepted.

This favors a controlled model ladder over an ambitious component that cannot
be tested properly. A modest competitive result with a careful report is more
defensible than a larger model whose gains cannot be attributed or reproduced.
The submission should still meet or beat the organizers' baselines; strong
analysis does not compensate for a predictor that does not work.

For oral or poster consideration, the final package must include:

- A NeurIPS-template technical report with roughly 4-8 pages of main text.
- Released development and submission code, plus any permitted training data
  that differ from `measurement-db`.
- An OpenReview submission containing links to the released materials.
- The selected Codabench submission ID.

The organizers may rerun the released method. The codebase and report are due
by **30 October 2026 at 23:59 Anywhere on Earth (AoE)**. Check the
[current competition page](https://aims.stanford.edu/competition/) before final
submission because the rules may change.

### Minimum report evidence

The report should include:

- A matched adaptation ladder: frozen AmortizedIRT, a regularized MAP ability
  update, CNP-AIRT, optional latent NP-AIRT, ANP-AIRT, and finally RAA-IRT.
  The older base-logit/global-shift/temperature/local-attention decomposition
  is retained as a component ablation inside the neural branch, not as a
  competing model ladder. Report intervals across benchmarks.
- Brier score at every label budget, not only aggregate Brier ALC, so the
  effect of acquired labels is visible.
- Calibration plots, per-benchmark failure analysis, and variation across
  random seeds.
- Negative attention results if the local term fails to help. Do not retain a
  component merely because it sounds novel.
- Disclosure of every training source and any use of LLM-based coding
  assistants.

## Proposed ablations

The proposed Neural Process extension is specified in
[the RAA-IRT design note](docs/raa_irt_design.md), with include-ready TikZ
figures under [`paper/neurips_2026/figures`](paper/neurips_2026/figures). Its
core comparison uses frozen AmortizedIRT and a regularized MAP ability update
as controls, followed by the neural sequence CNP-AIRT, latent NP-AIRT,
ANP-AIRT, and relation-aware ANP-AIRT. Published
GPIRT answers a different question; an optional GP residual adapter over the
same AIRT logits and item embeddings is the matched GP comparison. All of these
extensions are proposed and unimplemented.

Run the organizer-provided PAIEC predictors unchanged as reference baselines.
Record the baseline repository revision, configuration, API model, and random
seeds used for each run. These rows are not part of the cumulative ablation
branch.

The table below is the broader feature and calibration matrix, not the matched
NP-family ladder. Within any reported sub-branch, add one component at a time
and state its parent explicitly. The NCF calibration rows are alternatives:
`NCF + AnchorCalibrator` replaces the built-in Platt recalibration rather than
following it.

| Model name | New component under test |
|---|---|
| [PAIEC Empirical Mean](reference/paiec_baseline/empirical_mean/README.md) | Organizer baseline: mean acquired outcome for the same subject-benchmark pair, with 0.5 at budget 0 and default random acquisition |
| [PAIEC Bayesian Linguistic Evaluator (BLE)](reference/paiec_baseline/ble/README.md) | Organizer baseline: LLM evidence retrieval and belief revision with uncertainty acquisition |
| [PAIEC Empirical Mean + BLE Acquisition](reference/paiec_baseline/empirical_mean_ble_acquisition/README.md) | Organizer baseline: empirical-mean prediction with BLE choosing acquisition items |
| Beta-Binomial | Simple adaptive control condition |
| AmortizedIRT | Item-content embeddings predict IRT parameters |
| AmortizedIRT + Item Metadata | Add `item_features` and `interactors` |
| AmortizedIRT + Cognitive Scales | Add LLM-labelled reasoning and skill features |
| AmortizedIRT + Subject Metadata | Use subject metadata to initialize ability |
| AmortizedIRT + Online Ability Update | Update subject-benchmark ability from acquired outcomes |
| AmortizedIRT + Multidimensional Ability | Replace scalar ability with a cognitive skill vector |
| AmortizedIRT + AnchorCalibrator | Recalibrate logits using acquired anchors |
| AmortizedIRT + AnchorCalibrator + Fisher Acquisition | Select labels using expected Fisher information |
| NCF | Raw subject-item neural predictor |
| NCF + Platt Recalibration | Apply NCF's built-in online scaler |
| NCF + AnchorCalibrator | Replace Platt with grouped shrinkage calibration |

### Illustrative results table

**The values in this table are fabricated. They show the intended reporting
format and must not be cited as experimental results.** `B_n` is the Brier
score after acquiring at most `n` labels. Lower is better for every metric.

| Model | `B_0` | `B_1` | `B_3` | `B_7` | `B_15` | `B_31` | Brier ALC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PAIEC Empirical Mean | .250 | .270 | .235 | .210 | .195 | .185 | .2255 | .137 |
| PAIEC BLE | .205 | .200 | .192 | .185 | .179 | .175 | .1892 | .071 |
| PAIEC Empirical Mean + BLE Acquisition | .250 | .255 | .228 | .202 | .184 | .176 | .2164 | .119 |
| Beta-Binomial | .236 | .231 | .224 | .215 | .207 | .202 | .2192 | .112 |
| AmortizedIRT | .218 | .212 | .204 | .195 | .187 | .181 | .1995 | .084 |
| AmortizedIRT + Item Metadata | .213 | .207 | .199 | .190 | .183 | .178 | .1949 | .078 |
| AmortizedIRT + Cognitive Scales | .207 | .201 | .193 | .184 | .177 | .172 | .1889 | .070 |
| AmortizedIRT + Subject Metadata | .197 | .192 | .185 | .177 | .171 | .167 | .1814 | .061 |
| AmortizedIRT + Online Ability Update | .197 | .188 | .177 | .166 | .158 | .154 | .1729 | .052 |
| AmortizedIRT + Multidimensional Ability | **.193** | .183 | .172 | .161 | .153 | .149 | .1680 | .048 |
| AmortizedIRT + AnchorCalibrator | **.193** | .179 | .168 | .157 | .150 | .147 | .1648 | .030 |
| AmortizedIRT + AnchorCalibrator + Fisher Acquisition | **.193** | **.177** | **.165** | **.153** | **.146** | **.143** | **.1618** | **.028** |
| NCF | .190 | .190 | .190 | .190 | .190 | .190 | .1900 | .076 |
| NCF + Platt Recalibration | .190 | .185 | .179 | .172 | .166 | .163 | .1757 | .041 |
| NCF + AnchorCalibrator | .190 | .182 | .175 | .167 | .160 | .156 | .1714 | .033 |

The illustrative Brier ALC values use the competition weighting:

```text
Brier ALC = 0.1*B_0 + 0.2*B_1 + 0.2*B_3
          + 0.2*B_7 + 0.2*B_15 + 0.1*B_31
```

This layout separates four questions. Item and subject features should improve
the zero-label prediction. Online ability updates should improve later label
budgets. Calibration should reduce ECE and may reduce Brier error. Fisher
acquisition cannot change `B_0`; it should help after labels become available.

## AmortizedIRT: offline and online toy flow

"Offline" means before submission, using public Measurement Data Bank responses.
"Online" means during Codabench evaluation, using only the labels the evaluator
allows us to acquire.

### Offline: learn how content predicts difficulty

For each public item `j`, embed its content `x_j` and use responses from many
public subjects to train a neural difficulty function:

```text
item content -> model input x_j -> predicted difficulty b_j = f_phi(x_j)
```

In a one-parameter IRT model, the fitted success probability is:

```text
p_sj = sigmoid(theta_s - f_phi(x_j))
```

Training jointly estimates public subject abilities `theta_s` and the weights
`phi` of the content-to-difficulty network. A toy fitted function might behave
like this:

| Public item | Model input `x_j` | Predicted difficulty `b_j` |
|---|---:|---:|
| Easy | -1.0 | -1.0 |
| Medium | 0.0 | 0.0 |
| Hard | 1.0 | 1.0 |

The submission stores the trained embedding model and `f_phi`. The simple
online procedure below keeps them fixed.

### Online: estimate one test subject's ability

For a new subject `s` on a hidden benchmark, start with an ability prior:

```text
theta_hat_s = 0
```

When acquisition candidate `j` arrives, estimate its difficulty without seeing
its outcome:

```text
x_j     = Encode(item_j)
b_hat_j = f_phi(x_j)
p_hat_j = sigmoid(theta_hat_s - b_hat_j)
I_j     = p_hat_j * (1 - p_hat_j)
```

`I_j` is the Rasch information score. It is largest near `p_hat_j = 0.5`, where
the outcome is least predictable. Suppose the acquisition policy accepts items
with information near `0.25`:

| Candidate | `b_hat_j` | Current `p_hat_j` | Information | Decision | Revealed outcome |
|---|---:|---:|---:|---|---:|
| A | -1.5 | 0.82 | 0.15 | Reject | Hidden |
| B | 0.1 | 0.48 | 0.25 | Acquire | 1 |
| C | -1.5 | 0.87 | 0.11 | Reject | Hidden |
| D | 0.7 | 0.43 | 0.25 | Acquire | 0 |
| E | 0.2 | 0.48 | 0.25 | Acquire | 1 |

Using a regularized ability update, the illustrative trajectory is:

```text
start                      theta_hat_i*k = 0.00
B succeeds                 theta_hat_i*k = 0.42  <- budget 1
D fails                    theta_hat_i*k = 0.13
E succeeds                 theta_hat_i*k = 0.43  <- budget 3
```

The actual subject ability does not change. Only our estimate changes as labels
arrive.

### Predict the separate evaluation items

For an evaluation item `q` with embedding `e_q = 0.4`, suppose the offline
network gives `b_hat_q = 0.4`. The predictions are then:

```text
budget 0: sigmoid(0.00 - 0.40) = 0.40
budget 1: sigmoid(0.42 - 0.40) = 0.51
budget 3: sigmoid(0.43 - 0.40) = 0.51
```

The evaluator never reveals the evaluation outcome to the predictor. It uses
that outcome later to calculate Brier error.

```text
OFFLINE
public item content + public outcomes
                  |
                  v
learn Embed and content-to-difficulty function f_phi
                  |
                  v
save the fitted components

ONLINE
candidate content x_j
        |
        v
estimate difficulty b_hat_j
        |
        v
score information using current ability theta_hat_i*k
        |
        v
accept or reject candidate
        |
   accepted only
        v
observe y_i*j and update theta_hat_i*k
        |
        v
predict outcomes on separate evaluation items
```

NCF follows a different route. It learns a direct mapping from subject content
and item content to a probability, so it does not explicitly calculate an item
difficulty `b_j`.

## Official sources

- [AIMS competition page](https://aimslab.stanford.edu/competition/)
- [AIMS Foundations repository](https://github.com/aims-foundations/aims)
- [AIMS workshop](https://aimslab.stanford.edu/workshop)

Competition rules can change. The supplied official starting kit and current competition page take precedence over this repository.

## Collaboration

The intended GitHub repository visibility is private until redistribution rights and competition disclosure requirements have been checked. Follow the [collaboration setup](docs/collaboration-setup.md) to give each person separate GitHub, Modal, and WandB access without sharing tokens. See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, pull requests, tests, and rules for private competition material.

## License

Original code in this repository is available under the MIT License. Supplied reference kits, papers, templates, and other third-party files retain their own terms; the MIT License does not relicense them.
