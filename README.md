# AIMS predictive evaluation starter kit

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
submission/              self-contained competition submission
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
python reference/starting_kit/tools/run_smoke_test.py submission/

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
  --template submission_online \
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

Existing stores are kept. `bge_base_2pladapt` remains the best arm measured to
date (validation calibrated ALC 0.222584 against Beta 0.222838) and the residual
2PL adapter will be refitted on whichever new store wins, since that adapter is
the only change that has ever moved the number.

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

Best measured: `anp_attn` under the v4.1 sampler, **`0.219793`**, `-0.003045`
against Beta and `-0.0028` against the previous best arm ever run
(`bge_base_2pladapt`, `0.222584`). That is roughly 5% of the `+0.0621`
headroom.

Three readings worth carrying forward:

- **The encoder is the only robust effect.** Architecture gaps are all smaller
  than the largest per-seed spread (0.00171, `additive_loop3`), which was the
  best arm in the published loop ladder and is now last under the correct
  sampler. Single-seed architecture conclusions on this corpus are not safe.
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

### Open items

- Shared `x_encoder` for the masked and unmasked paths; the slice `(389, 773)` is hard-coded in ten places in `modal_faithful.py`; the 5 numeric columns are near-dead and unstandardized; the benchmark block is constant within an episode.
- ANP-AIRT and Stacked-MHSA-ANP-AIRT route $r^*$ into the logits through the decoder residual head at scale 0.1. $r^*$ is computed from subject-masked encodings, and the residual is a post-anchor correction; $r^*$ must not enter the MAP fit because context items would attend to their own labels.
- Pilot runs use 32-dim hashed item features and the main runs use the 1157-dim mixed features. Their numbers are not comparable.
- Order of work: deterministic track rerun on all benchmarks (Online-AIRT, CNP-AIRT, Beta-Binomial, 2 seeds), AnchorCalibrator on existing artifacts, organizer baselines plus a Codabench smoke submission, then one latent-track run.
- Encoder re-selection is run for me5 and bge-m3 and reported above; the two Qwen arms are still encoding. Not yet done: the achievable-ceiling ladder (refit the oracle 2PL with `b_i` constrained to a function of item text, within-benchmark and then held-out-benchmark, which separates the text-information limit from the transfer limit and decides whether further item-encoder work pays), posterior marginalization at inference (`marginalize_logits` exists in `r_c_attention.py` but no `forward` calls it and `map_theta` still discards the Hessian), within-benchmark centering of `b`, a 3PL lower asymptote, Fisher-information acquisition, and v4.1 runs at more than one seed.
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
