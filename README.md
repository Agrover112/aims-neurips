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

## Build and check a submission

```bash
python scripts/build_submission.py
python reference/starting_kit/tools/check_submission_zip.py dist/submission.zip
python reference/starting_kit/tools/run_smoke_test.py submission/
```

The generated archive has `model.py` and optional `labeling.py` at its root, as required by the official kit. Generated ZIPs, results, credentials, and local data are excluded from Git.

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
