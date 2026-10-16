# AIMS predictive evaluation starter kit

This repository is a collaboration-ready research workspace for the AIMS Predictive AI Evaluation Challenge (PAIEC). It preserves the official submission interface, adds an auditable adaptive baseline, reproduces the published Brier ALC calculation, and provides hooks for repeatable local, Modal, and Weights & Biases experiments.

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

For Modal, install the `modal` extra, authenticate, and run:

```bash
modal run modal_app.py --predictions examples/predictions.json
```

## Scientific workflow

Use grouped splits so repeated outcomes for the same subject-item pair never cross train and validation sets. Compare every adaptive method with random acquisition and a predictor that ignores labels. Report Brier score and calibration by budget, aggregate them with the published ALC weights, and retain uncertainty across seeds or folds.

Read the [concept dictionary](docs/concept-dictionary.md) for the competition's terminology and flow. Read [assumptions and limitations](docs/assumptions.md) before interpreting results. The preliminary literature review is input to hypothesis generation only. It does not limit candidate methods, and its claims require independent verification.

## AmortizedIRT: offline and online toy flow

"Offline" means before submission, using public Measurement Data Bank responses.
"Online" means during Codabench evaluation, using only the labels the evaluator
allows us to acquire.

### Offline: learn how content predicts difficulty

For each public item `j`, embed its content `x_j` and use responses from many
public subjects to train a neural difficulty function:

```text
item content x_j -> embedding e_j -> predicted difficulty b_j = f_phi(e_j)
```

In a one-parameter IRT model, the fitted success probability is:

```text
p_ij = sigmoid(theta_i - f_phi(e_j))
```

Training jointly estimates public subject abilities `theta_i` and the weights
`phi` of the content-to-difficulty network. A toy fitted function might behave
like this:

| Public item | Embedding `e_j` | Predicted difficulty `b_j` |
|---|---:|---:|
| Easy | -1.0 | -1.0 |
| Medium | 0.0 | 0.0 |
| Hard | 1.0 | 1.0 |

The submission stores the trained embedding model and `f_phi`. The simple
online procedure below keeps them fixed.

### Online: estimate one test subject's ability

For a new subject `i*` on hidden benchmark `k`, start with an ability prior:

```text
theta_hat_i*k = 0
```

When acquisition candidate `j` arrives, estimate its difficulty without seeing
its outcome:

```text
e_j     = Embed(x_j)
b_hat_j = f_phi(e_j)
p_hat_j = sigmoid(theta_hat_i*k - b_hat_j)
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
