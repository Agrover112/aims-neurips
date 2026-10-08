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

Read [assumptions and limitations](docs/assumptions.md) before interpreting results. The preliminary literature review is input to hypothesis generation only. It does not limit candidate methods, and its claims require independent verification.

## Official sources

- [AIMS competition page](https://aimslab.stanford.edu/competition/)
- [AIMS Foundations repository](https://github.com/aims-foundations/aims)
- [AIMS workshop](https://aimslab.stanford.edu/workshop)

Competition rules can change. The supplied official starting kit and current competition page take precedence over this repository.

## Collaboration

The intended GitHub repository visibility is private until redistribution rights and competition disclosure requirements have been checked. Invite collaborators through GitHub's repository settings. See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, pull requests, tests, and rules for private competition material.

## License

Original code in this repository is available under the MIT License. Supplied reference kits, papers, templates, and other third-party files retain their own terms; the MIT License does not relicense them.
