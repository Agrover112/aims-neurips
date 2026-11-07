# What Offline Fitting Buys Under Benchmark Shift

Code, results and paper for **What Offline Fitting Buys Under Benchmark Shift: Predicting LLM Item Outcomes from a Few Labels**
(Arshia Teymourtash and Ankit Grover, KTH Royal Institute of Technology), written for the NeurIPS 2026 Evaluations and Datasets track.

**Paper:** [`paper/eandd-drafts/eandd-v85.pdf`](paper/eandd-drafts/eandd-v85.pdf)

## What the paper asks
Given a few labelled answers of an LLM on a new benchmark, what can we fit offline on past benchmarks that still helps?
We compare Beta-Binomial, Rasch and kernel Rasch predictors that share tasks, labels and order, and change only where the prior comes from
(fixed in advance, tuned on validation, or fitted on training benchmarks), in an online simulation of the
Predictive AI Evaluation Competition.

Main findings:
- Offline item-level estimates help hugely on items seen in training (ID: ΔBrier ALC −0.0852 against Jeffreys).
- On unseen benchmarks only a shrinkage prior transfers, and its gain is modest and split-dependent: −0.005 averaged over 17 benchmarks, −0.004 over all 680 choices of three held-out benchmarks, harmful in 17.5% of them.
- The gain tracks how far a benchmark's accuracy lies from the prior mean (Spearman 0.96 over 17 benchmarks).
- Neural probes on item embeddings carry a small rank signal for item difficulty that does not improve prediction as a prior mean.
- Whether sharing labels across subjects helps is untested: each competition-shaped draw has one subject per benchmark.

## Competition entry
The final Codabench submission is **`1005155`**: [`dist/submission.zip`](dist/submission.zip), a Beta-Binomial predictor with a fixed prior and
subject-level shrinkage. Its source is `organized/submission/legacy/submission/`. Rebuild and check it with:

```bash
python scripts/build_submission.py
python reference/starting_kit/tools/check_submission_zip.py dist/submission.zip
```

The paper's own model (Beta-Binomial with the training-fitted prior) is a different predictor; its results come from the online simulation, not from the leaderboard.

## Repository layout
| Path | Contents |
|---|---|
| `paper/neurips_2026/eandd.tex` | LaTeX source of the paper; generated table inputs `eandd_*.tex`; `figures/eandd/` |
| `paper/eandd/` | `assemble.py` (builds the tables from the CSVs), `data/` (small result tables), `PROGRESS.md` (running log of results) |
| `paper/build-eandd.sh` | Rebuilds the tables, compiles the paper, saves the next numbered draft in `paper/eandd-drafts/` |
| `organized/testing/eandd/` | Online simulation, models, probes, robustness and split-sensitivity code |
| `organized/submission/` | Competition submissions and trained checkpoints |
| `dist/submission.zip` | The final Codabench entry |
| `literature/`, `docs/` | Papers used and design notes; the long development log is `docs/development-notes.md` |
| `Final-plan.md`, `Reviewer-updates-plan.md` | Work packages and the response to the workshop review |

## Reproducing the paper
```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev,experiment]'
paper/build-eandd.sh        # needs tectonic at .tools/tectonic/tectonic (not in git)
```
The tables and figures are rebuilt from the committed CSVs in `paper/eandd/data/`. A single run of the online simulation, for example:

```bash
.venv/bin/python organized/testing/eandd/run_replay.py --run my-check --shape competition --role test \
    --models jeffreys,rasch-retuned,kernel-old --no-preds
```

## Data
The competition response data cannot be redistributed and is not in this repository. The experiments read it from local paths
(`AIMS_RESPONSES`, `AIMS_REPLAY_CACHE`, `AIMS_EMBEDDINGS`, `AIMS_EANDD_DATA`; see `docs/collaboration-setup.md`).
The frozen draws (`paper/eandd/data/draws.parquet`), the small result tables and the saved checkpoints are included.
Run names are new every time: `save_run` refuses to overwrite an existing run.

## Contributing and license
See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`LICENSE`](LICENSE). Do not commit credentials, hidden outcomes or data whose licence does not allow redistribution.
