# Person B handoff

## Workspace for new paper runs

Use the shared `aims-neurips` workspace for new E&D paper experiments. Pass
`--profile aims-neurips` explicitly on every Modal launch; do not rely on the
active profile. Save new artifacts to the shared `aims-neurips-data` volume
under `/results/eandd-analysis/`. WP7 already finished there; its summary is
`/results/eandd-analysis/wp7-v1/summary.csv`.

The ID and OOD handoff artifacts described below were produced earlier in
`arshiaswe` and have not been copied to the shared workspace. A new shared run
that needs those exact files must wait until they are available there; do not
silently fall back to running it in `arshiaswe`.

The ID panel is in the user's Modal volume `aims-neurips-data`, under
`/results/eandd-analysis/id-v1/panel-cap20/`, and in the workspace at
`../eandd-data/id-v1/panel-cap20/`. The stand-alone split is in the sibling
`subjects/` directory. Person A should exclude every subject UID in
`heldout_subjects.parquet` from offline fitting, across all training benchmarks.
Use the original training response export for the fit; the ID panel contains
held-out subjects only. The panel uses the legacy replay's columns and marks
its source `benchmark_split=train`; the simulation's ID role belongs in draws.
Person A's new loader has not been tested against it yet.

Source: the arshiaswe canonical-filter cache, checked against the upstream
filter SHA, split seed, rounded source count, and split roles. The manifest
records the source SHA-256 so the team can compare it against Person A's
original export. Only training rows were used for the ID split and fit.
The holdout seed is 20261106. Pool assignment and order use the existing
replay hash recipe. The 20-task cap applies only to ID panel selection;
it does not change the held-out subject list. Only items observed among the
fit subjects within the benchmark are eligible for the ID panel.

Once the original OOD panel and Person A's frozen draws are available:

```text
python organized/testing/eandd/bb_grid.py --responses <export> --panel <ood-panel> --draws <draws.parquet> --output <new-output-directory>
python organized/testing/eandd/analysis.py --scores <score-files> --reference jeffreys --output paper/eandd/data/paired-scores.csv
```

The grid reads validation only for selection; its empirical prior uses training
only, with equal benchmark weights. The comparison reproduces the legacy
equal-task averaging convention: average repeated appearances within each task,
then average across tasks. The bootstrap resamples complete paired tasks.
Test confirmation should run only after the settings and inputs are frozen.
The empirical prior in `paper/eandd/data/empirical-beta-prior.json` uses the
whole training split for **OOD replay only**. Do not use it for the ID positive
control: the latter must exclude all of the ID holdout subjects from every fit.
The legacy item-probe target (0.09 Spearman / 0.53 pairwise) pooled six unseen
benchmarks (validation plus test); report the historical reproduction
separately from a test-only Table 2 comparison.

The documented 0.2494 Jeffreys row is a replay result, not a newly reproduced
score. The checked-in legacy submission template has mean .5, prior strength 4
and subject strength 8; this alone does not identify the actual uploaded ZIP
behind any leaderboard row. Verify an upload ID or package before attributing
its score to those constants.

The character-count result in the plan's open question was located in the
existing local `aims-tabpfn-lora/reports/reference-feature-power50-20261104-v1/`
report and completed JSON. It measures **Brier ALC** from a 50-token reference
feature experiment (24 training tasks, 390 validation tasks), not the paper's
OOD item-difficulty Spearman or pairwise accuracy. Do not use it as Table 2 data.

## WP9: rebuild from the Person A handoff

### Ownership of the replay code

`common.py` and `run_replay.py` are Person A's files. The analysis branch
has no independent edits to either file, and both now match `main` at
`45afc54` after the merge. The differences seen before the merge were
upstream changes: `e6320df` added the ID role, panel routing and separate
frozen draws; `5c4baa2` added the offline-fitted replay, keeping the
training benchmark rates and item difficulties fixed while updating ability.
Future replay changes should land on Person A's branch first.

Download `/results/eandd-analysis/person-a-v1/` from the shared
`aims-neurips` workspace into a private local directory. Then run:

```text
python organized/testing/eandd/wp9.py --handoff <person-a-v1> --output <new-output-directory>
```

This reads saved scores and main-run predictions, checks the replay keys
against the frozen draws, and verifies Brier from the prediction files. It
produces the nine-cell Table 1, matched task-bootstrap summaries, within-task
AUC and 10-bin calibration error, and four native-width vector figures.
Per-budget figures use the same equal-task averaging as the ALC table. No
model is fitted or replayed. Only the competition-shaped test predictions
were supplied, so AUC/calibration are limited to that run. AUC averages only
tasks with both outcomes in their query set.

On the handoff from 45afc54, the 150 prediction shards reproduced 27,000
Brier scores with maximum absolute error 2.61e-8. The three Beta prior curves
are within 0.001 Brier at 31 labels on the competition test set. This is a
descriptive threshold, not a test of statistical equivalence.

The figure PDFs and the small Table 1, AUC/calibration, and audit files are
in `paper/eandd/data/`. Figure 1(b) subtracts Jeffreys at each budget;
Figure 3 shows the training-fitted Beta–Binomial prior versus Jeffreys.
Figure 4 uses the 11 benchmark rows and paired task-bootstrap intervals
from `wp10-lobo-explained.csv`. Negative values favor the fitted prior;
the Spearman correlation with distance from 0.52 is +0.93 on this signed
scale. The distance uses observed benchmark outcomes and is an explanatory
diagnostic, not a feature available before acquiring labels.

To regenerate only the figures from a previously rebuilt WP9 directory,
replace `--handoff <person-a-v1>` with `--figures-from <wp9-directory>`.
