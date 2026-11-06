# Person B handoff

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
