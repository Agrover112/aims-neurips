# Progress log: Person-A (models and simulation)

Contract and code: `organized/testing/eandd/` (`common.py`, `run_replay.py`, `check_pooling.py`). Large outputs:
`~/aims-replay-cache/paper-eandd/` (`draws.parquet`, `scores/`, `preds/`, `runs/`). Small committed summaries: `paper/eandd/data/`.

## Done
| WP | What | Evidence |
|---|---|---|
| WP1 | Shared contract; frozen `draws.parquet` (single-pair 246 val / 450 test, multi-pair 60 / 150, competition 120 / 150 draws) | `common.py` prints a check that the competition test draws equal the existing ones: True |
| WP3 | `m0` is now a parameter (`cfg[6]`, backward compatible); runner saves scores and every prediction | Reproduces exactly: competition test -0.0032 / -0.0024 / -0.0011, Jeffreys 0.2499; multi-pair test -0.0035 / -0.0049, Jeffreys 0.2497. Saved predictions reproduce scores to 3e-8. 17 s per run |
| WP8 | Rasch-terms ladder via prior widths, test and validation, both shapes | `paper/eandd/data/wp8-ladder-*.csv` |

## Findings that change the plan
1. **Hole 1 confirmed.** In competition-shaped draws the joint Rasch fit equals fitting each task alone (max |dp| 1e-14,
   `paper/eandd/data/wp3-pooling-check.txt`). So Rasch there does not pool labels; its gain comes from its prior. In
   multi-pair draws the two differ (max 0.25), so pooling is real there. The wording "pooling" must be limited to multi-pair.
2. **The benchmark-rate prior flips sign between validation and test.** Pinning the benchmark rate at the training mean
   (`ladder-no-rate`): test -0.0148 [-0.0175, -0.0119] (multi-pair) and -0.0143 (competition); validation +0.0130 and
   +0.0116 (both CIs exclude zero). Cause: mean task accuracy on test benchmarks (0.54-0.68) is close to the training
   mean 0.512, while validation has `bird_sql` at 0.124. So how well a training-mean prior transfers is decided by which
   benchmarks are held out. Do **not** adopt `ladder-no-rate` as an improvement; it was not selected on validation, and
   validation says the opposite.
3. **Consequence for E2.** An offline-fitted prior (spread of training benchmark rates is narrower than the tau_b = 2 we
   use) will probably look good on test and bad on validation. Report both roles side by side. This is the paper's
   point if it holds, and a strong reason to state the three-benchmark limit plainly.
4. Competition shape, validation: nothing separates from Jeffreys except `ladder-no-rate` (worse). The test gains of
   -0.0011 to -0.0032 are small next to the benchmark effect in item 2.

## Next (Person-A)
- WP4b offline Rasch fit: waits for Person-B's held-out subject list (WP4a). The hyperparameter-only part (spreads of
  benchmark rates, abilities, difficulties across the 11 training benchmarks) needs no held-out list and can start.
- WP5, WP6 after WP4b.

## Needed from Person-B
`draws.parquet` is ready (WP1 handoff). Please send the held-out subject list for the ID panel and the Beta-Binomial prior
values from WP2.

## Handover: what is on GitHub and what is not
On GitHub: `Final-plan.md`, this file, `organized/testing/eandd/` (`common.py`, `run_replay.py`, `check_pooling.py`),
`paper/eandd/configs/`, `paper/eandd/data/*.csv` (summary CSVs: reproduction, ladder, pooling check; small),
`paper/eandd/data/draws.parquet` (the frozen draws, 13 KB: task ids and order seeds only; copy it to
`~/aims-replay-cache/paper-eandd/draws.parquet`, or set `AIMS_EANDD_DATA`, so `common.load_draws` finds it).

Not on GitHub, so Person-A hands them over separately:
- The saved scores and predictions (`~/aims-replay-cache/paper-eandd/`; `preds/` is about 117 MB; rerunnable in about 17 s per run), and the
  three input datasets (`~/aims-noitem-run/filtered-responses.parquet`, `~/aims-replay-cache/panel/`, the BGE-M3
  embeddings). These are large and were never meant for git.
- `paper/neurips_2026/` and `paper/versions/`: the paper source and compiled drafts, excluded locally through
  `.git/info/exclude`.

## Received from Person-B (2026-11-06)
- ID panel (114 tasks on the 11 training benchmarks, 10 orders each), held-out subject list, tasks table, Beta-Binomial
  priors: checked, all consistent with the source responses (0 label mismatches, every task subject on the held-out list).
- Placed at `~/aims-replay-cache/paper-eandd/id/`; new role `id` and frozen `draws-id.parquet` (150 draws, one subject
  per training benchmark). `run_replay.py --role id` uses them. First ID run of the online models (competition-shaped,
  not yet the offline Rasch): Jeffreys 0.2357, re-tuned Rasch -0.0024, old-priors Rasch -0.0007, kernel Rasch 0.0000.
- **Beta-Binomial grid** (WP2): prior mean {0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8} x strength {0.25, 0.5, 1, 2, 4,
  8, 16, 32, 64}. Validation picked Beta(0.4, 0.6) for all three shapes: interior on both axes, **not a grid edge**. It is
  Jeffreys' strength with the mean moved to 0.4, which `bird_sql` (accuracy 0.12) pulls on validation; test benchmarks sit
  at 0.54-0.68, so expect it to hurt on test. The empirical-Bayes prior (0.52, strength 3.7) is worse than Jeffreys on
  validation.
- **WP7 probes** (Modal `aims-neurips-data:/results/eandd-analysis/wp7-v1/`; rank correlation / pairwise accuracy):

| Features | ID (11 benchmarks) | OOD validation (3) | OOD test (3) |
|---|---|---|---|
| Item embedding | 0.400 / 0.649 | 0.196 / 0.570 | -0.020 / 0.494 |
| Item length | 0.107 / 0.539 | 0.058 / 0.525 | 0.061 / 0.521 |
| Random control | -0.003 / 0.499 | -0.024 / 0.490 | 0.076 / 0.529 |

  The earlier six-benchmark number reproduces (0.0879 / 0.5317). On test, neither feature beats the random control, so
  the OOD 0.088 in the existing paper is carried by the validation benchmarks, and the control's 0.076 shows how noisy a
  three-benchmark rank correlation is. The existing paper's "0.09, above chance" reading is not safe.
- `split-audit.json` shows the data filter in force: at least 15 responders per item, 80 items per subject, 10 tasks per
  benchmark (6,481 subjects, 123,035 items). Earlier project notes say the standard is 30 responders; check which is intended.
- Not copied: the older files from arshiaswe (blocked pending explicit approval).
