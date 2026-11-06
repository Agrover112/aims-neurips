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
