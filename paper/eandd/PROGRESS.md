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

## WP4b and WP5 (2026-11-06): offline fit and the prior-source matrix
Offline Rasch (`organized/testing/eandd/offline_rasch.py`, run `wp4b-offline-rasch-v1`): 4.34 M training responses
(11 benchmarks, 4,809 subjects, 109,987 items; Person-B's 1,204 held-out subjects removed), joint MAP with empirical-Bayes
hyperparameters averaged per benchmark. **266 s on this machine, so no Modal credit was needed.** Fitted:
m0 = +0.175, tau_b = 0.751, tau_s = 1.376, tau_i = 1.491 (we had been using m0 = 0.049, 2.0, 1.0, 1.0, tuned on
validation). Benchmark rate logits range from -1.35 (appworld) to +1.17 (afrieval). Positive control: at budget 0, the
offline fit alone (theta = 0) scores Brier 0.190 on the held-out ID items against 0.25 for a constant, and the
within-task AUC of its item difficulties is 0.836.

Table 1 matrix, Brier ALC difference to Jeffreys (95% CI, paired over tasks), competition-shaped draws:

| Prior source | Model | Test | Validation | ID (held-out subjects) |
|---|---|---|---|---|
| Fixed in advance | Beta-Binomial (Jeffreys) | reference | reference | reference |
| Fixed in advance | Rasch (m0 = 0, widths 1) | -0.0102 | +0.0031 (n.s.) | -0.0073 |
| Tuned on validation | Beta-Binomial (mean 0.4, strength 1) | +0.0055 | -0.0015 (n.s.) | +0.0041 |
| Tuned on validation | Rasch (re-tuned) | -0.0032 | +0.0004 | -0.0024 |
| Fitted on training | Beta-Binomial (empirical Bayes: mean 0.522, strength 3.68) | **-0.0138** | **+0.0079** | -0.0091 |
| Fitted on training | Rasch | -0.0119 | +0.0052 | -0.0081 |
| Fitted on training | Kernel Rasch | -0.0118 | +0.0035 (n.s.) | -0.0063 |

Files: `paper/eandd/data/wp5-*.csv` (summaries, per benchmark, per budget).

### What this says (my reading; pre-set criteria below)
1. **The Jeffreys baseline was too weak, and the old Rasch gain was mostly shrinkage.** A Beta-Binomial with a prior
   fitted on training (mean 0.52, strength 3.7) is better on test than every Rasch model and four times the old gain
   (-0.0138 vs -0.0032). The gap is at 0-3 labels: Jeffreys has Brier 0.2837 at one label on test, the fitted prior
   0.2460. At 31 labels every model is within 0.001.
2. **A training-fitted prior wins on test and loses on validation, and one benchmark causes the flip.** Test: all three
   benchmarks agree (-0.011 to -0.015). Validation: `ai2d_test` agrees (-0.011), `bird_sql` goes the other way
   (+0.025 for the Beta-Binomial prior). `bird_sql` accuracy is 0.12, below the lowest training benchmark (appworld, 0.26
   in the data, fitted rate 0.21). On the ID panel the fitted prior wins on 9 of 11 benchmarks.
3. **Pre-set criteria:** "fitted priors significantly win OOD" is met on test, so the thesis "offline does not transfer"
   fails as stated; the supported statement is "an offline prior transfers when the new benchmark's base rate is inside the
   training range, and fails outside it (`bird_sql`)". One benchmark outside the range is thin evidence.
4. **Not yet run:** WP6 (offline keeps its fitted mu_b and b_i, fits theta only online) on the ID panel, the real positive
   control for the item-difficulty part. And a held-out-benchmark check for the range claim.

## WP6 (2026-11-06): the offline model that keeps its fitted values
`run_replay.py` model `offline-kept`: benchmark rate and item difficulties from the offline fit, only the subject ability
fit online on the task's own labels (one-dimensional MAP). Benchmarks or items the offline fit never saw fall back to
m0 and 0. Brier ALC difference to Jeffreys, competition-shaped draws (CSVs `paper/eandd/data/wp6-*.csv`):

| Setting | Offline, fitted values kept | Offline prior only (Rasch) | Beta-Binomial, prior fitted on training |
|---|---|---|---|
| ID (held-out subjects, benchmark seen) | **-0.0852** [-0.0956, -0.0746] | -0.0081 | -0.0091 |
| Test (unseen benchmarks) | -0.0115 | -0.0119 | -0.0138 |
| Validation (unseen benchmarks) | +0.0047 | +0.0052 | +0.0079 |

Positive control passes: where the benchmark and its items were seen in training, keeping the fitted values is about
nine times better than any prior-only model. On unseen benchmarks the extra is gone (it falls back to the prior).

Pending: the leave-one-benchmark-out job (`lobo_prior.py`, run `wp10-lobo-v1`), which tests the training-range claim on
all 11 training benchmarks.

## WP10 (2026-11-06): leave-one-benchmark-out for fitted priors (`lobo_prior.py`, run `wp10-lobo-v1`)
For each of the 11 training benchmarks: fit the Beta-Binomial empirical-Bayes prior (my refit reproduces Person-B's full-data
values exactly when benchmarks are weighted equally: 0.5216, 3.676) and the offline Rasch hyperparameters on the other 10
benchmarks (every row of the held-out benchmark removed), then score that benchmark's held-out-subject tasks. Files:
`paper/eandd/data/wp10-lobo-v1.csv`, `wp10-lobo-explained.csv`. Brier ALC difference to Jeffreys, competition-shaped,
one task per draw:

- **Helps** (task-bootstrap CI below 0): `hallusionbench` -0.0151, `bridging_gap` -0.0142, `helm_afr` -0.0139,
  `afrimedqa` -0.0121, `afrieval` -0.0111. **Hurts with a CI above 0: none.** The other six have CIs that include 0
  (3-20 tasks each): `cruxeval` -0.0055, `brace` -0.0042, `agi_elo` +0.0004, `appworld` +0.0026, `alignment_faking`
  +0.0043, `harmbench` +0.0070.
- Macro mean over the 11 benchmarks: Beta-Binomial prior -0.0056, Rasch prior -0.0046. So the average gain is modest.
- **The "outside the training range" explanation was too narrow**: `appworld` is outside the range and does not gain, but
  `alignment_faking` and `harmbench` lose while inside it. What separates them is how far the benchmark's task accuracies
  sit from the prior mean (0.52): Spearman correlation between the gain and mean |task accuracy - 0.52| is **0.93**
  (n = 11), against 0.01 for the benchmark's mean accuracy. Shrinking toward 0.52 helps tasks near 0.52 and hurts tasks
  far from it. This is exploratory (computed after seeing the results, with the evaluation labels), so it is an
  explanation, not a predictor. The test benchmarks (task accuracy 0.54-0.68) sit near the prior mean, which is why they
  look good; `bird_sql` (0.12) is far, which is why validation looks bad.
