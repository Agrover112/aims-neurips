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

## Handover to Person-B for WP9 (2026-11-06)
Uploaded to the shared workspace: `aims-neurips-data:/results/eandd-analysis/person-a-v1/` (125 MB; `MANIFEST.md` inside).
Pull with `MODAL_PROFILE=aims-neurips modal volume get aims-neurips-data /results/eandd-analysis/person-a-v1 <local dir>`.
Contains: all `scores/` (wp3, wp5, wp6, wp8, wp10), `runs/`, the frozen draws, per-prediction files for the main run
(`preds/wp5-competition-test`, for AUC and calibration), and the offline hyperparameters. Not included: other preds
(multi-pair test is 194 MB), subject-level offline fits, the input datasets. Owner approved this transfer on 2026-11-06.

## Merge of Person-B's branch and follow-ups (2026-11-06)
`eandd-analysis` (3adb0ff, d4a9762) merged by fast-forward; `common.py` and `run_replay.py` are identical to main. Their
figures and tables are in `paper/eandd/data/` (`figure1..4`, `wp9-*`, `wp7-*`, `wp2-*`), their sections in
`paper/eandd/person-b-sections.tex`. Table 1, the 27,000-score reconstruction check and WP7 all agree with my numbers.

**ID leak found by Person-B's README, checked and fixed.** The empirical-Bayes Beta prior used in the ID row of Table 1 was
fitted on all training tasks, including the held-out subjects. Refit without them (mean 0.5183, strength 3.650 instead of
0.5216, 3.676) the ID gain is -0.0090 instead of -0.0091: no change in conclusion. Use `bb-eb-clean` for the ID cell
(`wp5b-competition-id-clean.csv`). The LOBO folds were not affected (the whole held-out benchmark is excluded).

**Rasch-terms ladder under the fitted priors** (`wp11-fitted-ladder-*.csv`; the first ladder used the old priors).
Brier ALC difference to Jeffreys, competition-shaped:

| Model (fitted priors) | Test | Validation |
|---|---|---|
| benchmark rate only | -0.0172 | +0.0211 |
| + subject ability | -0.0089 | +0.0029 (n.s.) |
| + item difficulty (= Rasch) | -0.0119 | +0.0052 |
| + kernel | -0.0118 | +0.0035 (n.s.) |
| without ability (rate + item) | -0.0174 | +0.0282 |
| without benchmark rate (ability + item) | -0.0140 | +0.0078 |

Reading: on test the simplest model is best (rate only, -0.0172, better than the fitted Beta-Binomial's -0.0138) and each
added term costs; on validation it is the worst (+0.0211) and the ability term is what rescues it (+0.0029), because it
lets a far-from-prior benchmark (`bird_sql`) move away from the prior. So the subject-ability term is insurance against a
benchmark far from the prior and costs variance when the benchmark is near it: the same bias-variance story as the
leave-one-benchmark-out result. The kernel adds nothing in either role. Multi-pair test gives the same ordering.

**Review of `person-b-sections.tex` (Person-B to fix):** (1) the word "replay" appears 7 times; our term is "online
simulation" (the file names in `code` may keep it); (2) the disclosure section lacks the LLM coding-assistant statement
and the training-data and pretrained-model statements the competition requires; (3) `\hline` tables vs booktabs (minor).
Table 2 values match `wp7-summary.csv`.

## Robustness and paired checks (2026-11-06)
Scripts: `organized/testing/eandd/robustness.py`, `paired_checks.py`. Files: `paper/eandd/data/wp12-*.csv`.

**Draw-seed robustness** (3 fresh sets of 150 competition-shaped draws per role, seeds 20261201-03; the frozen draws are
untouched). Every model's ALC difference to Jeffreys moves by at most 0.0007 across seeds, and the signs and order are the
same as on the frozen draws. Test: fitted Beta-Binomial -0.0137 / -0.0144 / -0.0139, fitted Rasch -0.0120 / -0.0124 /
-0.0123, validation-tuned Beta-Binomial +0.0059 to +0.0063. Validation: fitted Beta-Binomial +0.0078 to +0.0089. The draws
do not drive any result.

**Paired intervals between prior sources** (competition-shaped, tasks as the resampling unit; ID uses the clean prior):

| Comparison (a minus b) | Test | Validation | ID |
|---|---|---|---|
| Rasch fitted - Rasch fixed | -0.0017 [-0.0028, -0.0005] | +0.0021 [+0.0006, +0.0036] | - |
| Kernel fitted - Rasch fitted | +0.0001 (n.s.) | -0.0016 [-0.0021, -0.0012] | - |
| Rasch fitted - Beta-Binomial fitted | +0.0019 [+0.0007, +0.0031] | -0.0027 [-0.0041, -0.0012] | +0.0009 (n.s.) |

Every comparison between prior sources flips sign between test and validation; this is the same benchmark-distance effect.
The kernel only helps (significantly, -0.0016) on validation.

**Label budget after which the prior sources stop differing** (test, difference to Jeffreys; CI excludes zero = differs):

| Model | 0 | 1 | 3 | 7 | 15 | 31 |
|---|---|---|---|---|---|---|
| Beta-Binomial fitted | -0.0047 | -0.0359 | -0.0179 | -0.0093 | -0.0029 | -0.0009 (still differs) |
| Rasch fitted | -0.0085 | -0.0325 | -0.0146 | -0.0064 | -0.0016 | -0.0002 (n.s.) |
| Beta-Binomial validation-tuned | +0.0339 | +0.0082 | +0.0017 (n.s.) | n.s. | n.s. | n.s. |

The gap peaks at one label (Jeffreys jumps to 0.75 or 0.25 after one answer). The fitted Rasch is indistinguishable from
Jeffreys from 31 labels, the validation-tuned prior from 3 labels; the fitted Beta-Binomial still differs by 0.0009 at 31.

## Skipped on purpose (state in Limitations)
- **Item features as a prior mean for Rasch:** the item probes show no transfer on test (neither the embedding nor length
  beats a random control), so the prior mean would carry nothing; not run.
- **Trace entropy and trace embeddings, subject-FLOPS, PCA, trace-length ablation:** traces are unavailable at test time
  and were dropped; FLOPS is not in the subject record; PCA adds nothing to a probe that already shows no transfer.
- **Subject-metadata prior:** tested earlier in the old framework (provider and date priors lost on test); not rerun with the
  fitted-prior matrix.
- **Not "test read once":** test scores were inspected repeatedly while building the ladders and models. Selection was on
  validation only, but the test analyses are exploratory and the paper must say so.

## The paper source is now in git (2026-11-06, owner's instruction)
`paper/neurips_2026/` is excluded locally through `.git/info/exclude` (kept out of the public repository). The owner asked
for the E&D paper to be pushed anyway; the repository is private. Force-added: `paper/neurips_2026/eandd.tex`, the generated
`eandd_*.tex` inputs, `figures/eandd/`, the template `neurips_2026.sty`, and the compiled drafts in `paper/eandd-drafts/`.
Rebuild with `paper/build-eandd.sh "note"` (regenerates the tables from the CSVs, compiles, saves the next numbered PDF;
needs `.tools/tectonic/tectonic`, see `docs/collaboration-setup.md`). Before making the repository public, remove these files
from the history or move the paper to a private repository.

## Reviewer-updates plan, Person-A packages R1, R3, R4 (2026-11-07)
Scripts: `convergence_check.py`, `offline_rasch.py` (new `--maxiter --gtol --tol --configs-out`), `priors.py`,
`adaptive_prior.py`, `pilot_diagnostic.py`, `split_fits.py` (R6, still running when this was written). Files: `paper/eandd/data/r1-*`,
`r3-*`, `r4-*`.

**R1, convergence (E6): done, results unchanged.**
- Online Newton solver: 0 of 6,000 fits hit the 30-step limit, at most 7 steps, last step 0.0 (`r1-newton-convergence.csv`).
- Offline fit: the first fit stopped at the 400-iteration limit in three rounds. Refit with a real stopping rule (5 rounds,
  533 s) the hyperparameters move by +0.9% (m0), -0.1% (tau_b), +0.7% (tau_s), +1.2% (tau_i; just above the 1% bar), the
  benchmark rates by at most 0.011 logit, and no ALC result changes by more than 0.00007 (bar 0.0005; all four settings in
  `r1-tight-vs-earlier.csv`). Table 1 stands.

**R3, adaptive priors (E1): negative result against the criteria fixed in advance.** Gain over Jeffreys (negative is better);
17 benchmarks = 11 leave-one-out + 6 validation/test (`r3-adaptive-17-benchmarks.csv`):

| Model | Worst benchmark | Macro | Test | Verdict |
|---|---|---|---|---|
| fitted Beta prior (reference) | +0.0248 | -0.0050 | -0.0138 | - |
| Beta mixture, 2 or 3 components (EM) | +0.0269 | -0.0054 | -0.0163 | more gain, harm not reduced |
| average over strength at the fitted mean | +0.0122 | -0.0045 | -0.0110 | misses harm bar by 0.0002, test bar by 0.0008 |
| average over mean and strength (nothing learned) | +0.0034 | -0.0037 | -0.0063 | harm removed, gain halved |
| switch to Jeffreys on a far pilot, 3 labels (d = 0.1, chosen on validation) | +0.0136 | -0.0027 | -0.0078 | fails |
| the same with 7 pilot labels | +0.0202 | -0.0042 | -0.0114 | fails |

Criteria were: worst harm at most +0.0120, macro within 0.0010 of -0.0050, test within 0.0020 of -0.0138. No adaptive prior
meets all three. They move along a trade-off between gain and worst-case harm; none dominates the fitted prior. The EM mixtures
(2 and 3 components give the same predictions) have the best macro gain and the worst worst-case. Validation chose the smallest
switch threshold on the grid (d = 0.1), so that rule inherits the validation split's bias.

**R4, pilot diagnostic (E3).** From the first labels of the same order, distance of the pilot accuracy from the prior mean
predicts "the fitted prior hurts this task": pooled AUC **0.732 from 7 labels** (meets the 0.70 bar), 0.610 from 3 labels; the mean
per-benchmark AUC is only 0.636 (0.523 at 3 labels), so most of the signal is between benchmarks. Tasks with a 7-label pilot
distance above 0.4 are hurt 67% of the time (mean gain +0.0174); all closer bins gain on average (`r4-pilot-auc.csv`,
`r4-gain-by-pilot-distance-k7.csv`). As a switch rule with the threshold chosen on the 11 training benchmarks, the best
setting is never to switch (d = infinity): macro on the 6 unseen benchmarks -0.0037, worst +0.0252. Post hoc (not a valid
selection), 3 pilot labels with d = 0.5 gives worst +0.0132, 17-benchmark macro -0.0047, test -0.0122 (`r4-switch-sweep.csv`).
**Why the harm is hard to remove:** it sits at 0-3 labels, which carry half of the ALC weight (0.1 + 0.2 + 0.2), before any pilot
exists. A label-based diagnostic can only act from the budget where it has labels. A diagnostic available at budget zero would
have to come from other subjects' labels on the same benchmark (a benchmark-level pilot): only multi-subject draws have them.
