# Reviewer updates: Person B handoff

## R8: exact Beta-Binomial splits

All 680 triples use the same eligibility rules, label budgets, frozen outcomes
and acquisition orders as the saved WP5 (validation and test) or WP10 (original
training benchmarks, held-out subjects) trajectories. Each prior is refitted
by benchmark-weighted marginal likelihood on response counts from all 14
benchmarks outside its held-out triple. For a constant probability, Brier is
quadratic in the probability and linear in the query success rate; the paired
saved scores uniquely recover the query rate and acquisition prefix counts.
The script reconstructs both Brier curves and checks the five historical
reference scores before computing the split distribution.

| Protocol | Original test names | Percentile | Mean macro difference | 5th--95th percentile | Splits with harm |
|---|---:|---:|---:|---:|---:|
| Approximation reusing the saved gains | -0.01379 | 2.79% | -0.00496 | [-0.01334, +0.00560] | 17.79% |
| Refit each triple on the other 14 | -0.00874 | 12.79% | -0.00367 | [-0.00947, +0.00989] | 17.50% |

Negative is better than Jeffreys. The original paper's result used only the
11 original training benchmarks; the refit for the same test names also uses
the original validation benchmarks. Do not label the 3rd percentile as the
exact result. Macro means first give equal weight to tasks within a benchmark,
then equal weight to benchmarks. A task-weighted result is also in the CSV.
The 17-point explanation plot uses saved per-benchmark gains and a common
query-rate distance, so it is distinct from the split refits and from the
original 11-point figure. The R6 handoff has now been analyzed:
`data/r8-r6-split-sensitivity.csv` and
`data/figure-r8-r6-split-sensitivity.pdf`. R6 fits only on the remaining
original training benchmarks, so its Beta comparator is not the
14-benchmark refit above.

| R6 model, 40 triples | Mean macro difference | 5th--95th percentile | Harmful splits | Original trio |
|---|---:|---:|---:|---:|
| Beta-Binomial | -0.00514 | [-0.01282, +0.00667] | 20.0% | -0.01362 |
| Rasch | -0.00407 | [-0.00947, +0.00369] | 22.5% | -0.01186 |

The original trio is marked separately and is not added to the random sample.
R6 scores all available frozen orders rather than the original WP5 competition
draw frequencies, which accounts for the small difference in its original-trio
Beta result. These numbers reproduce Person A's supplied triple-gain CSV
to its five-decimal rounding.

## R9--R14

- R9 ran on the matched WP7 items and pairs, with settings selected in
  training-benchmark folds. The pass-rate MLP passed the planned pooled
  rank criterion (Spearman 0.090, 5/6 positive, random 0.026); its Ridge
  counterpart did not. **The Rasch-target MLP is now confirmed on Person A's
  original trimmed WP4b fit:** Spearman 0.102648, gap to random 0.076843,
  positive on 5/6 benchmarks. Its Ridge counterpart failed. These final
  original-fit numbers replace the earlier provisional local-refit result.
  The updated criterion JSON and per-benchmark scores are in `data/wp13-probe-*`.
  These are rank metrics, not downstream Brier gains.

  R5 exports are uploaded to shared workspace **aims-neurips**, volume
  **aims-neurips-data**, at
  `/results/eandd-analysis/person-b-r9-original-v1/`:

  - `r5-pass_rate_mlp-item-difficulties.parquet`
  - `r5-rasch_difficulty_mlp-item-difficulties.parquet`
  - `MANIFEST.md`, `MANIFEST.json`, `wp13-probe-criterion.json`
  - `probe-training-items.parquet` (exact training-item IDs).

  Each export has exactly 13,048 unique `benchmark_id,content_hash` keys,
  covering every acquisition and evaluation item in the supplied validation
  and test panel. Missing and extra keys: zero. **Both columns are Rasch
  difficulty, positive means harder; subtract from the success logit.**
  Rasch-target predictions are negated easiness. Pass-rate predictions are
  converted to logit-scale difficulty using the training baseline; the exact
  formula is in the manifest. The signed exports also pass the rank criterion.
  Both probes fit 8,462 WP7 sampled items from the 11 training benchmarks only,
  with no validation/test fitting or tuning. Uploaded files were verified by
  SHA-256 against the local checked exports. See `r5-original-fit-MANIFEST.md`.
- R10 is complete for test, validation and ID. Official
  log-loss ALC is 0.6656 for the fitted Beta-Binomial versus 0.7045 for Jeffreys;
  ten-bin task ECE ALC is 0.1113 versus 0.1426. The CSVs also contain uniform
  and late-only weights, per-budget AUC and Spearman for rankable tasks,
  rankable-task counts, and reliability bins. The validation result reverses:
  fitted versus Jeffreys log-loss ALC is 0.5404 versus 0.5159, and ECE ALC
  is 0.1677 versus 0.1405. The ID offline Rasch reaches 0.836 within-task
  AUC at 31 labels. All 270 uploaded R2 shards were scored and their Brier
  values checked against the saved scores. See `data/wp13-metrics.csv`,
  `data/wp13-metric-alc.csv`, and the generated paper appendix table.
- R11 figures are `data/figure-r11-adaptive-harm.pdf` (17 benchmarks,
  fitted prior and three adaptive variants) and `data/figure-r11-pilot-auc.pdf`
  (three and seven pilot labels; pooled and benchmark-macro AUC).
  The seven-label pooled AUC reaches 0.732, while benchmark-macro AUC is 0.636.
- R12 adds related work with references verified against the author copies,
  proceedings, or publisher records.
- R13 is optional. R14's split-specific limitations, Figure 4 caption and
  disclosure are updated; the item-difficulty paragraph and table have not
  yet been updated with the new R9 results.

Run `split_sensitivity.py --handoff <person-a-v1> --responses <canonical parquet>
--output <new directory>`, then `reviewer_figures.py <that directory>`.
Use `reviewer_metrics.py --handoff <handoff> --runs wp5-competition-test
--output <new directory>` for the original test predictions. These commands
are under `organized/testing/eandd/` and require no Modal execution.
