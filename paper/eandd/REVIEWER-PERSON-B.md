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
original 11-point figure. Rasch on 40 triples waits for Person A's R6.

## R9--R14

- R9 ran on the matched WP7 items and pairs, with settings selected in
  training-benchmark folds. The pass-rate MLP passed the planned pooled
  rank criterion (Spearman 0.090, 5/6 positive, random 0.026); its Ridge
  counterpart did not. The MLP using Rasch targets also passed on the local
  refit (0.112, 6/6 positive); its Ridge counterpart did not. On the three
  test benchmarks alone, Spearman was 0.109 for the pass-rate MLP, 0.104
  for the Rasch-target MLP, and 0.076 for the sampled random control.
  These are rank metrics, not downstream Brier gains. The Rasch results
  are provisional: the local refit matched the original response and
  item counts but used an item scale of 1.507 versus 1.491 in the stored
  WP4b fit. Aggregate outputs are in `data/wp13-probe-*`.
- R10 test metrics are complete on the existing prediction handoff. Official
  log-loss ALC is 0.6656 for the fitted Beta-Binomial versus 0.7045 for Jeffreys;
  ten-bin task ECE ALC is 0.1113 versus 0.1426. The CSVs also contain uniform
  and late-only weights, rankable-task counts, and reliability bins. Add
  validation and ID from the newly announced R2 prediction handoff before
  drawing cross-role conclusions.
- R11 figures remain to be built from Person A's newly merged R3/R4 outputs.
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
