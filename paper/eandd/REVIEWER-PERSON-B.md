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

- R9 uses the WP7 item population, matched sampled pairs and random control,
  with model settings selected by holding out groups of training benchmarks.
  Rasch targets must reproduce the stored WP4b fit or come from its artifact.
- R10 test metrics are complete on the existing prediction handoff. Official
  log-loss ALC is 0.6656 for the fitted Beta-Binomial versus 0.7045 for Jeffreys;
  ten-bin task ECE ALC is 0.1113 versus 0.1426. The CSVs also contain uniform
  and late-only weights, rankable-task counts, and reliability bins. Add
  validation and ID from the R2 predictions before drawing cross-role conclusions.
- R11 needs adaptive-prior and pilot diagnostic outputs from R3/R4.
- R12 adds related work with references verified against the author copies,
  proceedings, or publisher records.
- R13 is optional. R14's split-specific limitations, Figure 4 caption and
  disclosure are updated; the item-difficulty paragraph waits for R9.

Run `split_sensitivity.py --handoff <person-a-v1> --responses <canonical parquet>
--output <new directory>`, then `reviewer_figures.py <that directory>`.
Use `reviewer_metrics.py --handoff <handoff> --runs wp5-competition-test
--output <new directory>` for the original test predictions. These commands
are under `organized/testing/eandd/` and require no Modal execution.
