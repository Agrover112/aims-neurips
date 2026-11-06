# Reviewer-updates plan: revising the E&D paper after the workshop review (SR1.pdf)

## Context
The review is AI-generated (Stanford Agentic Reviewer, "may contain errors"). It recommends acceptance, summarises the
paper correctly, and credits the controlled design, the positive control, the open materials and the candour about test
peeking. This plan covers what the owner highlighted (6 points, all in "gaps" and "suggestions") and what we found while
reading it. Nothing below has been run. Costs are estimates; replays take about a minute each, all on local CPU.

## Read first: a finding from the 17 per-benchmark gains already saved (no new run)
Gain of the fitted Beta-Binomial prior over Jeffreys on each of the 17 benchmarks (11 leave-one-out, 6 val/test), then
averaged over every possible set of 3 test benchmarks (680 triples):
- **Our test triple (algotune, cmmlu, editbench) averages -0.0138, the 3rd percentile of all 680 triples.** It is one of the
  most favourable splits we could have drawn.
- Mean over all triples **-0.0050**; 5th to 95th percentile [-0.0133, +0.0056]; the prior **hurts in 18% of triples**;
  it helps by more than 0.005 in 55%. Macro mean over the 17 benchmarks -0.0050; worst benchmark `bird_sql` +0.0248.
- Caveat: LOBO gains use a prior fitted without that benchmark alone, not without the whole triple, so this is an
  approximation (E4 below makes it exact).
The abstract's "gains 0.014 on test" is the best case, not the typical one. The honest headline is the 17-benchmark
average and the spread across splits. This is the single most important revision and it answers the reviewer's
"sensitive to the particular split" point more strongly than they asked.

## Triage of the review
| Review point (orange = highlighted) | Verdict | Action |
|---|---|---|
| **Alternative shrinkage: hierarchical/mixture priors, adaptive empirical Bayes re-estimating (m, s) online after 1-3 labels** (also Q1, suggestions 1-2) | **Do. Core.** It targets the failure we found. But 1-3 labels cannot re-estimate a benchmark mean; a mixture prior whose weights update by marginal likelihood is the principled version (E1). | E1 |
| **Stronger amortized item-feature models from the amortized IRT literature** (also Q2) | **Do, scoped.** Train the probe on the fitted Rasch difficulties (the Truong et al. target), with ridge and a small MLP, under benchmark-grouped controls. | E5 |
| **Calibration, log loss, rank metrics** (also Q3) | **Do.** Rank metrics are uninformative by construction for constant-per-task predictors (within-task AUC is exactly 0.5 for the Beta-Binomial); say so and report them where they are not (kernel Rasch, offline model on ID). | E2 |
| **LOBO is post hoc; pre-deployment diagnostics** (also Q6) | **Do.** Replace the full-label distance by a diagnostic computable from the first labels; evaluate it as a switching rule. | E3 (with E1) |
| **MAP optimisation specifics, convergence checks** | **Do. We have a real gap**: 3 of 4 rounds of the offline fit stopped at the iteration limit (400) before the last warm-started round converged. | E6 |
| **More unseen benchmarks / more OOD splits** (page 4 highlight, suggestion 5) | **Do.** 17-benchmark analysis and exact split-sensitivity (E4). We cannot add benchmarks to the data. | E4 |
| Q4 per-family / per-metadata priors | Cite the earlier result (provider and date priors lost on test; leave-one-out over 14 benchmarks did not confirm) and rerun once in the new design if time allows. | E7 (P2) |
| Q5 more multi-subject draws, Rasch borrowing | The multi-pair draws and ladder exist. State the one-subject-per-benchmark logic more clearly; add a 2-3-subject setting only if E1 needs it. | E7 (P2) |
| "LOBO is based on held-out subjects, not unseen benchmarks" | **Partly a misreading**: in LOBO the prior is fitted with every row of the benchmark removed; only the scored tasks are held-out-subject tasks. | Say so in the Figure 4 caption and Section 4.3 |
| Related work: LSIRM, multilingual IRT, small-area smoothing, prediction-powered inference | Add 3-4 sentences. **Every citation must be verified before use** (see below). | Writing |

## Experiments
All use the existing contract (`organized/testing/eandd/common.py`, `run_replay.py`, frozen draws) and save scores and
predictions. Selection only on validation; criteria below are fixed before running.

| ID | Question | What runs | Protocol | Metrics | Owner | Cost |
|---|---|---|---|---|---|---|
| E1 | Does a prior that adapts to the first labels remove the harm without losing the gain? | (a) **mixture prior**: components fitted offline by EM on training tasks (K = 2 and 3 Beta components), prediction = posterior-weighted mean, weights updated by each component's Beta-Binomial marginal likelihood on the revealed labels (closed form); (b) **adaptive strength**: choose s per task from a grid by marginal likelihood after each label; (c) **switch rule** from E3. Each fitted without the held-out benchmark in LOBO folds | competition-shaped draws on test, validation, ID; LOBO over the 11 training benchmarks; reuse the 17 per-benchmark numbers | ALC difference to Jeffreys and to the fitted prior; per benchmark; worst-case harm | Person-A | 1 day (closed form, then minutes per run) |
| E2 | Do conclusions survive other scores? | Rerun val and ID with predictions saved (only test competition has them now). Log loss, 10-bin calibration error, reliability curves; within-task AUC and Spearman where defined; **ALC weight sensitivity** (uniform weights; weights on 15 and 31 labels) from saved scores | all main models, competition-shaped, test/val/ID | log loss ALC, ECE, AUC, ALC under 3 weightings | Person-A reruns, Person-B analyses | half a day |
| E3 | Can harm be anticipated before deployment? | Per task, estimate distance to the prior mean from the first 3 and 7 labels. Test (i) how well it predicts the realised per-task gain (AUC for "prior hurts this task"); (ii) a rule "use the fitted prior unless the pilot accuracy is more than d from the prior mean, else Jeffreys", d chosen on validation | tasks of all 17 benchmarks; 3 and 7 pilot labels | AUC of the diagnostic, ALC of the rule, per-benchmark harm | Person-A | half a day, from saved scores plus one run |
| E4 | How sensitive is the headline to which benchmarks are held out? | **Exact version**: for each of the 680 triples, refit the Beta prior and the Rasch hyperparameters without the triple (closed form for the Beta prior; 266 s each for Rasch is too slow, so Beta-Binomial exact for all 680 and Rasch for a random 40) and score the triple's tasks from the saved per-task scores. Plot the distribution of the test gain, mark our triple | existing per-task scores for all 17 benchmarks | distribution of gain; fraction of harmful splits; our percentile | Person-B (analysis), Person-A (fits) | half a day |
| E5 | Do stronger item-feature models transfer? | Ridge and a small MLP on BGE-M3 embeddings fitted to the **fitted Rasch item difficulties** (110k training items; the amortized-IRT target), plus the existing pass-rate target; benchmark-grouped selection; random control as before. If a probe passes the earlier criterion (OOD rank correlation at least 0.05 above the control and positive on 4 of 6 benchmarks), run it as a prior mean for item difficulty in Rasch | WP7 protocol, same items | rank correlation, pairwise accuracy; ALC if the criterion passes | Person-B (probes), Person-A (prior-mean option in `fit_predict`) | 1 day |
| E6 | Are the fits converged and do the results depend on it? | Refit the offline Rasch with a much higher iteration limit and a gradient-norm stopping rule; report iterations, final gradient norm, change in hyperparameters; rerun Table 1 with the tighter fit. Count online Newton fits that hit the 30-step limit | offline fit (about 10 min locally) and one replay set | hyperparameter change, ALC change | Person-A | 2 hours |
| E7 (P2) | Metadata or family priors; 2-3 subjects per benchmark | Rerun the metadata-prior Beta-Binomial in the new matrix; draws with 2-3 subjects per benchmark for Rasch vs Beta-Binomial | competition-shaped, multi-pair | ALC difference | Person-B | 1 day |

### Reruns (already-run experiments that change)
- Validation and ID replays: rerun **with predictions saved** (for log loss and AUC). About 1 minute each.
- Offline Rasch fit: rerun with tight convergence (E6). Table 1 follows.
- Abstract and Table 1 text: re-derived after E4.

### Criteria fixed before running
- **E1 adaptive prior succeeds** if, over the 17 benchmarks, (i) the worst per-benchmark harm falls from +0.0248 to at most +0.0120, and
  (ii) the macro gain is no worse than the fitted prior's (-0.0050) by more than 0.0010, and (iii) the test gain stays within
  0.0020 of -0.0138. Otherwise we report it as not helping.
- **E5 probe passes** if it clears the item-feature criterion above on the 6 unseen benchmarks. Otherwise report the null with the
  stronger probe, which is the answer to Q2.
- **E6** is fine if hyperparameters move by less than 1% and no ALC difference changes sign or by more than 0.0005.
- **E3 diagnostic is useful** if its AUC for "prior hurts this task" is at least 0.70 from the first 7 labels.

## Paper changes
1. Abstract and conclusion: lead with the 17-benchmark mean and the spread across splits; state that our test split is
   favourable (3rd percentile); keep -0.0138 as the best case.
2. New figure: distribution of the test gain over all splits (E4); extend Figure 4 to 17 benchmarks.
3. New results paragraph: adaptive and mixture priors (E1) and the pilot diagnostic (E3), answering Q1 and Q6.
4. Metrics: one table with Brier, log loss, calibration error (E2), and a sentence on why rank metrics are uninformative by construction.
5. Implementation paragraph or appendix: MAP details (Newton in the submission, L-BFGS in the simulation, iteration limits,
   convergence audit E6).
6. Figure 4 caption: the prior is fitted without any row of the held-out benchmark.
7. Related work (3-4 sentences), shrinkage and empirical Bayes (Efron and Morris; Robbins), small-area estimation (Rao and
   Molina), prediction-powered inference (Angelopoulos et al.), IRT variants (LSIRM; multilingual IRT), and the amortized IRT
   line. **Verify each reference exists and is cited correctly before adding it**; do not cite from memory.
8. Limitations: keep the test-peeking and three-benchmark statements; add that the 17-benchmark view reuses held-out
   subjects for 11 of the benchmarks.

## Work packages (the next steps, by person)
The shared contract of `Final-plan.md` still applies: frozen draws, the file schemas in `organized/testing/eandd/common.py`,
new run names every time (`save_run` refuses to overwrite), selection on validation only, small CSVs committed with
`git add -f paper/eandd/data/...`, large files shared through `aims-neurips-data:/results/eandd-analysis/<run>/`.
Branches: `eandd-models` (Person-A) and `eandd-analysis` (Person-B); merge after each handoff.

### Person-A (models and simulation)
| WP | Task | Needs | Effort | Done when |
|---|---|---|---|---|
| R1 | **E6 convergence audit.** Refit the offline Rasch with a high iteration limit and a gradient-norm stopping rule (`offline_rasch.py`: add `--maxiter`, `--gtol`, log final gradient norm); count online Newton fits that hit the 30-step limit (`kernel_rasch_replay.fit_predict`); rerun Table 1 with the tight fit | - | 2 hours | hyperparameters move < 1%, no ALC change > 0.0005 (else report the new numbers and rebuild Table 1) |
| R2 | **E2 reruns.** Rerun the validation and ID competition-shaped replays with predictions saved (`run_replay.py`, without `--no-preds`) for the models in Table 1 plus `offline-kept`; push to the shared volume | R1 (so the reruns use the final fit) | 1 hour | `preds/` for val and ID on the shared volume; Person-B notified |
| R3 | **E1 adaptive priors.** New `organized/testing/eandd/adaptive_prior.py`: (a) EM fit of a K-component Beta mixture to training tasks' (hits, n) counts, K = 2, 3, equal benchmark weights, refit without the held-out benchmark in every LOBO fold; (b) closed-form posterior-weighted prediction; (c) per-task adaptive strength by marginal likelihood. Register the models in `run_replay.py` (a new model kind `("mixture", ...)`); run on test, validation, ID and the 11 LOBO folds (`lobo_prior.py`) | R1 | 1 day | scores for all runs; the E1 criteria evaluated and written to `PROGRESS.md` |
| R4 | **E3 pilot diagnostic and switch rule.** From saved per-task scores and the first 3 and 7 labels, compute pilot distance to the prior mean per task; AUC for "fitted prior hurts this task"; switch rule `use fitted prior unless pilot accuracy is more than d from the prior mean, else Jeffreys`, d chosen on validation; run it as a model | R3 (shares the model registry) | half a day | AUC and ALC of the rule per benchmark in `paper/eandd/data/` |
| R5 | **E5 prior-mean option.** Add an optional per-item prior mean to `fit_predict` (`cfg` entry: path to a vector keyed by (benchmark, content_hash)); run it with Person-B's predicted difficulties only if the probe passes its criterion | Person-B's R9 | 2 hours | scores, or a written null |
| R6 | **Exact split sensitivity fits (for E4).** For random 40 of the 680 test triples refit the Rasch hyperparameters without the triple (`offline_rasch.py --exclude a,b,c`, about 4 minutes each, run 3 in parallel) and score the triple | - | half a day (runs in the background) | fits and scores saved; Person-B plots |
| R7 | **Paper sections.** Update Models and prior sources (mixture and adaptive priors, convergence details) and the Results text for E1, E3, E6; update the abstract and conclusion with the 17-benchmark headline | R1-R6 | 1 day | `eandd.tex` rebuilt, at most 8 pages |

### Person-B (data, analysis, figures)
| WP | Task | Needs | Effort | Done when |
|---|---|---|---|---|
| R8 | **E4 analysis.** All 680 test triples for the Beta-Binomial prior: refit the empirical-Bayes prior without the triple (closed form from training counts), score the triple from saved per-task scores; with Person-A's R6 for Rasch on 40 triples. New figure: distribution of the test gain over splits with our triple marked (3rd percentile) and the fraction of harmful splits; extend Figure 4 to the 17 benchmarks | saved scores (have them); R6 for the Rasch part | 1 day | figure and `wp13-split-sensitivity.csv` committed; numbers in the abstract checked |
| R9 | **E5 probes.** Ridge and a small MLP on BGE-M3 embeddings fitted to the **fitted Rasch item difficulties** (`fits/wp4b-offline-rasch-v1.parquet`, kind `b`) and to pass rate; benchmark-grouped selection; random control; same items and metrics as `wp7-summary.csv`; apply the criterion | WP7 code (`probes.py`) | 1 day | Table 2 extended; a pass/fail line for the criterion |
| R10 | **E2 analysis.** From the preds of R2: log loss ALC, 10-bin calibration error, reliability curves, within-task AUC and Spearman where defined, ALC under three weightings (uniform; late-budget heavy; the official) from saved scores. One table and one appendix figure | R2 | 1 day | `wp13-metrics.csv`, table generated by `paper/eandd/assemble.py` |
| R11 | **Figures for E1 and E3.** Per-benchmark harm before and after the adaptive prior, and the pilot-diagnostic AUC; same style rules (`plotstyle.py`, key on top, native width, no scaling in LaTeX) | R3, R4 | 1 day | figures in `paper/eandd/data/` |
| R12 | **Related work and references.** 3-4 sentences on shrinkage and empirical Bayes, small-area estimation, prediction-powered inference, IRT variants (LSIRM, multilingual IRT), the amortized IRT line. **Verify each reference first** (author, venue, year); write it into `paper/neurips_2026/eandd.tex` as the new Related work paragraph | - | half a day | each reference verified and listed |
| R13 | **E7 (only if time).** Metadata-prior Beta-Binomial in the new matrix; 2-3 subjects per benchmark for Rasch vs Beta-Binomial | R1 | 1 day | rows added, or stated as future work |
| R14 | **Paper sections.** Update Setting, Item-difficulty transfer, Limitations and the disclosure for the new analyses; Figure 4 caption clarifying that the prior is fitted without the held-out benchmark | R8-R10 | 1 day | sections merged into `eandd.tex` |

### Handoffs (the only waits)
1. R1 (Person-A) to R2, R3: final offline fit.
2. R2 (Person-A) to R10 (Person-B): predictions on the shared volume.
3. R6 (Person-A) to R8 (Person-B): Rasch fits for the triples.
4. R9 (Person-B) to R5 (Person-A): predicted item difficulties, only if the probe passes.
5. R3 and R4 (Person-A) to R11 (Person-B): scores for the new models.
6. Both: sections into `eandd.tex` (one section each), then Person-A rebuilds with `paper/build-eandd.sh`.

### Checkpoints
| Checkpoint | Contains | Decision |
|---|---|---|
| D1 | R1, R8 (Beta-Binomial part) | Does the fit converge, and is the headline the 17-benchmark mean? Rewrite the abstract if so |
| D2 | R2, R10, R3 | Do conclusions survive log loss and calibration? Does the adaptive prior meet its criteria? |
| D3 | R4, R6, R9, R11 | Is there a deployable diagnostic? Does a stronger probe transfer? |
| D4 | R7, R12, R14 | Submit |

Rough total: 3 to 4 working days each, in parallel, no GPU and no Modal spend. Cut in this order if time runs short:
R13, then R6 (use the 17-benchmark approximation and say so), then R9.

## Optional, high value if still possible
Submit the fitted Beta-Binomial (mean 0.52, strength 3.7; a few lines of code) to CodaBench: its hidden test set is a real
unseen-benchmark check on about 7 benchmarks. The leaderboard compares different pairs per submission, so it is only a
sanity check. Needs the owner's token and a check that submissions are still open (the deadline was 30 October).

## Verification
- Before new results: the replay must still reproduce the stored numbers (-0.0138, -0.0119, -0.0032, -0.0024; Jeffreys 0.2499).
- E1 and E3 fitted without the evaluated benchmark in every LOBO fold; the ID Beta-Binomial prior stays the clean one.
- Every new figure rendered and checked; numbers in the paper generated from the CSVs through `paper/eandd/assemble.py`.
- Recompile with `paper/build-eandd.sh`; main text at most 8 pages.

## Decisions for you
1. Do we reframe the headline around the 17-benchmark average now, before the new experiments? I recommend yes.
2. Is a CodaBench submission of the fitted Beta-Binomial still possible and wanted?
3. Do we spend the extra day on E7 (metadata priors, 2-3 subjects), or leave it as stated future work?
