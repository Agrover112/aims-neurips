# Final plan: offline vs online fitting for unseen LLM benchmarks

Short paper, NeurIPS template with the Evaluations & Datasets option (`\usepackage[eandd]{neurips_2026}`), at most 8 pages
of main text. Nothing below has been run. Effort figures are my estimates. Replay runtimes are unmeasured.

## 1. The paper in one paragraph
**Question:** when predicting item outcomes on an unseen benchmark from a few labels, does anything fitted offline help
beyond what is fitted online? **Design:** one model family, same tasks, same labels in the same order; only the source of
the prior changes. **Positive control:** offline must win in-distribution (ID), where its fitted values apply. No neural
models, so the claim covers the Beta-Binomial and the Rasch family only.

**Fixed before running:**
- **Claim holds:** fitted-on-training is not significantly better than fixed-in-advance on OOD test, and offline wins ID.
- **Claim fails:** fitted priors win OOD. We then report "training transfers a useful prior".
- **Control fails:** offline does not win ID. Debug before reading anything else.
- **Tuned Beta-Binomial matches Rasch:** we say so; it becomes a finding.

**To check first (my reasoning from the equations, tested in WP2 and WP3):**
1. With one subject per benchmark, "pooled" Rasch splits into independent per-task fits, so its gain there would come
   from its prior, not from sharing labels.
2. Jeffreys is a weak baseline: it ties a constant on test.
3. The leaderboard Beta-Binomial may use a different prior from the one we simulate.

## 2. What goes in the paper
| Item | Where | Source |
|---|---|---|
| Table 1: prior source x model (competition-shaped, test) | main | WP5 |
| Figure 1: Brier per budget by prior source; caption states how many labels until the source stops mattering | main | WP5, WP9 |
| Figure 2: offline vs online Rasch, ID and OOD panels | main | WP6, WP9 |
| Table 2: item-difficulty transfer (random, item length, item embedding) | main | WP7 |
| Baseline strength: one sentence and the reference row | main | WP2 |
| Baseline heatmap; Rasch-terms table; question-text entropy row; per-benchmark breakdown; leaderboard scores (labelled unpaired) | appendix | WP2, WP8, WP7, WP9 |

**Table 1 cells** (Brier ALC difference to Jeffreys, paired 95% CI):

| Prior source | Beta-Binomial | Rasch | Kernel Rasch |
|---|---|---|---|
| Fixed in advance: $m_0=0$, all widths 1; kernel $\rho=0.5, T=0.1$ | Jeffreys, reference | new | new |
| Tuned on validation | new (grid) | exists (-0.0032) | exists (-0.0024) |
| Fitted on the 11 training benchmarks | new | new | new |

**Out of scope, stated in Limitations:** neural models, traces, FLOPS, vLLM and new embedding models,
leave-one-benchmark-out, more acquisition policies. Also: three test benchmarks; task-level intervals; OOD mixes new
benchmarks with stronger subjects (0.62 vs 0.47); kernel Rasch uses a pretrained text model.

**Metrics:** Brier ALC and per-budget Brier, as paired differences (primary); AUC, labelled within-task or pooled
(secondary; within-task AUC tells nothing for the Beta-Binomial, which gives one value per task); within-benchmark
Spearman rank correlation and pairwise accuracy for item difficulty.

## 3. The shared contract (agree this first; it is what makes the split safe)
**Folders**
- `organized/testing/eandd/`: code. `results/` is gitignored everywhere, so committed summaries go to `paper/eandd/data/`.
- `$AIMS_EANDD_DATA` (default `~/aims-replay-cache/paper-eandd/`): large outputs, not in git.

**Inputs that are not in git.** Person-A must share these three with Person-B, or Person-B fetches them from the Modal
volume with their own account:
- `~/aims-noitem-run/filtered-responses.parquet` (6.34 M rows: `benchmark_id, subject_uid, item_uid, content_hash, label, benchmark_split`)
- `~/aims-replay-cache/panel/` (`items.parquet`, `candidate_orders.parquet`)
- `~/aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2/` (item embeddings)

**Files every run writes**
| File | Columns |
|---|---|
| `draws.parquet` (frozen, one file) | `shape, role, draw, task_id, order_seed` |
| `scores/<run>.parquet` | `run, model, shape, role, draw, task_id, order_seed, budget, brier` |
| `preds/<run>.parquet` (float32) | `model, task_id, order_seed, budget, item_uid, p, y` |
| `runs/<run>.json` | config, seeds, git commit, input file sizes, runtime |
| `fits/<run>.parquet` (offline fits) | `kind (mu, theta, b, hyper), key, value` |

**Rules:** model display names only from `organized/utils/model_names.py`. Figure scripts only read these files and never
run models. All selection on validation; test read once per final configuration. Each person uses their own accounts; no
shared tokens (`docs/collaboration-setup.md`).

**Figure style:** `organized/utils/plotstyle.py`, at `TEXTWIDTH` or `COLWIDTH`, **not rescaled in LaTeX**; vector PDF, key at
the top, solid vs dashed named in the key, no title or description inside the figure, bounded axes not truncated. Render
to PNG and check.

## 4. Work packages
| WP | Task | Owner | Needs | Effort | Done when |
|---|---|---|---|---|---|
| WP1 | Contract: `common.py` (load panel, save run in the schemas above), freeze draws for all shapes and roles | **Person-A** | - | 0.5 day | `draws.parquet` exists; Person-B can load it |
| WP2 | Beta-Binomial grid (mean, strength, tilted $\alpha\neq\beta$) and empirical Bayes from training tasks; check which prior the leaderboard submission used | **Person-B** | WP1 | 0.5 day | Jeffreys reproduces 0.2494 (single-pair test) |
| WP3 | Refactor replay: `m0` as a parameter (now a hard-coded `M0` in `fit_predict` and `score_draw`), save predictions; check joint fit equals per-task fits in competition shape | **Person-A** | WP1 | 1 day | reproduces -0.0035, -0.0049, -0.0032, -0.0024 |
| WP4a | ID panel: hold out 20% of subjects on the 11 training benchmarks, cap tasks per benchmark so `alignment_faking` does not dominate, 50/50 pool split, frozen orders; same format as `panel/` | **Person-B** | WP1 | 1 day | loads in Person-A's replay; no held-out subject in the fit list |
| WP4b | Offline Rasch: sparse joint MAP on training responses excluding the held-out subjects, save all fitted values; spreads with noise correction, averaged per benchmark | **Person-A** | WP4a's subject list | 1-2 days | ID fit predicts held-out items above chance; saved in `fits/` |
| WP5 | E2: run the Table 1 matrix on competition-shaped and multi-pair draws (fitted kernel settings chosen on the ID panel) | **Person-A** | WP3, WP4b, WP2 values | 1 day | all cells filled, scores and predictions saved |
| WP6 | E3: offline Rasch keeping fitted $\mu_b, b_i$, fitting only $\theta_s$ online (the scoring phase of Truong et al.), vs online Rasch, on ID; prior-only version OOD | **Person-A** | WP3, WP4a, WP4b | 0.5 day | ID and OOD scores saved |
| WP7 | E5: probes on random, item length, item embedding, on the same items; entropy row from existing results; rerun must match OOD 0.09 / 0.53 first | **Person-B** | inputs | 0.5 day | Table 2 data committed |
| WP8 | E4: Rasch terms via prior widths, multi-pair test | **Person-A** | WP3 | 0.5 day | appendix table data |
| WP9 | Analysis and figures: paired CIs, per-benchmark breakdown, AUC from `preds/`, Figures 1-3, tables as CSV | **Person-B** | schemas (mock data until WP5, WP6) | 1.5 days | figures rebuild with one command |
| WP10 | Paper: new `paper/neurips_2026/eandd.tex`, `paper/build.sh` takes a file name | **Person-A + Person-B** | all | 2 days | at most 8 pages; numbers match the CSVs |

**Paper sections by owner.**
- **Person-A:** Models and prior sources; Results on prior source and ID vs OOD.
- **Person-B:** Setting and online simulation; Results on item-difficulty transfer; Limitations; Disclosure and rerun
  instructions.
- **Person-A + Person-B:** each reviews the other's text. Person-B checks every number in the paper against the CSVs.

## 5. The split in short
| | Person-A: models and simulation | Person-B: data, analysis, figures |
|---|---|---|
| **Owns** | `common.py`, `offline_rasch.py`, `run_replay.py`, edits to `kernel_rasch_replay.py` | `bb_grid.py`, `build_id_panel.py`, `probes.py`, `analysis.py`, `plot_*.py` |
| **Produces** | `draws`, `scores`, `preds`, `fits` | ID panel, grid values, CIs, figures, tables |
| **Consumes** | ID panel and subject list, grid values | everything in the schemas |
| **Never edits** | Person-B's scripts and figures | the replay and fit code |

**Handoffs (the only points where one waits on the other)**
1. Person-A to Person-B: `common.py` and `draws.parquet` (after WP1).
2. Person-B to Person-A: held-out subject list and ID panel (WP4a). Do the list first, since it takes minutes and
   unblocks WP4b.
3. Person-B to Person-A: chosen Beta-Binomial prior values (WP2).
4. Person-A to Person-B: real `scores/` and `preds/` (WP5, WP6). Person-B builds on mock data with the same columns
   until then.

**Git:** branches `eandd-models` (Person-A) and `eandd-analysis` (Person-B); no shared file except `model_names.py`
(Person-A owns) and `eandd.tex` (one section each, merged often). Merge after each handoff.

## 6. Order and checkpoints
| Checkpoint | Person-A | Person-B | Decision it allows |
|---|---|---|---|
| C1 | WP1, WP3 | WP2 | Is the baseline still weak, and is competition-shape pooling real? Rewrite wording if not |
| C2 | WP4b, WP6 | WP4a | **Does the control pass?** If offline does not win ID, stop and debug |
| C3 | WP5 | WP7 | Does the claim hold? Decide the title and abstract |
| C4 | WP8, WP10 | WP9, WP10 | Submit |

Total: about 5 working days each, with Person-A and Person-B in parallel. If time runs short, cut in this order: WP8
(Person-A), the multi-pair column of WP5 (Person-A), the entropy row (Person-B).

## 7. Risks
- **Person-A, offline Rasch fit size:** 6.34 M rows and many parameters on a local CPU, untested. Fallback: subsample per benchmark.
- **Person-B, ID panel may look easier for reasons other than carried values.** Same draw shape and same averaging on both sides.
- **Person-B, few test benchmarks:** the per-benchmark view may show one benchmark carrying the result. We report it either way.
- **Person-A, data access for Person-B:** the inputs in section 3 must be shared before anything else.

## 8. Open question
The pasted result ("character count reduced Brier ALC by 0.00257", 14 question-side features) is not in the repo or the
side branches. It stays out unless its source is found.
