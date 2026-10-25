# Results report: what worked, what did not, what to do next

Status: 2026-10-25. Covers every result on the `aims-neurips-data` Volume
(both the `agrover112` and `aims-neurips` workspaces) and every results doc in
`docs/`. Nothing here is new training. Every paired number marked *(recomputed)*
was recomputed from the stored per-task results with
[`scripts/report_paired_stats.py`](../../scripts/report_paired_stats.py). The
rest are quoted from the source doc named next to them.

---

## 0. How to read this report

**Metric.** Brier ALC: the weighted mean Brier score over label budgets
`0, 1, 3, 7, 15, 31`, with weights `0.1, 0.2, 0.2, 0.2, 0.2, 0.1`. Lower is
better. A constant 0.5 forecast scores 0.25.

**Grid.** Unless stated otherwise, every number uses the same validation grid:

- 104 validation tasks (one task = one subject on one benchmark), from 5
  held-out benchmarks, none seen in training;
- 10 frozen acquisition orders;
- five-fold out-of-fold calibration;
- reference **Beta-Binomial = 0.2228** (exactly `0.222838` in every run, which
  confirms the runs share the grid).

The 22 test tasks are read once, for the final model only.

**Deltas** are arm minus reference, paired by task. Negative means the arm is
better. Two intervals are reported:

- *task CI*: 95% bootstrap over the 104 tasks. This is the primary interval.
- *bench CI*: 95% bootstrap over the 5 benchmark clusters. This is a
  sensitivity check. It is always wide, because one cluster has a single task.

**What counts as statistically sound (Parts 2–4).** At least three seeds with
a paired comparison, or a deterministic method scored on the full paired grid.
Single-seed results are one-offs. All the recent one-offs use the new
multilingual embeddings, so they are collected in **Part 5**.

**Power, read this first.** The paired standard error of "model minus Beta" is
about 0.0015, so the smallest effect this grid can reliably detect against
Beta (80% power) is about **0.004–0.005**. Comparisons between two similar
models are tighter, with a minimum detectable effect of 0.00005–0.0003 (see
`docs/faithful-ladder-stage1.md`). "CI includes zero" therefore means
*can't tell*, not *no effect*.

---

## 1. Summary

1. **No model has ever beaten Beta-Binomial with a CI that excludes zero**,
   on validation or on test. The best 3-seed result is `-0.0026` (task CI
   `[-0.0051, +0.0005]`). The best single run is `-0.0030`
   (`[-0.0060, +0.0007]`). Both are close to, but below, the grid's detection
   limit.
2. **The only replicated improvement is the new item store**:
   multilingual-e5 over `item_content` only. Against the old MiniLM store it
   gains `-0.0015` to `-0.0023` over 3 seeds; the task CI excludes zero on 2 of
   3 architectures. It is confounded with other store changes (§3.2).
3. **Calibration is what keeps neural models competitive.** Raw neural
   forecasts are 0.010–0.045 worse than Beta. Calibration, mostly the
   context-rate anchor, pulls every model onto Beta's curve.
4. **Almost every architecture change is noise.** This covers r_C variants,
   loops, latent NP/ANP, GP heads, multidimensional ability, NR, learned
   ability corrections, ability priors and hierarchical pooling. Seed spread
   (up to 0.0017) is larger than any architecture gap.
5. **The gap to the ceiling is item knowledge plus ability.** The ceiling is
   0.1608 and the best model is 0.2198. Item text predicts *which benchmark*
   an item is from, not *how hard* it is within a benchmark: R² ≈ 0 within a
   benchmark. Other subjects' responses to the same item predict outcomes very
   well (AUC 0.84 against 0.49 for text).
6. **Three promising one-offs need replication.** All three are single-seed
   on the new embeddings:
   - **the v4.1 training sampler for `cnp`/`anp_attn`** (`-0.0012`/`-0.0015`)
   - **D-optimal Fisher acquisition with MIRT** (`-0.0023` against random,
     both CIs exclude zero; not yet deployable as streamed)
   - **a tighter ability prior** (`tau = 1`)
7. **Benchmark mix matters more than architecture.** Gains come almost
   entirely from `cmmlu` (Chinese, about `-0.011`) and `afrimedqa` (about
   `-0.005`). Every me5 model loses on `appworld` (+0.004 to +0.009). One
   `cmmlu` task alone costs +0.10 to +0.13.

---

## 2. Where we stand

| Quantity | Validation ALC | Source |
|---|---:|---|
| Oracle item parameters and oracle ability | 0.1013 | `docs/feature-engineering.md` |
| Information ceiling (oracle items, ability from ≤31 labels) | 0.1608 | same |
| Best single run: `anp_attn`, me5, v4.1 sampler, 1 seed | 0.2198 | `r-c-ablation` *(recomputed)* |
| Best 3-seed mean: `direct_loop0`, me5 | 0.2202 | `r-c-ablation` *(recomputed)* |
| Best result on the old MiniLM store (`meta_gp_2pl`, 3 seeds) | 0.2218 | `docs/ablation-results.md` |
| **Beta-Binomial** | **0.2228** | every run |
| Raw (uncalibrated) neural models | 0.226–0.42 | various |

The best model closes about **5%** of the 0.062 gap between Beta and the
ceiling.

**Test** (22 tasks, read once): Beta **0.2182**. The selected CNP-AIRT scored
0.2199 calibrated (+0.0018, task CI `[+0.0004, +0.0030]`). Source:
`docs/uniform-audit.md`.

**Codabench hidden scores.** These are descriptive only. Each submission drew
different subject-benchmark groups (8–9 per submission), so they are **not
paired** and cannot rank methods.

| ID | Method | Hidden ALC |
|---|---|---:|
| 1005155 | Beta-Binomial with active labeling (current final entry) | **0.1486** |
| 1005839 | Flat budget shrinkage | 0.1856 |
| 1005187 | CNP-AIRT, random acquisition | 0.1868 |
| 1005210 | Online-AIRT, active acquisition | 0.2011 |
| 1005225 | CNP-AIRT, active acquisition | 0.2142 |
| 1005650 | Empirical mean | 0.2190 |

---

## 3. What has worked (sound evidence)

### 3.1 Calibration is essential, and the context-rate anchor does the work

Source: `docs/uniform-audit.md`. Four seed/mode configurations, out-of-fold.

| Variant (selected CNP) | ALC | vs full BCE calibrator | task CI |
|---|---:|---:|---|
| Raw | 0.2349 | +0.0124 | [+0.0089, +0.0158] |
| Platt only | 0.2276 | +0.0050 | [+0.0032, +0.0069] |
| Context rate only | 0.2245 | +0.0019 | [+0.0005, +0.0031] |
| Anchor only | 0.2246 | +0.0020 | [+0.0007, +0.0032] |
| **Full BCE (submitted form)** | **0.2226** | — | — |

- Without calibration, every neural model loses to Beta by 0.010–0.045.
- The fitted calibrator shrinks logits by 65–78%. It does this because the
  ranking of items inside a task is near chance (validation AUC about 0.48,
  test 0.51–0.56). See `docs/why-calibration-collapses.md`.
- **Rejected:** a Brier-loss calibrator scored `-0.0016` (CI excludes zero) but
  got there with *negative* scales, inverting a validation-specific ranking.
  It does not transfer to test.

### 3.2 The new item store: multilingual-e5, `item_content` only

*(Recomputed.)* Same architecture and the same 3 seeds; only the store
changes.

| Architecture | me5 minus MiniLM@512 | task CI | bench CI | wins | per seed |
|---|---:|---|---|---:|---|
| `scalar` | **-0.0022** | [-0.0040, -0.0001] | [-0.0048, +0.0012] | 80/104 | -0.0019 / -0.0027 / -0.0020 |
| `direct_loop0` | **-0.0023** | [-0.0042, -0.0000] | [-0.0048, +0.0010] | 79/104 | -0.0010 / -0.0034 / -0.0027 |
| `additive_loop3` | -0.0015 | [-0.0034, +0.0009] | [-0.0050, +0.0039] | 75/104 | -0.0020 / -0.0022 / -0.0001 |

- All 9 seed-level differences point the same way. This is the only change in
  the project that holds across seeds and architectures.
- **Where it helps** (per-benchmark delta against Beta, me5 `scalar`, 3-seed
  mean):

  | | cmmlu | afrimedqa | cruxeval | appworld |
  |---|---:|---:|---:|---:|
  | me5 | **-0.0105** | -0.0041 | +0.0011 | **+0.0045** |
  | MiniLM@512 | -0.0046 | -0.0012 | +0.0034 | +0.0008 |

  The largest gain is on the Chinese benchmark, which is the case the
  multilingual encoder was chosen for.
- **Confound (important):** the "me5 store" changed several things at once:
  - the encoder (MiniLM → me5);
  - the text (content plus metadata → content only);
  - the training cap (first 128 by acquisition rank → random 128);
  - very probably the calibrator. The MiniLM runs finished 2026-10-21 to
    2026-10-22 03:36. Calibration v2 (nonnegative scale) is dated 2026-10-22.
    The me5 runs are from 2026-10-25. The result files don't record the
    calibrator version, so this is inferred from dates.

  The effect is real for the *bundle*. It is not yet attributable to the
  encoder alone (see backward step B2).

### 3.3 Weak positives that did not survive scrutiny

| Result | Evidence | Why it is not a win |
|---|---|---|
| GP-2PL vs 2PL (MiniLM) | -0.0008, CI [-0.0012, -0.0005], 3 seeds | Per-seed -0.0022 / -0.0002 / -0.0000: one seed drives it. Under the corrected v4.1 trainer, `gp_2pl - 2pl` = +0.00002 (can't tell). |
| Loops: additive K=3 vs K=0 (MiniLM) | -0.0003, CI [-0.0005, -0.0002] | Tiny. On me5 the same arm has the largest seed spread (0.0017) and is last under v4.1. |
| v4.1 Stage 1: `gp_anp` vs `2pl` | -0.00006, better in 3/3 seeds | One third of its MDE; flips sign across benchmarks. |

---

## 4. What has not worked (sound evidence)

Every row is either multi-seed or a deterministic method on the full paired
grid. Delta is against the matched control named in the row.

| # | Experiment | Seeds | Delta (CI) | Verdict | Source |
|---|---|---|---|---|---|
| 1 | Learned ability correction Δθ(C) in CNP (full minus zeroed) | 4 configs | +0.00001 [-0.00003, +0.00006] | redundant; `map_theta` is `no_grad`, so it can't train | `uniform-audit.md` |
| 2 | r_C ladder: scalar / additive / ANP / CNP vs 2PL | 3 | -0.00005 to +0.00007, all CIs span 0 | tie | `ablation-results.md` |
| 3 | r_C direct / residual vs 2PL | 3 | +0.0004 / +0.0008, CIs exclude 0 | worse | same |
| 4 | NR-2PL, U-NR-2PL (± prior) vs 2PL | 3 | +0.0006 to +0.0007 | worse | same |
| 5 | Multidimensional ability, D=4 vs scalar twin (me5) | 3 | -0.0001 / +0.0001 / +0.0001, all CIs span 0; sign flips by seed | no evidence | *(recomputed)* |
| 6 | v4.1 faithful ladder Stage 1 (`2pl_anp`, `gp_2pl`, `gp_anp` vs `2pl`) | 3 | all below MDE | can't tell; nothing promoted | `faithful-ladder-stage1.md` |
| 7 | 2PL vs 1PL after calibration | 4 configs (CI: selected config, test) | -0.00008 [-0.0013, +0.0012]; 4-config mean 0.2196 vs 0.2193 | 2PL's raw loss (+0.028) is erased; 2PL adds nothing | README (encoder ladder) |
| 8 | Disjoint context/target training (CNP) | 4 configs | +0.00013 [-0.000002, +0.00026] | tie | `uniform-audit.md` |
| 9 | Active acquisition with CNP: Fisher / uncertainty / disagreement vs random | 2 | Fisher +0.0058 [+0.0014, +0.0111]; others span 0 | Fisher worse; others tie | same |
| 10 | Threshold acquisition rules (Codabench-style) | 10 streams | all within ±0.0002 | rules rarely fire; reserve rule decides | README |
| 11 | Budget-specific shrinkage κ | full grid | 0.000000 | null by construction (κ = 4 = flat prior) | `uniform-audit.md` |
| 12 | Hierarchical empirical Bayes (subject + benchmark effects) | full grid | reduces to flat prior | 0/104 validation subjects seen; benchmarks held out | `results/hierarchical-eb-v1` |
| 13 | Ability prior from benchmark mean or subject-metadata ridge | 3 (pilot) | 0.000 to +0.016 | ridge α pins at the grid ceiling; metadata carries no signal | README |
| 14 | Latent NP / ANP / Stacked-MHSA / TfNP-D (BeaverTails pilot) | 3 | all worse than Beta (0.2062–0.2145 vs 0.1942) | latent rank 1; KL ≈ 2e-7 | README |
| 15 | Frozen-encoder ablation v1 (MiniLM, bge-base, bge-m3, Llama-3.1-8B, adapter) | 1 each | calibrated 0.2226–0.2237 | calibration erases a 0.044 raw spread | `why-calibration-collapses.md` |
| 16 | Final test, selected CNP vs Beta | 1 readout | +0.0018 [+0.0004, +0.0030] | Beta wins on test | `uniform-audit.md` |

Row 15 is single-seed but sits here because it uses the old stores.

**Pattern.** Each row changes *how ability is inferred* (r_C, loops, latent,
multidimensional, priors, hierarchy, corrections) or *how labels are bought*
(with a predictor that has no item signal). None changes *what the model knows
about items*. Calibration then removes whatever ranking noise remains.

---

## 5. Why: diagnostics

These are measurements, not ALC ablations.

| Diagnostic | Finding | Source |
|---|---|---|
| Gap decomposition | ceiling 0.1608 minus oracle 0.1013 = **0.0595** is the cost of not knowing θ; best 0.2198 minus 0.1608 = **0.0590** is the cost of not knowing item parameters | `me5-encoder-and-v41-sampler.md` |
| Item text → difficulty (v1 store, with metadata) | R² 0.78 overall, **≈ 0 within benchmark**, **negative** on held-out benchmarks | `results/difficulty-model` |
| Collaborative signal | ranking validation items by *other validation subjects'* revealed labels: AUC **0.84** (0.99 of tasks above 0.5) vs text **0.49**; validation items are answered by a median of 4 tasks | `results/cf-difficulty` |
| Identity overlap | 6,701 subjects for 6,702 training tasks; 0/104 validation subjects seen; 0/5 benchmarks | `hierarchical-eb-v1` |
| Variance components | task η² 0.32; item η² 0.23 after shrinkage | `feature-diagnostics/replication-variance.json` |
| Training abilities | θ SD 3.1–3.5; **17% of tasks sit outside the ±4 clamp** | `feature-diagnostics/theta-prior-variance.json` |
| Item replication | 45% of training items answered once (median 2) | same |
| Language | 9% of items non-English across 24 languages; Chinese in 9 benchmarks | README |

The single most important number here is the R² of ≈ 0 within a benchmark.
It was measured **on the old v1 embeddings, which included the metadata that
leaks benchmark identity**. It has **not** been re-measured on the new
content-only stores. That makes it backward step B1.

---

## 6. One-off experiments with the new multilingual embeddings

All rows are **seed 20261018 only**, on the me5 / bge-m3-long / Qwen3 stores
(`item_content` only). Paired against the stated control on the same 104
tasks. *(Recomputed.)* Treat each as a hypothesis to replicate, not a result.

### 6.1 v4.1 training sampler (full task-budget coverage, disjoint C/T)

| Architecture | v4.1 minus random sampler | task CI | bench CI | wins |
|---|---:|---|---|---:|
| `anp_attn` | **-0.0015** | [-0.0025, -0.0005] | [-0.0036, +0.0022] | 78/104 |
| `cnp` | **-0.0012** | [-0.0019, -0.0005] | [-0.0029, +0.0021] | 80/104 |
| `direct_loop0` | -0.0004 | [-0.0011, +0.0004] | — | 66/104 |
| `scalar` | +0.0000 | [-0.0008, +0.0009] | — | 60/104 |
| `additive_loop3` | +0.0009 | [+0.0001, +0.0017] | — | 46/104 |

This makes sense mechanically. The models that *learn* adaptation from the
context set (`cnp`, `anp_attn`) were trained on about 7.7% of task-budget
pairs under the old sampler. The IRT rungs get adaptation from a Newton solve
and don't care.

### 6.2 bge-m3 (8192 tokens) loses to me5 (512 tokens)

| Architecture | bge-m3-long minus me5 | task CI | bench CI | wins |
|---|---:|---|---|---:|
| `scalar` | +0.0027 | [+0.0007, +0.0043] | [+0.0008, +0.0050] | 30/104 |
| `direct_loop0` | +0.0026 | [+0.0002, +0.0045] | [+0.0007, +0.0046] | 28/104 |
| `additive_loop3` | +0.0031 | [+0.0006, +0.0050] | [+0.0008, +0.0059] | 30/104 |
| `mirt_direct_loop0_d4` | +0.0032 | [+0.0008, +0.0052] | [+0.0006, +0.0065] | 28/104 |
| `mirt_scalar_d4` | +0.0030 | [+0.0007, +0.0049] | [+0.0003, +0.0064] | 31/104 |

All five architectures land above Beta, and both CIs exclude zero in every
row. bge-m3 truncates 94× fewer items, so context length is not the
bottleneck. This is the strongest one-off, but it is still one seed.

### 6.3 Architecture ranking under v4.1

`anp_attn` (0.2198) is best. Against the others, same seed:

- `cnp`: -0.0003 [-0.0006, +0.00002]
- `mirt_scalar_d4`: -0.0002 [-0.0007, +0.0003]
- `scalar`: -0.0010 [-0.0019, -0.00004]
- `additive_loop3`: -0.0014 [-0.0021, -0.0006]

Under the old sampler, the IRT anchor looked better than a structure-free
model. Under v4.1 that ordering reverses. With a seed spread of up to 0.0017
on this store, no architecture ranking is safe yet.

### 6.4 Ability prior variance τ (MAP prior N(0, τ))

| Rung | τ=1 | τ=4 | τ=9 | τ=16 |
|---|---:|---:|---:|---:|
| `2pl` | **0.2203** | 0.2209 | 0.2211 | 0.2208 |
| `rasch` | **0.2210** | 0.2219 | 0.2224 | 0.2226 |
| `scalar` | **0.2203** (τ=1) | 0.2208 (default) | — | — |

The tightest prior wins, even though training abilities have SD about 3. A
reading: with noisy item parameters, a strong pull toward 0 is useful
shrinkage. The same direction shows in §6.5 (`zero_tau1 - zero` = -0.0003).

### 6.5 Subject priors: FLOPs and model identity (`modal_flops_prior_2pl.py`)

The MAP prior mean is set from training abilities on the `2pl_tau4`
checkpoint. Coverage: 40/104 validation tasks have FLOPs; 33 have a seen model
name.

| Arm minus zero-mean prior | delta | task CI | bench CI | wins |
|---|---:|---|---|---:|
| log-FLOPs prior | +0.00015 | [-0.0006, +0.0011] | [-0.0005, +0.0007] | 51/104 |
| same-model prior (ability on other benchmarks) | **+0.0008** | [+0.00001, +0.0016] | [+0.0001, +0.0013] | 34/104 |
| FLOPs and model | +0.0009 | [-0.00003, +0.0017] | [+0.0003, +0.0015] | 40/104 |

Within a benchmark, log-FLOPs explains R² = 0.035 of training ability
(r = 0.19). The FLOPs prior does nothing. **The same-model prior makes things
worse:** a model's ability on other benchmarks is not a good prior for a new
one. Together with row 13 of §4, this closes the subject-prior channel for
now.

### 6.6 Multidimensional ability, D=4 vs D=1 (same reparametrization)

On seed 18, D=4 beats D=1 by -0.0007 (CI excludes zero) for `scalar` and
`additive_loop3`. This **contradicts** the 3-seed twin comparison in §4 row 5,
which shows no effect. Treat it as noise.

### 6.7 D-optimal Fisher acquisition with MIRT (`modal_mirt_acquisition.py`)

Predictor `mirt_scalar_d4`, 10 candidate streams, out-of-fold calibration.

| Policy minus random | delta | task CI | bench CI | wins |
|---|---:|---|---|---:|
| **Fisher, D-optimal** | **-0.0023** | [-0.0040, -0.0007] | [-0.0059, -0.0006] | 63/104 |
| Fisher, A-optimal | -0.0023 | [-0.0040, -0.0006] | [-0.0055, -0.0005] | 62/104 |
| Fisher, E-optimal | -0.0008 | [-0.0022, +0.0006] | — | 58/104 |
| Max-min embedding coverage | +0.0000 | [-0.0007, +0.0008] | — | 53/104 |
| Uncertainty (closest to 0.5) | +0.0005 | [-0.0009, +0.0020] | — | 49/104 |

This is the first acquisition result in the project where both intervals
exclude zero. Earlier, Fisher with CNP was *worse* (§4 row 9). Two caveats:

- **Single seed.**
- **Not deployable as tested.** The policy picks the best item from the whole
  remaining stream. The competition's streaming hook sees one candidate at a
  time and must accept or skip it on the spot. A streamed version, such as
  "accept if the Fisher score beats the running quantile, adjusted by
  `items_remaining`", has not been tested.

### 6.8 Bayesian MIRT (HMC) across four encoders (`modal_bayesian_mirt.py`)

Calibrated ALC against Beta 0.2228. Every arm is within ±0.0009, and every
paired CI spans zero.

| Store | D=1 | D=2 | D=4 | Raw ALC (best D) |
|---|---:|---:|---:|---:|
| bge-base (v1) | 0.2235 | 0.2235 | — | 0.41 |
| bge-m3 (v1) | 0.2237 | 0.2237 | — | 0.28 |
| bge-m3-long | 0.2226 | 0.2228 | 0.2230 | 0.29 |
| me5 | — | 0.2223 | 0.2232 | 0.28 |
| qwen3-4b | 0.2229 | 0.2231 | 0.2229 | **0.231** |
| qwen3-8b | **0.2222** | — | — | **0.234** |

Calibration ties everything. One observation stands out: **Qwen stores give
raw forecasts that are close to calibrated already** (0.231–0.236), against
0.28–0.42 for the others. That is the first sign that an encoder changes the
probability scale rather than just the ranking. Qwen has not yet been run
through the r_C ladder.

### 6.9 Context-cook ensemble (`aims-neurips` workspace, `paiec-context-cook`)

A stacked ensemble of Beta, kernel-local, mixture and text-kNN priors, with
weights chosen by cross-validation over training benchmarks only:

- `cook-gpu-v1`: -0.0027 vs Beta, bench CI [-0.0043, -0.0018]
- `cook-crossfit-v2` (cross-fitted, more conservative): **-0.0009**, bench CI
  [-0.0022, -0.0002]

**Not comparable to the grid.** It uses 3 orders, not 10; its Beta reference
is 0.2233, not 0.2228; and its code is not in this repo. The best single
component was a text-kNN prior (`text_a10_k4`, -0.0017). Worth porting onto
the grid (step F5).

### 6.10 Status of the four new stores

| Store | Encoded | Copied to `aims-neurips` | r_C ladder | Bayesian MIRT |
|---|---|---|---|---|
| me5 | yes | yes | 6 rungs × 3 seeds, plus v4.1 | yes |
| bge-m3-long | yes | yes | 5 rungs × 1 seed | yes |
| qwen3-4b | yes | yes | **not run** | yes |
| qwen3-8b | yes (verified: no NaN, unit norm, order matches 4b) | yes | **not run** | yes |

---

## 7. Next steps

Ordered by what each step *decides*, cheapest first. "Backward" fixes or
re-checks the measurement pipeline. "Forward" adds new capability.

### Backward

| ID | Step | Cost | Decides |
|---|---|---|---|
| **B1** | Re-measure item-difficulty R² **within benchmark** and **on held-out benchmarks** for the content-only stores (me5, bge-m3-long, qwen3-4b/8b). Use the 6,450 anchor items and the existing ridge probe in `results/difficulty-model`. | minutes, CPU | Whether item-text work can close any of the 0.059 item gap. If R² stays ≈ 0, stop encoder work and go to F2/F1. |
| **B2** | Re-run MiniLM@512 `scalar` (3 seeds) under the **current** code: current calibrator, random-cap store builder. | ~1 GPU-h | Whether the me5 gain (§3.2) is the encoder or the bundled pipeline changes. |
| **B3** | Replicate the four one-offs at seeds 19 and 20: v4.1 `anp_attn`/`cnp` (§6.1), bge-m3-long `scalar` (§6.2), `2pl_tau1`/`scalar_tau1` (§6.4), Fisher-D acquisition (§6.7). | a few GPU-h | Which one-offs become results. Nothing should be promoted before this. |
| ~~B4~~ | **Done, see §7.1.** The outlier and the `appworld` losses are the same failure: subjects that almost always fail. | — | — |
| **B5** | Get statistical power. Five validation benchmarks cannot resolve 0.003 effects against Beta. Add grouped cross-validation over the 41 training benchmarks, as context-cook already does, as a second readout with many more clusters. | moderate | Lets real 0.002–0.003 effects be confirmed instead of reading as "can't tell". |
| B6 | Structural fixes already identified: remove `@torch.no_grad()` from the ability path; marginalize over the θ posterior at inference (`marginalize_logits` exists but is unused); revisit the ±4 clamp (17% of training θ lie outside it). | moderate | Whether the learned-adaptation path can work at all; currently it can't receive a gradient. |

### Forward

| ID | Step | Cost | Decides |
|---|---|---|---|
| **F1** | Build a **streaming-deployable** Fisher acquisition hook. Use a quantile threshold on the Fisher score from candidates seen so far, plus a reserve rule driven by `items_remaining`. Score it on the same 10 streams against random. | ~1 day | Whether the -0.0023 from §6.7 survives the real hook. This is the most direct path to a better Codabench entry. |
| **F2** | **Cross-pair pooling at test time.** The rules allow it: the predictor receives "all evidence purchased so far, across sampled pairs", and `benchmark_id` is shared. Other subjects' labels give item AUC 0.84 against 0.49 for text (§5). First, simulate a realistic draw: about 1,000 source items and about 12 pairs, preferring under-represented benchmarks. Measure how often two pairs share a benchmark and an item. | ~1 day | Whether the largest measured signal in the project is reachable on the leaderboard, or almost never available because the test draw rarely shares items. |
| F3 | Run qwen3-4b and qwen3-8b through the r_C ladder (`scalar`, `direct_loop0`, `anp_attn` v4.1), 3 seeds. | a few GPU-h | Whether Qwen's better raw calibration (§6.8) turns into a calibrated gain, and whether it beats me5 off Chinese. |
| F4 | Fold in τ=1 and the v4.1 sampler only if B3 confirms them. Then rebuild one candidate submission: me5 + best confirmed architecture + calibration v2 (+ F1 hook if confirmed). | ~half day | The final entry. Until something beats Beta on the grid with a CI excluding zero (or on the B5 readout), **Beta (1005155) stays the safe submission.** |
| F5 | Port the context-cook text-kNN prior onto the 10-order grid with the standard paired readout. | ~half day | Whether its -0.0009 to -0.0017 is real on the shared grid. |

### 7.1 B4 result: the models fail on subjects that almost always fail

**The outlier.** `subject_46df718ebae142b2a22d::cmmlu` is "CMMLU /
deepseek-r1" (OpenCompass, `prompt_condition=nocot`). It is correct on 0.8% of
evaluation items and 0.66% of 5,790 acquisition items. The other 21 cmmlu
tasks sit at 0.44–0.86. For a strong model this is almost certainly a broken
run, probably failed answer extraction without CoT. It is still a scored
outcome, and the test draw can contain the same kind of run.

Per-budget Brier on this task, seed 18:

| Budget | 0 | 1 | 3 | 7 | 15 | 31 |
|---|---:|---:|---:|---:|---:|---:|
| Beta | 0.250 | 0.162 | 0.095 | 0.042 | 0.021 | 0.011 |
| 2PL (τ=4), calibrated | 0.474 | 0.393 | 0.215 | 0.100 | 0.043 | 0.019 |
| 2PL (τ=4), raw | 0.494 | 0.371 | 0.219 | 0.107 | 0.091 | **0.091** |
| 2PL (τ=1), raw | 0.494 | 0.460 | 0.391 | 0.292 | 0.185 | 0.097 |
| MIRT D=4, calibrated | 0.513 | 0.447 | 0.275 | 0.139 | 0.060 | 0.028 |

Three causes are visible:

1. **Budget 0 says "easy".** The item head predicts p ≈ 0.7 for cmmlu items,
   because most subjects pass them. Beta predicts 0.5.
2. **Ability adapts too slowly.** After 3 failures Beta predicts 0.29; the IRT
   models still predict about 0.5. A tighter prior (τ=1) is worse here.
3. **The θ clamp sets a floor.** Raw Brier stops improving at 0.091 from
   budget 15 on, which means p ≈ 0.3 with θ pinned at -4. Only the
   calibrator's context-rate anchor rescues budget 31. (17% of training θ lie
   outside ±4; see §5.)

**It is not one task; it is a regime.** *(recomputed)* Delta vs Beta,
grouped by the task's evaluation base rate:

| Base rate | Tasks | Beta ALC | me5 `scalar` (3 seeds) | `anp_attn` v4.1 | `2pl_tau1` v4.1 | MiniLM `scalar` (3 seeds) |
|---|---:|---:|---:|---:|---:|---:|
| < 0.1 | 6 | 0.114 | **+0.028** | **+0.030** | **+0.035** | +0.001 |
| 0.1–0.3 | 15 | 0.205 | +0.001 | +0.003 | +0.004 | +0.002 |
| 0.3–0.7 | 53 | 0.251 | -0.000 | -0.002 | -0.002 | +0.003 |
| 0.7–0.9 | 30 | 0.204 | **-0.013** | **-0.015** | **-0.015** | -0.007 |

- The six tasks below 0.1 are the cmmlu outlier and five `appworld` tasks
  (0.01–0.09). That explains the consistent `appworld` losses in §3.2.
- **The me5 gain is largely a better "most subjects pass" level.** It helps
  the 30 high-base-rate tasks (-0.013 to -0.015) and hurts the 6 near-zero
  tasks (+0.028 to +0.035). MiniLM did not have that level, so it neither won
  nor lost there.
- Excluding the 6 tasks, me5 `scalar` goes from -0.0023 to **-0.0042** and
  `anp_attn` from -0.0030 to **-0.0051**. The low-base-rate tasks cancel about
  60–70% of the gain. These numbers are a diagnosis, not a valid score: the
  tasks can't be dropped from the leaderboard.

**Fixes to test.** All are cheap re-scores of existing checkpoints, except the
last:

- (a) widen or remove the ±4 θ clamp;
- (b) use a heavy-tailed θ prior (Student-t), or a two-component prior with a
  "broken run" mode near p = 0, so a few failures can move θ far;
- (c) a budget-aware fallback toward the Beta context rate when the revealed
  rate is extreme, for example 0 of n with n ≥ 3;
- (d) budget 0: shrink the item-head level toward 0.5. This is the
  budget-0 = 0.5 control, already shown to help on test in 3 of 4 CNP
  configurations.

Score each fix by base-rate bin as well as overall. It should recover the
low-rate loss without giving back the -0.013 on high-rate tasks.

### Stop (closed by sound evidence; reopen only with new information)

- Subject-metadata channels: embeddings, ridge, FLOPs and same-model priors
  (§4 row 13, §6.5).
- Multidimensional ability D > 1 (§4 row 5).
- Latent NP/ANP, NR, residual and direct r_C variants (§4).
- Hierarchical or empirical-Bayes pooling by identity (§4 rows 11–12).
- Acquisition rules driven by a predictor without item signal (§4 rows 9–10).
- Longer context windows as a goal in themselves (§6.2).

### Timeline

The report and code are due **30 October 2026, 23:59 AoE**; the Codabench
phase closes **31 October 11:59 UTC**.

- **Days 1–2:** B1–B4 (all cheap). They determine whether F2/F1 or more
  encoder work is the better use of the remaining days.
- **Day 3:** F1 and F2.
- **Final day:** F4.

---

## Appendix: reproduce the recomputed numbers

```bash
# 1. pull the small result files (JSON + row parquet) from the Volume
MODAL_PROFILE=agrover112 modal volume get aims-neurips-data /results/r-c-ablation/<run>.json ./pulled/results/r-c-ablation/
#    (also /results/flops-prior-2pl/rows-calibrated.parquet and
#     /results/mirt-acquisition/*-6arms.rows.parquet)

# 2. recompute every paired delta, task CI, benchmark CI and win count
python scripts/report_paired_stats.py --results ./pulled/results
```

Bootstrap: 10,000 task resamples and 2,000 benchmark-cluster resamples, seed
0. Multi-seed rows average the seeds within a task before resampling.
