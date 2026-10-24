# Feature engineering: measured state and open work

Status: 2026-10-23. Every number here is measured, not assumed. Sources are
`modal_feature_diagnostics.py`, `scripts/information_ceiling.py`, and
`scripts/analyze_benchmark_errors.py`.

## 1. Item features

| Aspect | Current | Measured finding | Action |
|---|---|---|---|
| Text template | `item: {content}\nmetadata: {item_features}` | identical in store and ablation; not confounded | keep |
| Encoder | MiniLM-L6-v2, 384-d | trained for paraphrase similarity, not difficulty | bge-m3, then fine-tune |
| Token cap, v1 store | 256 | **44.1% of all tokens discarded**, 49.8% of items cut | superseded |
| Token cap, v3 / bge-m3 store | 512 | **20.9% of tokens discarded**, 15.1% of items cut | insufficient |
| Recommended cap | 2048 | 93% of tokens kept, 1.0% of items cut | **re-encode** |
| Length distribution | — | p50 251, p90 620, p99 2031, max 255,270 | heavy right tail |
| Replication | — | **44.8% of items answered once**; median 2, mean 9.4, max 825 | difficulty unmeasurable for those |
| Anchor items | — | **6,450 items with >= 10 responses** | clean supervision set |
| Variance share | — | eta^2 = **0.23** shrunk (0.41 raw is memorization) | real signal |

## 2. Subject features

Current subject text is eight fields newline-joined, then MiniLM to 384-d.

| Field | Filled | Bridges benchmarks | Verdict |
|---|---:|---:|---|
| `provider` | 66% | **65%** | best bridge; one-hot, 59 levels |
| `release_date` | 40% | **39%** | scalar, days since epoch |
| `normalized_name` | 66% | **32%** | model-family one-hot |
| `harness` | 100% | 7% | only 2 of 66 span; proxy for benchmark |
| `subject_features_extra` | 100% | ~0% | 7,951 distinct of 8,104 — a near-unique fingerprint. **Drop from text**, but regex-mine `model_identifier=` (present in 18% of blank-name subjects, ~500 names recoverable) |
| `harness_version` | 6% | 0% | drop |
| `access_date` | 1% | 0% | drop |
| `reasoning_effort` | **0%** | 0% | **empty — drop** |

Proposed replacement vector: provider one-hot + days-since-release +
model-family one-hot + an `unknown` flag per channel. About 70 interpretable
dimensions in place of 384 opaque ones.

## 3. Structural problems with the subject channel

| Problem | Detail |
|---|---|
| Constant within task | `task_id = subject_uid :: benchmark_id`, so one subject per task |
| `visible` mode | `b_i = b(item) + beta(subject)` is degenerate with `theta`; ability is fitted twice |
| `masked` mode | 384 of 768 dims zeroed, and `CNPAIRT.logits` never reads them — dead weight |
| Correct home | subject features belong in the **prior mean of `theta`**, not in `a(x)` / `b(x)` |
| Variance share | eta^2 = **0.32**, larger than the item channel |

`adaptation.py::fit_ability_prior` already implements the correct
factorization. The NP/CNP models do not use it.

## 4. Identifiability

| Scope | Separable? | Why |
|---|---|---|
| Within a benchmark | **Yes** | items are answered by ~9 subjects on average |
| Across benchmarks, by `subject_uid` | No | no uid spans two benchmarks (train, validation, and test all confirmed) |
| Across benchmarks, by model identity | **Yes, partially** | 255 of 437 model names span >1 benchmark; GPT-4 appears on 11 |

The `subject_uid` hash combines name, provider, date, harness, effort, and
version, so the same model under two harnesses becomes two subjects. The
bridge exists at the model-identity level, not the hash level.

Residual limitation: roughly 35% of subjects have no usable name or provider
and remain unbridged.

## 5. Headroom

| Quantity | Validation (104 tasks, OOF, 96% item coverage) |
|---|---:|
| Oracle item parameters + oracle ability | 0.1013 |
| **Information ceiling `ALC_min`** | **0.1608** |
| Beta-Binomial baseline | 0.2228 |
| Best CNP to date | ~0.2226 |
| **Headroom** | **+0.0621**, 95% CI [+0.0563, +0.0677] |

Beta beats the ceiling on 1 of 104 tasks. The ablation ladder moves ALC by
1e-4 to 1e-3; available headroom is ~100x larger. The project is not bounded
by information content — it is bounded by item-difficulty knowledge.

The test-split ceiling (0.2057 vs Beta 0.2182) is **not** trustworthy: item
parameter coverage is only 59.6% there, so the oracle is not an oracle.

## 6. Error structure

Tasks split into two regimes, visible under within-benchmark clustering:

| Regime | Budget profile | vs Beta |
|---|---|---|
| Flat near 0.25 | `[.254 .260 .264 .265 .258 .253]` | **loses** +0.002 to +0.009 |
| Steeply declining | `[.162 .181 .098 .050 .037 .029]` | **wins** -0.005 to -0.011 |

The aggregate null is these two regimes cancelling, not uniform
indifference. A gating rule that detects a flat profile and falls back to Beta
would keep the wins and drop the losses, using existing checkpoints.

## 7. Status

Implemented:

- token-length and truncation diagnostic (`modal_feature_diagnostics.py`)
- item replication and variance components (same file)
- information ceiling with out-of-fold 2PL (`scripts/information_ceiling.py`)
- per-benchmark error analysis and clustering (`scripts/analyze_benchmark_errors.py`)
- ceiling derivation (`docs/information-ceiling-note.md`)

Not implemented:

- bge-m3 re-encode at 2048 tokens
- engineered subject vector (provider, release date, family, unknown flags)
- regex mining of `model_identifier=` for blank-name subjects
- removal of `subject_features_extra` from the subject text
- moving subject features into the `theta` prior
- flat-profile gating with Beta fallback
- item-signal diagnostic: regress empirical `b_hat` on the embedding, read out-of-sample R^2
- encoder fine-tuned on the IRT objective
- anchor-item calibration on the 6,450 well-replicated items

## 8. Proposed successor model

The v5 design that builds on this inventory is in
[`model-v5-design.md`](model-v5-design.md).
