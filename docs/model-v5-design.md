# Model v5: factorized ability with full-corpus features

Status: 2026-10-23. **Specified, not implemented.** No numbers in this file are
results. Every measurement cited here comes from `modal_feature_diagnostics.py`,
`modal_field_audit.py`, `scripts/information_ceiling.py`, and
`scripts/analyze_benchmark_errors.py`. Feature inventory lives in
[`feature-engineering.md`](feature-engineering.md).

This design deliberately leaves strict 2PL/IRT behind. It keeps exactly one
idea from IRT -- a low-dimensional ability state updated from the acquired
context -- because at most 31 binary responses genuinely cannot identify more
than that. Everything else becomes a factorization with side information.

## 1. Why the previous model stalled

Three separate failures, all measured.

**Ability was fitted twice.** In `visible` mode, subject features reach
`b(x)`. Since a task is one `(subject, benchmark)` pair, the subject vector is
constant within a task, so a task-constant shift in `b` is algebraically
indistinguishable from a shift in `theta`:

```text
a_i (theta_j - b_item(i) - beta(s_j))  ==  a_i ((theta_j - beta(s_j)) - b_item(i))
```

Only the difference is identified. In `masked` mode the same 384 dimensions are
zeroed and never read by `CNPAIRT.logits`, so subject information is simply
discarded. Neither mode uses it correctly.

**The learned correction measured as nothing.** `cnp_full` minus
`cnp_no_delta` is `+0.000014` with interval `[-0.000029, +0.000056]`. The cause
is structural: `map_theta` is decorated `@torch.no_grad()`, so the outer loss
never reaches the context encoder through the ability path. The encoder was
trained only through a delta term whose contribution is indistinguishable from
zero.

**The target was two orders of magnitude too small.** The information ceiling
is `0.1608` on validation against a Beta baseline of `0.2228`, so roughly
`0.062` of headroom exists. Every rung of the ablation ladder moved Brier ALC
by `1e-4` to `1e-3`. Separately, the aggregate null hides two opposite regimes:
tasks whose error curve declines beat Beta by `0.005`-`0.011`, tasks whose curve
is flat near `0.25` lose by `0.002`-`0.009`, and they cancel.

## 2. Features the old pipeline ignored

Measured in `modal_field_audit.py` and a follow-up probe.

| Source | Finding |
|---|---|
| `benchmarks` table | **entirely unused.** `domain` is 100% filled with 21 atomic skills (reasoning 18, safety 12, knowledge 7, software_engineering 6, science 5, general 5, agents_and_tool_use 5, multilingual 5, medicine 4, mathematics 3, plus 11 singletons), mean 1.5 per benchmark, max 3 |
| `modality` | 100% filled, 6 values: text 54, image 10, audio 2, video 1, gui_screenshot 1, time_series 1 |
| `item_features` | 98% filled and **parseable `key=value`**: `context_length` (276k), `input_context_status`, `task`, `lang` (78k), `split`, `mmlu_subject` (14k), `dataset`, `prompt_origin`, `category`, and an explicit **`difficulty`** key on 1,534 items |
| `verifier` | real cardinality is **2**, not 1,877: `exact_matcher` (1,290,450) and `judge` (65,862); both span benchmarks |
| `content_hash` | spans benchmarks for only 215 of 769,888 values, so there is no item-level bridge; 47% of rows share a hash with another row inside a benchmark |
| `content_chars` | correlates with empirical difficulty at `r = -0.29` overall but only `-0.077` within benchmark -- length mostly proxies benchmark identity |
| anchor items | 6,450 items carry >= 10 responses; their empirical difficulty has **SD 2.28 logits** |

## 3. Feature plan

| Source | Feature | Encoding | Fill |
|---|---|---|---|
| item | `content` | bge-m3, **2048-token cap**, fine-tuned in stage 3 | 100% |
| item | `context_length` | numeric, log1p | 20% |
| item | `lang`, `mmlu_subject`, `dataset`, `task`, `category`, `split` | learned categorical embeddings | varies |
| item | `difficulty` (explicit) | ordinal + missing flag; auxiliary target | 0.1% |
| item | `verifier_class` | binary (`exact_matcher` / `judge`) | 100% |
| item | `log(content_chars)` | numeric | 100% |
| item | duplicate count by `content_hash` | numeric | 100% |
| subject | `provider` | embedding, 59 levels | 66% |
| subject | model family (`normalized_name` + regex-mined `model_identifier=`) | embedding | ~70% |
| subject | days since `release_date` | numeric, standardized | 40% |
| subject | `harness` | embedding, 66 levels | 100% |
| subject | per-channel missing flags | binary | 100% |
| benchmark | `domain` | multi-hot, 21 | 100% |
| benchmark | `modality` | multi-hot, 6 | 100% |
| benchmark | `has_reference_answer`, `max_trial`, `log n_items`, `coverage`, release date | numeric / binary | 94-100% |

Dropped: `subject_features_extra` (7,951 distinct of 8,104 -- a near-unique
fingerprint that prevents sharing), `reasoning_effort` (0% filled),
`harness_version` (6%, no bridge), `access_date` (1%, no bridge).
Off-limits by competition contract: `grading_criterion`, `asset_manifest`.

## 4. The model

Towers, with `K` the ability rank:

$$\mathbf{v}_i = \mathrm{MLP}_{\text{item}}\big(f_\phi(\text{text}_i) \,\|\, \mathbf{c}_i \,\|\, \mathbf{n}_i\big) \in \mathbb{R}^{K}, \qquad b_i \in \mathbb{R}$$

$$\mathbf{u}_j = \mathrm{MLP}_{\text{subj}}\big(\mathbf{e}_{\text{prov}} \,\|\, \mathbf{e}_{\text{family}} \,\|\, \mathbf{e}_{\text{harness}} \,\|\, t_j \,\|\, \text{flags}\big)$$

$$\mathbf{g}_b = \mathrm{MLP}_{\text{bench}}\big(\mathbf{d}_b \,\|\, \mathbf{m}_b \,\|\, \mathbf{n}_b\big)$$

Prior ability, available before any label is revealed:

$$\boldsymbol{\theta}_j^{(0)} = W\big[\mathbf{u}_j \,\|\, \mathbf{g}_b \,\|\, \mathbf{u}_j \odot \mathbf{g}_b\big] \in \mathbb{R}^{K}$$

Online update from the `n` acquired responses, permutation invariant, no Newton
solve and no `no_grad` barrier:

$$\boldsymbol{\theta}_j^{(n)} = \boldsymbol{\theta}_j^{(0)} + \sigma(\alpha_n)\cdot \rho\Big(\tfrac{1}{n}\sum_{k \in C}\psi\big(\mathbf{v}_k, b_k, y_k\big)\Big)$$

Prediction:

$$\mathrm{logit}\,P(y_{ij}=1) = \boldsymbol{\theta}_j^{(n)\top}\mathbf{v}_i - b_i + \gamma_b$$

### What each term repairs

| Term | Repairs |
|---|---|
| `theta^T v` with `K` in `[4,16]` | difficulty is relative to the subject: a safety model and a math model are not ordered on one axis |
| `g_b` built from `domain` | makes a `K`-dimensional ability learnable without needing `K` responses per task |
| `u_j (*) g_b` | ability is relative to the benchmark |
| `theta^(0)` from metadata | budget 0 stops being a coin flip; it is currently Brier `0.25` by construction |
| `gamma_b` | absorbs the per-benchmark constant that is genuinely unidentifiable, instead of letting the model fight it |
| set encoder, gradients flow | removes the `@torch.no_grad()` barrier that made the old correction untrainable |
| `sigma(alpha_n)` gate | lets the model stay at its prior on flat-profile tasks, which is the measured loss regime |
| fine-tuned `f_phi` | attacks the `0.062` item-difficulty gap directly |

## 5. Training stages

| Stage | Content | Rationale |
|---|---|---|
| 0 | Re-encode item text with bge-m3 at 2048 tokens, frozen | recovers 14 percentage points of text over the current 512 store |
| 1 | Fit `v_i`, `b_i` against empirical difficulty on the 6,450 anchor items | real supervision with SD 2.28 logits to predict |
| 2 | Joint training of all towers, text encoder frozen | learn the `K`-dimensional structure cheaply |
| 3 | Unfreeze bge-m3, fine-tune on the IRT-style loss | the actual prize |
| 4 | Padded/masked batching and full task coverage | removes the sequential-episode bottleneck |

Loss:

```text
BCE(held-out half) + lambda * || theta^(n) - theta^(0) ||^2 + mu * anchor-difficulty regression
```

The shrinkage term keeps the gate honest: the online update must earn its
deviation from the metadata prior.

## 6. Known limitations

- **`K` is the main risk.** A 16-dimensional online update cannot be identified
  from 31 responses. The *prior* may be full rank; the *update* must be small.
  Start at `K = 4` and ablate upward.
- **`difficulty=` is 0.1% filled.** Useful as an auxiliary target, never as a
  relied-upon feature.
- **Length is a trap.** `r = -0.29` overall collapses to `-0.077` within
  benchmark. Without `gamma_b` the model will learn "long means hard", which is
  benchmark identity in disguise.
- **About 35% of subjects have no provider and no name**, so their
  `theta^(0)` falls back to harness and benchmark priors only.
- **`gamma_b` is identified only through subjects that span benchmarks.**
  Provider coverage is 65%, so it is estimable but noisy, and the remaining
  benchmarks lean on weak evidence.
- **This is no longer IRT.** It is matrix factorization with side information.
  The clean "item difficulty" interpretation weakens, which is a real cost to
  the paper narrative and should be stated rather than hidden.
- Mean domain count is 1.5 per benchmark and 11 of 21 domains appear in a
  single benchmark, so for those the domain signal and `gamma_b` are the same
  thing.

## 7. Related

Why a better encoder has not moved calibrated ALC:
[`why-calibration-collapses.md`](why-calibration-collapses.md).
