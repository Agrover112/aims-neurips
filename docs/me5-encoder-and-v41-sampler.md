# me5 encoder and v4.1 sampler: measured results

Status: 2026-10-25. Every number is measured on the 104 validation tasks with
out-of-fold calibration, masked mode, acquisition order 0, the competition's
six budgets and weights. Beta-Binomial reference is `0.222838` throughout and
matches `docs/ablation-results.md` to six decimals, so these rows are directly
comparable to the published r_C ladder.

Sources: `modal_embed_v2.py`, `modal_compact_store.py`, `modal_r_c_ablation.py`.
Result files: `/data/results/r-c-ablation/r-c-{rung}-{seed}-masked-{store}.json`,
where a `-v41e1` suffix marks the v4.1 sampler.

## 1. What changed

Two independent changes were made and measured separately.

**Encoder.** Items are now encoded as `item_content` alone with
`intfloat/multilingual-e5-large-instruct` (1024-d, 512 tokens, mean pooling,
instruction prefix), replacing `all-MiniLM-L6-v2` at 256/512 tokens on
`f"item: {content}\nmetadata: {item_features}"`. Rationale and selection
evidence are in the README decision log; the corpus language audit is in
`docs/language-distribution.pdf`.

**Sampler.** `modal_r_c_ablation.py` gained a `--sampler v41` path that uses
`build_pair_epoch_batches` and `sample_pair_episode` from
`aims_starter.faithful_ladder`, matching the v4.1 training contract already
used by `modal_faithful_ladder.py`.

| | `random` (previous) | `v41` |
|---|---|---|
| Episodes | 800 steps x 4 = 3,200 draws | 2,514 batches x 16 = **40,212 visits** |
| Coverage of (task, budget) | ~7.7% | **100%, once per epoch** |
| Context/target | overlapping | **disjoint**, rotating split |
| Split key | n/a | `(task, budget, pass)`, model-seed independent |
| Optimizer | Adam + cosine | AdamW + warmup + cosine |

The old sampler's docstring states its intent plainly: *"Match the production
CNP episode sampler (context included in targets)."* It was faithful to the
pre-v4.1 behaviour, which is what the v4.1 work replaced.

## 2. Encoder effect: the only result that replicates

Validation ALC, me5 store, `random` sampler, three seeds.

| Rung | s18 | s19 | s20 | mean | spread | vs Beta |
|---|---:|---:|---:|---:|---:|---:|
| `direct_loop0` | 0.220859 | 0.219836 | 0.220042 | **0.220245** | 0.00102 | **-0.002593** |
| `mirt_direct_loop0_d4` | 0.220152 | 0.220723 | 0.220014 | 0.220296 | 0.00071 | -0.002542 |
| `mirt_scalar_d4` | 0.220291 | 0.220468 | 0.220338 | 0.220366 | **0.00018** | -0.002472 |
| `scalar` | 0.220791 | 0.220253 | 0.220440 | 0.220494 | 0.00054 | -0.002344 |
| `additive_loop3` | 0.220358 | 0.220487 | 0.222066 | 0.220970 | **0.00171** | -0.001868 |
| `mirt_additive_loop3_d4` | 0.220678 | 0.221018 | 0.221583 | 0.221093 | 0.00091 | -0.001745 |
| Beta-Binomial | -- | -- | -- | 0.222838 | -- | -- |

Against the published `minilm512` numbers for the same rungs
(`additive_loop3` 0.222431, `direct_loop0` 0.222587, `scalar` 0.222726), the
encoder swap is worth **-0.0017 to -0.0023** on every architecture tested. It
is the only effect in this document that holds across seeds, samplers, and
architectures.

## 3. v4.1 sampler effect: real, and only for the structure-free models

me5 store, seed 20261018.

| Rung | v4.1 ALC | vs Beta | vs `random`, same seed |
|---|---:|---:|---:|
| `anp_attn` | **0.219793** | **-0.003045** | **-0.001540** |
| `mirt_scalar_d4` | 0.219945 | -0.002893 | -0.000346 |
| `cnp` | 0.220088 | -0.002750 | **-0.001189** |
| `direct_loop0` | 0.220496 | -0.002342 | -0.000363 |
| `mirt_direct_loop0_d4` | 0.220668 | -0.002170 | +0.000516 |
| `scalar` | 0.220820 | -0.002018 | +0.000029 |
| `additive_loop3` | 0.221208 | -0.001630 | +0.000850 |

Paired task-level bootstrap, 10,000 resamples, v4.1 minus random:

| Rung | delta | 95% CI | tasks improved |
|---|---:|---|---:|
| `anp_attn` | -0.001540 | [-0.002457, -0.000535] | 78/104 |
| `cnp` | -0.001189 | [-0.001893, -0.000456] | 80/104 |
| `scalar` | +0.000029 | [-0.000785, +0.000876] | 60/104 |

**The split follows where ability comes from.** `cnp` and `anp_attn` must
*learn* adaptation from `r_C`, so they need to see many `(task, budget)` pairs;
at 7.7% coverage they never did. The IRT-anchored rungs get adaptation from a
Newton MAP solve at inference, so training coverage barely moves them.

This reverses an earlier reading of these runs. Under the old sampler the IRT
anchor looked worth about `0.0005` over a structure-free CNP. Under the correct
sampler `anp_attn` (0.219793) beats `scalar` (0.220820): the anchor's apparent
advantage was an artifact of undertrained baselines.

## 4. bge-m3 is worse, and context length is not the problem

bge-m3 store, `random` sampler, seed 20261018. Same corpus, same cap, same
sampler; only the encoder differs.

| Rung | ALC | vs Beta |
|---|---:|---:|
| `mirt_scalar_d4` | 0.223336 | +0.000498 |
| `mirt_direct_loop0_d4` | 0.223378 | +0.000540 |
| `additive_loop3` | 0.223423 | +0.000585 |
| `scalar` | 0.223462 | +0.000624 |
| `direct_loop0` | 0.223492 | +0.000654 |

All five land **above** Beta, in a 0.00016-wide band.

bge-m3 truncates **1,838** items at its 8192-token cap against me5's **172,297**
at 512 -- a 94-fold difference -- and loses anyway. Token truncation is not what
was limiting the item channel. The ordering matches the independent benchmarks
that selected me5 in the first place (AfriMTEB-Lite 62.0 against 55.0; MMTEB
63.2 against 59.6), so encoder quality, not context length, is the operative
variable.

## 5. Multidimensional ability: withdrawn

`MultidimContextIRT` replaces the scalar ability with a vector `theta` in R^D
and a compensatory logit `a_i . theta - d_i`, with a D-dimensional Newton MAP.
On seed 20261018 two of three paired comparisons had 95% intervals excluding
zero, which would have been the first such result in this project.

It does not survive more seeds. MIRT(D=4) minus its scalar twin, paired by task:

| Pair | s18 | s19 | s20 |
|---|---:|---:|---:|
| `mirt_additive_loop3_d4` | +0.000320 | +0.000531 | -0.000482 |
| `mirt_direct_loop0_d4` | **-0.000706** | **+0.000887** | -0.000028 |
| `mirt_scalar_d4` | **-0.000500** | +0.000215 | -0.000101 |

The sign flips. `mirt_direct_loop0_d4` swings by `0.0016` between seeds 18 and
19 on a quantity whose claimed effect was `0.0007`.

The `D=1` controls, which use the same `a . theta - d` reparametrization at one
dimension, land at 0.220515 / 0.220998 / 0.221400 on seed 18 -- interleaved with
the `D=4` arms rather than separated from them. Nothing distinguishes
multidimensionality from the reparametrization.

**Verdict: no evidence for multidimensional ability on this corpus.** The code
is kept and the rungs stay registered, but the result is not claimed.

## 6. Architecture is inside seed noise

`additive_loop3` was the best arm in the published r_C loop ladder (3-seed mean
0.222431, the only comparison there whose CI excluded zero). On the me5 store it
has the **largest** seed spread of any arm (0.00171), is last under the v4.1
sampler, and its mean is worse than four other rungs.

Seed spread across the six main rungs runs 0.00018 to 0.00171. Every
architecture gap in Table 2 is smaller than the largest of those spreads. No
architecture conclusion from single-seed runs on this corpus should be trusted,
including the ones in this document that are marked single-seed.

## 7. The gap, decomposed

| Quantity | ALC |
|---|---:|
| Oracle item parameters + oracle ability | 0.1013 |
| Information ceiling (oracle items, 31-label ability) | 0.1608 |
| Best measured here (`anp_attn`, v4.1) | 0.219793 |
| Beta-Binomial | 0.222838 |

```
0.1608 - 0.1013 = 0.0595   cost of not knowing theta
0.2198 - 0.1608 = 0.0590   cost of not knowing (a_i, b_i)
```

The two error sources are equal to within 1%. Work to date has addressed only
the second. The ceiling is also conditional on assumptions that are not all
fixed: a correctly specified 2PL, the given acquisition order, and a posterior
integrated by quadrature. The models here plug a MAP point estimate into the
response function instead of integrating, so they do not use the method the
bound assumes.

## 8. Status

Implemented and measured:

- four 1024-d `item_content`-only embedding stores (`modal_embed_v2.py`);
  me5 and bge-m3 complete, both Qwen arms still encoding
- per-cap training stores (`modal_compact_store.py`)
- v4.1 coverage sampler in the r_C runner (`--sampler v41 --pair-epochs N`)
- `MultidimContextIRT` and its `d1`/`d4`/`d8` rungs

Implemented, not yet wired in:

- `marginalize_logits` in `r_c_attention.py`. The helper exists; `map_theta`
  still discards the Hessian and no `forward` calls it. No results.

Not started:

- the achievable-ceiling ladder: refit the oracle 2PL with `b_i` constrained to
  a function of item text, fit within benchmark and then on held-out
  benchmarks. This separates the text-information limit from the transfer
  limit and is the measurement that decides whether further item-encoder work
  is worth anything
- within-benchmark centering of `b`
- a 3PL lower asymptote
- Fisher-information acquisition
- Qwen arms, and v4.1 runs at more than one seed
