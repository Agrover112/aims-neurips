# Completed ablations (current campaign)

Status: 2026-10-22. This is the working record, not the paper.

Version: **v3-postfix** (MiniLM@512, masked subject channel). Protocol: 104
validation tasks, 10 frozen orders, budgets `0,1,3,7,15,31`, five-fold OOF
calibration, calibrated Brier ALC. Lower is better.

**Preliminary trainer warning:** all rows below use `800` steps x `4` episodes.
That is only 3,200 episodes for 6,702 train tasks. The numbers are
version-correct but not training-converged. Do not compare them to a future
coverage-trained run.

**Additive anchor fix (2026-10-22):** the CNP-additive rows below predate the
fix. They used context-edited `a_C,b_C` in the ability MAP. The fixed code
uses item-only base parameters and applies additive offsets to targets only,
so those rows are not comparable to a post-fix rerun.

**Next campaign:** the corrected-protocol ladder is specified in
[`faithful-ablation-ladder.md`](faithful-ablation-ladder.md). It is **not run**
yet. No row below belongs to that ladder, and its numbers must not be mixed
with these tables.

Beta-Binomial reference: **0.222838**.

## r_C ladder, 3 seeds

| arm | seed 18 | seed 19 | seed 20 | mean |
|---|---:|---:|---:|---:|
| 2PL | 0.222429 | 0.223325 | 0.222570 | 0.222775 |
| scalar | 0.222727 | 0.222998 | 0.222453 | **0.222726** |
| additive | 0.222108 | 0.223415 | 0.222748 | 0.222757 |
| ANP | 0.222287 | 0.223689 | 0.222412 | 0.222796 |
| CNP | 0.222475 | 0.223445 | 0.222627 | 0.222849 |
| direct | 0.222549 | 0.223630 | 0.223309 | 0.223163 |
| residual | 0.223335 | 0.223763 | 0.223557 | 0.223552 |

Paired task deltas, seeds averaged within task:

| arm minus 2PL | delta | 95% task CI | reading |
|---|---:|---:|---|
| scalar | -0.000049 | [-0.000114, +0.000022] | tie |
| additive | -0.000018 | [-0.000158, +0.000121] | tie |
| ANP | +0.000021 | [-0.000178, +0.000225] | tie |
| CNP | +0.000074 | [-0.000174, +0.000323] | tie |
| direct | +0.000388 | [+0.000120, +0.000636] | worse |
| residual | +0.000777 | [+0.000264, +0.001279] | worse |

**Verdict:** r_C does not earn its keep. The best point estimate (`scalar`,
-0.000049) is inside noise. Direct and residual are reliably worse than 2PL.

## Loops, 3 seeds

Training uses held-out-only targets. Compare within loop family only; do not
compare these rows directly to the r_C ladder.

| family | K=0 | K=1 | K=3 |
|---|---:|---:|---:|
| additive | 0.222737 | 0.222782 | **0.222431** |
| direct | **0.222587** | 0.222688 | 0.222543 |

Per-seed values:

| arm | seed 18 | seed 19 | seed 20 | mean |
|---|---:|---:|---:|---:|
| additive K=0 | 0.222346 | 0.223364 | 0.222500 | 0.222737 |
| additive K=1 | 0.221970 | 0.223261 | 0.223117 | 0.222782 |
| additive K=3 | 0.222378 | 0.222702 | 0.222213 | 0.222431 |
| direct K=0 | 0.221811 | 0.223258 | 0.222691 | 0.222587 |
| direct K=1 | 0.222227 | 0.223213 | 0.222625 | 0.222688 |
| direct K=3 | 0.222000 | 0.222664 | 0.222966 | 0.222543 |

Paired task deltas vs K=0:

| comparison | delta | 95% task CI | reading |
|---|---:|---:|---|
| additive K=1 | +0.000046 | [-0.000062, +0.000147] | tie |
| additive K=3 | -0.000306 | [-0.000455, -0.000158] | small win |
| direct K=1 | +0.000102 | [-0.000032, +0.000233] | tie |
| direct K=3 | -0.000044 | [-0.000162, +0.000076] | tie |

**Verdict:** only additive K=3 survives, and the gain is tiny. Not worth the
extra complexity unless a coverage-trained replication shows a larger effect.

## AR, single seed, held-out-only

Training uses held-out-only targets. Only one seed (`20261018`) is complete.
Compare AR with carried state against the matched no-AR control; do not compare
these rows to the r_C ladder.

| arm | ALC | delta vs no-AR | 95% task CI |
|---|---:|---:|---:|
| additive no-AR | 0.222142 | — | — |
| additive AR | 0.222474 | +0.000331 | [+0.000047, +0.000609] |
| direct no-AR | 0.222638 | — | — |
| direct AR | **0.222044** | -0.000594 | [-0.000824, -0.000371] |

**Verdict:** decoder-dependent. AR hurts the additive decoder and helps the
direct decoder. Single seed; not evidence until seeds 19/20 replicate.

## GP/metadata prior and NR, 3 seeds, non-disjoint

Training targets include context items (`heldout_only=False`), so these rows
are matched to the 2PL/additive rows but not to the disjoint AR/loop protocol.

| arm | ALC | matched control | delta | 95% task CI |
|---|---:|---:|---:|---:|
| 2PL | 0.222775 | — | — | — |
| GP-2PL | **0.221941** | 2PL | -0.000833 | [-0.001155, -0.000512] |
| additive | 0.222757 | — | — | — |
| GP-additive | **0.222031** | additive | -0.000726 | [-0.001064, -0.000388] |
| Meta-GP | **0.221793** | GP-2PL | -0.000148 | [-0.000400, +0.000104] |
| NR-2PL | 0.223461 | 2PL | +0.000686 | [+0.000089, +0.001283] |
| NR-2PL + prior | 0.223302 | NR-2PL | -0.000159 | [-0.000265, -0.000052] |
| U-NR-2PL | 0.223497 | NR-2PL | +0.000037 | [-0.000042, +0.000115] |
| U-NR-2PL + prior | 0.223419 | U-NR-2PL | -0.000079 | [-0.000134, -0.000024] |

GP-2PL per-seed deltas vs 2PL are `-0.002231`, `-0.000230`, `-0.000040`.
Seed 18 drives the mean; the other two seeds are essentially ties. Meta-GP is
a tie with GP-2PL, not a reliable gain. NR/U-NR are worse than 2PL and are
terminal negatives under this protocol.

**Verdict:** the GP point estimate is still the only positive signal, but it is
fragile and not a full GPIRT posterior. The NR family is not promoted.

## Not completed or stopped

- AR seeds 19/20: stopped; do not rerun.
- Disjoint GP-2PL and metadata-prior GP-2PL: now part of the v4.1 ladder, not a
  v3 patch.
- Full paper GPIRT: not run and not identifiable for unseen target items
  (train/eval item overlap is zero).
