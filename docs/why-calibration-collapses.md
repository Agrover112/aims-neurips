# Why calibration collapses every embedding onto Beta

Status: 2026-10-23. Measured, not speculative. Sources:
`results/allbench-embedding-ablation/embedding-*-masked.json` (end-to-end ALC
for six embedding arms) and a ridge probe of item difficulty on the anchor
items.

## Definitions

- **Item** -- one question. A task holds roughly 500 evaluation items.
- **Task** -- one `(subject, benchmark)` pair, i.e. one model evaluated on one
  benchmark.
- **Between-item difficulty** -- which questions are harder *in general*,
  averaged across all models.
- **Within-task ranking** -- for *this particular model*, which of *its own*
  evaluation items will it get wrong.

Within-task ranking is the quantity the competition scores. Between-item
difficulty is necessary for it but not sufficient.

## 1. Every embedding was scored end to end

Validation, 104 tasks, masked mode. Beta-Binomial reference `0.222838`.

| Embedding arm | Predictor | Raw ALC | Calibrated ALC |
|---|---|---:|---:|
| `bge_base_2pladapt` | cnp | 0.228785 | **0.222584** |
| `bge_base` | cnp | 0.225759 | 0.222822 |
| Beta (all arms) | beta | 0.222838 | 0.222838 |
| `minilm` | cnp | 0.235004 | 0.222894 |
| `minilm512` | cnp | 0.233960 | 0.223095 |
| `bge_m3` | cnp | 0.228605 | 0.223662 |

`bge_base_2pladapt` is the best arm ever run and beats Beta by `0.000254`,
which is indistinguishable from zero.

## 2. Calibration erases the differences

| | Spread across the six arms |
|---|---|
| Raw ALC | `0.2241` to `0.2684` -- **0.044** |
| Calibrated ALC | `0.2226` to `0.2237` -- **0.001** |

Calibration removes roughly 97% of the difference between embeddings. A better
encoder improves raw ALC substantially and is then shrunk back into
irrelevance.

## 3. The mechanism: the fitted scale

`AnchorCalibrator` applies `p = sigmoid(scale * logit + bias)`. The fitted
values on these runs are `scale` about `0.22` to `0.36` and `bias` about
`0.22`, so logits are shrunk by 65-78%.

With `scale = 0.26`, `bias = 0.22`:

| Model output (logit) | Raw probability | Calibrated probability |
|---|---:|---:|
| `+2.0` confident correct | 0.88 | **0.68** |
| `0.0` no information | 0.50 | 0.55 |
| `-2.0` confident wrong | 0.12 | **0.43** |

```text
raw:        0.12 ------------------------ 0.88
calibrated:              0.43 ---- 0.68
```

The usable range collapses toward `0.5`, which is the corpus base rate
(`grand mean label = 0.486`). Predicting the base rate everywhere is exactly
what the Beta-Binomial baseline does, so the calibrated neural model converges
onto Beta by construction.

## 4. The shrinkage is correct, not a bug

`uniform-audit.md` records validation raw AUC near `0.48`. AUC measures
within-task ranking: `1.0` is a perfect ordering, `0.5` is random. At `0.48`
the model orders a task's items slightly worse than chance.

When a ranking carries no information, the prediction minimizing Brier and
log-loss is the base rate for every item. The calibrator detects this and
shrinks accordingly. It is behaving correctly given what it is handed.

## 5. The contradiction to resolve

| Measurement | Value |
|---|---:|
| Ridge on `bge_base_2pladapt`, predicting item difficulty (anchor items, n >= 20) | corr **0.79**, R^2 **0.62** |
| Deployed model, within-task item ranking | AUC **0.48** |

Both cannot describe the same pipeline. Two candidate explanations:

**(1) Between-item difficulty does not transfer to within-task ranking.**
Knowing that one benchmark is harder than another does not help rank 500 items
*inside* a single task. If the evaluation items within a task carry little
difficulty spread, globally accurate difficulty estimates buy nothing locally.
This is the same trap already observed with `content_chars`, whose correlation
with difficulty is `-0.29` overall but only `-0.077` within benchmark.

**(2) The `a(x)`, `b(x)` head fails to read the embedding.** The ridge probe
proves the difficulty signal is linearly decodable from the embedding. The item
head is a small network trained only through the outer loss, with `map_theta`
under `@torch.no_grad()` and 800 steps over 6,702 tasks. The information is
present; the head never learns to extract it.

These have opposite remedies, so they must be distinguished before any further
training:

- If **within-benchmark** and **benchmark-held-out** R^2 stay high, the signal
  is local and explanation (2) holds. The fix is the head and the training
  loop, not the encoder.
- If they collapse toward zero, explanation (1) holds. The fix requires
  subject-by-item interaction terms, because a single scalar difficulty cannot
  produce within-task ordering.

## 6. Consequence for the roadmap

No encoder change is justified until this is settled. The adapter already
proves a better representation moves raw ALC and not calibrated ALC, so
representation is not the binding constraint. The binding constraint is
whichever of (1) or (2) is true.
