# AIMS concept dictionary

This glossary separates the AI system being evaluated, our submitted predictor,
the task, the observed result, and the probability we predict. Those roles are
easy to mix up because they are all called "models" or "responses" in different
contexts.

## The five objects to remember

| Term | Plain meaning | Example |
|---|---|---|
| **Subject** | The AI system being evaluated. It may be a model, agent, or model-plus-agent configuration. | A particular model running through OpenHands. |
| **Item** | One task presented to the subject. | One coding issue, math problem, or web task. |
| **Outcome** | What actually happened when that subject attempted that item. For competition scoring, this is `0` or `1`. | `1` means success; `0` means failure. |
| **Predictor** | Our submitted program. It estimates the outcome without running the subject. | `submission/model.py`. |
| **Prediction** | The predictor's probability that the outcome will be `1`. | `0.72` means an estimated 72% chance of success. |

The central relationship is:

```text
evaluated AI subject + evaluation item
                  |
                  v
          our submitted predictor
                  |
                  v
       predicted probability p
                  |
                  v
   compared with actual outcome y
                  |
                  v
             Brier error
              (p - y)^2
```

The **subject is not our predictor**. Our predictor predicts how the subject
will perform.

## Dataset terms

| Term | Meaning |
|---|---|
| **Benchmark** | A collection of related items with a shared evaluation setup. |
| **Subject-benchmark pair** | One subject evaluated on one benchmark. Scores are calculated separately for each such pair before the leaderboard averages them. |
| **Response** | The recorded grade for one subject-item attempt. In the Data Bank it is stored as a number and may be binary or fractional. |
| **Trial** | The attempt number when a subject attempts the same item more than once. |
| **Trace** | The raw output or interaction record that produced a response. A trace is not the grade itself. |
| **Test condition** | A recorded variation in how the evaluation was run, such as text-only versus multimodal. |
| **Interactors** | Other agents, tools, or systems involved in the attempt. |
| **Anonymous benchmark ID** | A stable name such as `benchmark_274926`. It lets us group items from the same hidden benchmark without revealing its real name. |
| **Response matrix** | A table with subjects as rows, items as columns, and observed outcomes as cells. Missing cells mean that combination was not observed. |

### How the Data Bank tables connect

```text
subjects ── subject_id ──┐
                        ├── responses ── response_id ── traces
items ───── item_id ─────┘
  |
  └── benchmark_id ── benchmarks
```

- `subjects` describes evaluated AI systems.
- `items` describes tasks.
- `responses` contains recorded grades.
- `traces` contains available raw outputs.
- `benchmarks` describes each task collection.

## Test-time labels

| Term | Meaning |
|---|---|
| **Acquisition pool** | Items whose outcomes may be revealed to us. These items are not scored. |
| **Evaluation pool** | Separate items on which our predictions are scored. Their outcomes remain hidden. |
| **Acquire a label** | Spend one unit of label budget to reveal an acquisition item's actual `0` or `1` outcome. No money is involved. |
| **Label** | A revealed outcome supplied back to our predictor as evidence. |
| **Label budget** | The maximum number of outcomes we may reveal. The checkpoints are `0, 1, 3, 7, 15, 31`. |
| **Nested labels** | Higher budgets retain the labels acquired at lower budgets. Budget 7 includes the earlier budget-1 and budget-3 evidence. |
| **Acquisition policy** | The logic in `labeling.py` that accepts or rejects each arriving acquisition candidate. |
| **Test-time adaptation** | Updating predictions using labels revealed during evaluation. It does not permit access to evaluation outcomes. |
| **Adaptive testing** | Choosing informative acquisition items so that a small label budget teaches us as much as possible about the subject. |

The competition's flow is:

```text
candidate acquisition item arrives
              |
              v
 labeling.py accepts or rejects it
              |
       accepted: reveal outcome
              |
              v
 add outcome to labeled evidence
              |
              v
predict evaluation items using that evidence
```

## Probabilities and calibration

| Term | Meaning |
|---|---|
| **Probability** | A number from 0 to 1 representing estimated chance of success. |
| **Logit** | A probability expressed on an unrestricted score scale: `log(p / (1-p))`. A probability of `0.5` has logit `0`. |
| **Sigmoid** | Converts a logit back into a probability: `1 / (1 + exp(-z))`. |
| **Calibration** | Agreement between predicted probabilities and observed frequencies. Among many cases predicted near `0.7`, about 70% should succeed. |
| **Overconfidence** | Predictions are too extreme relative to observed results. |
| **Underconfidence** | Predictions stay too close to `0.5` relative to observed results. |
| **Platt scaling** | Learns a slope and bias on raw logits: `sigmoid(a*z + b)`. It changes confidence and the overall upward or downward tendency. |
| **Temperature scaling** | Divides logits by `T`. `T > 1` softens probabilities; `T < 1` makes them more extreme. |
| **Anchor** | An acquired example for which both the original prediction and actual outcome are known. |
| **AnchorCalibrator** | Uses acquired anchors to estimate temperature and bias, optionally by category. Sparse category estimates are blended with a pooled estimate. |
| **Shrinkage** | Pulling an estimate based on few observations toward a more stable pooled estimate instead of trusting the small sample completely. |
| **Category** | A group chosen for calibration, such as a benchmark or evaluation condition. It is not automatically a class label. |

Calibration does not determine whether one answer is correct. It asks whether
probabilities are honest across many predictions.

## Item response theory

Item response theory, or **IRT**, models success using subject ability and item
difficulty. A common form is:

```text
P(success) = c + (1-c) * sigmoid(a * (theta - b))
```

| Symbol | IRT meaning |
|---|---|
| `theta` | Subject ability. Higher values imply greater capability. |
| `b` | Item difficulty. Higher values reduce success probability. |
| `a` | Item discrimination. Larger values make the item separate stronger and weaker subjects more sharply. |
| `c` | Guessing or lower-asymptote parameter. It sets a nonzero success floor for very weak subjects. |

The symbols are local to their formulas. In IRT, `b` means difficulty and is
subtracted from ability. In Platt scaling and AnchorCalibrator, `b` means a
calibration bias and moves logits upward or downward.

| Model | What it does |
|---|---|
| **Rasch / 1PL** | Learns subject ability and item difficulty. |
| **2PL** | Adds item discrimination. |
| **3PL** | Adds discrimination and guessing. |
| **AmortizedIRT** | Learns to predict IRT item parameters from item embeddings, allowing difficulty estimates for unseen items. It then combines those parameters with subject ability to predict success. |
| **NCF** | Neural Collaborative Filter. It embeds subject and item descriptions, combines them with a neural network, and predicts success probability. The `torch_measure` version can apply Platt calibration using acquired labels. |

## Competition scoring

| Term | Meaning |
|---|---|
| **Brier score** | Mean squared probability error: the average of `(prediction - outcome)^2`. Lower is better. |
| **Learning curve** | Brier score plotted against the number of acquired labels. It shows whether predictions improve as evidence arrives. |
| **ALC** | Area under the learning curve. The competition uses a normalized trapezoidal average across its six budgets. |
| **Brier ALC** | The primary score: `0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31`. Lower is better. |
| **ECE** | Expected calibration error. It compares mean predicted probability with observed success rate inside probability bins. Lower is better, but it is diagnostic rather than the primary score. |

## One-sentence mental model

We use public response data and a few labels acquired at test time to predict the
probability that a particular AI subject will succeed on a particular hidden
evaluation item, and Brier ALC measures how accurate those probabilities are as
the label budget grows.
