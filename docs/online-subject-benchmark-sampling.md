# How the Codabench hidden test set is sampled for online subject–benchmark evaluation

**Status:** reconstruction, 2026-10-28
**Evidence:** the official starting-kit README plus the scoring logs of six completed submissions.
**Scope:** how the platform appears to draw subject–benchmark pairs, split their items, and score them online. This is not an official specification; the starting kit is authoritative where it speaks.

---

## 0. Bottom line

The platform does **not** create a fresh test set per submission.

It appears to:

1. Start from a fixed hidden pool of eligible subject–benchmark pairs.
2. Draw **complete pairs** under a limit of about **1,000 distinct source items**.
3. Give each selected pair a **persistent random 50/50 split**: an acquisition pool and a fixed, unqueryable evaluation pool.
4. Let the submission buy up to **31 labels** from the acquisition pool, one candidate at a time.
5. Score the **same fixed evaluation pool** at budgets `0, 1, 3, 7, 15, 31`.
6. Average the six-budget Brier curve **unweighted across pairs**.

The six logs show 8–9 pairs per submission, seven benchmark IDs, and evaluation-pool counts that sum to **489–496** per submission. Under the official 50/50 rule, that implies **979–994 distinct source items** per draw — almost exactly the ~1,000-item cap. That is the strongest evidence for the reconstruction.

---

## 1. Evidence and confidence

| Claim | Evidence | Confidence |
|---|---|---|
| A pair must have at least **80 distinct items** to be eligible | Starting kit, “Streaming test-time adaptation” | High |
| Each pair has a persistent random **50/50** acquisition/evaluation split | Starting kit, same section | High |
| Repeated response runs of an item stay in the same pool | Starting kit, same section | High |
| Odd item counts give the extra item to **acquisition** | Starting kit, same section | High |
| The draw limit is about **1,000 distinct source items**, applied **before** the split | Starting kit, interface section | High |
| The platform draws **complete subject–benchmark groups**, not individual items | Starting kit, interface section | High |
| Groups that would exceed the remaining draw budget are skipped | Starting kit, interface section | High |
| Preference goes to benchmarks/subjects represented less often in the draw; ties are random | Starting kit, interface section | High |
| Logs show **8–9 pairs** per submission | Six scoring logs | High |
| Logs show **seven benchmark IDs**, and every submission contains all seven | Six scoring logs | High |
| Same pair recurring across submissions has the **same evaluation-item count** | Six scoring logs | High |
| Evaluation counts sum to **489–496** per submission | Six scoring logs | High |
| Implied total source items are **979–994** | Logs + official 50/50 rule | High |
| The hidden benchmark pool is exactly seven | Logs show only seven; total pool not observable | Medium |
| The exact balancing algorithm | Official text says “preference” and “random tie-breaking,” but not the full rule | Low / unknown |

---

## 2. Reconstructed sampling pipeline

### 2.1 Eligibility

A subject–benchmark pair is eligible only if the subject has at least **80 distinct items** in that benchmark.

Eligibility counts **distinct items**, not response runs. A subject that answered the same item several times still contributes one item to the eligibility count.

A subject is a base model plus its evaluation settings (provider, harness, reasoning effort, and so on). The same base model under different settings is a different subject. Subjects are chosen by the platform and are not announced in advance; their anonymous IDs are stable across submissions.

### 2.2 Draw complete pairs under the item cap

The platform draws **complete `(benchmark, subject)` groups** from the hidden pool. It does not sample individual items first.

The draw is constrained by a limit of approximately **1,000 distinct source items**. Each selected group contributes all of its source items to the draw. A group that would exceed the remaining budget is skipped.

The cap applies **before** the acquisition/evaluation split, not to the sum of predictor calls across the six budgets.

The stated selection preference is:

- benchmarks represented less often in the current draw;
- subjects represented less often in the current draw;
- random tie-breaking.

There is **no fixed subject count per benchmark**. The number of pairs in a draw is whatever fits under the item cap.

### 2.3 Persistent 50/50 split per pair

Each selected pair is split randomly into:

- **acquisition pool** — outcomes may be revealed;
- **evaluation pool** — outcomes are never revealed and are used only for scoring.

The split is **50/50 by distinct item**. If the item count is odd, the extra item goes to acquisition.

Important properties:

- Repeated response runs of an item stay in the same pool.
- Existing assignments persist across submissions and snapshot updates.
- New items are assigned to balance the pools without moving old items.
- Acquisition items are never scored.

### 2.4 Online acquisition

For each pair, acquisition candidates arrive **one at a time in a fixed random order**.

The submission’s `labeling.py` hook sees:

- the current `[subject, item]` input;
- the current prediction;
- previously revealed labels;
- `labels_remaining`, `labels_acquired`, `max_labels`, and `items_remaining`.

The hook returns `True` to reveal the current item’s 0/1 outcome, or `False` to skip it permanently.

The maximum is **31 labels per pair**. Budgets are nested: `0, 1, 3, 7, 15, 31`. All pairs finish a checkpoint before any pair receives higher-budget labels. If the acquisition stream is exhausted, later caps use the labels actually acquired.

### 2.5 Scoring

At every budget, the platform scores **all stored response rows in the fixed evaluation pool**.

- Repeated response runs of the same item use the same predicted probability.
- The logged “Number of Unique Subject-Item Pairs” counts distinct items, not response runs.
- Pair-level Brier curves use weights `0.1, 0.2, 0.2, 0.2, 0.2, 0.1` at budgets `0, 1, 3, 7, 15, 31`.
- The aggregate Brier ALC is the **unweighted mean across subject–benchmark pairs**, not an item-weighted mean.

---

## 3. What the six scoring logs show

The logs are from:

- `1005155` — shrunken mean;
- `1005187` — CNP-AIRT, random acquisition;
- `1005210` — Online-AIRT, active acquisition;
- `1005225` — CNP-AIRT, active acquisition;
- `1005650` — empirical mean;
- `1005839` — flat budget shrinkage.

### 3.1 Submission-level view

| Submission | Method | Pairs | Distinct benchmarks | Evaluation items per pair | Brier ALC |
|---:|---|---:|---:|---:|---:|
| 1005155 | Shrunken mean (Beta-Binomial active) | 9 | 7 | 44–71 | 0.148605 |
| 1005187 | CNP-AIRT · random | 9 | 7 | 44–71 | 0.186848 |
| 1005210 | Online-AIRT · active | 8 | 7 | 44–113 | 0.201107 |
| 1005225 | CNP-AIRT · active | 8 | 7 | 44–113 | 0.214205 |
| 1005650 | Empirical mean | 9 | 7 | 44–76 | 0.219005 |
| 1005839 | Flat budget shrinkage | 9 | 7 | 44–74 | 0.185556 |

`1005839` is not in the aggregate summary CSV; its ALC is taken directly from its scoring log.

Two facts stand out:

- Every submission contains **all seven benchmark IDs**.
- The two 8-pair submissions both contain an unusually large **113-item** pair; the 9-pair submissions do not.

That is exactly what a packing constraint predicts: a larger pair consumes more of the ~1,000-item budget, so fewer pairs fit.

### 3.2 Benchmark-level view

| Benchmark ID | Pair appearances | Submissions | Observed evaluation counts |
|---|---:|---:|---|
| `benchmark_119137` | 8 | 6 | 44 |
| `benchmark_213669` | 9 | 6 | 58 |
| `benchmark_342307` | 6 | 6 | 60 |
| `benchmark_800730` | 8 | 6 | 50, 53, 54 |
| `benchmark_830060` | 7 | 6 | 53, 71, 74, 76 |
| `benchmark_869474` | 6 | 6 | 56, 113 |
| `benchmark_893575` | 8 | 6 | 44 |

Interpretation:

- Each submission has **one base pair for every benchmark**, plus 1–2 extra pairs on some benchmarks.
- The extra pairs are additional **subjects** on an already represented benchmark, not additional benchmarks.
- The same benchmark can have different evaluation counts for different subjects. That is expected: different subjects answered different item sets.
- The same **pair** has the same evaluation count whenever it recurs.

### 3.3 Recurring pairs prove the split is persistent

These pairs appear in more than one submission with the **same evaluation-item count**:

| Pair | Submissions | Evaluation items | Pair-level Brier ALC |
|---|---|---:|---|
| `subject_102340` × `benchmark_830060` | 1005225, 1005839 | 74 | 0.211405, 0.214823 |
| `subject_312659` × `benchmark_213669` | 1005155, 1005650 | 58 | 0.061082, 0.025000 |
| `subject_524441` × `benchmark_213669` | 1005225, 1005839 | 58 | 0.230441, 0.233571 |
| `subject_556050` × `benchmark_342307` | 1005155, 1005210, 1005839 | 60 | 0.260777, 0.253981, 0.254807 |
| `subject_672159` × `benchmark_869474` | 1005210, 1005225 | 113 | 0.243911, 0.230670 |
| `subject_941033` × `benchmark_119137` | 1005187, 1005225 | 44 | 0.249955, 0.258559 |

The item count is identical; the Brier differs because the predictor differs. This is what a fixed evaluation pool plus different predictions looks like. It is not a fresh split per submission.

The official kit also states that anonymous IDs are assigned once and never reused, so matching IDs always identify the same subject or benchmark.

### 3.4 The ~1,000-item cap is visible in the logs

The official rule says each pair is split 50/50, with the extra item going to acquisition on odd counts. Therefore:

```text
evaluation_items = floor(source_items / 2)
source_items     = 2 * evaluation_items + odd
```

where `odd` is 1 when the pair has an odd number of distinct source items, else 0. Equivalently, the acquisition pool is the same size as the evaluation pool, or one item larger on odd counts.

Summing the logged evaluation counts and adding one for each odd-sized pair gives:

| Submission | Pairs | Evaluation items | Odd pairs | Implied source items |
|---:|---:|---:|---:|---:|
| 1005155 | 9 | 495 | 1 | 991 |
| 1005187 | 9 | 489 | 1 | 979 |
| 1005210 | 8 | 491 | 1 | 983 |
| 1005225 | 8 | 491 | 1 | 983 |
| 1005650 | 9 | 493 | 1 | 987 |
| 1005839 | 9 | 496 | 2 | 994 |

The implied totals are **979–994**, essentially the stated ~1,000-item draw limit. The small shortfall is also expected: the official text says the draw may include fewer than 1,000 items when no further complete group fits.

This is the cleanest quantitative confirmation of the reconstruction.

### 3.5 Why the pair count is 8 or 9

The platform does not target “8 pairs” or “9 pairs.” It packs complete pairs until the next pair would exceed the item budget.

- A typical pair has 44–76 evaluation items, so about 88–152 source items.
- Eight or nine such pairs fit under ~1,000 items.
- A pair with 113 evaluation items is about 226 source items; it consumes enough budget to reduce the draw to eight pairs.

The two 8-pair submissions both contain the 113-item pair. The 9-pair submissions do not. This is consistent with a packing rule, not a fixed pair count.

### 3.6 The aggregate is an unweighted pair mean

For each submission, the reported aggregate Brier ALC equals the arithmetic mean of the pair-level ALCs, not the item-weighted mean.

Example, `1005155`:

- 9 pairs;
- unweighted mean `0.148605`;
- item-weighted mean would be `0.150216`.

The official scoring section states the same rule: every evaluated subject–benchmark combination has equal weight, regardless of its number of responses.

This matters because a draw with a few easy or hard pairs can move the aggregate, and because hidden scores across submissions are not directly comparable unless the pair sets match.

---

## 4. What is inferred, and what is not

### High confidence

- The evaluation unit is the **subject–benchmark pair**.
- Each pair has a persistent 50/50 acquisition/evaluation split.
- The evaluation pool is fixed across budgets.
- Acquisition items are never scored.
- The draw is constrained by ~1,000 distinct source items.
- The draw packs complete pairs and skips pairs that do not fit.
- The aggregate is an unweighted mean across pairs.
- The logs' evaluation counts sum to about half of the draw limit.

### Medium confidence

- In these six draws, every submission contains all seven observed benchmarks once, plus 1–2 extra subjects.
- The hidden pool for these submissions contains at least these seven benchmarks.
- The balancing preference in the official text explains the one-pair-per-benchmark pattern, but the exact algorithm is not published.

### Not observable from the logs

- The total number of hidden benchmarks. The logs show seven; they do not prove the pool contains only seven.
- The acquisition-pool size per pair. It is inferred from the 50/50 rule, not logged.
- The exact item assignments and acquisition order.
- The mapping from anonymous benchmark IDs to public benchmark names.
- Whether the hidden benchmarks were present in the public training data.
- The exact subject/benchmark balancing algorithm.

---

## 5. Implications for offline replication

To mimic the hidden protocol, do **not** evaluate on a single fixed set of 8–9 pairs. Simulate the draw.

Conceptual recipe:

1. Build all eligible pairs with at least 80 distinct items.
2. For each simulated submission draw:
   - start with an empty draw and a remaining budget of ~1,000 distinct source items;
   - prefer benchmarks and subjects used less often in the draw;
   - break ties randomly;
   - add a complete pair only if it fits the remaining budget;
   - stop when no complete pair fits.
3. For each selected pair:
   - split distinct items randomly 50/50;
   - give the extra item to acquisition on odd counts;
   - keep repeated response runs in the same pool;
   - persist the split.
4. For each pair:
   - order acquisition candidates randomly;
   - reveal up to 31 labels with nested budgets `0, 1, 3, 7, 15, 31`;
   - score the same fixed evaluation pool at every budget.
5. Score each pair’s Brier curve, then average **unweighted across pairs**.
6. Repeat the draw many times and report the distribution across draws.

For paired comparisons, match on **subject ID and benchmark ID**, not just benchmark. The logs show only six recurring pairs across 52 pair appearances, so most hidden comparisons are unpaired.

---

## 6. Relation to “leave one benchmark out, then split that benchmark in half”

That description is close, but not exact.

- The platform’s draw unit is the **subject–benchmark pair**, not the benchmark.
- The 50/50 split is **per pair**, not per benchmark.
- The logs show all seven benchmarks in every submission, so the hidden test is not “hold out one benchmark and test only on it.”
- The logs do not reveal whether the hidden benchmarks were excluded from public training data. The platform may be testing new **subject–benchmark pairs** on benchmarks that were visible in public data.

So:

- **Platform-like protocol:** hold out subject–benchmark pairs; for each pair, use half the items as acquisition and half as evaluation.
- **Stronger generalization protocol:** also hold out entire benchmarks from training, then split the held-out benchmark’s pairs 50/50.
- **Different question:** fine-tuning on half of a held-out benchmark’s items and scoring the other half. That is transductive adaptation, not the platform’s fixed offline-training + online-label-acquisition setup.

The platform is best described as **benchmark-mixed, pair-held-out, few-shot online evaluation**, not strict leave-one-benchmark-out.

---

## 7. Source files

- Official protocol: [`reference/starting_kit/README.md`](../reference/starting_kit/README.md)
  - subject definition: line 36;
  - interface and ~1,000-item draw: line 39;
  - eligibility and 50/50 split: lines 72–77;
  - acquisition and nested budgets: lines 79–109;
  - scoring and unweighted pair mean: lines 143–156.
- Scoring logs:
  - [`results/codabench-1005155/scoring_stdout.txt`](../results/codabench-1005155/scoring_stdout.txt)
  - [`results/codabench-1005187/scoring_stdout.txt`](../results/codabench-1005187/scoring_stdout.txt)
  - [`results/codabench-1005210/scoring_stdout.txt`](../results/codabench-1005210/scoring_stdout.txt)
  - [`results/codabench-1005225/scoring_stdout.txt`](../results/codabench-1005225/scoring_stdout.txt)
  - [`results/codabench-1005650/scoring_stdout.txt`](../results/codabench-1005650/scoring_stdout.txt)
  - [`results/codabench-1005839/scoring_stdout.txt`](../results/codabench-1005839/scoring_stdout.txt)
- Aggregate summary: [`results/codabench-comparison/submission_summary.csv`](../results/codabench-comparison/submission_summary.csv)
