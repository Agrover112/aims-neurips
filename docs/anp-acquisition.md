# Frozen ANP acquisition pilot

The four 3,000-update OOD runs selected these checkpoints using the same 64-task validation bank:

| Variant | Best update | Selection Brier ALC | Full validation Brier ALC |
|---|---:|---:|---:|
| Balanced ERM | 100 | 0.18778594 | 0.19512895 |
| GroupDRO | 100 | 0.18777168 | 0.19509057 |
| SWA | 800 | 0.19707625 | 0.20619662 |
| SAM | 700 | 0.18648396 | 0.19431762 |

Lower is better. The late curves generally degrade, so there is no evidence here supporting a blind extension to 100k updates. Each update has 11 task episodes. These are one-seed results, not evidence of statistically significant superiority. Tests were not rescored.

`modal_anp_acquisition.py` freezes SAM's step-700 best checkpoint and compares random, Bernoulli variance, BALD (16 prior latent draws), and approximate expected Brier reduction. Inference uses the prior latent mean to match checkpoint selection; BALD alone samples latents for its score. The Brier score uses hypothetical labels with current model probabilities, and measures squared prediction changes on 32 training reference items; this is a heuristic, not a guarantee of Bayesian risk reduction.

Pilot: 24 validation tasks stratified equally over the three validation benchmarks, with task-population weights; three stream permutations (17, 43, 89). All arms share fixed 50/50 acquisition/evaluation items, stream order and six nested label budgets. No test selection or scoring. Evaluation features enter only scoring snapshots, never the policy. Future stream features are not inspected. Acquisition outcomes are read only on accepted items. Each stream must purchase 31 labels.

Random accepts with probability remaining budget / remaining candidates. Scored arms use the candidate's smoothed randomized upper-tail rank among 32 training-only probe scores under the **current acquired context** and the same budget fraction. Recompute probes after a purchase; force purchase when remaining budget equals remaining opportunities. Randomized ties handle collapsed latent scores. This is a concrete approximate pacing policy; results compare score plus pacing, not unrestricted pool ranking. The reference/probe banks have no held-out item IDs, and are fixed without validation outcomes. Their limited relevance to new benchmarks is a limitation.

Each arm uses 2 CPU cores and 8 GiB, no GPU, a six-hour cap, injected Modal `wandb` secret, and Forge project `aims-predictive-evaluation`. The workflow runs three smoke episodes per arm before launching full pilots. Results and per-task resume state are in `/results/anp-acquisition/anp-sam-acquisition-20261029-v1/<policy>/`. Resume occurs between trajectories. JSON records every budget, runtime, score summaries, purchase positions, and forced purchases. Forge logs running aggregates; interpret final comparisons only after all matched trajectories finish. No new training occurs.

Launch from the worktree:

```powershell
..\.venv-aims\Scripts\python.exe -X utf8 -m modal run --detach --profile arshiaswe organized/training/modal_anp_acquisition.py
```

Four local tests cover BALD disagreement, the Brier-change calculation, causal label access/nested snapshots, and score ties/budget forcing.
