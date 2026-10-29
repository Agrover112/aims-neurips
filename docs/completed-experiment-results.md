# Aggregate completed experiment results

Lower Brier ALC is better. Different cohorts/protocols below must not be ranked together. Canonical filtering and 11/3/3 benchmark split apply.

## Matched test cohort: 78 subject-benchmark tasks, random acquisition

| Predictor | Test Brier ALC |
|---|---:|
| Feature Rasch | 0.235079 |
| Linear regression | 0.235597 |
| TabPFN LoRA | 0.236410 |
| Fixed Beta(2,2) | 0.237026 |
| Learned Beta-Binomial | 0.243423 |
| Response-only Rasch | 0.244100 |
| Bayesian RBF regression | 0.257082 |

## Full validation: 390 tasks, fixed random contexts

| Predictor | Validation Brier ALC |
|---|---:|
| Bayesian ANP correction | 0.192153 |
| TabPFN LoRA | 0.193521 |
| Response-only Rasch | 0.193560 |
| Learned Beta-Binomial | 0.193596 |
| Linear regression | 0.193623 |
| SAM ANP, step 700 | 0.194318 |
| GroupDRO ANP, step 100 | 0.195091 |
| Balanced ERM ANP, step 100 | 0.195129 |
| Feature Rasch | 0.197070 |
| SWA ANP, step 800 | 0.206197 |
| Bayesian RBF regression | 0.207408 |
| Bernoulli GP classifier | 0.208470 |

## Frozen SAM ANP acquisition: 390 validation tasks x 3 stream orders

| Policy | Task-weighted ALC | Benchmark-macro ALC | Mean forced purchases |
|---|---:|---:|---:|
| Random | 0.193744 | 0.182713 | 0.175 |
| Uncertainty | 0.195512 | 0.184825 | 2.635 |
| Expected Brier change | 0.190547 | 0.183906 | 0.644 |

Expected Brier improved task-weighted ALC by 0.003197, or approximately 1.65%. It improved two validation benchmarks and worsened the third. Equal benchmark weighting slightly favors random. Its prediction-change score is an approximation for ANP; improvement on TabPFN is untested. Only three validation benchmarks are represented, so more tasks do not establish broad transfer.

Detailed reports and training-derived bank embeddings are intentionally retained locally/in Modal. Fetch scripts are provided under `scripts/`; protocol documentation is linked in `docs/benchmark-experiment-index.md`.
