# Zhang et al.: interpolation and extrapolation claim check

Checked against *How Benchmark Prediction from Fewer Data Misses the Mark*, arXiv 2506.07673v2 (2026-07-21), Sections 4.2-4.3 and Figure 3. Check date: 2026-11-07.

Source: https://arxiv.org/html/2506.07673v2#S4.SS3

## Finding

The manuscript's wording that methods successful at interpolation **can fail** on higher-performing target models is supported. It should not be expanded to say that all learned methods fail: the paper's AIPW estimator is an explicit exception.

- Interpolation randomly assigns 75% of models to sources and 25% to targets within each benchmark.
- Extrapolation uses the lowest-performing 50% as sources and the top 30% as targets, ranked by full-benchmark accuracy.
- The target is aggregate benchmark accuracy, estimated from 50 responses in the main comparison. Error is the absolute gap between estimated and actual full-benchmark performance, averaged over target models.
- Random-Search-Learn, the strongest interpolation method in the reported comparison, is worse than the random-sample mean on every benchmark under extrapolation.
- AIPW still beats the random-sample mean on all 19 benchmarks in that comparison, although its gains are smaller than under interpolation.

## Recommended wording

> Zhang et al. find that most benchmark-prediction methods lose their advantage over random sampling when extrapolating to higher-performing models, while their AIPW estimator retains an advantage.

Their experiment shifts the target **models within benchmarks**. Our experiment holds out **entire benchmarks** and scores per-item probabilities with Brier ALC. Their result motivates examining transfer; it does not directly establish our result.

## Citation convention

The library entry is `zhang2026benchmark`, following the current official arXiv export. The manuscript's existing `zhang2025benchmark` uses the first-submission year. They refer to the same paper; retain one consistent convention when integrating citations.
