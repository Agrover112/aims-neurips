# R5 original-fit item difficulty handoff

Rasch difficulty: positive means harder; enters logit as minus difficulty.

Both probes used 8462 deterministic WP7 sampled items (at most 800 per benchmark) from the 11 training benchmarks only; no validation or test items or outcomes were used for fitting, scaling, or hyperparameter selection.

Each exported file exactly covers every unique benchmark/content_hash key in Person A's validation/test replay panel, including acquisition and evaluation items.

Original trimmed fit SHA-256: aec4b6951131edaaf94ec3bbc7be3ab65f070df5b4fc5eb252d22e1ccd97136a

The pass-rate arm is mapped from a centered probability correction to the logit scale using only the training-derived baseline; the Rasch-target arm negates the predicted easiness. Positive values mean harder items in both files.

| Arm | Spearman | Random | Gap | Positive benchmarks | Pass | File |
|---|---:|---:|---:|---:|---|---|
| pass_rate_mlp | 0.089967 | 0.025805 | 0.064162 | 5/6 | True | r5-pass_rate_mlp-item-difficulties.parquet |
| pass_rate_ridge | 0.065195 | 0.025805 | 0.039390 | 4/6 | False | No export |
| rasch_easiness_mlp | 0.102648 | 0.025805 | 0.076843 | 5/6 | True | r5-rasch_difficulty_mlp-item-difficulties.parquet |
| rasch_easiness_ridge | 0.047332 | 0.025805 | 0.021527 | 4/6 | False | No export |
