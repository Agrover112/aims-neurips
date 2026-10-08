# Assumptions and limitations

## Confirmed competition behavior

The repository targets the predictor interface and evaluation rules published by AIMS on October 8, 2026. A submission contains `model.py` at its ZIP root. The optional `labeling.py` may acquire outcomes during evaluation. The primary score is Brier ALC at label budgets 0, 1, 3, 7, 15, and 31, with weights 0.1, 0.2, 0.2, 0.2, 0.2, and 0.1.

The official starting kit remains the authority if the website, local kit, and this repository disagree. Recheck the rules before each upload.

## Modeling assumptions

The reference predictor treats revealed outcomes as exchangeable within a subject and anonymous benchmark. It uses beta-binomial shrinkage and does not model item text, item difficulty, response dependence, temporal drift, or repeated-outcome correlation. These choices make the baseline easy to audit, but they are unlikely to be competitive on their own.

The predictor uses evidence from the same subject across benchmarks as a weak intermediate estimate. This assumes that a subject's observed performance carries information across benchmarks. Ablate `subject_strength` to test whether that transfer helps.

## Test-time adaptation

The acquisition policy selects predictions near 0.5 and forces purchases near the end of the candidate stream so the budget can be filled. Uncertainty sampling need not produce a representative sample. It may bias a simple group mean toward harder items. Compare it with the platform's default random policy and report both predictive accuracy and calibration.

No prediction may use future acquisition candidates, hidden evaluation outcomes, or state left by earlier worker calls. Every prediction must depend only on its input, bundled artifacts, and the supplied `labeled` evidence.

## Offline evaluation

An offline split is a development estimate, not a reconstruction of the hidden leaderboard. Group splits must keep repeated responses for a subject-item pair together. Model selection performed on the same split introduces optimism, so use nested validation or a final untouched holdout when comparing many configurations.

Report variability across seeds or folds. The aggregate leaderboard gives equal weight to subject-benchmark combinations, while Brier and ECE within each combination use all included response rows. Do not silently replace this with row-weighted aggregation.

## Literature review

The files in `literature/` are leads, not constraints. Candidate methods should be judged against explicit assumptions, relevant evidence, and controlled experiments. Their inclusion does not imply that their claims have been independently verified.

