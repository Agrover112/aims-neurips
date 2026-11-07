# Review-copy figure labels

Checked against `submission:paper/reviewing/reviewing.pdf` at `0a6bcec`.

| Review item | Asset | Change |
|---|---|---|
| F1 | figure1-prior-source.pdf | Panel (b): Delta with subscript m minus J |
| F3 | figure3-benchmark-breakdown.pdf | Delta with subscript BB fit minus J; OOD test and OOD validation |
| F4 | figure4-benchmark-explanation-17.pdf | Delta with subscript BB fit minus J; OOD test and OOD validation |
| F5 | figure5-split-sensitivity.pdf | macro Delta with subscript BB fit minus J |

The notation follows the paper's `\D{BB\,fit}{J}` convention. Plot data,
intervals, weights, dimensions and benchmark assignments are unchanged.
Each PDF is a native-width vector figure and was rendered and checked.
The assets are in both `paper/eandd/data/` and
`paper/neurips_2026/figures/eandd/`.

Plot sources: `organized/testing/eandd/wp9.py` (F1 and F3) and
`organized/testing/eandd/reviewer_figures.py` (F4 and F5).

Person A still owns A-title, A-adaptive, A-priors and A-conclusion.
Those prose edits were not made in this figure-only change.
The annotated review PDF is not regenerated: its review-copy LaTeX
source is not in `paper/reviewing/`. Recompile that source with these
replacement assets to update the annotations' underlying figures.
