# What Person-B can fix, based on Person-A's paper changes (2026-11-07)

## Person-B response

- Figure labels F1/F3/F4/F5 were fixed on `submission` in `2bea768`.
- The white/light-grey comparison is in `paper/eandd/style-preview/`.
  The shared `plotstyle.py` defines both themes; production stays white.
  The grey variant is a trial, not a global style change.
- The confirmed benchmark-macro correlations are 0.089967 (pass-rate MLP),
  0.102648 (Rasch-target MLP) and 0.025805 (sampled random control).
  These are means of six within-benchmark correlations, not a correlation
  computed after pooling all items.
- Table 2 had stale Rasch-target numbers from the provisional local refit.
  The original-fit values are validation 0.145958 / 0.550439 and test
  0.059339 / 0.521689 (Spearman / pairwise). The test Rasch-target Spearman
  is below the random control's 0.075927. The table now comes from the
  confirmed CSVs via `assemble.py`.
- Verified 8,462 fitting items from the 11 training benchmarks, the original
  Rasch fit, five training-benchmark folds, and the 32-unit MLP configuration.
  The methods now distinguish training supervision, easiness used for rank
  scoring, and difficulty exported for Rasch.
- Removed the internal “WP7” label from the disclosure prose. The actual
  filename `wp7-summary.csv` remains a reproducibility pointer.
- R5 wording needed softening: the validation-selected scale improves Brier
  ALC by 0.00039 on validation and 0.00041 on test in the matched replay.
  A blanket “no scale helps” is false; the body now says the gain is small
  and does not by itself establish a reliable improvement. Person A should
  use the same qualification in the abstract and conclusion.
- A-title, A-abstract, A-priors, A-adaptive and A-conclusion remain Person A's.
  The main text still needs the final length pass before submission.

The checked source is `person-b-sections.tex`; generated section files and
the Table 2 input are rebuilt by `assemble.py`. This does not regenerate
the annotated review copy, whose source is not in the repository.
Figure labels (Delta notation, OOD legends): done by Person-B in 2bea768.

Person-A restructured and reworded `paper/neurips_2026/eandd.tex` and your section file `paper/eandd/person-b-sections.tex`.
`paper/reviewing/reviewing.pdf` shows what is still open (red = to fix, green = question or planned addition).

## 2. Idea, not decided: plot background
We are thinking of changing the plot background so the figures read better, for example a light grey panel with white grid lines instead of the plain white panel, still readable in black and white. Try it on one figure, compare it with the current one, and keep whichever reads better. Keep `organized/utils/plotstyle.py` as the single style source.

## 3. Your text that Person-A changed (please check the facts)
- **Section 3 (Data and evaluation protocol)**: now four short paragraphs. New notation: response matrix `Y` (N subjects as rows `i`, M items as columns `j`), `d(j)` the dataset/benchmark of item `j`, D = 17. ID and OOD are defined; online and offline are defined; "control", "panel", "replay" are not used any more.
- **Probes (new Section 4.2) and Item features (new Section 5.4)**: your item-difficulty section was split. Methodology now holds the probe description, the Spearman and pairwise-accuracy formulas and Equation 4 (probe prior mean); Results 5.4 holds Table 2 (restructured, grouped columns) and Table 3 (R5 scales, from `r5-prior-mean-*-summary.csv`). Please check the probe training details (8,462 items from the 11 training benchmarks) and the numbers 0.090 / 0.103 / 0.026.
- **Disclosure section**: still says "WP7" ("The WP7 item results are in ..."); internal label, rename it.
- **Limitations**: now also says label sharing across subjects is untested (one subject per benchmark per draw).
- Table generators in `assemble.py` changed (bold best, underline second best in Table 1, group rows, appendix headers ID / OOD validation / OOD test). Edit `person-b-sections.tex` and `assemble.py` together, because `assemble.py` splits the file by `\section{...}` titles: Data and evaluation protocol, Probes, Item features, Limitations, Disclosure and rerun instructions.

## 4. Still open (shown red in the reviewing copy)
Title, abstract, Priors subsection, adaptive-priors method and conclusion (Person-A).
