# What Person-B can fix, based on Person-A's paper changes (2026-11-07)

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
