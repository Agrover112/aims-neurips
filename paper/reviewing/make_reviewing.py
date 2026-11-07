"""Build paper/neurips_2026/eandd_reviewing.tex from the current eandd.tex.

Light red  = existing text I would delete or rewrite, or the OLD wording of an edit already made ([S1 was], ...).
Light green = text/analyses I would add, or the NEW wording of an edit already made ([S1 now], ...).
"""
import re
src = open("eandd.tex").read()
idf = open("eandd_b_itemfeat.tex").read()
lim = open("eandd_b_limits.tex").read()


def wrap(tag, text):
    return "\\par\\begin{chg}{%s}%s\\end{chg}\\par\n" % (tag, text)


def note(tag, text):
    return "\\par\\begin{addnote}{%s}%s\\end{addnote}\\par\n" % (tag, text)


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (s.count(old), old[:80])
    return s.replace(old, new)


s = src
PRE = r"""\usepackage{float}
\setlength{\textfloatsep}{12pt plus 2pt minus 4pt}\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\usepackage{tcolorbox}\tcbuselibrary{breakable,skins}
\newtcolorbox{chg}[1]{breakable,enhanced,colback=red!14,colframe=red!14,boxrule=0pt,arc=0pt,left=2pt,right=2pt,top=1.5pt,bottom=1.5pt,before skip=2pt,after skip=2pt,before upper={\textcolor{red!70!black}{\small\bfseries[#1]}\ }}
\newtcolorbox{oldtxt}[1]{breakable,enhanced,colback=red!14,colframe=red!14,boxrule=0pt,arc=0pt,left=2pt,right=2pt,top=1.5pt,bottom=1.5pt,before skip=2pt,after skip=2pt,before upper={\textcolor{red!70!black}{\small\bfseries[S#1 was]}\ }}
\newtcolorbox{newtxt}[1]{breakable,enhanced,colback=green!14,colframe=green!14,boxrule=0pt,arc=0pt,left=2pt,right=2pt,top=1.5pt,bottom=1.5pt,before skip=2pt,after skip=2pt,before upper={\textcolor{green!45!black}{\small\bfseries[S#1 now]}\ }}
\newtcolorbox{addnote}[1]{breakable,enhanced,colback=green!14,colframe=green!55!black,boxrule=0.6pt,arc=1pt,left=3pt,right=3pt,top=2pt,bottom=2pt,before skip=4pt,after skip=4pt,fontupper=\small\itshape,before upper={\textcolor{green!45!black}{\upshape\bfseries NEW [#1]:}\ }}
\newtcolorbox{qnote}[1]{breakable,enhanced,colback=green!14,colframe=green!55!black,boxrule=0.6pt,arc=1pt,left=3pt,right=3pt,top=2pt,bottom=2pt,before skip=4pt,after skip=4pt,fontupper=\small,before upper={\textcolor{green!45!black}{\bfseries QUESTION [#1]:}\ }}
\newtcolorbox{fixnote}[1]{colback=red!14,colframe=red!55!black,boxrule=0.6pt,arc=1pt,left=3pt,right=3pt,top=2pt,bottom=2pt,before skip=4pt,after skip=4pt,fontupper=\small,before upper={\textcolor{red!70!black}{\bfseries TO FIX (needs Person-B's plot script) [#1]:}\ }}
\newtcolorbox{improve}[1]{colback=red!14,colframe=red!55!black,boxrule=0.6pt,arc=1pt,left=3pt,right=3pt,top=2pt,bottom=2pt,before skip=4pt,after skip=4pt,fontupper=\small,before upper={\textcolor{red!70!black}{\bfseries STILL TO IMPROVE [#1]:}\ }}
\newcommand{\oldin}[1]{\colorbox{red!14}{#1}}
\newcommand{\newin}[1]{\colorbox{green!14}{#1}}
"""
s = sub(s, "\\usepackage{float}\n", PRE)
BANNER = r"""\maketitle
\begin{tcolorbox}[colback=white,colframe=black,boxrule=0.5pt,arc=0pt,left=3pt,right=3pt,top=2pt,bottom=2pt]
\small\textbf{REVIEW COPY, not a paper version.} \textbf{Light red} = existing text I would delete or rewrite, or (marked [S1 was] to [S21 was]) the old wording of an edit already made. \textbf{Light green} = text or an analysis I would add, or (marked [S1 now] to [S21 now]) the new wording of an edit already made. Numbers in brackets refer to the list \emph{What I would change and why}; [S1] to [S21] to the list of edits already made. Both lists are at the end.
\end{tcolorbox}
"""
s = sub(s, "\\maketitle\n", BANNER)

# ---- edits already made (S1, S2 accepted) ----

# ---- planned changes ----
a1 = "On unseen benchmarks that gain disappears, and only a shrinkage prior transfers: fitted on training it gains $0.014$ on test but loses $0.008$ on validation."
n1 = " Averaged over all 17 benchmarks it gains $0.005$, and over all 680 choices of three held-out benchmarks it gains $0.004$ and hurts in $17.5\\%$ of them, so our test split is a favourable one."
s = sub(s, "  \\item \\textbf{Open materials.}",
        "  \\item \\par\\begin{addnote}{5}A bullet for the negative results the reviewer asked about: adaptive priors (mixtures, strength averaging, a pilot switch rule) did not remove the harm without losing the gain, and the first labels warn only weakly (pooled AUC 0.73, within-benchmark 0.64).\\end{addnote}\n  \\item \\textbf{Open materials.}")
s = sub(s, "\\paragraph{Offline model that keeps its fitted values.}",
        note("5", r"A short subsection on adaptive priors, all closed form: Beta mixtures fitted by EM whose weights update by marginal likelihood on the revealed labels; averaging over the prior parameters $(\alpha,\beta)$; and a switch to Jeffreys when the first $k$ labels sit far from the prior mean, with the threshold chosen on validation only.")
        + note("7", r"Convergence paragraph: the online Newton solver converged in all 6,000 audited fits (at most 7 steps); the offline fit first stopped at an iteration limit, and a tight refit moved the hyperparameters by at most $1.2\%$ and no result by more than $0.0001$.")
        + "\\paragraph{Offline model that keeps its fitted values.}")
r8 = r"On test, a Beta-Binomial with a prior fitted on training is the best predictor ($\D{BB\,fit}{J}=-0.0138$ [$-0.0158$, $-0.0116$]), four times the gain of the validation-tuned Rasch, and better than any Rasch model ($\D{R\,fit}{BB\,fit}=+0.0019$ [$+0.0007$, $+0.0031$])."
s = sub(s, r8, "\\par\\begin{chg}{8}" + r8 + "\\end{chg}\\par")
s = sub(s, "\\begin{figure}[t]\n  \\centering\n  \\includegraphics{\\figdir/figure1-prior-source.pdf}",
        note("9", r"One sentence and a pointer: log loss and calibration error give the same ordering on test and the same reversal on validation (log loss ALC $0.6656$ against $0.7045$ for Jeffreys on test; $0.5404$ against $0.5159$ on validation), and rank metrics are uninformative by construction for a constant prediction.")
        + "\\begin{figure}[t]\n  \\centering\n  \\includegraphics{\\figdir/figure1-prior-source.pdf}")
s = sub(s, "Figure~\\ref{fig:benchmarks} shows the test and validation benchmarks separately",
        "\\par\\begin{qnote}{Q1}Section 5.3 and Figure 4: the text reported the 11-benchmark view (Spearman $0.93$, helps on 5 of 11) while the Figure 4 caption already showed 17 benchmarks, and the abstract's Spearman matched the 11-benchmark text. Is the paper now consistent on 17 benchmarks everywhere (abstract, contributions, Section 5.3, Figure 4)?\\end{qnote}\\par\nFigure~\\ref{fig:benchmarks} shows the test and validation benchmarks separately")
s = sub(s, "Under the fitted priors, the simplest Rasch model is also the best on test",
        note("11", r"A short subsection, \emph{How sensitive is the test gain to which benchmarks are held out?}, moved from the appendix: over 680 triples the mean gain is $-0.0037$ (prior refit on the other 14 benchmarks) or $-0.0051$ (training benchmarks only), the prior hurts in 17.5\%--20\% of splits, and our test triple gives $-0.0138$ when the prior is fitted on the 11 training benchmarks but $-0.0087$ when it is fitted on the other 14. State which fit set is used.")
        + note("12", r"The first-principles version of the explanation: for a task of true accuracy $p$ the expected Brier excess of the posterior mean is $\frac{(\alpha-p(\alpha+\beta))^2+n\,p(1-p)}{(\alpha+\beta+n)^2}$, so the gain over Jeffreys is an exact function of $p$ and of the budget; compare the predicted gain per benchmark with the observed one instead of citing a post hoc correlation.")
        + note("5", r"Results of the adaptive priors on the 17 benchmarks (worst harm, macro gain, test gain), and the pilot diagnostic: from 7 labels the distance of the pilot accuracy from the prior mean predicts harm with pooled AUC $0.73$ (within a benchmark $0.64$); the harm sits at 0--3 labels, before any pilot exists.")
        + "Under the fitted priors, the simplest Rasch model is also the best on test")
figfix = {}
for fn, (tag, txt) in figfix.items():
    key = "\\includegraphics{\\figdir/" + fn + "}"
    s = sub(s, key, key + "\n\\begin{fixnote}{" + tag + "}" + txt + "\\end{fixnote}")
s = sub(s, "\\section{Leave-one-benchmark-out for the fitted prior}\\label{app:lobo}",
        "\\section{Leave-one-benchmark-out for the fitted prior}\\label{app:lobo}\n" + note("16", r"Extend this table to all 17 benchmarks (the figure already shows 17), with the Spearman correlation for 17 ($0.96$) beside the one for the 11 training benchmarks ($0.93$)."))
s = sub(s, "\\section{Sensitivity to the benchmark split}\\label{app:split-sensitivity}",
        note("7", r"New appendix tables: the convergence audit; the adaptive-prior results per benchmark; the pilot-diagnostic AUC and the switch-rule sweep; and the per-split distribution for the Rasch prior (40 triples).")
        + "\\section{Sensitivity to the benchmark split}\\label{app:split-sensitivity}")

END = r"""
\clearpage
\section*{What I would change and why}
\small
\begin{enumerate}\setlength{\itemsep}{2pt}
\item \textbf{Headline.} $0.014$ is the best case: our test split is one of the most favourable we could have drawn. The average over splits is about $0.004$--$0.005$, and the prior hurts in roughly one split in five.
\item \textbf{17 benchmarks.} Figure~4 already shows 17 benchmarks (Spearman $0.96$, helps on 9, hurts on 1); the text, the abstract and the appendix table still say 11 (Spearman $0.93$, 5 of 11).
\item \textbf{Item features.} A pass-rate MLP and a Rasch-target MLP pass the rank criterion set in advance (pooled $0.090$ and $0.112$ against $0.026$ for a random control); ridge probes do not. On the three test benchmarks alone the margin over the control is small ($0.109$ against $0.076$). The claim of no signal must be softened; the Brier test of the prior-mean option decides whether it matters.
\item \textbf{Label sharing.} ``Adds nothing to a good prior'' holds on test and with one subject per benchmark. On validation the fitted Rasch beats the fitted Beta-Binomial, and multi-subject draws are barely tested.
\item \textbf{Negative results the reviewer asked for.} Adaptive priors and the pilot switch rule do not meet the criteria fixed in advance; the first labels warn only weakly.
\item \textbf{Specificity.} Say which widths and kernel settings the validation-tuned Rasch priors took.
\item \textbf{Convergence and extra appendix tables} (answers a highlighted reviewer point).
\item \textbf{Wording of the best result.} Add that it is the favourable split and that the validation split reverses.
\item \textbf{Other metrics.} Log loss and calibration agree with Brier (reviewer question 3).
\item \textbf{Correct the 11-benchmark numbers} in the running text to the 17-benchmark view.
\item \textbf{Sensitivity to the split in the main text,} with the fit set stated: $-0.0138$ (11 training benchmarks) against $-0.0087$ (the other 14).
\item \textbf{An exact risk formula} so the explanation is a derivation, not a correlation found after the fact.
\item \textbf{Limitations.} Quantify the favourable-split statement with the 680 splits.
\item \textbf{Conclusion.} Rewrite around the average, the spread across splits and the limits of what a label-based diagnostic can do.
\item \textbf{Terminology.} Done: the table caption now says ``online simulation'' (see S5).
\item \textbf{Appendix table} to 17 benchmarks.
\end{enumerate}
\section*{Edits already made to the paper (old wording red, new wording green)}
\begin{enumerate}\setlength{\itemsep}{2pt}
\item[\textbf{S1}] \textbf{Beta-Binomial notation.} Conventional symbols (checked against standard references): success probability $p$, prior $\mathrm{Beta}(\alpha,\beta)$, $n$ trials with $k$ successes, posterior $\mathrm{Beta}(\alpha+k,\beta+n-k)$ and posterior mean $(\alpha+k)/(\alpha+\beta+n)$, with $\alpha+\beta$ read as the number of pseudo-labels the prior is worth. The old text used $\pi$, a mean $m$ and a strength $s$ (which clashed with the subject index $s$) and $h$ for the correct labels. The prior settings are now stated as $(\alpha,\beta)$: Jeffreys $(\tfrac12,\tfrac12)$, validation-tuned $(0.4,0.6)$, training-fitted $(1.92,1.76)$.
\item[\textbf{S2}] \textbf{Rasch equation in IRT convention.} $p_{ij}=\sigma(\theta_i-b_j)$, replacing $\operatorname{logit}p=\mu_b+\theta_s-b_i$. The item difficulty is $b_j$ (see S4 for its split); the IRT symbol $\beta$ is left to the Beta prior.
\item[\textbf{S3}] \textbf{Response matrix.} Section 2 now defines $Y\in\{0,1\}^{N\times M}$ with rows $i$ (subjects) and columns $j$ (items), benchmarks as blocks of columns, a task as one row restricted to one block, and the splits as disjoint column blocks; the ID control holds out rows.
\item[\textbf{S4}] \textbf{Symbols.} Item difficulty $b_j=\delta_j-\mu_{d(j)}$: $\delta_j$ is the deviation from the benchmark's average, $\mu_d$ the benchmark rate, $d(j)$ the dataset (benchmark) of item $j$, $D=17$ benchmarks. Prior widths are standard deviations $\sigma_\mu,\sigma_\theta,\sigma_\delta$ (formerly $\tau$); the kernel uses an indicator $\mathbf 1_{j=j'}$.
\item[\textbf{S6}] \textbf{Explicit differences.} Every difference is $\Delta_{a-b}$, the Brier ALC of model $a$ minus model $b$, with J (Jeffreys), BB, R and KR (Beta-Binomial, Rasch, kernel Rasch) and fix, tune, fit or off as the subscript, in the Results text, table headers, the paired and robustness tables and the captions. \textbf{Not yet done:} the axis labels inside Person-B's figure files still read ``difference to Jeffreys'' or ``Fitted minus Jeffreys''.
\item[\textbf{S7}] \textbf{Related work} is a numbered section after the Introduction instead of a bold run-in paragraph.
\item[\textbf{S8}] \textbf{Title block.} Authors, affiliation, address and emails added (were \emph{Anonymous Author(s)} and the repository link). The page footer keeps the NeurIPS 2026 conference line.
\item[\textbf{S9}] \textbf{Methodology.} Section 4, formerly \emph{Models and prior sources}, is now \emph{Methodology} with two subsections: \emph{Models} (Beta-Binomial, Rasch, kernel Rasch as its own paragraph, the offline model) and \emph{Priors} (a sentence listing each model's prior parameters, then the three sources: fixed, tuned, fitted).
\item[\textbf{S10}] \textbf{Appendix.} Same symbols as the main text: rows of the Rasch-terms table use $\mu_d,\theta_i,\delta_j$ and the width $\sigma$; table rows and the paired, robustness and metrics tables use the model labels J, BB, R, KR (fix, tune, fit, off) with $\Delta_{a-b}$; ``panel'' removed.
\item[\textbf{S11}] \textbf{Section 3 structure.} The protocol text is split into bold run-in paragraphs: Data, Splits, Draws, Label acquisition, Scoring, In-distribution control. The wording is unchanged apart from S3.
\item[\textbf{S12}] \textbf{Kernel Rasch.} The two item priors are shown side by side as equations: Rasch $\boldsymbol\delta\sim\mathcal N(\mathbf 0,\sigma_\delta^2 I)$ and kernel Rasch $\boldsymbol\delta\sim\mathcal N(\mathbf 0,K)$ with $K=\sigma_\delta^2[(1-\rho)I+\rho S]$, so the only difference is visible; they coincide at $\rho=0$.
\item[\textbf{S13}] \textbf{Abstract.} ``language model'' is now ``LLM''; the rest of the abstract is unchanged, so its claims still await [1], [3] and [4].
\item[\textbf{S14}] \textbf{Abstract, authors.} Added the 17-benchmark average and the 680-split average with the share of harmful splits (the favourable-split point of [1]); replaced the item-features sentence with the probe result and the R5 null; added the equal-contribution mark (asterisk on both names, footnote ``Work done equally.'').
\item[\textbf{S15}] \textbf{Locked in.} Items [2], [3], [10], [13] and [14] are now in the paper and no longer shown as red or green: the 17-benchmark view (Spearman $0.96$, helps on 9 of 17, hurts on 1, macro $-0.0050$) in the abstract, the contributions and Section 5.3; the two neural probes and the R5 null in Section 6; the 680-split sentence in the limitations; the rewritten conclusion.
\item[\textbf{Q1, Q2}] \textbf{Open questions} on the 17-benchmark consistency and on Section 6 are shown as green question boxes in the text.
\item[\textbf{S16}] \textbf{Embeddings and kernel settings.} A new \emph{Item embeddings} paragraph (frozen BGE-M3, 1024 dimensions, 8,192-token context, unit length) is in Section 3 and Kernel Rasch refers to it; Section 3 is four bold paragraphs (Data, Item embeddings, Splits and draws, Label acquisition and scoring); the kernel paragraph states how unlabelled items are predicted; the settings moved to \emph{Priors}: tuned values for Rasch and kernel Rasch, and that the fitted kernel Rasch keeps $\rho=0.8$, $T=0.2$ from validation.
\item[\textbf{S17}] \textbf{Honest claims.} Abstract and conclusion lead with the average over 17 benchmarks and over the 680 splits and the share of harmful splits, then give the test and validation gains as the two examples. The label-sharing claim is replaced by a statement that it is untested (one subject per benchmark in every draw), also added to the limitations. The item-feature claim is limited to the two probes we trained (Section 6 notes a stronger predictor could do better).
\item[\textbf{S18}] \textbf{Item-difficulty section dissolved.} The separate Section 6 is gone. \emph{Probes} is a new subsection of Methodology (4.2: the probes, how they are scored, and Equation 5, the probe prior mean); the two tables and the discussion are Results 5.4, \emph{Do item features help?}; the scale $\gamma$ and the training-only probes are noted in \emph{Priors}; a contribution bullet was added.
\item[\textbf{S19}] \textbf{Research questions and headings.} The Results subsections are statements of what each shows (no separate list of research questions: the Introduction keeps its one governing question and the contribution bullets). Table headers read ID, OOD validation, OOD test (Table 1 reordered to match; Table 2 caption gives the benchmark counts).
\item[\textbf{F1, F3, F4, F5}] \textbf{Figure labels: done by Person-B (commit 2bea768).} The axis labels and legends inside Figures 1, 3, 4 and 5 still use the old wording (``difference to Jeffreys'', ``Fitted minus Jeffreys'', ``Test'', ``Validation''). They are drawn by Person-B's plot scripts, so Person-B must regenerate them with $\Delta_{a-b}$ labels and OOD legends; shown as red boxes at each figure.
\item[\textbf{S20}] \textbf{Tables.} All tables follow NeurIPS style (booktabs, captions above, no vertical rules). Table 1 and Tables 3, 4 and 8 bold the best value per column with a note in the caption; Table 1 shows the reference as ``0 (reference)''; Table 2 has grouped columns (Spearman, Pairwise under ID, OOD validation, OOD test) with the random score as a baseline row; Table 4 separates the added terms from the ablations; Table 7 has a mean row; Table 8 has a rule between splits and the split label on the first row of each block.
\item[\textbf{S21}] \textbf{Person-B's update incorporated.} Original-fit probe numbers (Rasch-target test Spearman $0.059$, below the random control's $0.076$), Table 2 rebuilt from the confirmed CSVs, and the R5 wording softened: the validation-selected scale improves Brier ALC by only about $0.0004$ on each split. The abstract, the contribution bullet, the Section 5.4 heading and the conclusion now say this instead of ``no scale helps''.
\item[\textbf{S5}] \textbf{Wording.} Section 2 is titled \emph{Data and evaluation protocol} (was \emph{Setting and online simulation}); ``panel'' became ``ID control set'' and ``response data''; ``replay'' became ``online simulation''.
\end{enumerate}
\end{document}"""
s = sub(s, "\\end{document}", END)
open("eandd_rv_setting.tex", "w").write(open("eandd_b_setting.tex").read())
s = sub(s, "\\input{eandd_b_setting}", "\\input{eandd_rv_setting}")
s = sub(s, "\\input{eandd_b_itemfeat}", "\\input{eandd_rv_itemfeat}")
s = sub(s, "\\input{eandd_b_limits}", "\\input{eandd_rv_limits}")
imp = [
 ("\\begin{abstract}\n", "A-title", "Title: still undecided. Option on the table: a title that states the finding, e.g. ``Offline Fitting Transfers a Prior, Not Item Estimates: Predicting LLM Item Outcomes on Unseen Benchmarks''.", "before"),
 ("\\begin{abstract}\n", "A-abstract", "Abstract: still long and number-heavy. Tighten it to one claim per sentence, check every number against the tables, and settle the headline (the average over splits) before finalising.", "after"),
 ("\\subsection{Priors}", "A-priors", "Priors (Section 4.3): the bullets are dense. Separate prior parameters from prior sources more clearly, consider a small table of the settings per model and source, and state plainly that the fitted kernel Rasch keeps $\\rho$ and $T$ from validation.", "after"),
 ("\\subsection{Priors}", "A-adaptive", "Methodology: adaptive priors are missing. Add the mixture prior, strength averaging and the pilot switch rule as models, with their equations (see NEW [5] above), before the results on them can be reported.", "before"),
 ("\\section{Conclusion}\n", "A-conclusion", "Conclusion: still to improve. The adaptive-priors sentence is unsupported until the body reports it; rewrite after the headline and the adaptive-priors result are settled, and keep it ranked and short.", "after"),
]
for key, tag, txt, where in imp:
    box = "\\par\\begin{improve}{" + tag + "}" + txt + "\\end{improve}\\par\n"
    assert s.count(key) >= 1, key
    s = s.replace(key, (box + key) if where == "before" else (key + box), 1)
open("eandd_reviewing.tex", "w").write(s)

idf = sub(idf, "Embeddings show a clear signal on known benchmarks.", "\\begin{qnote}{Q2}Section 5.4: the probe table and the prior-mean table now sit in Results, with the probes defined in Section 4.2. Does the abstract say no more than these two tables show?\\end{qnote}\n" + "Embeddings show a clear signal on known benchmarks.")
open("eandd_rv_itemfeat.tex", "w").write(idf)
open("eandd_rv_limits.tex", "w").write(lim)
print("written")
