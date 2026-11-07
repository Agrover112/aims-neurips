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
BANNER = "\\maketitle\n"
s = sub(s, "\\maketitle\n", BANNER)

# ---- edits already made (S1, S2 accepted) ----

# ---- planned changes ----
a1 = "On unseen benchmarks that gain disappears, and only a shrinkage prior transfers: fitted on training it gains $0.014$ on test but loses $0.008$ on validation."
n1 = " Averaged over all 17 benchmarks it gains $0.005$, and over all 680 choices of three held-out benchmarks it gains $0.004$ and hurts in $17.5\\%$ of them, so our test split is a favourable one."
figfix = {}
for fn, (tag, txt) in figfix.items():
    key = "\\includegraphics{\\figdir/" + fn + "}"
    s = sub(s, key, key + "\n\\begin{fixnote}{" + tag + "}" + txt + "\\end{fixnote}")

END = r"""\end{document}"""
s = sub(s, "\\end{document}", END)
open("eandd_rv_setting.tex", "w").write(open("eandd_b_setting.tex").read())
s = sub(s, "\\input{eandd_b_setting}", "\\input{eandd_rv_setting}")
s = sub(s, "\\input{eandd_b_itemfeat}", "\\input{eandd_rv_itemfeat}")
s = sub(s, "\\input{eandd_b_limits}", "\\input{eandd_rv_limits}")
imp = [
]
for key, tag, txt, where in imp:
    box = "\\par\\begin{improve}{" + tag + "}" + txt + "\\end{improve}\\par\n"
    assert s.count(key) >= 1, key
    s = s.replace(key, (box + key) if where == "before" else (key + box), 1)
open("eandd_reviewing.tex", "w").write(s)

open("eandd_rv_itemfeat.tex", "w").write(idf)
open("eandd_rv_limits.tex", "w").write(lim)
print("written")
