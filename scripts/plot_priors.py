"""docs/model-results-visualization.pdf: the ability priors explained visually.

Pages: (1) what a prior is and the three shapes, (2) how hard each prior pulls,
(3) a worked example using the real solver, (4) the measured results.
Usage: plot_priors.py <heavytail-result.json> <out.pdf>
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aims_starter.r_c_attention import MultidimContextIRT

RES = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
OUT = sys.argv[2]
TAU2 = 4.0
SCALE = math.sqrt(TAU2)  # 2.0
GREY, BLUE, ORANGE, RED, DARK = "#7f8c9b", "#2b6ca3", "#d98324", "#c0392b", "#33414e"
COLORS = {"gauss": BLUE, "t3": ORANGE, "cauchy": RED}
NAMES = {"gauss": "Gaussian (the old default)", "t3": "Student-t, 3 df  (t3)",
         "cauchy": "Cauchy = Student-t, 1 df"}


def density(kind, x):
    if kind == "gauss":
        return np.exp(-0.5 * (x / SCALE) ** 2) / (SCALE * math.sqrt(2 * math.pi))
    nu = 3.0 if kind == "t3" else 1.0
    c = math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2) * SCALE)
    return c * (1 + (x / SCALE) ** 2 / nu) ** (-(nu + 1) / 2)


def pull(kind, x):
    """Force dragging theta back to 0: derivative of -log(prior)."""
    if kind == "gauss":
        return x / TAU2
    nu = 3.0 if kind == "t3" else 1.0
    return (nu + 1) * x / (nu * TAU2 + x ** 2)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def header(fig, title, sub):
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.925, sub, fontsize=9.5, color="#44505c", va="top", wrap=True)


def note(fig, x, y, text, size=9.5, color=DARK, **kw):
    fig.text(x, y, text, fontsize=size, color=color, va="top", **kw)


pdf = PdfPages(OUT)
xs = np.linspace(-14, 14, 1400)

# ---------------- page 1: what is a prior, the three shapes -----------------
fig = plt.figure(figsize=(11, 8.5))
header(fig, "What is an ability prior?",
       "Every subject gets one number, theta, for how good it is (higher = better). "
       "Before we see any of its answers, we still have to guess a value.\n"
       "The prior is that guess, drawn as a hill: the higher the hill at a value, the more "
       "plausible that value is BEFORE seeing any labels.")
ax = fig.add_axes([0.07, 0.50, 0.40, 0.33])
for k in ("gauss", "t3", "cauchy"):
    ax.plot(xs, density(k, xs), color=COLORS[k], lw=2.4, label=NAMES[k])
ax.set_xlim(-10, 10)
ax.set_title("A. The three hills (same width, different tails)", fontsize=10, loc="left",
             fontweight="bold")
ax.set_xlabel("theta  (ability)"); ax.set_ylabel("how plausible"); ax.legend(fontsize=8,
                                                                            frameon=False)
style(ax)
ax = fig.add_axes([0.57, 0.50, 0.38, 0.33])
for k in ("gauss", "t3", "cauchy"):
    ax.plot(xs, density(k, xs), color=COLORS[k], lw=2.4)
ax.set_yscale("log"); ax.set_xlim(-14, 14); ax.set_ylim(1e-8, 0.3)
for edge, lab in ((4, "old clamp\n(+-4)"), (12, "wide clamp\n(+-12)")):
    for sgn in (-1, 1):
        ax.axvline(sgn * edge, color=GREY, ls="--", lw=1)
    ax.text(edge + 0.2, 3e-8, lab, fontsize=7.5, color=GREY)
ax.set_title("B. Same picture, log scale: zoom into the tails", fontsize=10, loc="left",
             fontweight="bold")
ax.set_xlabel("theta  (ability)"); ax.set_ylabel("how plausible (log)"); style(ax)
note(fig, 0.055, 0.40, "How to read it", 11, DARK, fontweight="bold")
note(fig, 0.055, 0.37,
     "- Near the middle all three look alike. They only disagree about EXTREME subjects.\n"
     "- Gaussian: extreme values are very unlikely. At theta = 8 it is about 3,000 times less "
     "plausible than at 0.\n"
     "- Student-t / Cauchy: extreme values are rare but possible. They have 'heavy tails'.\n"
     "- Cauchy has the heaviest tails, t3 is in between, Gaussian has the lightest.\n"
     "- The dashed lines are the CLAMP: a hard wall. The solver is not allowed to put theta "
     "beyond it,\n  no matter what the answers say. 'Old clamp' = +-4, 'wide clamp' = +-12.",
     10)
note(fig, 0.055, 0.17, "Why we care", 11, DARK, fontweight="bold")
note(fig, 0.055, 0.14,
     "Some subjects get almost everything wrong (one deepseek run scored 0.8%). Their true "
     "theta is very negative.\nIf the prior says 'that is nearly impossible', it keeps pulling "
     "them back toward the middle, and we predict they will do better than they do.",
     10)
pdf.savefig(fig); plt.close(fig)

# ---------------- page 2: how hard each prior pulls -------------------------
fig = plt.figure(figsize=(11, 8.5))
header(fig, "How hard does each prior pull?",
       "The prior acts like a rubber band tied to zero. The further the subject's theta is "
       "from zero, the more the band pulls it back.\nThe question is how the pull changes as "
       "theta gets more extreme.")
ax = fig.add_axes([0.08, 0.40, 0.55, 0.42])
px = np.linspace(0, 14, 600)
for k in ("gauss", "t3", "cauchy"):
    ax.plot(px, pull(k, px), color=COLORS[k], lw=2.6, label=NAMES[k])
ax.set_xlim(0, 14); ax.set_ylim(0, 3.6)
ax.axvline(4, color=GREY, ls="--", lw=1); ax.text(4.1, 3.3, "old clamp", fontsize=8, color=GREY)
ax.set_xlabel("how extreme the subject's theta is (distance from 0)")
ax.set_ylabel("pull back toward 0 (rubber-band force)")
ax.legend(fontsize=9, frameon=False, loc="upper left"); style(ax)
note(fig, 0.67, 0.80, "What the curves say", 11, DARK, fontweight="bold")
note(fig, 0.67, 0.77,
     "Gaussian (blue)\nThe pull keeps growing the\nfurther out you go. At theta = 12\n"
     "it is tugging 3 units hard.\nEvidence has to shout louder\nand louder to move it.\n\n"
     "Student-t (orange, red)\nThe pull rises, peaks, then\nFADES. Far out, the band goes\n"
     "slack and the answers decide.\nCauchy fades the fastest.",
     9.5)
note(fig, 0.08, 0.30, "In plain words", 11, DARK, fontweight="bold")
note(fig, 0.08, 0.27,
     "- Ordinary subjects (theta near 0): all three priors pull about the same. No difference.\n"
     "- Extreme subjects: the Gaussian keeps dragging them back. The heavy-tailed priors let go.\n"
     "- That is the whole idea: shrink small, uncertain estimates, but do not crush a subject "
     "whose labels clearly say 'always fails'.\n"
     "- Trade-off: letting go also means a few unlucky labels can push a normal subject too far.\n"
     "  That is why we also tried the middle option (t3) and not only Cauchy.",
     10)
pdf.savefig(fig); plt.close(fig)

# ---------------- page 3: worked example with the real solver --------------
fig = plt.figure(figsize=(11, 8.5))
header(fig, "A worked example: a subject that fails everything",
       "We reveal labels one by one. Every label is a FAIL, on items that most subjects pass "
       "(easy items).\nThe plots below use the real solver from our model, so this is what the "
       "code would do, not a cartoon.")
ARMS = [("gauss_box2", "Gaussian, old clamp (+-4)", BLUE, "-", None, 2.0),
        ("gauss_box6", "Gaussian, wide clamp (+-12)", BLUE, "--", None, 6.0),
        ("t3_box6", "t3, wide clamp", ORANGE, "-", 3.0, 6.0),
        ("cauchy_box6", "Cauchy, wide clamp", RED, "-", 1.0, 6.0)]
ns = list(range(32))
thetas = {}
for key, _, _, _, nu, box in ARMS:
    path = []
    for n in ns:
        a = torch.ones(n, 1); d = torch.full((n,), -1.0); y = torch.zeros(n)
        th = MultidimContextIRT.map_theta(a, d, y, prior_variance=TAU2, prior_nu=nu,
                                          box_sigmas=box, steps=30)
        path.append(float(th[0]))
    thetas[key] = np.array(path)
ax = fig.add_axes([0.08, 0.50, 0.40, 0.33])
for key, lab, c, ls, _, _ in ARMS:
    ax.plot(ns, thetas[key], color=c, ls=ls, lw=2.4, label=lab)
ax.set_title("A. What the solver believes theta is", fontsize=10, loc="left",
             fontweight="bold")
ax.set_xlabel("number of failed labels revealed"); ax.set_ylabel("estimated theta")
ax.legend(fontsize=7.5, frameon=False); style(ax)
ax = fig.add_axes([0.58, 0.50, 0.38, 0.33])
for key, lab, c, ls, _, _ in ARMS:
    ax.plot(ns, 1 / (1 + np.exp(-(thetas[key] + 1.0))), color=c, ls=ls, lw=2.4)
ax.axhline(0.0, color="k", lw=0.6)
ax.set_title("B. So the chance it predicts a pass on a typical item", fontsize=10,
             loc="left", fontweight="bold")
ax.set_xlabel("number of failed labels revealed"); ax.set_ylabel("predicted chance of passing")
ax.set_ylim(-0.02, 0.6); style(ax)
note(fig, 0.055, 0.40, "What to notice", 11, DARK, fontweight="bold")
g0 = thetas["gauss_box2"][31]; t3v = thetas["t3_box6"][31]; cv = thetas["cauchy_box6"][31]
note(fig, 0.055, 0.37,
     f"- After 31 straight failures: Gaussian with the old clamp says theta = {g0:.1f}, "
     f"Gaussian with the wide clamp {thetas['gauss_box6'][31]:.1f},\n"
     f"  t3 says {t3v:.1f}, Cauchy says {cv:.1f}.\n"
     "- The flat part of the solid blue line at -4 is the CLAMP (a wall), not the prior.\n"
     "- Remove the wall (dashed blue) and the Gaussian still slows down a lot: its own pull "
     "keeps growing,\n  so it only creeps lower. Widening the clamp alone changes little. "
     "That matches our result.\n"
     "- The heavy-tailed priors keep sliding down as the failures pile up, so the predicted "
     "pass chance keeps shrinking, as it should.\n"
     "- At 0 labels all four are identical: with no evidence, the prior is the guess.",
     10)
note(fig, 0.055, 0.17, "Setup: items of difficulty -1 (an average subject passes about 73%), "
     "one ability dimension, prior width 2 (variance 4).", 8.5, GREY)
pdf.savefig(fig); plt.close(fig)

# ---------------- page 4: measured results ---------------------------------
br = RES["base_rate"]; arms = list(RES["task_alc"])
groups = {"hard\n(pass rate < 0.1)": lambda b: b < 0.1,
          "0.1 - 0.5": lambda b: 0.1 <= b < 0.5,
          "0.5 - 0.7": lambda b: 0.5 <= b < 0.7,
          "easy\n(>= 0.7)": lambda b: b >= 0.7}
base = RES["task_alc"]["gauss_box2"]["oof"]
shown = [("gauss_box6", "Gaussian + wide clamp", BLUE), ("t3_box6", "t3 + wide clamp", ORANGE),
         ("cauchy_box6", "Cauchy + wide clamp", RED)]
fig = plt.figure(figsize=(11, 8.5))
header(fig, "Did it help? Measured results",
       "Change in Brier ALC against the old setup (Gaussian, +-4 clamp). Lower is better, so "
       "BARS GOING DOWN = BETTER.\nTasks are grouped by how often the subject passes. Model: "
       "me5 embeddings + 4-dimensional ability. Validation set, one seed.")
ax = fig.add_axes([0.08, 0.40, 0.88, 0.43])
width = 0.26
for i, (key, lab, c) in enumerate(shown):
    vals = []
    for f in groups.values():
        ts = [t for t in base if t in RES["task_alc"][key]["oof"] and f(br[t])]
        vals.append(np.mean([RES["task_alc"][key]["oof"][t] - base[t] for t in ts]))
    bars = ax.bar(np.arange(4) + (i - 1) * width, vals, width, color=c, label=lab)
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + (-0.00025 if v < 0 else 0.0001),
                f"{v:+.4f}", ha="center", va="top" if v < 0 else "bottom", fontsize=7.5)
counts = [sum(f(br[t]) for t in base) for f in groups.values()]
ax.set_xticks(range(4))
ax.set_xticklabels([f"{g}\n{n} tasks" for g, n in zip(groups, counts, strict=True)],
                   fontsize=9)
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("change in ALC vs old setup")
ax.legend(fontsize=9, frameon=False, loc="lower right"); style(ax)
note(fig, 0.055, 0.30, "What it says", 11, DARK, fontweight="bold")
note(fig, 0.055, 0.27,
     "- Widening the clamp alone (blue) does almost nothing. The prior is the problem, not "
     "the wall.\n"
     "- Heavy tails help the hard tasks a lot: Cauchy improves them by about 0.0045, t3 by "
     "about 0.0025.\n"
     "- The easy tasks are not hurt after calibration (all within 0.0005 of the old setup).\n"
     "- Overall the gain is small, about 0.0004 to 0.0006, because only 6 of 104 tasks are "
     "hard.",
     10)
note(fig, 0.055, 0.13, "Be careful", 11, RED, fontweight="bold")
note(fig, 0.055, 0.10,
     "- One seed and only 6 hard tasks: a lead, not a proven result.\n"
     "- Before calibration, Cauchy made the easy tasks worse (+0.0038). It looks fine only "
     "after calibration, so t3 is the safer pick.",
     10)
pdf.savefig(fig); plt.close(fig)
# ---------------- page 5: train / inference prior mismatch -----------------
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(11, 8.5))
header(fig, "The mismatch: one prior in training, another at test time",
       "Our model runs the same ability solver (Newton MAP) in two places. Only one of them "
       "got the new prior.")


def box(ax, x, y, w, h, text, face, edge=DARK, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=face, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=DARK)


ax = fig.add_axes([0.04, 0.50, 0.92, 0.36]); ax.axis("off"); ax.set_xlim(-0.04, 1.04)
ax.set_ylim(0, 1)
ax.text(0.0, 0.97, "TRAINING  (item network learns difficulty and discrimination)",
        fontsize=10, fontweight="bold", color=BLUE, va="top")
box(ax, 0.00, 0.50, 0.20, 0.30, "labels of the\ncontext items", "#eef3f8")
box(ax, 0.26, 0.50, 0.22, 0.30, "ability solver\nprior = GAUSSIAN\n(hard-wired)", "#d6e4f2")
box(ax, 0.54, 0.50, 0.20, 0.30, "predict the\ntarget items", "#eef3f8")
box(ax, 0.80, 0.50, 0.20, 0.30, "loss -> update the\nitem network", "#eef3f8")
for x0, x1 in ((0.20, 0.26), (0.48, 0.54), (0.74, 0.80)):
    ax.annotate("", xy=(x1, 0.65), xytext=(x0, 0.65), arrowprops={"arrowstyle": "->", "lw": 1.5})
ax.text(0.0, 0.40, "TEST TIME  (what we changed)", fontsize=10, fontweight="bold",
        color=RED, va="top")
box(ax, 0.00, 0.00, 0.20, 0.30, "labels of the\nacquired items", "#fbeeee")
box(ax, 0.26, 0.00, 0.22, 0.30, "ability solver\nprior = t3 / CAUCHY\n(swapped in)", "#f6d3cf")
box(ax, 0.54, 0.00, 0.20, 0.30, "predict the\nevaluation items", "#fbeeee")
box(ax, 0.80, 0.00, 0.20, 0.30, "item network:\nFROZEN, never saw\nthis prior", "#f4e1b8")
for x0, x1 in ((0.20, 0.26), (0.48, 0.54)):
    ax.annotate("", xy=(x1, 0.15), xytext=(x0, 0.15), arrowprops={"arrowstyle": "->", "lw": 1.5})
ax.annotate("", xy=(0.74, 0.15), xytext=(0.80, 0.15),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "ls": "--", "color": GREY})

ax = fig.add_axes([0.07, 0.10, 0.42, 0.30])
ax.plot(ns, thetas["gauss_box2"], color=BLUE, lw=2.6, label="what training assumed (Gaussian)")
ax.plot(ns, thetas["t3_box6"], color=ORANGE, lw=2.4, label="what test time now does (t3)")
ax.plot(ns, thetas["cauchy_box6"], color=RED, lw=2.4, label="what test time now does (Cauchy)")
ax.fill_between(ns, thetas["gauss_box2"], thetas["cauchy_box6"], color=RED, alpha=0.12)
ax.set_xlabel("failed labels revealed"); ax.set_ylabel("estimated theta")
ax.set_title("The shaded gap is the mismatch", fontsize=10, loc="left", fontweight="bold")
ax.legend(fontsize=7.5, frameon=False); style(ax)
note(fig, 0.53, 0.40, "Why it matters", 11, DARK, fontweight="bold")
note(fig, 0.53, 0.37,
     "- The item network was trained while the Gaussian\n  kept shrinking theta. It may have learned\n"
     "  difficulties that quietly make up for that shrinkage.\n"
     "- At test time we remove the shrinkage, so those\n  difficulties are slightly off.\n"
     "- Even so, heavy tails helped the hard tasks\n  (one seed, 6 tasks). Retraining with the new\n"
     "  prior MIGHT help more, but we have not tested it.\n\n"
     "Not done yet: retraining with the t prior\nin the training loop (needs a GPU run).",
     9.5)
pdf.savefig(fig); plt.close(fig)
pdf.close()
print("saved", OUT)
