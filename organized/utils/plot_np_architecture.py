#!/usr/bin/env python3
"""Architecture of the shipped ANP, in the two conventional NP figure styles.

    page 1  block diagram in the shape of Kim et al. (2019) Fig. 2: context
            pairs on the left, a deterministic path through target-to-context
            cross-attention and a latent path through a mean pool, both feeding
            one decoder.
    page 2  plate diagram in the style of Garnelo et al. (2018b) Fig. 1: what
            is observed, what is latent, what is deterministic, per task.

Drawn from ``organized/training/anp_crossattn_blocks.py`` as configured for
the bge_m3_long runs (width 64, one self-attention block, 4 heads, z in R^32).
Two facts of that code the paper figure does not have: the self-attention
block is shared by both paths, and keys and queries see the item embedding
only -- the label enters through the values alone.

    python organized/utils/plot_np_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import HAIRLINE, INK, MUTED, SUBTLE, TEXTWIDTH  # noqa: E402

DET = MUTED["indigo"]   # deterministic path
LAT = MUTED["rose"]     # latent path
SHARED = MUTED["sand"]  # shared by both
TARGET = MUTED["teal"]  # target inputs and output


def exact_page(fig) -> Bbox:
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


def canvas(height: float, width_units: float = 110):
    fig = plt.figure(figsize=(TEXTWIDTH, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width_units)
    ax.set_ylim(0, width_units * height / TEXTWIDTH)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, colour, fs=6.2, alpha=0.28, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=colour, ec=colour, lw=0.9, alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=colour, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def node(ax, x, y, text, colour, r=2.6, filled=False, dashed=False, fs=7):
    ax.add_patch(Circle((x, y), r, fc=colour if filled else "white", alpha=1 if not filled else 0.35,
                        ec=colour, lw=0.9, ls=(0, (2.5, 1.5)) if dashed else "-"))
    if filled:
        ax.add_patch(Circle((x, y), r, fc="none", ec=colour, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, pts, colour=INK, dashed=False, lw=0.8):
    """Polyline whose last segment carries the head; elbows stay square."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=lw,
                ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=6,
                                 color=colour, lw=lw, shrinkA=0, shrinkB=0,
                                 ls=(0, (3, 2)) if dashed else "-"))


def label(ax, x, y, text, colour=SUBTLE, fs=5.6, **kw):
    ax.text(x, y, text, fontsize=fs, color=colour, **kw)


# ------------------------------------------------------------- Kim Fig. 2 --
def stack(ax, x, y, w, h, text, colour, n=3, step=0.7):
    """A box drawn as a short deck: one module applied to every context pair."""
    for j in reversed(range(1, n)):
        ax.add_patch(FancyBboxPatch((x + j * step, y + j * step), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    fc="white", ec=colour, lw=0.7))
    box(ax, x, y, w, h, text, colour)


def np_panel(ax, y0: float, attentive: bool):
    """One panel of Kim et al. (2019) Fig. 2; y0 is the panel's baseline.

    Deterministic path on top, latent path below, decoder on the right. The
    attentive panel inserts self-attention on both paths and swaps the
    deterministic mean for cross-attention queried by the target input.
    """
    yd, yl = y0 + 24, y0 + 6          # path centre lines
    ax.text(1.5, y0 + 37, "(b) Attentive Neural Process" if attentive else "(a) Neural Process",
            fontsize=7.5, fontweight="bold", va="top")

    # Context pairs, then the split into the two paths.
    for j, yy in enumerate((yd - 1, y0 + 12.5, yl - 3)):
        box(ax, 2, yy, 11, 4.5, f"$x_{j + 1}, y_{j + 1}$" if j < 2 else "$x_n, y_n$", INK, alpha=0.06, fs=5.8)
    ax.text(7.5, y0 + 9.4, "$\\vdots$", ha="center", va="center", fontsize=6.5)
    arrow(ax, [(13, yd + 1.25), (19, yd + 1.25)], DET)
    arrow(ax, [(13, yl - 0.75), (19, yl - 0.75)], LAT)
    ax.plot([15.5, 15.5], [yd + 1.25, yl - 0.75], color=INK, lw=0.6)
    ax.plot([13, 15.5], [y0 + 14.75, y0 + 14.75], color=INK, lw=0.6)

    x = 19
    for yy, colour in ((yd, DET), (yl, LAT)):
        stack(ax, x, yy - 3.5, 9, 7, "MLP", colour)
    x += 13
    if attentive:
        for yy, colour in ((yd, DET), (yl, LAT)):
            stack(ax, x, yy - 3.5, 12, 7, "self-\nattention", colour)
            arrow(ax, [(x - 2.0, yy), (x, yy)], colour)
        x += 16
    else:
        for yy, colour in ((yd, DET), (yl, LAT)):
            ax.plot([x - 2.0, x + 3], [yy, yy], color=colour, lw=0.8)
        x += 3
    label(ax, x - 1.4, yd + 1.2, "$r_i$", colour=DET, fs=5.6, ha="center")
    label(ax, x - 1.4, yl + 1.2, "$s_i$", colour=LAT, fs=5.6, ha="center")

    # Latent path: mean -> s_C -> z.
    box(ax, x, yl - 3.5, 9, 7, "mean", LAT)
    arrow(ax, [(x - 3, yl), (x, yl)], LAT)
    node(ax, x + 15, yl, "$s_C$", LAT, r=2.5, fs=6.2)
    arrow(ax, [(x + 9, yl), (x + 12.5, yl)], LAT)
    node(ax, x + 25, yl, "$z$", LAT, r=2.5, dashed=True, fs=6.5)
    arrow(ax, [(x + 17.5, yl), (x + 22.5, yl)], LAT)
    label(ax, x + 20, yl + 3.2, "$\mathcal{N}$", colour=LAT, fs=5.6, ha="center")

    # Deterministic path: mean -> r_C, or cross-attention -> r_*.
    dec_x = 89
    if attentive:
        box(ax, x, yd - 4, 20, 8, "cross-attention\nK: $x_i$   V: $r_i$", DET)
        arrow(ax, [(x - 3, yd), (x, yd)], DET)
        node(ax, x + 25, yd, "$r_*$", DET, r=2.5, fs=6.2)
        arrow(ax, [(x + 20, yd), (x + 22.5, yd)], DET)
        box(ax, x + 5, y0 + 31, 10, 4.5, "$x_*$", TARGET, fs=6)
        arrow(ax, [(x + 10, y0 + 31), (x + 10, yd + 4)], DET)
        label(ax, x + 11, yd + 5, "Q", colour=DET, fs=5.4)
        xq = x + 15
    else:
        box(ax, x, yd - 3.5, 9, 7, "mean", DET)
        arrow(ax, [(x - 3, yd), (x, yd)], DET)
        node(ax, x + 25, yd, "$r_C$", DET, r=2.5, fs=6.2)
        arrow(ax, [(x + 9, yd), (x + 22.5, yd)], DET)
        box(ax, x + 5, y0 + 31, 10, 4.5, "$x_*$", TARGET, fs=6)
        xq = x + 15

    # Decoder.
    box(ax, dec_x, yl - 4, 11, yd - yl + 8, "decoder\nMLP", INK, alpha=0.08)
    arrow(ax, [(x + 27.5, yd), (dec_x, yd)], DET)
    arrow(ax, [(x + 27.5, yl), (dec_x, yl)], LAT)
    arrow(ax, [(xq, y0 + 33.25), (dec_x + 5.5, y0 + 33.25), (dec_x + 5.5, yd + 4)], TARGET)
    node(ax, 106, (yd + yl) / 2, "$y_*$", TARGET, r=2.4, dashed=True, fs=6.2)
    arrow(ax, [(dec_x + 11, (yd + yl) / 2), (103.6, (yd + yl) / 2)], TARGET)


def kim_page(pdf):
    fig, ax = canvas(4.55)
    top = 110 * 4.55 / TEXTWIDTH
    ax.text(1.5, top - 1.5, "Neural Process vs Attentive Neural Process", fontsize=9,
            fontweight="bold", va="top")
    label(ax, 1.5, top - 6, "After Kim et al. (2019), Fig. 2.  Blue: deterministic.  Red: latent.  "
          "Teal: target.  Stacked boxes: applied to every context pair.",
          fs=6.2, va="top")
    np_panel(ax, 41, attentive=False)
    np_panel(ax, 1, attentive=True)
    ax.plot([1.5, 108.5], [40.5, 40.5], color=HAIRLINE, lw=0.6)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


# ------------------------------------------------------------------ page 2 --
def block_page(pdf):
    """Ours, on a strict grid: every module the same height, two path rows.

    The two differences from panel (b) are structural: one value encoder and
    one self-attention block are shared by both paths, and the keys come from
    the item embedding alone, so labels reach the target only via the values.
    """
    H = 2.75
    fig, ax = canvas(H)
    top = 110 * H / TEXTWIDTH
    ax.text(1.5, top - 1.2, "Our Attentive Neural Process", fontsize=9, fontweight="bold", va="top")
    label(ax, 1.5, top - 5.6, "Left: context $C$.  Sand: shared by both paths.  Blue: deterministic.  Red: latent.  "
          "Teal: target.  $d=64$, 4 heads, $z\in\mathbb{R}^{32}$.", fs=6.0, va="top")

    yd, yl = 27, 9            # path rows
    yc = (yd + yl) / 2        # shared trunk
    h = 8                     # module height

    # Context column.
    for yy, t in ((yc + 6, "$x_1, y_1$"), (yc - 0.5, "$x_2, y_2$"), (yc - 10.5, "$x_n, y_n$")):
        box(ax, 2, yy - 2.25, 11, 4.5, t, INK, alpha=0.06, fs=5.8)
    ax.text(7.5, yc - 5.3, "$\\vdots$", ha="center", va="center", fontsize=6.5)

    # Shared trunk.
    box(ax, 18, yc - h / 2, 15, h, "value\nencoder", SHARED, alpha=0.35)
    box(ax, 37, yc - h / 2, 13, h, "self-\nattention", SHARED, alpha=0.35)
    arrow(ax, [(13, yc), (18, yc)])
    arrow(ax, [(33, yc), (37, yc)])
    label(ax, 25.5, yc - h / 2 - 3.2, "$v_i = P(x_i) + W(2y_i{-}1)$", ha="center", fs=5.4)

    # Split.
    ax.plot([50, 53.5], [yc, yc], color=INK, lw=0.8)
    ax.plot([53.5, 53.5], [yl, yd], color=INK, lw=0.8)

    # Deterministic row.
    box(ax, 57, yd - h / 2, 19, h, "cross-\nattention", DET)
    arrow(ax, [(53.5, yd), (57, yd)], DET)
    label(ax, 55.2, yd + 0.8, "V", ha="center", colour=DET, fs=5.4)
    node(ax, 81, yd, "$r_*$", DET, r=2.4, fs=6.2)
    arrow(ax, [(76, yd), (78.6, yd)], DET)
    # Keys: straight from the context embeddings, entering from above.
    arrow(ax, [(7.5, yc + 8.25), (7.5, 36), (60, 36), (60, yd + h / 2)], DET)
    label(ax, 61, 33, "K", colour=DET, fs=5.4)
    label(ax, 33, 36.8, "keys $k(x_i)$: embedding only, never the label", ha="center", colour=DET, fs=5.4)
    # Query: the target item.
    box(ax, 66, 38, 10, 4.5, "$x_*$", TARGET, fs=6)
    arrow(ax, [(71, 38), (71, yd + h / 2)], DET)
    label(ax, 72, 33, "Q", colour=DET, fs=5.4)

    # Latent row.
    box(ax, 57, yl - h / 2, 10, h, "mean", LAT)
    arrow(ax, [(53.5, yl), (57, yl)], LAT)
    node(ax, 71.5, yl, "$s_C$", LAT, r=2.4, fs=6.2)
    arrow(ax, [(67, yl), (69.1, yl)], LAT)
    node(ax, 81, yl, "$z$", LAT, r=2.4, dashed=True, fs=6.5)
    arrow(ax, [(73.9, yl), (78.6, yl)], LAT)
    label(ax, 76.2, yl + 1.4, "$\\mathcal{N}$", colour=LAT, fs=5.4, ha="center")

    # Decoder.
    dx = 88
    box(ax, dx, yl - h / 2, 11, yd - yl + h, "decoder\nMLP", INK, alpha=0.08)
    arrow(ax, [(83.4, yd), (dx, yd)], DET)
    arrow(ax, [(83.4, yl), (dx, yl)], LAT)
    arrow(ax, [(76, 40.25), (dx + 5.5, 40.25), (dx + 5.5, yd + h / 2)], TARGET)
    node(ax, 106, yc, "$y_*$", TARGET, r=2.4, dashed=True, fs=6.2)
    arrow(ax, [(dx + 11, yc), (103.6, yc)], TARGET)

    label(ax, 1.5, 0.8, "Train: $z\\sim q(z\\mid C\\cup T)$; loss BCE + KL$(q(z\\mid C\\cup T)\\,\\|\\,q(z\\mid C))$.  "
          "Prior-anchor variant: + beta-binomial logit of $y_C$.", fs=5.4)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


# ------------------------------------------------------------------ page 2 --
def plate(ax, x, y, w, h, text, colour=INK):
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=colour, lw=0.8))
    ax.text(x + w - 1, y + 1, text, ha="right", va="bottom", fontsize=6, color=colour)


def diamond(ax, x, y, text, colour, r=3.0):
    ax.add_patch(Polygon([(x - r, y), (x, y + r), (x + r, y), (x, y - r)], fc="white", ec=colour, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5)


def plate_page(pdf):
    fig, ax = canvas(3.0)
    ax.text(1.5, 58.5, "Graphical model of the ANP", fontsize=9, fontweight="bold", va="top")
    label(ax, 1.5, 54, "Shaded: observed.  Dashed circle: latent.  Diamond: deterministic.  "
          "Solid edge: generative.  Dashed edge: inference $q(z\\mid C)$.", fs=6.2, va="top")

    plate(ax, 6, 4, 98, 44, "tasks $\\tau$ = subject $\\times$ benchmark")
    plate(ax, 12, 9, 30, 30, "$i \\in C$")
    plate(ax, 58, 9, 40, 30, "$t \\in T$")

    node(ax, 50, 42, "$z$", LAT, r=3, dashed=True)
    node(ax, 20, 28, "$x_i$", INK, r=3, filled=True)
    node(ax, 34, 28, "$y_i$", INK, r=3, filled=True)
    node(ax, 66, 28, "$x_t$", INK, r=3, filled=True)
    diamond(ax, 78, 20, "$r_t$", DET)
    node(ax, 90, 28, "$y_t$", TARGET, r=3)

    arrow(ax, [(23, 28), (31, 28)])
    arrow(ax, [(69, 28), (87, 28)])
    arrow(ax, [(68.2, 25.9), (75.8, 21.8)], DET)
    arrow(ax, [(81, 20), (88, 25.6)], DET)
    # Context reaches r_t through cross-attention: deterministic, so dashed blue.
    ax.plot([20, 20], [25, 14], color=DET, lw=0.8, ls=(0, (3, 2)))  # keys
    arrow(ax, [(34, 25), (34, 14), (78, 14), (78, 17)], DET, dashed=True)
    ax.plot([20, 34], [14, 14], color=DET, lw=0.8, ls=(0, (3, 2)))
    label(ax, 52, 15, "cross-attention (keys $x_i$, values $x_i, y_i$)", colour=DET, ha="center", fs=5.4)
    # z generates every label; inference runs from the context to z.
    arrow(ax, [(53, 40.7), (87.4, 29.8)], LAT)
    arrow(ax, [(47, 40.7), (36.6, 29.8)], LAT)
    arrow(ax, [(21.5, 30.6), (47.1, 42.6)], LAT, dashed=True)
    label(ax, 24, 37.2, "$q(z\\mid C)$", colour=LAT, fs=5.6)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def main() -> int:
    out = ROOT / "docs" / "np-architecture-final.pdf"
    png = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    with PdfPages(out) as pdf:
        figs = [("kim-fig2", kim_page(pdf)), ("block", block_page(pdf)), ("plate", plate_page(pdf))]
    if png:
        png.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            fig.savefig(png / f"{name}.png", dpi=150, bbox_inches=exact_page(fig))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
