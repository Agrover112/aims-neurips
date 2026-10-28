"""Render a compact architecture map for the NP experiments.

The first three pages explain the model families used in this repository.  The
fourth appends the existing attention-complexity figure unchanged, so the
architecture choice and its cost stay in one document.

    .venv/bin/python docs/generate_np_design_pdf.py

The diagrams are deliberately implementation-facing.  They describe the
modules in ``anp_crossattn_blocks.py`` and ``aea_np_blocks.py`` rather than a
generic neural-process sketch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "organized" / "utils"))

from plotstyle import BRIGHT, HAIRLINE, INK, MUTED, SUBTLE, TEXTWIDTH  # noqa: E402


OUT = ROOT / "docs" / "np-design.pdf"
COMPLEXITY = ROOT / "docs" / "np-design-complexity.pdf"
_TEMP = ROOT / "docs" / ".np-design-architectures.tmp.pdf"

DETERMINISTIC = MUTED["cyan"]
LATENT = MUTED["purple"]
SHARED = MUTED["grey"]
SET_YELLOW = BRIGHT["yellow"]
WHITE = "white"
OBSERVED_RV = MUTED["green"]
LATENT_RV = MUTED["purple"]
PREDICTED_RV = MUTED["rose"]


def _box(ax, x, y, w, h, text, *, color=WHITE, edge=INK, fontsize=5.35,
         lw=0.65, weight="normal", rounded=0.018, text_color=INK):
    """Draw one labelled module in axes coordinates."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={rounded}",
        facecolor=color, edgecolor=edge, linewidth=lw,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color=text_color,
            fontweight=weight, linespacing=1.15, zorder=3)
    return (x, y, w, h)


def _circle(ax, x, y, radius, text, *, color, observed: bool, fontsize=6.7):
    """Draw a stochastic variable. Solid circles are observed; dashed are not."""
    patch = Circle(
        (x, y), radius, transform=ax.transAxes, clip_on=False,
        facecolor=to_rgba(color, 0.23) if observed else WHITE,
        edgecolor=color, linewidth=0.9,
        linestyle="-" if observed else (0, (3, 2)), zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3)
    return (x, y, radius)


def _arrow(ax, start, end, *, color=INK, lw=0.7, style="-"):
    """Draw a thin directed connector between two axes-coordinate points."""
    ax.annotate(
        "", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style,
                        shrinkA=0, shrinkB=0, mutation_scale=8),
        zorder=1,
    )


def _label(ax, x, y, text, *, color=SUBTLE, size=6.2, ha="center",
           weight="normal"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="center", fontsize=size,
            color=color, fontweight=weight, linespacing=1.12, zorder=3)


def _title(ax, title, subtitle):
    ax.text(0.02, 0.965, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold")
    ax.text(0.02, 0.915, subtitle, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=SUBTLE)


def _path_tag(ax, y, text, color):
    _box(ax, 0.025, y - 0.015, 0.105, 0.038, text, color=color, edge=color,
         fontsize=5.1, weight="bold", rounded=0.01)


def _new_page():
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _legend(ax):
    """Small visual key; keep it out of the diagrams themselves."""
    _box(ax, 0.05, 0.035, 0.040, 0.032, r"$x$", color=WHITE,
         edge=HAIRLINE, fontsize=5.6, rounded=0.007)
    _label(ax, 0.098, 0.051, "input", ha="left", size=5.6)
    _circle(ax, 0.225, 0.051, 0.014, r"$y_C$", color=OBSERVED_RV,
            observed=True, fontsize=5.3)
    _label(ax, 0.245, 0.051, "observed", ha="left", size=5.6)
    _circle(ax, 0.410, 0.051, 0.014, r"$z$", color=LATENT_RV,
            observed=False, fontsize=5.3)
    _label(ax, 0.430, 0.051, "unobserved / predicted", ha="left", size=5.6)


def page_paper_anp(pdf, preview_dir: Path | None = None):
    """Paper-faithful ANP structure: separate encoders on both paths."""
    fig, ax = _new_page()
    _title(
        ax,
        "Attentive Neural Process (paper architecture)",
        "Self-attention appears separately on the deterministic and latent paths. "
        "Targets attend to labelled contexts only on the deterministic path.",
    )

    # Two intentionally separate encoders, as in Figure 8 of the ANP paper.
    _path_tag(ax, 0.720, "Deterministic\npath", DETERMINISTIC)
    _box(ax, 0.155, 0.675, 0.070, 0.065, r"$x_C$", color=WHITE, edge=HAIRLINE)
    _circle(ax, 0.270, 0.708, 0.030, r"$y_C$", color=OBSERVED_RV, observed=True, fontsize=5.3)
    _box(ax, 0.320, 0.650, 0.130, 0.115, "pair MLP\n" r"$\{r_i\}$", color=DETERMINISTIC)
    _box(ax, 0.490, 0.650, 0.145, 0.115, "self-attention\non $r_i$", color=DETERMINISTIC)
    _box(ax, 0.680, 0.650, 0.160, 0.115, "cross-attention\n" r"$q(x_T),k(x_C),v(r_C)$",
         color=DETERMINISTIC, fontsize=5.7)
    _box(ax, 0.880, 0.650, 0.070, 0.115, r"$r_T$", color=DETERMINISTIC)
    for start, end in [((0.225, 0.708), (0.320, 0.708)), ((0.300, 0.708), (0.320, 0.708)),
                       ((0.450, 0.708), (0.490, 0.708)),
                       ((0.635, 0.708), (0.680, 0.708)), ((0.840, 0.708), (0.880, 0.708))]:
        _arrow(ax, start, end, color=DETERMINISTIC)
    _box(ax, 0.710, 0.815, 0.105, 0.048, r"$x_T$", color=WHITE, edge=DETERMINISTIC)
    _arrow(ax, (0.762, 0.815), (0.762, 0.765), color=DETERMINISTIC)

    _path_tag(ax, 0.420, "Latent\npath", LATENT)
    _box(ax, 0.155, 0.375, 0.070, 0.065, r"$x_C$", color=WHITE, edge=HAIRLINE)
    _circle(ax, 0.270, 0.408, 0.030, r"$y_C$", color=OBSERVED_RV, observed=True, fontsize=5.3)
    _box(ax, 0.320, 0.350, 0.130, 0.115, "pair MLP\n" r"$\{s_i\}$", color=LATENT)
    _box(ax, 0.490, 0.350, 0.145, 0.115, "self-attention\non $s_i$", color=LATENT)
    _box(ax, 0.680, 0.350, 0.115, 0.115, "mean\npool", color=LATENT)
    z = _circle(ax, 0.890, 0.408, 0.043, r"$z$", color=LATENT_RV, observed=False)
    for start, end in [((0.225, 0.408), (0.320, 0.408)), ((0.300, 0.408), (0.320, 0.408)),
                       ((0.450, 0.408), (0.490, 0.408)),
                       ((0.635, 0.408), (0.680, 0.408)), ((0.795, 0.408), (0.847, 0.408))]:
        _arrow(ax, start, end, color=LATENT)
    _label(ax, 0.890, 0.485, r"$q(z\mid C)$", color=LATENT)

    _box(ax, 0.450, 0.145, 0.235, 0.105,
         "decoder MLP\n" r"$[x_T,r_T,z] \mapsto \mathrm{logit}\ p(y_T{=}1)$",
         color=WHITE, edge=INK, fontsize=6.1, weight="bold")
    _circle(ax, 0.785, 0.197, 0.037, r"$y_T$", color=PREDICTED_RV, observed=False)
    _arrow(ax, (0.950, 0.650), (0.685, 0.230), color=DETERMINISTIC)
    _arrow(ax, (z[0] - z[2], z[1]), (0.685, 0.170), color=LATENT)
    _arrow(ax, (0.685, 0.197), (0.748, 0.197), color=PREDICTED_RV)
    _label(ax, 0.02, 0.100, r"At training, the latent encoder also forms $q(z\mid C\cup T)$.",
           ha="left", size=6.2)
    _legend(ax)

    pdf.savefig(fig)
    if preview_dir:
        fig.savefig(preview_dir / "np-design-page-1-paper-anp.png")
    plt.close(fig)


def page_our_anp(pdf, preview_dir: Path | None = None):
    """The exact shared-value ANP and its one-block ISAB ablation."""
    fig, ax = _new_page()
    _title(
        ax,
        "Our ANP and the clean Set-ANP ablation",
        "One shared labelled-context encoder feeds both paths.  Set-ANP changes only the one context-mixing block.",
    )

    _label(ax, 0.50, 0.855, "Shared encoder and mixer; yellow is the Set-ANP replacement.",
           color=SUBTLE, size=5.8)
    _box(ax, 0.055, 0.630, 0.090, 0.115, r"$x_C$", color=WHITE, edge=HAIRLINE)
    _circle(ax, 0.180, 0.687, 0.027, r"$y_C$", color=OBSERVED_RV, observed=True, fontsize=5.2)
    _box(ax, 0.245, 0.630, 0.135, 0.115, "shared\nencoder", color=SHARED)
    _box(ax, 0.425, 0.630, 0.150, 0.115, "MHSA (ANP)\nISAB (Set-ANP)", color=SET_YELLOW,
         edge=SET_YELLOW, fontsize=5.0, weight="bold")
    _box(ax, 0.620, 0.630, 0.105, 0.115, "context\n" r"$\tilde h_C$", color=SHARED)
    for start, end in [((0.145, 0.688), (0.153, 0.688)), ((0.207, 0.688), (0.245, 0.688)),
                       ((0.380, 0.688), (0.425, 0.688)), ((0.575, 0.688), (0.620, 0.688))]:
        _arrow(ax, start, end)

    _path_tag(ax, 0.410, "Deterministic\npath", DETERMINISTIC)
    _box(ax, 0.120, 0.315, 0.105, 0.095, r"$x_T$", color=WHITE, edge=DETERMINISTIC)
    _box(ax, 0.310, 0.295, 0.190, 0.135, "target-to-context\ncross-attention", color=DETERMINISTIC,
         fontsize=5.15)
    _box(ax, 0.550, 0.315, 0.080, 0.095, r"$r_T$", color=DETERMINISTIC)
    _arrow(ax, (0.225, 0.362), (0.310, 0.362), color=DETERMINISTIC)
    _arrow(ax, (0.500, 0.362), (0.550, 0.362), color=DETERMINISTIC)
    _arrow(ax, (0.672, 0.630), (0.485, 0.430), color=DETERMINISTIC)

    _path_tag(ax, 0.185, "Latent\npath", LATENT)
    _box(ax, 0.310, 0.105, 0.115, 0.085, "mean\npool", color=LATENT)
    _box(ax, 0.475, 0.105, 0.160, 0.085, "$p(z\\mid C)$\n$q(z\\mid C\\cup T)$", color=LATENT,
         fontsize=5.0)
    z = _circle(ax, 0.690, 0.147, 0.033, r"$z$", color=LATENT_RV, observed=False)
    _arrow(ax, (0.672, 0.630), (0.367, 0.190), color=LATENT)
    _arrow(ax, (0.425, 0.147), (0.475, 0.147), color=LATENT)
    _arrow(ax, (0.635, 0.147), (0.657, 0.147), color=LATENT)

    _box(ax, 0.800, 0.220, 0.105, 0.160, "pointwise\ndecoder\nBernoulli", color=WHITE,
         edge=INK, fontsize=5.0, weight="bold")
    y_t = _circle(ax, 0.955, 0.300, 0.034, r"$y_T$", color=PREDICTED_RV, observed=False)
    _arrow(ax, (0.630, 0.362), (0.800, 0.345), color=DETERMINISTIC)
    _arrow(ax, (z[0] + z[2], z[1]), (0.800, 0.255), color=LATENT)
    _arrow(ax, (0.905, 0.300), (y_t[0] - y_t[2], 0.300), color=PREDICTED_RV)
    _legend(ax)

    pdf.savefig(fig)
    if preview_dir:
        fig.savefig(preview_dir / "np-design-page-2-our-set-anp.png")
    plt.close(fig)


def page_aea_np(pdf, preview_dir: Path | None = None):
    """Full AEA-NP: the broader Set Transformer architecture."""
    fig, ax = _new_page()
    _title(
        ax,
        "AEA-NP: full Set-Transformer latent Neural Process",
        "This is not the one-block Set-ANP ablation.  It also changes latent pooling and lets target rows mix in the decoder.",
    )

    _box(ax, 0.050, 0.625, 0.085, 0.120, r"$x_C$", color=WHITE, edge=HAIRLINE)
    _circle(ax, 0.170, 0.685, 0.027, r"$y_C$", color=OBSERVED_RV, observed=True, fontsize=5.2)
    _box(ax, 0.235, 0.625, 0.115, 0.120, "item / label\nlinears", color=SHARED, fontsize=5.0)
    _box(ax, 0.395, 0.625, 0.105, 0.120, "pair linear\n" r"$d=32$", color=SHARED, fontsize=5.0)
    _box(ax, 0.545, 0.625, 0.155, 0.120, "ISAB encoder\n" r"$\times 2$, $m=16$", color=SET_YELLOW,
         edge=SET_YELLOW, fontsize=4.8, weight="bold")
    _box(ax, 0.745, 0.625, 0.110, 0.120, "context\n" r"$h_C$", color=SHARED)
    for start, end in [((0.135, 0.685), (0.143, 0.685)), ((0.197, 0.685), (0.235, 0.685)),
                       ((0.350, 0.685), (0.395, 0.685)), ((0.500, 0.685), (0.545, 0.685)),
                       ((0.700, 0.685), (0.745, 0.685))]:
        _arrow(ax, start, end)

    _path_tag(ax, 0.415, "Deterministic\npath", DETERMINISTIC)
    _box(ax, 0.110, 0.315, 0.110, 0.095, r"$x_T$", color=WHITE, edge=DETERMINISTIC)
    _box(ax, 0.305, 0.295, 0.190, 0.135, "target-to-context\ncross-attention", color=DETERMINISTIC,
         fontsize=5.15)
    _box(ax, 0.545, 0.315, 0.080, 0.095, r"$r_T$", color=DETERMINISTIC)
    _arrow(ax, (0.220, 0.362), (0.305, 0.362), color=DETERMINISTIC)
    _arrow(ax, (0.495, 0.362), (0.545, 0.362), color=DETERMINISTIC)
    _arrow(ax, (0.800, 0.625), (0.480, 0.430), color=DETERMINISTIC)

    _path_tag(ax, 0.185, "Latent\npath", LATENT)
    _box(ax, 0.305, 0.105, 0.125, 0.085, "learned\nMHA pool", color=LATENT)
    _box(ax, 0.475, 0.105, 0.160, 0.085, "$p(z\\mid C)$\n$q(z\\mid C\\cup T)$", color=LATENT,
         fontsize=5.0)
    z = _circle(ax, 0.690, 0.147, 0.033, r"$z$", color=LATENT_RV, observed=False)
    _arrow(ax, (0.800, 0.625), (0.367, 0.190), color=LATENT)
    _arrow(ax, (0.430, 0.147), (0.475, 0.147), color=LATENT)
    _arrow(ax, (0.635, 0.147), (0.657, 0.147), color=LATENT)

    _box(ax, 0.785, 0.220, 0.140, 0.160, "linear fuse\nISAB decoder\n" r"$\times 2$ · Bernoulli",
         color=SET_YELLOW, edge=SET_YELLOW, fontsize=4.55, weight="bold")
    y_t = _circle(ax, 0.970, 0.300, 0.027, r"$y_T$", color=PREDICTED_RV, observed=False)
    _arrow(ax, (0.625, 0.362), (0.785, 0.345), color=DETERMINISTIC)
    _arrow(ax, (z[0] + z[2], z[1]), (0.785, 0.255), color=LATENT)
    _arrow(ax, (0.925, 0.300), (y_t[0] - y_t[2], 0.300), color=PREDICTED_RV)
    _label(ax, 0.50, 0.500, "Target rows mix after fusion in the set-valued decoder.",
           color=SUBTLE, size=5.8)
    _legend(ax)

    pdf.savefig(fig)
    if preview_dir:
        fig.savefig(preview_dir / "np-design-page-3-aea-np.png")
    plt.close(fig)


def render(output: Path = OUT, preview_dir: Path | None = None) -> None:
    """Render three architecture pages and append the existing complexity page."""
    if not COMPLEXITY.exists():
        raise FileNotFoundError(f"missing complexity figure: {COMPLEXITY}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(_TEMP) as pdf:
        page_paper_anp(pdf, preview_dir)
        page_our_anp(pdf, preview_dir)
        page_aea_np(pdf, preview_dir)

    writer = PdfWriter()
    writer.append(PdfReader(_TEMP))
    writer.append(PdfReader(COMPLEXITY))
    writer.add_metadata({
        "/Title": "NP architecture designs and attention complexity",
        "/Author": "AIMS-NeurIPS",
        "/Subject": "ANP, Set-ANP, AEA-NP, and context-attention complexity",
    })
    with output.open("wb") as handle:
        writer.write(handle)
    _TEMP.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--preview-dir", type=Path,
        help="Optional directory for page PNGs used for visual inspection.",
    )
    args = parser.parse_args()
    render(args.output, args.preview_dir)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
