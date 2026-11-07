"""Shared publication style for every figure in this repo.

Import this before creating figures; it sets rcParams globally and exposes the
palette plus two helpers. Palettes are Paul Tol's qualitative schemes, which are
colourblind-safe and stay distinguishable in greyscale print.

    from plotstyle import MUTED, BRIGHT, style, header, TEXTWIDTH

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.0))
    ax.bar(x, y, color=MUTED["indigo"])
    style(ax)

Figure widths are NeurIPS column measurements in inches: use TEXTWIDTH for a
full-width figure and COLWIDTH for a half-width one, and never rescale the
result in LaTeX (\\includegraphics without a width keeps the font sizes here
matching the body text).
"""

from __future__ import annotations

import matplotlib as mpl

# Paul Tol qualitative schemes. https://sronpersonalpages.nl/~pault/
# Ordered so that taking the first n colours stays distinguishable.
BRIGHT = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

MUTED = {
    "indigo": "#332288",
    "cyan": "#88CCEE",
    "teal": "#44AA99",
    "green": "#117733",
    "olive": "#999933",
    "sand": "#DDCC77",
    "rose": "#CC6677",
    "wine": "#882255",
    "purple": "#AA4499",
    "grey": "#DDDDDD",
}

# Three-colour high-contrast set for figures that must survive fax-quality print.
HIGH_CONTRAST = {"blue": "#004488", "yellow": "#DDAA33", "red": "#BB5566"}

INK = "#000000"
SUBTLE = "#555555"
HAIRLINE = "#CCCCCC"

# Default cycle: order chosen so adjacent series differ in both hue and lightness.
CYCLE = [MUTED["indigo"], MUTED["rose"], MUTED["teal"], MUTED["sand"],
         MUTED["wine"], MUTED["cyan"], MUTED["olive"], MUTED["purple"]]

# Sequential maps that are perceptually uniform and colourblind-safe.
SEQUENTIAL = "viridis"
DIVERGING = "BrBG"

TEXTWIDTH = 5.5  # NeurIPS \textwidth, inches
COLWIDTH = 2.65  # half of the above, minus the inter-column gap

RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.2,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType so the PDF is editable and searchable
    "ps.fonttype": 42,
}

mpl.rcParams.update(RC)


PANEL_THEMES = {
    "white": {"facecolor": "white", "grid_color": HAIRLINE, "grid_alpha": .35, "grid_width": .4},
    "light-grey": {"facecolor": "#F0F0F0", "grid_color": "white", "grid_alpha": 1., "grid_width": .7},
}


def style(ax, grid_axis: str = "y", panel_theme: str = "white") -> None:
    """Drop the top/right frame and lay a hairline grid behind the data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    theme = PANEL_THEMES[panel_theme]
    ax.set_facecolor(theme["facecolor"])
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, alpha=theme["grid_alpha"],
                lw=theme["grid_width"], color=theme["grid_color"])
    ax.set_axisbelow(True)


def header(fig, title: str, subtitle: str = "") -> None:
    """Left-aligned title block for multi-panel overview pages.

    Only for the standalone PDF reports -- a figure destined for the paper gets
    its title from the LaTeX caption instead.
    """
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.02, ha="left", y=0.995)
    if subtitle:
        fig.text(0.02, 0.962, subtitle, fontsize=7.5, color=SUBTLE, va="top")


def errorbar_kw(**overrides):
    """Thin capped error bars; heavy defaults otherwise swamp small figures."""
    kw = dict(elinewidth=0.7, capsize=1.6, capthick=0.7, ecolor=INK)
    kw.update(overrides)
    return kw
