# Plotting style

**Every figure in this repo goes through `organized/utils/plotstyle.py`.** It is the
single source of truth for fonts, palette, figure widths and axis furniture. Do
not set `rcParams`, pick hex colours, or hand-roll a `style_axis()` helper in an
individual plot script — if something is missing from `plotstyle`, add it there.

Worked example, rendering every rule below on synthetic data:
`docs/generate_neurips_style_samples.py` → `docs/neurips-sample-plots.pdf`.
Read that file before writing a new plot script.

## Usage

`plotstyle` sets `rcParams` on import, so import it before creating figures.
It is not on the path by default; plot scripts under `organized/utils/` can
import it directly, anything else inserts the directory first:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "organized" / "utils"))

import matplotlib.pyplot as plt
from plotstyle import MUTED, INK, SUBTLE, TEXTWIDTH, style, errorbar_kw

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.0))
ax.bar(x, y, 0.38, yerr=e, error_kw=errorbar_kw(), color=MUTED["indigo"],
       edgecolor=INK, lw=0.4)
ax.set_ylim(0, 1)
style(ax)
fig.tight_layout(pad=0.3)
```

### What it exports

| name | what it is |
|---|---|
| `MUTED`, `BRIGHT`, `HIGH_CONTRAST` | Paul Tol qualitative palettes, as `{name: hex}` |
| `CYCLE` | default colour cycle, already installed into `rcParams` |
| `SEQUENTIAL`, `DIVERGING` | `"viridis"`, `"BrBG"` — for continuous fields |
| `INK`, `SUBTLE`, `HAIRLINE` | black, grey annotation text, grid/neutral grey |
| `TEXTWIDTH`, `COLWIDTH` | 5.5in and 2.65in — NeurIPS full and half column |
| `style(ax, grid_axis=...)` | drops top/right spines, lays a hairline grid behind the data |
| `header(fig, title, subtitle)` | left-aligned title block, **standalone PDF reports only** |
| `errorbar_kw(**overrides)` | thin capped error bars |

`grid_axis` takes `"y"` (default), `"x"`, `"both"`, or `"none"` for image panels.

## Rules

Colour:

- **Use the Tol palettes, nothing else.** They are colourblind-safe *and* stay
  distinguishable in greyscale print. `MUTED` is the default; `HIGH_CONTRAST`
  is for figures that must survive bad printing.
- Pick colours by meaning and keep them stable across figures — if subject
  holdout is `MUTED["indigo"]` in one plot, it is indigo everywhere.
- Never encode a variable by colour alone. Pair it with a linestyle or marker,
  as the training-curve page does (`-`, `--`, `-.` for 1PL/2PL/3PL).
- Continuous fields use `SEQUENTIAL`; use `DIVERGING` only for signed
  quantities with a meaningful zero.

Axes:

- **Never truncate a bounded axis.** AUC, Brier, ECE and probabilities start at
  0; for AUC draw the chance line at 0.5. A zoomed y-axis that makes a 0.02
  difference look decisive is the single thing reviewers flag hardest.
- `style(ax)` on every axis, immediately after creating it.
- Error bars go through `errorbar_kw()`. Matplotlib's defaults are heavy enough
  to swamp a 2-inch-wide figure.
- Grouped bars at `width=0.38`; a narrower bar leaves the panel looking empty.

Typography and size:

- Figures are sized in real column inches (`TEXTWIDTH` / `COLWIDTH`) and
  included in LaTeX with **no** `width=` — rescaling breaks the match between
  figure text and body text. Body is 8pt serif, ticks 7pt.
- Titles via `ax.set_title(..., loc="left", fontweight="bold")`, panels labelled
  `(a)`, `(b)`. A figure headed for the paper gets its caption from LaTeX, not
  from `header()`.
- `pdf.fonttype=42` is set, so text stays selectable and searchable.

Honesty — these are real mistakes caught in review, not hypotheticals:

- **Subsample evenly, never a contiguous block.** The before/after filtering
  panel originally sliced `mat[:, :90]`, the hardest items, which faked a
  pass-rate jump that filtering does not cause. Use `mat[:, ::2]`.
- Show the denominator. Panels that drop data annotate what fraction survived.
- If two implementations of a metric disagree, plot both rather than quietly
  picking one.

Layout traps that cost a re-render each:

- `fig.colorbar(im, ax=ax)` steals width from that panel, so the last column
  ends up narrower than its neighbours. Use an inset cax instead:
  `fig.colorbar(im, cax=ax.inset_axes([1.07, 0, 0.08, 1.0]))`.
- A figure-level legend needs its strip reserved first, or it lands on the
  x labels: `fig.tight_layout(rect=(0, 0.12, 1, 1))`, then
  `fig.legend(..., loc="lower center", bbox_to_anchor=(0.5, 0.0))`.
- Check legend placement against the data — `loc="lower right"` sat directly on
  the chance line in the acquisition panel.

## Checking a figure

There is no PDF rasteriser on this machine (`pdftoppm` is absent), so **look at
a PNG before claiming a figure is done**. Call the page function with a shim
that saves a PNG instead of writing to the `PdfPages`, and actually read the
image back. Several of the defects above only showed up in the render.

Text extraction from the PDFs under `literature/` works via `pypdf` in `.venv`.

## Migration status

`plotstyle.py` is new. As of 2026-10-26 the only consumer is
`docs/generate_neurips_style_samples.py`; **all 21 `organized/utils/plot_*.py`
scripts still carry their own private styling** and have not been ported. When
you touch one of them, port it: delete its local palette and `style_axis()`,
import `plotstyle`, and re-render to confirm nothing truncated or collided.
