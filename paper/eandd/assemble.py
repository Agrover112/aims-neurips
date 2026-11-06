#!/usr/bin/env python3
"""Prepare the inputs of paper/neurips_2026/eandd.tex from the committed results.

    .venv/bin/python paper/eandd/assemble.py

Copies Person-B's figures, builds Table 1 from the summary CSVs (nothing is typed by hand), and splits
person-b-sections.tex into one file per section so eandd.tex can place each where it belongs.
"""
from __future__ import annotations
import re, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "paper/eandd/data", ROOT / "paper/neurips_2026"
FIG = OUT / "figures" / "eandd"
FIG.mkdir(parents=True, exist_ok=True)
for f in DATA.glob("figure*.pdf"):
    shutil.copy(f, FIG / f.name)


def cell(d, m):
    r = d[d.model == m].iloc[0]
    if m == "jeffreys":
        return "reference"
    value = f"${r['diff']:+.4f}$"
    interval = f"[${r.lo:+.4f}$, ${r.hi:+.4f}$]"
    return (r"\shortstack[r]{" + value + r" \\{}" + interval + "}").replace(
        "$-", "$\\minus").replace("+0.0000", "0.0000")


tests = pd.read_csv(DATA / "wp5-competition-test.csv")
vals = pd.read_csv(DATA / "wp5-competition-val.csv")
ids = pd.read_csv(DATA / "wp5-competition-id.csv")
idc = pd.read_csv(DATA / "wp5b-competition-id-clean.csv")
rows = [("Fixed in advance", [("jeffreys", "Beta-Binomial (Jeffreys)"), ("rasch-fixed", "Rasch"), ("kernel-fixed", "Kernel Rasch")]),
        ("Tuned on validation", [("bb-tuned", "Beta-Binomial"), ("rasch-retuned", "Rasch"), ("kernel-old", "Kernel Rasch")]),
        ("Fitted on training", [("bb-eb", "Beta-Binomial"), ("rasch-fitted", "Rasch"), ("kernel-fitted", "Kernel Rasch")])]
lines = []
for gi, (src, ms) in enumerate(rows):
    for mi, (m, name) in enumerate(ms):
        idd = idc[idc.model == "bb-eb-clean"].assign(model="bb-eb") if m == "bb-eb" else ids
        if m == "bb-eb":
            id_cell = cell(pd.concat([idd]), "bb-eb")
        else:
            id_cell = cell(ids, m)
        first = f"\\multirow{{3}}{{*}}{{{src}}}" if mi == 0 else ""
        lines.append(f"{first} & {name} & {cell(tests, m)} & {cell(vals, m)} & {id_cell} \\\\")
    if gi < 2:
        lines.append("\\midrule")
(OUT / "eandd_table1.tex").write_text("\n".join(lines) + "\n")

b = (ROOT / "paper/eandd/person-b-sections.tex").read_text()
parts = re.split(r"(?=\\section\{)", b)
names = {"Setting and online simulation": "setting", "Item-difficulty transfer": "itemdiff", "Limitations": "limits",
         "Disclosure and rerun instructions": "disclosure"}
for p in parts:
    m = re.match(r"\\section\{(.*?)\}", p)
    if m and m.group(1) in names:
        (OUT / f"eandd_b_{names[m.group(1)]}.tex").write_text(re.sub(r"^\\section\{.*?\}\s*", "", p, count=1))
print("wrote", sorted(x.name for x in OUT.glob("eandd_*.tex")))


# ---- appendix tables, all generated from the committed CSVs ----
MINUS = "$\\minus"


def one(x):
    return f"${x:+.4f}$".replace("$-", MINUS)


def fmt(x, lo=None, hi=None, stack=False):
    if lo is None:
        return one(x)
    interval = f"[{one(lo)}, {one(hi)}]"
    return r"\shortstack[r]{" + one(x) + r" \\{}" + interval + "}" if stack else f"{one(x)} {interval}"


lad = {"fl-rate-only": "Benchmark rate only", "fl-plus-ability": "+ subject ability", "fl-plus-item": "+ item difficulty (Rasch)",
       "fl-kernel": "+ kernel", "fl-no-ability": "Rate and item, no ability", "fl-no-rate": "Ability and item, no rate"}
t = pd.read_csv(DATA / "wp11-fitted-ladder-competition-test.csv").set_index("model")
v = pd.read_csv(DATA / "wp11-fitted-ladder-competition-val.csv").set_index("model")
mt = pd.read_csv(DATA / "wp11-fitted-ladder-multi-pair-test.csv").set_index("model")
(OUT / "eandd_ladder.tex").write_text("\n".join(
    f"{n} & {fmt(t.loc[m,'diff'], t.loc[m,'lo'], t.loc[m,'hi'], stack=True)} & {fmt(v.loc[m,'diff'], v.loc[m,'lo'], v.loc[m,'hi'], stack=True)} & {fmt(mt.loc[m,'diff'], mt.loc[m,'lo'], mt.loc[m,'hi'], stack=True)} \\\\"
    for m, n in lad.items()) + "\n")

pp = pd.read_csv(DATA / "wp12-paired-prior-sources.csv")
nm = {"rasch-fitted": "Rasch fitted", "rasch-fixed": "Rasch fixed", "kernel-fitted": "Kernel fitted", "bb-eb": "Beta-Binomial fitted"}
pairs = [("rasch-fitted", "rasch-fixed"), ("kernel-fitted", "rasch-fitted"), ("rasch-fitted", "bb-eb")]
lines = []
for a, b in pairs:
    cells = []
    for role in ("test", "validation", "id"):
        r = pp[(pp.role == role) & (pp.model_a == a) & (pp.model_b == b)]
        cells.append(fmt(r.a_minus_b.iloc[0], r.lo.iloc[0], r.hi.iloc[0], stack=True) if len(r) else "--")
    lines.append(f"{nm[a]} $-$ {nm[b]} & " + " & ".join(cells) + " \\\\")
(OUT / "eandd_paired.tex").write_text("\n".join(lines) + "\n")

rb = pd.read_csv(DATA / "wp12-draw-seed-robustness.csv")
lines = []
for m, n in [("bb-eb", "Beta-Binomial fitted"), ("rasch-fitted", "Rasch fitted"), ("bb-tuned", "Beta-Binomial tuned")]:
    cells = []
    for role in ("test", "validation"):
        x = rb[(rb.role == role) & (rb.model == m)].sort_values("seed")["diff"]
        cells.append(" / ".join(one(v_) for v_ in x))
    lines.append(f"{n} & {cells[0]} & {cells[1]} \\\\")
(OUT / "eandd_robust.tex").write_text("\n".join(lines) + "\n")

lo = pd.read_csv(DATA / "wp10-lobo-explained.csv").sort_values("bb_eb_diff")
(OUT / "eandd_lobo.tex").write_text("\n".join(
    f"\\code{{{r.held_out.replace('_', chr(92) + '_')}}} & {int(r.tasks)} & {fmt(r.bb_eb_diff, r.lo, r.hi)} & {r.mean_abs_acc_minus_0_52:.2f} \\\\"
    for r in lo.rename(columns={"mean_abs_acc_minus_0.52": "mean_abs_acc_minus_0_52"}).itertuples()) + "\n")
print("appendix tables written")


# ---- wrap each generated body in a complete tabular (avoids \input inside a tabular) ----
def wrap(name, spec, header):
    f = OUT / name
    body = f.read_text().rstrip("\n")
    f.write_text("\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


wrap("eandd_table1.tex", "llccc", "Prior source & Model & Test (OOD) & Validation (OOD) & ID")
wrap("eandd_ladder.tex", "lccc", "Model & Test & Validation & Multi-pair test")
wrap("eandd_paired.tex", "lccc", "Comparison (a $-$ b) & Test & Validation & ID")
wrap("eandd_robust.tex", "lcc", "Model & Test & Validation")
wrap("eandd_lobo.tex", "lccc", "Held-out benchmark & Tasks & Gain [95\\% CI] & Distance from $0.52$")

split = pd.read_csv(DATA / "wp13-split-sensitivity.csv")
d = split.refit_macro_delta
orig = d.loc[split.original_test_triple].item()
p05, p95 = d.quantile([.05, .95])
(OUT / "eandd_split_sensitivity.tex").write_text(
    f"We held out each of the {len(split)} triples, refitted the benchmark-weighted empirical Beta prior on the other "
    f"14 benchmarks, and rescored the frozen trajectories. The mean macro difference was {one(d.mean())}, "
    f"with 5th--95th percentiles [{one(p05)}, {one(p95)}]. The fitted prior hurt on {100*(d > 0).mean():.1f}\\% of splits. "
    f"The original test benchmark names gave {one(orig)} (percentile {100*(d <= orig).mean():.0f}) under refitting. "
    "This refit includes the former validation benchmarks among the 14 eligible fit benchmarks for the original "
    "test triple, so its result differs from the original 11-benchmark fit. The 11 original training benchmarks "
    "are scored on held-out-subject tasks from the existing leave-one-out panel. These overlapping triples describe "
    "split sensitivity within this dataset.\n"
)

metric = pd.read_csv(DATA / "wp13-metric-alc.csv")
metric = metric.loc[metric.weighting.eq("official")].set_index(["run", "model"])
rank = pd.read_csv(DATA / "wp13-metrics.csv").query("budget == 31").set_index(["run", "model"])
names = {"jeffreys": "Jeffreys", "bb-eb": "Fitted Beta-Binomial", "bb-eb-clean": "Fitted Beta-Binomial",
         "rasch-fitted": "Fitted Rasch",
         "kernel-fitted": "Fitted kernel Rasch", "offline-kept": "Offline Rasch"}
lines = []
for run, role in [("wp5-competition-test", "Test"), ("r2-competition-val", "Validation"),
                  ("r2-competition-id", "ID")]:
    for model, label in names.items():
        key = (run, model)
        if key not in metric.index:
            continue
        r = metric.loc[key]
        auc = rank.loc[key, "auc"]
        auc_text = f"{auc:.3f}" if pd.notna(auc) else "--"
        lines.append(f"{role} & {label} & {r.brier:.4f} & {r.log_loss:.4f} & {r.ece10:.4f} & {auc_text} \\\\")
(OUT / "eandd_reviewer_metrics.tex").write_text("\n".join(lines) + "\n")
wrap("eandd_reviewer_metrics.tex", "llrrrr", "Split & Model & Brier & Log loss & ECE & AUC at 31")
