#!/usr/bin/env python3
"""Prepare the inputs of paper/neurips_2026/eandd.tex from the committed results.

    .venv/bin/python paper/eandd/assemble.py

Copies Person-B's figures, builds Table 1 from the summary CSVs (nothing is typed by hand), and splits
person-b-sections.tex into one file per section so eandd.tex can place each where it belongs.
"""
from __future__ import annotations
import json, re, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "paper/eandd/data", ROOT / "paper/neurips_2026"
FIG = OUT / "figures" / "eandd"
FIG.mkdir(parents=True, exist_ok=True)
for f in DATA.glob("figure*.pdf"):
    shutil.copy(f, FIG / f.name)


GREY = ""


def cell(d, m, bold=False, second=False):
    r = d[d.model == m].iloc[0]
    if m == "jeffreys":
        return "0 (reference)"
    value = f"${r['diff']:+.4f}$"
    if bold:
        value = f"$\\mathbf{{{r['diff']:+.4f}}}$".replace("{-", "{\\minus")
    if second:
        value = r"\underline{" + value + "}"
    interval = f"[${r.lo:+.4f}$, ${r.hi:+.4f}$]"
    return (value + r" {\scriptsize " + interval + "}").replace("$-", "$\\minus").replace("+0.0000", "0.0000")


tests = pd.read_csv(DATA / "wp5-competition-test.csv")
vals = pd.read_csv(DATA / "wp5-competition-val.csv")
ids = pd.read_csv(DATA / "wp5-competition-id.csv")
idc = pd.read_csv(DATA / "wp5b-competition-id-clean.csv")
rows = [("Fixed in advance", [("jeffreys", "Beta-Binomial (Jeffreys)"), ("rasch-fixed", "Rasch"), ("kernel-fixed", "Kernel Rasch")]),
        ("Tuned on validation", [("bb-tuned", "Beta-Binomial"), ("rasch-retuned", "Rasch"), ("kernel-old", "Kernel Rasch")]),
        ("Fitted on training", [("bb-eb", "Beta-Binomial"), ("rasch-fitted", "Rasch"), ("kernel-fitted", "Kernel Rasch")])]
lines = []
ids_all = pd.concat([ids[ids.model != "bb-eb"], idc[idc.model == "bb-eb-clean"].assign(model="bb-eb")])
allm = [m for _, ms in rows for m, _ in ms if m != "jeffreys"]
rank_ = {k: sorted(allm, key=lambda m: d[d.model == m].iloc[0]["diff"]) for k, d in (("id", ids_all), ("val", vals), ("test", tests))}
best = {k: v_[0] for k, v_ in rank_.items()}
second = {k: v_[1] for k, v_ in rank_.items()}
for gi, (src, ms) in enumerate(rows):
    for mi, (m, name) in enumerate(ms):
        id_cell = cell(ids_all, m, bold=(m == best["id"]), second=(m == second["id"]))
        if mi == 0:
            lines.append(("\\addlinespace\n" if gi else "") + f"\\multicolumn{{4}}{{l}}{{\\textit{{{src}}}}} \\\\")
        lines.append(f"\\hspace{{1em}}{name} & {id_cell} & {cell(vals, m, bold=(m == best['val']), second=(m == second['val']))} & {cell(tests, m, bold=(m == best['test']), second=(m == second['test']))} \\\\")
(OUT / "eandd_table1.tex").write_text("\n".join(lines) + "\n")

# Table 2: generate from the confirmed CSVs, never from a provisional refit.
probe_old = pd.read_csv(DATA / "wp7-summary.csv").set_index(["setting", "role", "feature"])
probe_new = pd.read_csv(DATA / "wp13-probe-summary.csv").set_index(["role", "arm"])
probe_rows = []
for label, old_feature, new_arm in [
    ("Random score", "random", "random"),
    ("Log character count", "length", None),
    ("Ridge on embeddings", "embedding", None),
    ("Pass-rate ridge", None, "pass_rate_ridge"),
    ("Rasch-target ridge", None, "rasch_easiness_ridge"),
    ("Pass-rate MLP", None, "pass_rate_mlp"),
    ("Rasch-target MLP", None, "rasch_easiness_mlp"),
]:
    cells = []
    for role in ("train", "validation", "test"):
        if role == "train" and old_feature is None:
            cells.extend(["--", "--"])
            continue
        if role != "train" and new_arm is not None:
            result = probe_new.loc[(role, new_arm)]
        else:
            result = probe_old.loc[("ID" if role == "train" else "OOD", role, old_feature)]
        cells.extend(f"${result[metric]:.3f}$" for metric in ("rho", "pairwise_acc"))
    probe_rows.append(label + " & " + " & ".join(cells) + r" \\")
(OUT / "eandd_probe_table.tex").write_text(
    "\\begin{tabular}{lcccccc}\n\\toprule\n"
    " & \\multicolumn{2}{c}{ID} & \\multicolumn{2}{c}{OOD validation} & \\multicolumn{2}{c}{OOD test}\\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    "Probe & Spearman & Pairwise & Spearman & Pairwise & Spearman & Pairwise\\\\\n"
    "\\midrule\n" + "\n".join(probe_rows) + "\n\\bottomrule\n\\end{tabular}\n")

b = (ROOT / "paper/eandd/person-b-sections.tex").read_text()
parts = re.split(r"(?=\\section\{)", b)
names = {"Data and evaluation protocol": "setting", "Probes": "probes", "Item features": "itemfeat", "Limitations": "limits",
         "Disclosure": "disclosure"}
for p in parts:
    m = re.match(r"\\section\{(.*?)\}", p)
    if m and m.group(1) in names:
        (OUT / f"eandd_b_{names[m.group(1)]}.tex").write_text(re.sub(r"^\\section\{.*?\}\s*", "", p, count=1))
print("wrote", sorted(x.name for x in OUT.glob("eandd_*.tex")))


# ---- appendix tables, all generated from the committed CSVs ----
MINUS = "$\\minus"


def one(x):
    return f"${x:+.4f}$".replace("$-", MINUS)


def onebf(x):
    return f"$\\mathbf{{{x:+.4f}}}$".replace("{-", "{\\minus")


def fmt(x, lo=None, hi=None, stack=False, bold=False):
    val = onebf(x) if bold else one(x)
    if lo is None:
        return val
    interval = f"[{one(lo)}, {one(hi)}]"
    pre = GREY if bold else ""
    return pre + r"\begin{tabular}[c]{@{}r@{}}" + val + r" \\{}" + interval + r"\end{tabular}" if stack else f"{pre}{val} {interval}"


lad = {"fl-rate-only": "Benchmark rate $\\mu_d$ only", "fl-plus-ability": "+ subject ability $\\theta_i$", "fl-plus-item": "+ item deviation $\\delta_j$ (Rasch)",
       "fl-kernel": "+ kernel", "fl-no-ability": "$\\mu_d$ and $\\delta_j$, no $\\theta_i$", "fl-no-rate": "$\\theta_i$ and $\\delta_j$, no $\\mu_d$"}
t = pd.read_csv(DATA / "wp11-fitted-ladder-competition-test.csv").set_index("model")
v = pd.read_csv(DATA / "wp11-fitted-ladder-competition-val.csv").set_index("model")
bv, bt = min(lad, key=lambda m: v.loc[m, "diff"]), min(lad, key=lambda m: t.loc[m, "diff"])
ladder_lines = []
for m, n in lad.items():
    if m == "fl-no-ability":
        ladder_lines.append("\\midrule")
    ladder_lines.append(f"{n} & {fmt(v.loc[m,'diff'], v.loc[m,'lo'], v.loc[m,'hi'], stack=True, bold=(m == bv))} & {fmt(t.loc[m,'diff'], t.loc[m,'lo'], t.loc[m,'hi'], stack=True, bold=(m == bt))} \\\\")
(OUT / "eandd_ladder.tex").write_text("\n".join(ladder_lines) + "\n")

pp = pd.read_csv(DATA / "wp12-paired-prior-sources.csv")
nm = {"rasch-fitted": "R\\,fit", "rasch-fixed": "R\\,fix", "kernel-fitted": "KR\\,fit", "bb-eb": "BB\\,fit"}
pairs = [("rasch-fitted", "rasch-fixed"), ("kernel-fitted", "rasch-fitted"), ("rasch-fitted", "bb-eb")]
lines = []
for a, b in pairs:
    cells = []
    for role in ("id", "validation", "test"):
        r = pp[(pp.role == role) & (pp.model_a == a) & (pp.model_b == b)]
        cells.append(fmt(r.a_minus_b.iloc[0], r.lo.iloc[0], r.hi.iloc[0], stack=True) if len(r) else "--")
    lines.append(f"$\\Delta_{{\\mathrm{{{nm[a]}}}-\\mathrm{{{nm[b]}}}}}$ & " + " & ".join(cells) + " \\\\")
(OUT / "eandd_paired.tex").write_text("\n".join(lines) + "\n")

rb = pd.read_csv(DATA / "wp12-draw-seed-robustness.csv")
lines = []
for m, n in [("bb-eb", "BB\\,fit"), ("rasch-fitted", "R\\,fit"), ("bb-tuned", "BB\\,tune")]:
    cells = []
    for role in ("validation", "test"):
        x = rb[(rb.role == role) & (rb.model == m)].sort_values("seed")["diff"]
        cells.append(" / ".join(one(v_) for v_ in x))
    lines.append(f"$\\mathrm{{{n}}}$ & {cells[0]} & {cells[1]} \\\\")
(OUT / "eandd_robust.tex").write_text("\n".join(lines) + "\n")

from scipy.stats import spearmanr
ex = pd.read_csv(DATA / "wp13-benchmark-explanation.csv").sort_values("delta")
DIST = "mean_abs_query_acc_minus_0.52"
VAL, TEST = {"ai2d_test", "bird_sql", "financebench"}, {"algotune", "cmmlu", "editbench"}
role = lambda b: "OOD validation" if b in VAL else ("OOD test" if b in TEST else "Training")
CITE = {d["benchmark_id"]: d["bibtex_keys"] for d in json.load(open(ROOT / "paper/eandd/benchmark-citation-map.json"))}
lobo_lines = [f"\\code{{{r.benchmark.replace('_', chr(92) + '_')}}}~\\citep{{{','.join(CITE[r.benchmark])}}} & {role(r.benchmark)} & {int(r.tasks)} & {fmt(r.delta, r.ci_low, r.ci_high)} & {getattr(r, 'dist'):.2f} \\\\"
              for r in ex.rename(columns={DIST: "dist"}).itertuples()]
lobo_lines.append("\\midrule")
tr = ex[ex.benchmark.map(role) == "Training"]
lobo_lines.append(f"Mean over the 17 benchmarks & & {int(ex.tasks.sum())} & {one(ex.delta.mean())} & {ex[DIST].mean():.2f} \\\\")
lobo_lines.append(f"Mean over the 11 training benchmarks & & {int(tr.tasks.sum())} & {one(tr.delta.mean())} & {tr[DIST].mean():.2f} \\\\")
lobo_lines.append("\\midrule")
lobo_lines.append(f"\\multicolumn{{5}}{{l}}{{Spearman $\\rho$ between gain and distance: ${spearmanr(ex.delta, ex[DIST])[0]:.2f}$ over the 17 benchmarks, ${spearmanr(tr.delta, tr[DIST])[0]:.2f}$ over the 11 training benchmarks.}} \\\\")
(OUT / "eandd_lobo.tex").write_text("\n".join(lobo_lines) + "\n")
print("appendix tables written")


# ---- wrap each generated body in a complete tabular (avoids \input inside a tabular) ----
def wrap(name, spec, header):
    f = OUT / name
    body = f.read_text().rstrip("\n")
    f.write_text("\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


wrap("eandd_table1.tex", "lccc", "Model $m$ & ID & OOD validation & OOD test")
wrap("eandd_ladder.tex", "lcc", "Model $m$ & OOD validation & OOD test")
wrap("eandd_paired.tex", "lccc", "Comparison & ID & OOD validation & OOD test")
wrap("eandd_robust.tex", "lcc", "Model $m$ & OOD validation & OOD test")
wrap("eandd_lobo.tex", "llccc", "Benchmark & Split & Tasks & $\\Delta_{\\mathrm{BB\\,fit}-\\mathrm{J}}$ [95\\% CI] & Mean distance from $0.52$")


# ---- R5: probe difficulty as the Rasch item prior mean (both CSVs committed) ----
r5 = pd.concat([pd.read_csv(DATA / "r5-prior-mean-v1-summary.csv"), pd.read_csv(DATA / "r5-prior-mean-signcheck-summary.csv")])
r5 = r5.drop_duplicates(["role", "model"]).set_index(["role", "model"])["diff"]
lines = []
gs = (0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5)
colkeys = [(arm, role) for arm in ("pass_rate_mlp", "rasch_difficulty_mlp") for role in ("validation", "test")]
bestg = {ck: min(gs, key=lambda g: r5[(ck[1], "rasch-fitted" if g == 0 else f"pm-{ck[0]}-{g}")]) for ck in colkeys}
for g in gs:
    cells = []
    for ck in colkeys:
        key = "rasch-fitted" if g == 0 else f"pm-{ck[0]}-{g}"
        cells.append(GREY + onebf(r5[(ck[1], key)]) if g == bestg[ck] else one(r5[(ck[1], key)]))
    lines.append(f"${g:g}$ & " + " & ".join(cells) + " \\\\")
    if g == 0:
        lines.append("\\midrule")
(OUT / "eandd_r5.tex").write_text("\n".join(lines) + "\n")
f = OUT / "eandd_r5.tex"
f.write_text("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Pass-rate MLP} & \\multicolumn{2}{c}{Rasch-target MLP} \\\\\n"
             "Scale $\\gamma$ & OOD validation & OOD test & OOD validation & OOD test \\\\\n\\midrule\n" + f.read_text() + "\\bottomrule\n\\end{tabular}\n")


# ---- adaptive priors (R3): worst benchmark, macro mean over 17 benchmarks and test gain, from the committed CSV ----
ad = pd.read_csv(DATA / "r3-adaptive-17-benchmarks.csv").set_index("model")
ad_rows = [("Fitted Beta-Binomial (reference)", "bb-eb"), ("Beta mixture, 2 or 3 components", "mix2"),
           ("Average over strength", "bma-s"), ("Average over mean and strength", "bma-ms"),
           ("Switch to Jeffreys, 3 labels", "switch-3"), ("Switch to Jeffreys, 7 labels", "switch-7")]
b_worst = min(ad.loc[m, "worst"] for _, m in ad_rows[1:]); b_macro = min(ad.loc[m, "macro"] for _, m in ad_rows[1:]); b_test = min(ad.loc[m, "test_gain"] for _, m in ad_rows[1:])
def adcell(x, best):
    return onebf(x) if abs(x - best) < 1e-12 else one(x)
ad_lines = []
for name, m in ad_rows:
    ad_lines.append(f"{name} & {adcell(ad.loc[m,'worst'], b_worst)} & {adcell(ad.loc[m,'macro'], b_macro)} & {adcell(ad.loc[m,'test_gain'], b_test)} \\\\")
    if m == "bb-eb":
        ad_lines.append("\\midrule")
(OUT / "eandd_adaptive.tex").write_text("\\begin{tabular}{lccc}\n\\toprule\nModel $m$ & Worst benchmark & Mean over 17 benchmarks & OOD test \\\\\n\\midrule\n" + "\n".join(ad_lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---- appendix: adaptive-prior details (R3, R4) and the Rasch split distribution (R6) ----
pb = pd.read_csv(DATA / "r3-adaptive-per-benchmark.csv")
pb = pb[~pb.model.isin(["jeffreys", "mix3"])]
cols = [("bb-eb", "Fitted BB"), ("mix2", "Mixture"), ("bma-s", "Avg.\\ strength"), ("bma-ms", "Avg.\\ mean, strength"), ("switch-3", "Switch, 3"), ("switch-7", "Switch, 7")]
piv = pb.pivot_table(index=["benchmark", "source"], columns="model", values="gain").reset_index()
order = {"lobo": 0, "validation": 1, "test": 2}; label = {"lobo": "Training", "validation": "OOD validation", "test": "OOD test"}
piv["o"] = piv.source.map(order)
piv = piv.sort_values(["o", "benchmark"])
rows = []
for r in piv.to_dict("records"):
    rows.append(f"\\code{{{r['benchmark'].replace('_', chr(92) + '_')}}} & {label[r['source']]} & " + " & ".join(one(r[m]) for m, _ in cols) + " \\\\")
(OUT / "eandd_adaptive_bench.tex").write_text("\\begin{tabular}{llcccccc}\n\\toprule\nBenchmark & Split & " + " & ".join(n for _, n in cols) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

pa = pd.read_csv(DATA / "r4-pilot-auc.csv")
pl = [f"{int(r.pilot_labels)} & {100*r.share_hurt:.1f}\\% & {r.auc_pooled:.2f} & {r.auc_mean_per_benchmark:.2f} \\\\" for r in pa.itertuples()]
(OUT / "eandd_pilot.tex").write_text("\\begin{tabular}{cccc}\n\\toprule\nPilot labels & Tasks hurt & AUC, pooled & AUC, within benchmark \\\\\n\\midrule\n" + "\n".join(pl) + "\n\\bottomrule\n\\end{tabular}\n")

sw = pd.read_csv(DATA / "r4-switch-sweep.csv")
sl = []
for r in sw.itertuples():
    eta = "$\\infty$" if r.d >= 9 else f"{r.d:g}"
    sl.append(f"{int(r.k)} & {eta} & {one(r.macro_17)} & {one(r.worst_17)} & {one(r.test_gain)} \\\\")
(OUT / "eandd_switch.tex").write_text("\\begin{tabular}{ccccc}\n\\toprule\nPilot labels $k_0$ & Threshold $\\eta$ & Mean over 17 & Worst benchmark & OOD test \\\\\n\\midrule\n" + "\n".join(sl) + "\n\\bottomrule\n\\end{tabular}\n")

tg = pd.read_csv(DATA / "r6-triple-gains.csv")
tl = []
for m, nm in (("bb-eb", "Beta-Binomial"), ("rasch-fitted", "Rasch")):
    v = tg[tg.model == m].gain
    q05, q95 = v.quantile([.05, .95])
    tl.append(f"{nm} & {len(v)} & {one(v.mean())} & [{one(q05)}, {one(q95)}] & {100*(v > 0).mean():.1f}\\% \\\\")
(OUT / "eandd_rasch_triples.tex").write_text("\\begin{tabular}{lcccc}\n\\toprule\nFitted prior & Triples & Mean macro $\\Delta_{m-\\mathrm{J}}$ & 5th--95th percentile & Hurts\\\\\n\\midrule\n" + "\n".join(tl) + "\n\\bottomrule\n\\end{tabular}\n")

split = pd.read_csv(DATA / "wp13-split-sensitivity.csv")
d = split.refit_macro_delta
orig = d.loc[split.original_test_triple].item()
p05, p95 = d.quantile([.05, .95])
(OUT / "eandd_split_sensitivity.tex").write_text(
    f"We held out each of the {len(split)} triples, refitted the benchmark-weighted empirical Beta prior on the other "
    f"14 benchmarks, and rescored the frozen trajectories. The mean macro $\\Delta_{{\\mathrm{{BB\\,fit}}-\\mathrm{{J}}}}$ was {one(d.mean())}, "
    f"with 5th--95th percentiles [{one(p05)}, {one(p95)}]. The fitted prior hurt on {100*(d > 0).mean():.1f}\\% of splits. "
    f"The original test benchmark names gave {one(orig)} (percentile {100*(d <= orig).mean():.0f}) under refitting. "
    "This refit includes the former validation benchmarks among the 14 eligible fit benchmarks for the original "
    "test triple, so its result differs from the original 11-benchmark fit. The 11 original training benchmarks "
    "are scored on held-out-subject tasks from the existing leave-one-out data. These overlapping triples describe "
    "split sensitivity within this dataset. The main text uses the prior fitted on the 11 training benchmarks only. "
    f"Under that prior the mean macro $\\Delta_{{\\mathrm{{BB\\,fit}}-\\mathrm{{J}}}}$ over the same triples is {one(split.historical_macro_delta.mean())}, "
    f"the prior hurts on {100*(split.historical_macro_delta > 0).mean():.1f}\\% of them, and the original test benchmarks give "
    f"{one(split.historical_macro_delta.loc[split.original_test_triple].item())}.\n"
)

metric = pd.read_csv(DATA / "wp13-metric-alc.csv")
metric = metric.loc[metric.weighting.eq("official")].set_index(["run", "model"])
rank = pd.read_csv(DATA / "wp13-metrics.csv").query("budget == 31").set_index(["run", "model"])
names = {"jeffreys": "J", "bb-eb": "BB\\,fit", "bb-eb-clean": "BB\\,fit",
         "rasch-fitted": "R\\,fit",
         "kernel-fitted": "KR\\,fit", "offline-kept": "R\\,off"}
lines = []
for ri, (run, role) in enumerate([("r2-competition-id", "ID"), ("r2-competition-val", "OOD validation"),
                  ("wp5-competition-test", "OOD test")]):
    present = [mm for mm in names if (run, mm) in metric.index]
    lo_ = {c_: metric.loc[[(run, mm) for mm in present], c_].min() for c_ in ("brier", "log_loss", "ece10")}
    aucs = {mm: rank.loc[(run, mm), "auc"] for mm in present}
    hi_auc = max((a_ for a_ in aucs.values() if pd.notna(a_)), default=None)
    if ri:
        lines.append("\\midrule")
    for model, label in names.items():
        key = (run, model)
        if key not in metric.index:
            continue
        r = metric.loc[key]
        auc = rank.loc[key, "auc"]
        b_ = lambda x, c_: (GREY + f"\\textbf{{{x:.4f}}}" if abs(x - lo_[c_]) < 1e-12 else f"{x:.4f}")
        auc_text = (GREY + f"\\textbf{{{auc:.3f}}}" if auc == hi_auc else f"{auc:.3f}") if pd.notna(auc) else "--"
        lines.append(f"{role if model == present[0] else ''} & $\\mathrm{{{label}}}$ & {b_(r.brier, 'brier')} & {b_(r.log_loss, 'log_loss')} & {b_(r.ece10, 'ece10')} & {auc_text} \\\\")
(OUT / "eandd_reviewer_metrics.tex").write_text("\n".join(lines) + "\n")
wrap("eandd_reviewer_metrics.tex", "llrrrr", "Split & Model $m$ & Brier & Log loss & ECE & AUC at 31 labels")
