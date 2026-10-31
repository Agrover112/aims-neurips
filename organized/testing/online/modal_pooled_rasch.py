"""Pooled hierarchical Rasch model using every pair's labels, replayed offline.

CodaBench gives the predictor all evidence purchased so far across the sampled
subject-benchmark pairs, not only the current pair's. This model uses it:

    logit p(s, i) = mu_b + theta_s - b_h

    mu_b     benchmark base rate, pooled over every pair on that benchmark
    theta_s  subject ability, from that subject's labels (shared across
             benchmarks when one subject appears on several)
    b_h      item difficulty, from other subjects' labels on the same item,
             matched by content hash; 0 for an item nobody has labelled

All three are MAP estimates under Gaussian priors, refitted from scratch at
every budget on the evidence available at that checkpoint. Budget 0 has no
evidence anywhere, so it predicts sigmoid(m0).

The CodaBench draw is simulated from the frozen replay panel: each draw picks
one or two subjects per benchmark, and every pair acquires its labels in the
beta-binomial run's random-prefix order, so the evidence is exactly what the
existing replay rows saw. Prior scales are chosen on validation draws and the
chosen setting is scored once on test draws.

    modal run --detach organized/testing/online/modal_pooled_rasch.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-neurips-pooled-rasch"
VOLUME_NAME = "aims-neurips-data"
PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"
REPLAY = Path("/data/results/online-replay")
RESPONSES = Path("/data/export/bhold-bge_m3_long-n15-m80-s20261018/filtered-responses.parquet")
OUT = REPLAY / "pooled-rasch"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = dict(zip(BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
SEEDS = {"validation": "s3", "test": "s10"}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scipy>=1.13,<2",
)


def _logit(p):
    import numpy as np

    return float(np.log(p / (1 - p)))


def _fit(evidence, m0: float, tau_b: float, tau_s: float, tau_i: float, theta0=None):
    """MAP of (mu_b, theta_s, b_h) from evidence rows (bench, subject, hash, y)."""
    import numpy as np
    from scipy.optimize import minimize

    benches = sorted({e[0] for e in evidence})
    subjects = sorted({e[1] for e in evidence})
    hashes = sorted({e[2] for e in evidence})
    bi = {b: j for j, b in enumerate(benches)}
    si = {s: j for j, s in enumerate(subjects)}
    hi = {h: j for j, h in enumerate(hashes)}
    B = np.array([bi[e[0]] for e in evidence])
    S = np.array([si[e[1]] for e in evidence])
    H = np.array([hi[e[2]] for e in evidence])
    y = np.array([e[3] for e in evidence], dtype=float)
    nb, ns, nh = len(benches), len(subjects), len(hashes)
    use_items = tau_i > 0
    th0 = np.array([(theta0 or {}).get(s, 0.0) for s in subjects])

    def unpack(v):
        return v[:nb], v[nb:nb + ns], (v[nb + ns:] if use_items else np.zeros(nh))

    def f(v):
        mu, th, bh = unpack(v)
        z = mu[B] + th[S] - bh[H]
        p = 1 / (1 + np.exp(-z))
        nll = np.sum(np.logaddexp(0, z) - y * z)
        r = p - y
        g_mu = np.bincount(B, r, nb) + (mu - m0) / tau_b ** 2
        g_th = np.bincount(S, r, ns) + (th - th0) / tau_s ** 2
        prior = 0.5 * np.sum((mu - m0) ** 2) / tau_b ** 2 + 0.5 * np.sum((th - th0) ** 2) / tau_s ** 2
        if use_items:
            g_bh = -np.bincount(H, r, nh) + bh / tau_i ** 2
            prior += 0.5 * np.sum(bh ** 2) / tau_i ** 2
            return nll + prior, np.concatenate([g_mu, g_th, g_bh])
        return nll + prior, np.concatenate([g_mu, g_th])

    x0 = np.concatenate([np.full(nb, m0), th0, np.zeros(nh if use_items else 0)])
    v = minimize(f, x0, jac=True, method="L-BFGS-B").x
    mu, th, bh = unpack(v)
    return ({b: mu[j] for b, j in bi.items()}, {s: th[j] for s, j in si.items()},
            {h: bh[j] for h, j in hi.items()} if use_items else {})


def _load(role: str):
    import pandas as pd

    panel = REPLAY / "panels" / PANEL
    items = pd.read_parquet(panel / "items.parquet")
    items = items[items["benchmark_split"] == role]
    run = REPLAY / "runs" / PANEL / f"{role}-beta-random_prefix-{SEEDS[role]}-v2"
    d = pd.read_parquet(run / "acquisition_decisions.parquet")
    d = d[d["accepted"]].sort_values(["task_id", "order_seed", "labels_acquired_before"])
    hash_of = items.set_index(["task_id", "item_uid"])["content_hash"]
    streams = {}
    for (task, seed), g in d.groupby(["task_id", "order_seed"]):
        hashes = [hash_of[(task, u)] for u in g["item_uid"]]
        streams[(task, int(seed))] = list(zip(hashes, g["revealed_label"].astype(int)))
    ev = items[items["pool"] == "evaluation"]
    evaluation = {t: (g["content_hash"].to_numpy(), g["label"].to_numpy()) for t, g in ev.groupby("task_id")}
    meta = items.drop_duplicates("task_id").set_index("task_id")[["benchmark_id", "subject_uid"]]
    seeds = sorted({s for _, s in streams})
    return streams, evaluation, meta, seeds


def _draws(meta, seeds, n_draws: int, rng):
    """Each draw: one or two subjects per benchmark, one candidate order each."""
    by_bench = {b: g.index.tolist() for b, g in meta.groupby("benchmark_id")}
    draws = []
    for _ in range(n_draws):
        pairs = []
        for b, tasks in by_bench.items():
            k = min(len(tasks), int(rng.integers(1, 3)))
            for t in rng.choice(tasks, k, replace=False):
                pairs.append((t, int(rng.choice(seeds))))
        draws.append(pairs)
    return draws


def _score(draws, streams, evaluation, meta, settings: dict):
    """Pair-level ALC for each setting, plus Jeffreys beta on the same pairs."""
    import numpy as np

    out = {name: [] for name in settings} | {"jeffreys": []}
    keys = []
    for pairs in draws:
        per = {name: {t: [] for t, _ in pairs} for name in out}
        for k in BUDGETS:
            evidence = []
            for t, seed in pairs:
                b, s = meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"]
                evidence += [(b, s, h, y) for h, y in streams[(t, seed)][:k]]
            for t, seed in pairs:
                hashes, y = evaluation[t]
                own = [lab for _, lab in streams[(t, seed)][:k]]
                p = (sum(own) + 0.5) / (k + 1)
                per["jeffreys"][t].append(float(((p - y) ** 2).mean()))
            for name, setting in settings.items():
                m0, tb, ts, ti = setting[:4]
                theta0 = setting[4] if len(setting) > 4 else {}
                if evidence:
                    mu, th, bh = _fit(evidence, m0, tb, ts, ti, theta0)
                for t, seed in pairs:
                    hashes, y = evaluation[t]
                    b, s = meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"]
                    if evidence:
                        z = (mu.get(b, m0) + th.get(s, theta0.get(s, 0.0))
                             - np.array([bh.get(h, 0.0) for h in hashes]))
                    else:
                        z = np.full(len(hashes), m0 + theta0.get(s, 0.0))
                    p = 1 / (1 + np.exp(-z))
                    per[name][t].append(float(((p - y) ** 2).mean()))
        for t, _ in pairs:
            keys.append(t)
            for name in out:
                out[name].append(per[name][t])
    return keys, {name: np.array(v) for name, v in out.items()}


@app.function(image=image, cpu=8, memory=16_384, volumes={"/data": volume}, timeout=4 * 60 * 60)
def run(n_val_draws: int = 60, n_test_draws: int = 150) -> dict:
    import itertools

    import numpy as np
    import pandas as pd

    w = np.array([WEIGHTS[b] for b in BUDGETS])
    r = pd.read_parquet(RESPONSES, columns=["benchmark_split", "label"])
    m0 = _logit(float(r.loc[r["benchmark_split"] == "train", "label"].mean()))
    print(f"m0 = logit(train mean) = {m0:.3f}", flush=True)

    rng = np.random.default_rng(20261018)
    data = {role: _load(role) for role in SEEDS}

    # Diagnostic: how predictive is another subject's label on the same item?
    for role, (streams, evaluation, meta, _) in data.items():
        acq = {}
        for (t, _), stream in streams.items():
            for h, lab in stream:
                acq.setdefault((meta.loc[t, "benchmark_id"], h), {})[meta.loc[t, "subject_uid"]] = lab
        pairs, matched, total = [], 0, 0
        for t, (hashes, y) in evaluation.items():
            b, s = meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"]
            for h, lab in zip(hashes, y):
                total += 1
                others = [v for subj, v in acq.get((b, h), {}).items() if subj != s]
                if others:
                    matched += 1
                    pairs.append((np.mean(others), lab))
        if pairs:
            x, yy = np.array(pairs).T
            pos, neg = x[yy == 1], x[yy == 0]
            auc = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
                   if len(pos) and len(neg) else float("nan"))
            print(f"{role}: {matched / total:.1%} of evaluation items have another subject's label "
                  f"(whole panel, all 31 labels); AUC of that label {auc:.3f}", flush=True)

    grid = {f"tau_b={tb},tau_s={ts},tau_i={ti}": (m0, tb, ts, ti)
            for tb, ts, ti in itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0))}
    streams, evaluation, meta, seeds = data["validation"]
    _, val = _score(_draws(meta, seeds, n_val_draws, rng), streams, evaluation, meta, grid)
    val_alc = {name: float((v @ w).mean()) for name, v in val.items()}
    best = min(grid, key=val_alc.get)
    best_no_items = min((g for g in grid if g.endswith("tau_i=0.0")), key=val_alc.get)
    print(f"validation Jeffreys {val_alc['jeffreys']:.4f}; best {best} {val_alc[best]:.4f}; "
          f"best without item term {best_no_items} {val_alc[best_no_items]:.4f}", flush=True)

    chosen = {"pooled Rasch": grid[best], "pooled Rasch, no item term": grid[best_no_items]}
    streams, evaluation, meta, seeds = data["test"]
    keys, test = _score(_draws(meta, seeds, n_test_draws, rng), streams, evaluation, meta, chosen)
    out = {"m0": m0, "validation_alc": val_alc, "chosen": {k: list(v) for k, v in chosen.items()}, "test": {}}
    for name in chosen:
        diff = pd.Series((test[name] - test["jeffreys"]) @ w, index=keys).groupby(level=0).mean().to_numpy()
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
        per_budget = (test[name] - test["jeffreys"]).mean(0)
        out["test"][name] = {"alc": float((test[name] @ w).mean()), "jeffreys_alc": float((test["jeffreys"] @ w).mean()),
                             "diff": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                             "tasks": int(len(diff)),
                             "per_budget_diff": {str(b): round(float(x), 4) for b, x in zip(BUDGETS, per_budget)}}
        print(name, json.dumps(out["test"][name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    volume.commit()
    return out


def _provider_theta0(m0: float, kappa: float = 20.0) -> dict:
    """Provider's typical ability on training benchmarks, as a logit offset
    from m0, shrunk toward 0 by ``kappa`` tasks' worth of prior weight. Keyed
    by subject_uid for every subject with a provider."""
    import numpy as np
    import pandas as pd

    subjects = pd.read_parquet("/data/datasets/full/subjects.parquet").drop_duplicates("subject_uid")
    prov = (subjects["provider"].astype("string").str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    subjects = pd.DataFrame({"subject_uid": subjects["subject_uid"], "provider": prov.replace("", pd.NA)})
    r = pd.read_parquet(RESPONSES, columns=["subject_uid", "benchmark_id", "benchmark_split", "label"])
    tasks = (r[r["benchmark_split"] == "train"].groupby(["subject_uid", "benchmark_id"], as_index=False)["label"]
             .mean().merge(subjects, on="subject_uid").dropna(subset=["provider"]))
    tasks["z"] = np.log(tasks["label"].clip(0.02, 0.98) / (1 - tasks["label"].clip(0.02, 0.98))) - m0
    g = tasks.groupby("provider")["z"].agg(["sum", "size"])
    offset = (g["sum"] / (g["size"] + kappa)).to_dict()
    return {u: float(offset.get(p, 0.0)) for u, p in zip(subjects["subject_uid"], subjects["provider"])
            if isinstance(p, str)}


@app.function(image=image, cpu=8, memory=16_384, volumes={"/data": volume}, timeout=4 * 60 * 60)
def combined(n_val_draws: int = 60, n_test_draws: int = 150) -> dict:
    """Pooled Rasch with each subject's ability prior centred on its provider."""
    import numpy as np
    import pandas as pd

    w = np.array([WEIGHTS[b] for b in BUDGETS])
    r = pd.read_parquet(RESPONSES, columns=["benchmark_split", "label"])
    m0 = _logit(float(r.loc[r["benchmark_split"] == "train", "label"].mean()))
    theta0 = _provider_theta0(m0)
    print(f"provider offsets for {len(theta0)} subjects", flush=True)
    rng = np.random.default_rng(20261018)
    data = {role: _load(role) for role in SEEDS}
    base = (m0, 2.0, 1.0, 1.0)  # the pooled Rasch setting validation chose
    settings = {"pooled Rasch": base}
    for alpha in (0.5, 1.0):
        settings[f"combined, provider prior x{alpha}"] = base + ({k: alpha * v for k, v in theta0.items()},)
    streams, evaluation, meta, seeds = data["validation"]
    cover = meta["subject_uid"].isin(theta0).mean()
    _, val = _score(_draws(meta, seeds, n_val_draws, rng), streams, evaluation, meta, settings)
    val_alc = {k: float((v @ w).mean()) for k, v in val.items()}
    print("validation", json.dumps(val_alc), f"(provider coverage {cover:.0%})", flush=True)
    best = min((k for k in settings if k.startswith("combined")), key=val_alc.get)
    chosen = {"pooled Rasch": settings["pooled Rasch"], best: settings[best]}
    streams, evaluation, meta, seeds = data["test"]
    cover = meta["subject_uid"].isin(theta0).mean()
    keys, test = _score(_draws(meta, seeds, n_test_draws, rng), streams, evaluation, meta, chosen)
    out = {"validation_alc": val_alc, "chosen": best, "test_provider_coverage": float(cover), "test": {}}
    for name in chosen:
        diff = pd.Series((test[name] - test["jeffreys"]) @ w, index=keys).groupby(level=0).mean().to_numpy()
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
        out["test"][name] = {"alc": float((test[name] @ w).mean()), "jeffreys_alc": float((test["jeffreys"] @ w).mean()),
                             "diff": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                             "per_budget_diff": {str(b): round(float(x), 4) for b, x in zip(BUDGETS, (test[name] - test["jeffreys"]).mean(0))}}
        print(name, json.dumps(out["test"][name]), flush=True)
    d = pd.Series((test[best] - test["pooled Rasch"]) @ w, index=keys).groupby(level=0).mean().to_numpy()
    boots = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    out["combined_minus_pooled"] = {"diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    print("combined minus pooled", json.dumps(out["combined_minus_pooled"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "combined.json").write_text(json.dumps(out, indent=2))
    volume.commit()
    return out


@app.function(image=image, cpu=8, memory=16_384, volumes={"/data": volume}, timeout=4 * 60 * 60)
def curves(n_val_draws: int = 150, n_test_draws: int = 150) -> dict:
    """Per (draw, task, budget) Brier for the report's multi-pair curves.

    Saves the draws themselves, so models that ignore other pairs can be
    placed on exactly the same draws from their saved single-pair scores.
    Settings are the ones validation already chose; nothing is re-tuned.
    """
    import numpy as np
    import pandas as pd

    r = pd.read_parquet(RESPONSES, columns=["benchmark_split", "label"])
    m0 = _logit(float(r.loc[r["benchmark_split"] == "train", "label"].mean()))
    theta0 = _provider_theta0(m0)
    base = (m0, 2.0, 1.0, 1.0)
    settings = {"constant": (m0, 1e-6, 1e-6, 0.0),
                "pooled-rasch": base,
                "pooled-rasch-no-item": (m0, 2.0, 1.0, 0.0),
                "combined-provider": base + (theta0,)}
    rng = np.random.default_rng(20261018)
    rows = []
    for role, n in (("validation", n_val_draws), ("test", n_test_draws)):
        streams, evaluation, meta, seeds = _load(role)
        draws = _draws(meta, seeds, n, rng)
        flat = [(d, t, seed) for d, pairs in enumerate(draws) for t, seed in pairs]
        keys, scores = _score(draws, streams, evaluation, meta, settings)
        assert keys == [t for _, t, _ in flat]
        for (d, t, seed), j in zip(flat, range(len(flat))):
            for name, v in scores.items():
                for b, x in zip(BUDGETS, v[j]):
                    rows.append((role, d, t, seed, name, b, float(x)))
        print(f"{role}: {len(draws)} draws, {len(flat)} pair-draws", flush=True)
    f = pd.DataFrame(rows, columns=["role", "draw", "task_id", "order_seed", "model", "budget", "brier"])
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(OUT / "curves.parquet", index=False)
    volume.commit()
    return {"rows": len(f)}


@app.local_entrypoint()
def main(n_val_draws: int = 60, n_test_draws: int = 150, mode: str = "pooled") -> None:
    fn = {"combined": combined, "curves": curves}.get(mode, run)
    call = fn.spawn(n_val_draws=n_val_draws, n_test_draws=n_test_draws)
    print(f"spawned Modal call {call.object_id}")
