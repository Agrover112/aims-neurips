"""Hierarchical beta-binomial with a subject-metadata prior, replayed offline.

Each task (one subject on one benchmark) has an accuracy drawn from
Beta(kappa * mu_s, kappa * (1 - mu_s)). With k acquired labels and h hits the
prediction for every scored item is

    p = (kappa * mu_s + h) / (kappa + k),

so budget 0 predicts mu_s and labels pull it toward the task's own hit rate.
mu_s comes from what CodaBench sends with every subject:

    mu_s = sigmoid(b + name_emb[name] + provider_emb[provider] + w * date + v * date_missing)

``name`` and ``provider`` are lower-cased and punctuation-collapsed into
categories; unseen ones map to a learned "unknown" (names are dropped at
random during fitting, so "unknown" is the typical subject, not noise).
``date`` is release date in years since 2024-01-01. kappa is one learned
number. Everything is fitted by beta-binomial marginal likelihood on the
TRAIN benchmarks' tasks only; the L2 strength is chosen on validation and the
chosen fit is scored once on test.

Scoring reuses the frozen replay panel and the beta-binomial run's random-prefix
acquisition orders, so every arm sees exactly the same labels in the same
order as the existing replay rows.

    modal run --detach organized/testing/online/modal_hier_beta.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-neurips-hier-beta"
VOLUME_NAME = "aims-neurips-data"
PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"
RESPONSES = Path("/data/export/bhold-bge_m3_long-n15-m80-s20261018/filtered-responses.parquet")
SUBJECTS = Path("/data/datasets/full/subjects.parquet")
REPLAY = Path("/data/results/online-replay")
OUT = REPLAY / "hier-beta"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = dict(zip(BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
SEEDS = {"validation": "s3", "test": "s10"}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24",
).pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")


def _norm(series):
    import pandas as pd

    out = (series.astype("string").str.lower()
           .str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    return out.where(out.notna() & (out != ""), pd.NA)


def _subject_table():
    import pandas as pd

    s = pd.read_parquet(SUBJECTS).drop_duplicates("subject_uid").replace("", pd.NA)
    date = pd.to_datetime(s["release_date"], errors="coerce")
    return pd.DataFrame({
        "subject_uid": s["subject_uid"],
        "name": _norm(s["normalized_name"]),
        "provider": _norm(s["provider"]),
        "date": (date - pd.Timestamp("2024-01-01")).dt.days / 365.25,
    })


def _fit(tasks, features: tuple, l2: float, seed: int = 0, steps: int = 1500):
    """Fit mu_s(features) and kappa by beta-binomial marginal likelihood."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    names = {n: i + 1 for i, n in enumerate(sorted(tasks["name"].dropna().unique()))}
    provs = {p: i + 1 for i, p in enumerate(sorted(tasks["provider"].dropna().unique()))}
    name_ix = torch.tensor(tasks["name"].map(names).fillna(0).astype(int).to_numpy())
    prov_ix = torch.tensor(tasks["provider"].map(provs).fillna(0).astype(int).to_numpy())
    date = torch.tensor(tasks["date"].fillna(0.0).to_numpy(), dtype=torch.float32)
    miss = torch.tensor(tasks["date"].isna().to_numpy(), dtype=torch.float32)
    h = torch.tensor(tasks["hits"].to_numpy(), dtype=torch.float32)
    n = torch.tensor(tasks["n"].to_numpy(), dtype=torch.float32)

    p = {"b": torch.zeros(1, requires_grad=True), "log_kappa": torch.ones(1, requires_grad=True),
         "name": torch.zeros(len(names) + 1, requires_grad=True),
         "prov": torch.zeros(len(provs) + 1, requires_grad=True),
         "w": torch.zeros(2, requires_grad=True)}
    opt = torch.optim.Adam(p.values(), lr=0.05)

    def mu_logit(ni, pi, d, m):
        z = p["b"].expand_as(d)
        if "name" in features:
            z = z + p["name"][ni]
        if "provider" in features:
            z = z + p["prov"][pi]
        if "date" in features:
            z = z + p["w"][0] * d + p["w"][1] * m
        return z

    for _ in range(steps):
        # Name dropout teaches the "unknown" slot what an unseen model looks like.
        ni = torch.where(torch.rand(len(name_ix)) < 0.2, torch.zeros_like(name_ix), name_ix)
        mu = torch.sigmoid(mu_logit(ni, prov_ix, date, miss)).clamp(1e-4, 1 - 1e-4)
        k = p["log_kappa"].exp()
        a, b = k * mu, k * (1 - mu)
        ll = (torch.lgamma(h + a) + torch.lgamma(n - h + b) - torch.lgamma(n + a + b)
              - torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(a + b))
        loss = -ll.mean() + l2 * (p["name"].pow(2).sum() + p["prov"].pow(2).sum()) / len(h)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def predict_mu(frame):
        ni = torch.tensor(frame["name"].map(names).fillna(0).astype(int).to_numpy())
        pi = torch.tensor(frame["provider"].map(provs).fillna(0).astype(int).to_numpy())
        d = torch.tensor(frame["date"].fillna(0.0).to_numpy(), dtype=torch.float32)
        m = torch.tensor(frame["date"].isna().to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(mu_logit(ni, pi, d, m)).numpy()

    return predict_mu, float(p["log_kappa"].exp()), names


def _replay(role: str, mu_by_task: dict, kappa: float):
    """Task x budget Brier under the beta run's random-prefix orders."""
    import numpy as np
    import pandas as pd

    panel = REPLAY / "panels" / PANEL
    items = pd.read_parquet(panel / "items.parquet")
    ev = {t: g["label"].to_numpy() for t, g in items[items["pool"] == "evaluation"].groupby("task_id")}
    run = REPLAY / "runs" / PANEL / f"{role}-beta-random_prefix-{SEEDS[role]}-v2"
    d = pd.read_parquet(run / "acquisition_decisions.parquet")
    d = d[d["accepted"]].sort_values(["task_id", "order_seed", "labels_acquired_before"])
    rows = []
    for (task, seed), g in d.groupby(["task_id", "order_seed"]):
        lab = g["revealed_label"].astype(int).to_numpy()
        y = ev[task]
        mu = mu_by_task[task]
        for k in BUDGETS:
            hits = lab[:k].sum()
            p = (kappa * mu + hits) / (kappa + k)
            rows.append((task, seed, k, float(((p - y) ** 2).mean())))
    f = pd.DataFrame(rows, columns=["task_id", "seed", "budget", "brier"])
    return f.groupby(["task_id", "budget"])["brier"].mean().unstack()


@app.function(image=image, cpu=4, memory=16_384, volumes={"/data": volume}, timeout=2 * 60 * 60)
def run(arm_names: str = "", tag: str = "summary") -> dict:
    import numpy as np
    import pandas as pd

    r = pd.read_parquet(RESPONSES, columns=["subject_uid", "benchmark_id", "benchmark_split", "label"])
    subjects = _subject_table()
    tasks = (r.groupby(["subject_uid", "benchmark_id", "benchmark_split"], as_index=False)["label"]
             .agg(hits="sum", n="size").merge(subjects, on="subject_uid", how="left"))
    tasks["task_id"] = tasks["subject_uid"] + "::" + tasks["benchmark_id"]
    train = tasks[tasks["benchmark_split"] == "train"].reset_index(drop=True)
    by_task = tasks.set_index("task_id")
    print(f"train tasks {len(train)}; name coverage {train['name'].notna().mean():.2f}", flush=True)

    rng = np.random.default_rng(0)

    def alc(tb):
        return sum(w * tb[b] for b, w in WEIGHTS.items())

    def ci(x):
        boots = [rng.choice(x, len(x)).mean() for _ in range(4000)]
        return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    arms = {
        "hier, no metadata": (),
        "hier + provider + date": ("provider", "date"),
        "hier + name + provider + date": ("name", "provider", "date"),
        # Release date dropped: test models are newer than any training model,
        # so a linear date term extrapolates.
        "hier + provider": ("provider",),
        "hier + name + provider": ("name", "provider"),
    }
    if arm_names:
        arms = {k: arms[k] for k in arm_names.split(";")}
    jeffreys = {role: _replay(role, {t: 0.5 for t in by_task.index}, 1.0) for role in SEEDS}
    results, chosen, task_rows = [], {}, []
    for arm, feats in arms.items():
        # L2 chosen on validation ALC only; test is scored once with that choice.
        best = None
        for l2 in ((0.0,) if not feats else (0.1, 1.0, 10.0)):
            predict_mu, kappa, names = _fit(train, feats, l2)
            mu = dict(zip(by_task.index, predict_mu(by_task)))
            val = alc(_replay("validation", mu, kappa)).mean()
            print(f"{arm:<32} l2={l2:<5} kappa={kappa:6.2f} validation ALC {val:.4f}", flush=True)
            if best is None or val < best[0]:
                best = (val, l2, mu, kappa, names)
        _, l2, mu, kappa, names = best
        chosen[arm] = {"l2": l2, "kappa": kappa}
        for role in SEEDS:
            tb = _replay(role, mu, kappa)
            # Per task x budget, for the report's curves and per-budget CIs.
            long = tb.stack().rename("brier").reset_index()
            long[["role", "arm"]] = role, arm
            task_rows.append(long)
            diff = (alc(tb) - alc(jeffreys[role])).to_numpy()
            per_budget = (tb - jeffreys[role]).mean().round(4).to_dict()
            row = {"arm": arm, "role": role, "l2": l2, "kappa": kappa,
                   "alc": float(alc(tb).mean()), "jeffreys_alc": float(alc(jeffreys[role]).mean()),
                   "diff": ci(diff), "per_budget_diff": {str(k): v for k, v in per_budget.items()}}
            if role == "test" and "name" in feats:
                seen = np.array([by_task.loc[t, "name"] in names for t in tb.index])
                row["diff_seen_name"] = ci(diff[seen]) if seen.sum() > 2 else None
                row["diff_unseen_name"] = ci(diff[~seen]) if (~seen).sum() > 2 else None
                row["n_seen_name"] = int(seen.sum())
            results.append(row)
            print(json.dumps(row), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps({"results": results, "chosen": chosen}, indent=2))
    for role in SEEDS:
        long = jeffreys[role].stack().rename("brier").reset_index()
        long[["role", "arm"]] = role, "jeffreys"
        task_rows.append(long)
    pd.concat(task_rows).to_parquet(OUT / f"{tag}-task-budget.parquet", index=False)
    volume.commit()
    return {"results": results, "chosen": chosen}


@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume}, timeout=4 * 60 * 60)
def lobo(tag: str = "lobo", tasks_per_benchmark: int = 200, orders: int = 3) -> dict:
    """Leave-one-benchmark-out over the 14 non-test benchmarks.

    For each benchmark B, fit every arm on the tasks of the other 13 and score
    B's tasks: each task's items are shuffled, the first 31 are the context
    stream and the rest are scored, so the protocol matches the replay. Test
    benchmarks are never read. L2 strengths are the ones validation chose.
    """
    import numpy as np
    import pandas as pd

    r = pd.read_parquet(RESPONSES, columns=["subject_uid", "benchmark_id", "benchmark_split", "item_uid", "label"])
    r = r[r["benchmark_split"] != "test"]
    subjects = _subject_table()
    tasks = (r.groupby(["subject_uid", "benchmark_id"], as_index=False)["label"]
             .agg(hits="sum", n="size").merge(subjects, on="subject_uid", how="left"))
    arms = {"hier, no metadata": ((), 0.0), "hier + provider": (("provider",), 0.1),
            "hier + name + provider": (("name", "provider"), 10.0)}
    rng = np.random.default_rng(20261018)
    rows = []
    for bench in sorted(tasks["benchmark_id"].unique()):
        fit_tasks = tasks[tasks["benchmark_id"] != bench].reset_index(drop=True)
        held = tasks[tasks["benchmark_id"] == bench]
        held = held.sample(min(len(held), tasks_per_benchmark), random_state=0)
        fitted = {arm: _fit(fit_tasks, feats, l2) for arm, (feats, l2) in arms.items()}
        labels = r[r["benchmark_id"] == bench].groupby("subject_uid")["label"].apply(np.asarray)
        for _, t in held.iterrows():
            y_all = labels[t["subject_uid"]]
            if len(y_all) < 40:
                continue
            mus = {arm: float(f[0](held.loc[[t.name]])[0]) for arm, f in fitted.items()}
            for o in range(orders):
                y = rng.permutation(y_all)
                ctx, ev = y[:31], y[31:]
                for k in BUDGETS:
                    h = ctx[:k].sum()
                    row = {"benchmark": bench, "task": t["subject_uid"], "order": o, "budget": k,
                           "jeffreys": float((((h + 0.5) / (k + 1) - ev) ** 2).mean())}
                    for arm, (_, kappa, _) in fitted.items():
                        row[arm] = float((((kappa * mus[arm] + h) / (kappa + k) - ev) ** 2).mean())
                    rows.append(row)
        print(f"{bench}: {len(held)} tasks scored", flush=True)
    f = pd.DataFrame(rows)
    tb = f.groupby(["benchmark", "task", "budget"]).mean(numeric_only=True).drop(columns="order")
    alc = sum(w * tb.xs(b, level="budget") for b, w in WEIGHTS.items())
    # Benchmark is the unit: average within each, then across benchmarks.
    per_bench = alc.groupby(level="benchmark").mean()
    out = {"per_benchmark": per_bench.round(5).to_dict(orient="index"), "summary": {}}
    for arm in arms:
        d = (per_bench[arm] - per_bench["jeffreys"]).to_numpy()
        boots = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        wins = int((d < 0).sum())
        out["summary"][arm] = {"mean_diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))], "benchmarks_better": f"{wins}/{len(d)}"}
        print(arm, out["summary"][arm], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2))
    volume.commit()
    return out


@app.local_entrypoint()
def main(arms: str = "", tag: str = "summary", mode: str = "replay") -> None:
    call = lobo.spawn(tag=tag) if mode == "lobo" else run.spawn(arm_names=arms, tag=tag)
    print(f"spawned Modal call {call.object_id}")
