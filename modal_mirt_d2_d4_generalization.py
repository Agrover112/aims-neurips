"""MIRT (multidimensional IRT) subject/item split-half generalization, D=2 and D=4.
Filtered data (n>=30), 20% row/column holdout, split-half per held-out unit,
AdamW optimizer, 3 seeds.

Run:  modal run --detach modal_mirt_d2_d4_generalization.py::main
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-mirt-d2-d4-generalization"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT = Path("/data/results/data-analysis/mirt-d2-d4-generalization.parquet")

MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HELD_OUT_FRAC = 0.20
STEPS = 400
MIN_MATRIX = 20
MIN_HALF = 4

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu118")


def _fit_core(mat, kind, d, steps, seed):
    """Joint MLE on every non-NaN cell of ``mat``."""
    import numpy as np
    import torch

    s_idx, i_idx = np.nonzero(~np.isnan(mat))
    y = torch.as_tensor(mat[s_idx, i_idx], dtype=torch.float32)
    s_t = torch.as_tensor(s_idx, dtype=torch.long)
    i_t = torch.as_tensor(i_idx, dtype=torch.long)
    n_s, n_i = mat.shape

    torch.manual_seed(seed)
    theta = torch.randn(n_s, d, requires_grad=True) * 0.1
    b = torch.zeros(n_i, requires_grad=True)
    log_a = torch.zeros(n_i, requires_grad=True)
    params = [theta, b] + ([log_a] if kind in ("2pl", "3pl") else [])

    opt = torch.optim.AdamW(params, lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i)
        z = a[i_t] * (theta[s_t].sum(dim=1) - b[i_t])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        loss = loss + 0.125 * (theta.square().mean() + b.square().mean())
        loss.backward()
        opt.step()
    return theta.detach(), b.detach(), log_a.detach()


def _fit_one_theta(a_known, b_known, y_half, d, steps=200):
    """New subject: fit multidim theta against frozen (a, b) on revealed cells."""
    import torch

    theta = torch.randn(1, d, requires_grad=True) * 0.1
    opt = torch.optim.AdamW([theta], lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        z = a_known * (theta.sum(dim=1) - b_known)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_half)
        loss = loss + 0.25 * theta.square().mean()
        loss.backward()
        opt.step()
    return theta.detach()


def _fit_one_item(theta_known, y_half, kind, steps=200):
    """New item: fit (a, b) against frozen theta on revealed cells."""
    import torch

    b = torch.zeros(1, requires_grad=True)
    log_a = torch.zeros(1, requires_grad=True)
    params = [b] + ([log_a] if kind in ("2pl", "3pl") else [])
    opt = torch.optim.AdamW(params, lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(1)
        z = a * (theta_known.sum(dim=1) - b)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_half)
        loss = loss + 0.25 * b.square().mean()
        loss.backward()
        opt.step()
    a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(1)
    return a.detach(), b.detach()


@app.function(image=image, gpu="L4", memory=131_072, volumes={"/data": volume},
              timeout=12 * 60 * 60)
def run(seed: int = 20261018) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    items_meta = pd.read_parquet(PROCESSED / "items.parquet",
                                 columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")
    d = obs.merge(items_meta, on="item_uid", how="left")
    d = d.drop_duplicates(["content_hash", "subject_uid"])
    var = d.groupby("item_uid")["label"].agg(["mean", "size"])
    keep_items = var[(var["mean"] > 0) & (var["mean"] < 1)].index
    d = d[d["item_uid"].isin(keep_items)]
    n_sub_per_item = d.groupby("item_uid")["subject_uid"].nunique()
    d = d[d["item_uid"].isin(n_sub_per_item[n_sub_per_item >= MIN_SUBJECTS].index)]
    n_item_per_task = d.groupby("subject_uid")["item_uid"].nunique()
    d = d[d["subject_uid"].isin(n_item_per_task[n_item_per_task >= MIN_ITEMS_PER_TASK].index)]
    print(f"filtered (n>=30): {len(d):,} rows, {d.benchmark_id.nunique()} benchmarks", flush=True)

    rows = []
    rng_global = np.random.default_rng(seed)
    for bench, dd in d.groupby("benchmark_id"):
        subs = sorted(dd["subject_uid"].unique())
        its = sorted(dd["item_uid"].unique())
        if len(subs) < 10 or len(its) < 10:
            continue
        s_map = {s: i for i, s in enumerate(subs)}
        i_map = {s: i for i, s in enumerate(its)}
        mat = np.full((len(subs), len(its)), np.nan, dtype=np.float32)
        mat[dd["subject_uid"].map(s_map).to_numpy(), dd["item_uid"].map(i_map).to_numpy()] = \
            dd["label"].to_numpy()

        rng = np.random.default_rng([seed, hash(bench) % (2**31)])
        held_s = rng.choice(len(subs), size=max(1, int(len(subs) * HELD_OUT_FRAC)), replace=False)
        held_i = rng.choice(len(its), size=max(1, int(len(its) * HELD_OUT_FRAC)), replace=False)
        train_s = np.setdiff1d(np.arange(len(subs)), held_s)
        train_i = np.setdiff1d(np.arange(len(its)), held_i)
        if len(train_s) < MIN_MATRIX or len(train_i) < MIN_MATRIX:
            continue
        core = mat[np.ix_(train_s, train_i)]
        if np.sum(~np.isnan(core)) < MIN_MATRIX:
            continue

        for kind in ("1pl", "2pl", "3pl"):
            for d_dim in (2, 4):
                theta_core, b_core, loga_core = _fit_core(core, kind, d_dim, STEPS, int(rng_global.integers(0, 2**31)))
                a_core = torch.exp(loga_core) if kind in ("2pl", "3pl") else torch.ones(len(train_i))

                # subject generalization
                for row in held_s:
                    cols = np.flatnonzero(~np.isnan(mat[row, train_i]))
                    if len(cols) < 2 * MIN_HALF:
                        continue
                    perm = rng.permutation(cols)
                    half1, half2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
                    y1 = torch.as_tensor(mat[row, train_i][half1], dtype=torch.float32)
                    theta_new = _fit_one_theta(a_core[half1], b_core[half1], y1, d_dim)
                    with torch.no_grad():
                        z2 = a_core[half2] * (theta_new.sum(dim=1) - b_core[half2])
                        p2 = torch.sigmoid(z2).numpy()
                    y2 = mat[row, train_i][half2]
                    if len(np.unique(y2)) < 2 or not np.isfinite(p2).all():
                        continue
                    rows.append({"benchmark_id": bench, "model": f"{kind}_d{d_dim}", "seed": seed, "axis": "subject",
                                "n_fit": len(half1), "n_score": len(half2),
                                "auc": float(roc_auc_score(y2, p2)),
                                "brier": float(np.mean((p2 - y2) ** 2))})

                # item generalization
                for col in held_i:
                    rows_idx = np.flatnonzero(~np.isnan(mat[train_s, col]))
                    if len(rows_idx) < 2 * MIN_HALF:
                        continue
                    perm = rng.permutation(rows_idx)
                    half1, half2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
                    y1 = torch.as_tensor(mat[train_s, col][half1], dtype=torch.float32)
                    a_new, b_new = _fit_one_item(theta_core[half1], y1, kind)
                    with torch.no_grad():
                        z2 = a_new * (theta_core[half2].sum(dim=1) - b_new)
                        p2 = torch.sigmoid(z2).numpy()
                    y2 = mat[train_s, col][half2]
                    if len(np.unique(y2)) < 2 or not np.isfinite(p2).all():
                        continue
                    rows.append({"benchmark_id": bench, "model": f"{kind}_d{d_dim}", "seed": seed, "axis": "item",
                                "n_fit": len(half1), "n_score": len(half2),
                                "auc": float(roc_auc_score(y2, p2)),
                                "brier": float(np.mean((p2 - y2) ** 2))})
        print(f"{bench:<20}subs={len(subs):<5}items={len(its):<6}"
              f"held_s={len(held_s)}  held_i={len(held_i)}", flush=True)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"{OUT.stem}-seed{seed}{OUT.suffix}")
    frame.to_parquet(out, index=False)
    volume.commit()
    return {"rows": len(frame), "benchmarks": frame["benchmark_id"].nunique(), "seed": seed}


SEEDS = (20261018, 20261019, 20261020)


@app.local_entrypoint()
def main() -> None:
    handles = [run.spawn(seed) for seed in SEEDS]
    for h in handles:
        print(h.get())
