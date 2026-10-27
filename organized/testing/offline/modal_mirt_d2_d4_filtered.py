"""MIRT (multidimensional IRT) 1PL/2PL/3PL with D=2 and D=4 latent dimensions.
Filtered data (n>=30), 20% cell holdout, AdamW optimizer, 3 seeds.

Run:  modal run --detach modal_mirt_d2_d4_filtered.py::main
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-mirt-d2-d4-filtered"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT = Path("/data/results/data-analysis/mirt-d2-d4-filtered.parquet")

MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HOLDOUT_FRAC = 0.20
STEPS = 400
MIN_MATRIX = 20

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2", "tqdm>=4.66,<5")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")


def _fit_and_score(mat, mask_train, mask_test, kind, d, steps, rng):
    """Joint MLE fit of multidimensional theta/b/a(/c) on training cells, scored
    on held-out cells.

    Compensatory MIRT: discrimination is a vector a_j in R^d per item, so
    z = <a_j, theta_i> - b_j. A scalar a times theta.sum() would make every
    d > 1 algebraically identical to d = 1 -- that was the bug here before.
    3PL gets a real guessing parameter c_j via logit_c, previously missing
    entirely so "3PL" was silently a duplicate 2PL.
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    s_idx, i_idx = np.nonzero(mask_train)
    y_train = torch.as_tensor(mat[s_idx, i_idx], dtype=torch.float32)
    s_t = torch.as_tensor(s_idx, dtype=torch.long)
    i_t = torch.as_tensor(i_idx, dtype=torch.long)
    n_s, n_i = mat.shape

    torch.manual_seed(int(rng.integers(0, 2**31)))
    theta = (torch.randn(n_s, d) * 0.1).requires_grad_(True)
    b = torch.zeros(n_i, requires_grad=True)
    log_a = torch.zeros(n_i, d, requires_grad=True)
    logit_c = torch.full((n_i,), -2.0, requires_grad=True)
    params = [theta, b]
    if kind in ("2pl", "3pl"):
        params.append(log_a)
    if kind == "3pl":
        params.append(logit_c)

    opt = torch.optim.AdamW(params, lr=0.1)
    loss_history = []
    for _ in range(steps):
        opt.zero_grad()
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i, d)
        z = (a[i_t] * theta[s_t]).sum(dim=1) - b[i_t]
        if kind == "3pl":
            c = torch.sigmoid(logit_c) * 0.5
            p = (c[i_t] + (1 - c[i_t]) * torch.sigmoid(z)).clamp(1e-6, 1 - 1e-6)
            loss = -(y_train * torch.log(p) + (1 - y_train) * torch.log(1 - p)).mean()
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_train)
        loss = loss + 0.125 * (theta.square().mean() + b.square().mean())
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu().numpy()))

    s_idx, i_idx = np.nonzero(mask_test)
    if len(s_idx) < 10 or len(np.unique(mat[s_idx, i_idx])) < 2:
        return None
    with torch.no_grad():
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i, d)
        z = (a[i_idx] * theta[torch.as_tensor(s_idx)]).sum(dim=1) - b[i_idx]
        p = torch.sigmoid(z)
        if kind == "3pl":
            c = torch.sigmoid(logit_c) * 0.5
            p = c[i_idx] + (1 - c[i_idx]) * p
        p = p.numpy()
    y_test = mat[s_idx, i_idx]
    try:
        auc = float(roc_auc_score(y_test, p))
    except ValueError:
        auc = float("nan")
    brier = float(((p - y_test) ** 2).mean())
    weights = {"theta": theta.detach().numpy(), "b": b.detach().numpy(),
              "log_a": log_a.detach().numpy(), "logit_c": logit_c.detach().numpy(),
              "loss_history": loss_history}
    return auc, brier, len(y_test), weights


def _mask_and_prune(mat, rng):
    """Random 20% cell holdout, drop degenerate rows/cols in train."""
    import numpy as np

    answered = ~np.isnan(mat)
    s_idx, i_idx = np.nonzero(answered)
    n_test = int(len(s_idx) * HOLDOUT_FRAC)
    test_pos = rng.choice(len(s_idx), size=n_test, replace=False)
    test_mask = np.zeros_like(answered)
    test_mask[s_idx[test_pos], i_idx[test_pos]] = True
    train_mask = answered & ~test_mask

    def _degenerate(axis):
        sums = np.nansum(np.where(train_mask, mat, np.nan), axis=axis)
        counts = train_mask.sum(axis=axis)
        return (counts > 0) & ((sums == 0) | (sums == counts))

    bad_s = _degenerate(axis=1)
    bad_i = _degenerate(axis=0)
    keep_s = ~bad_s
    keep_i = ~bad_i
    if keep_s.sum() < 5 or keep_i.sum() < 5:
        return None
    return (mat[np.ix_(keep_s, keep_i)], train_mask[np.ix_(keep_s, keep_i)],
            test_mask[np.ix_(keep_s, keep_i)])


# CPU: free-parameter fits, no embeddings or MLP -- a GPU buys nothing.
@app.function(image=image, cpu=16, memory=65_536, volumes={"/data": volume},
              timeout=6 * 60 * 60)
def run(seed: int = 20261018) -> dict:
    import numpy as np
    import pandas as pd

    from tqdm import tqdm
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
    weights_all = {}
    rng_global = np.random.default_rng(seed)
    _benches = list(d.groupby("benchmark_id"))
    for bench, dd in tqdm(_benches, desc=f"seed={seed}", unit="bench", mininterval=5.0):
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
        pruned = _mask_and_prune(mat, rng)
        if pruned is None:
            continue
        m2, train_mask, test_mask = pruned

        for kind in ("1pl", "2pl", "3pl"):
            for d_dim in (2, 4):
                result = _fit_and_score(m2, train_mask, test_mask, kind, d_dim, STEPS, rng)
                if result is None:
                    continue
                auc, brier, n_test, weights = result
                weights_all[f"{bench}_{kind}_d{d_dim}"] = weights
                rows.append({"benchmark_id": bench, "model": f"{kind}_d{d_dim}", "seed": seed,
                            "n_subjects": len(subs), "n_items": len(its),
                            "n_subjects_fit": m2.shape[0], "n_items_fit": m2.shape[1],
                            "n_test_cells": n_test, "auc": auc, "brier": brier})
        print(f"{bench:<20}subs={len(subs):<6}items={len(its):<8}fit={m2.shape}", flush=True)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"{OUT.stem}-seed{seed}{OUT.suffix}")
    frame.to_parquet(out, index=False)
    import torch
    weights_out = OUT.parent / f"mirt-d2-d4-filtered-weights-seed{seed}.pt"
    torch.save(weights_all, weights_out)
    volume.commit()
    return {"rows": len(frame), "benchmarks": frame["benchmark_id"].nunique(), "seed": seed, "weights_saved": len(weights_all)}


SEEDS = (20261018, 20261019, 20261020)


@app.local_entrypoint()
def main() -> None:
    handles = [run.spawn(seed) for seed in SEEDS]
    for h in handles:
        print(h.get())
