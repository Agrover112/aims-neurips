"""Bayesian MIRT (with Gaussian priors) D=1, D=2, D=4, filtered data (n>=30).
20% random cell holdout, AdamW optimizer, 3 seeds.

Run:  modal run --detach modal_bayesian_mirt_d1_d2_d4_filtered.py::main
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-bayesian-mirt-d1-d2-d4-filtered"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT = Path("/data/results/data-analysis/bayesian-mirt-d1-d2-d4-filtered.parquet")

MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HOLDOUT_FRAC = 0.20
STEPS = 400
MIN_MATRIX = 20
PRIOR_SCALE = 1.0

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu118")


def _fit_and_score(mat, mask_train, mask_test, kind, d, steps, rng):
    """Joint MLE with Gaussian priors on theta and item params."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    s_idx, i_idx = np.nonzero(mask_train)
    y_train = torch.as_tensor(mat[s_idx, i_idx], dtype=torch.float32)
    s_t = torch.as_tensor(s_idx, dtype=torch.long)
    i_t = torch.as_tensor(i_idx, dtype=torch.long)
    n_s, n_i = mat.shape

    torch.manual_seed(int(rng.integers(0, 2**31)))
    theta = torch.randn(n_s, d, requires_grad=True) * 0.1
    b = torch.zeros(n_i, requires_grad=True)
    log_a = torch.zeros(n_i, requires_grad=True)
    params = [theta, b] + ([log_a] if kind in ("2pl", "3pl") else [])

    opt = torch.optim.AdamW(params, lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i)
        z = a[i_t] * (theta[s_t].sum(dim=1) - b[i_t])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_train)
        # Bayesian: Gaussian priors on theta, b, log_a
        loss = loss + (theta.square().sum() + b.square().sum() + 0.5*log_a.square().sum()) / (2 * PRIOR_SCALE**2)
        loss.backward()
        opt.step()

    s_idx, i_idx = np.nonzero(mask_test)
    if len(s_idx) < 10 or len(np.unique(mat[s_idx, i_idx])) < 2:
        return None
    with torch.no_grad():
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i)
        z = a[i_idx] * (theta[torch.as_tensor(s_idx)].sum(dim=1) - b[i_idx])
        p = torch.sigmoid(z).numpy()
    y_test = mat[s_idx, i_idx]
    try:
        auc = float(roc_auc_score(y_test, p))
    except ValueError:
        auc = float("nan")
    brier = float(((p - y_test) ** 2).mean())
    return auc, brier, len(y_test)


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


@app.function(image=image, gpu="L4", memory=131_072, volumes={"/data": volume},
              timeout=12 * 60 * 60)
def run(seed: int = 20261018) -> dict:
    import numpy as np
    import pandas as pd

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
        pruned = _mask_and_prune(mat, rng)
        if pruned is None:
            continue
        m2, train_mask, test_mask = pruned

        for kind in ("1pl", "2pl", "3pl"):
            for d_dim in (1, 2, 4):
                import torch
                theta_fit, b_fit, loga_fit = _fit_core(m2, kind, d_dim, STEPS, int(rng_global.integers(0, 2**31)))
                result = _fit_and_score(m2, train_mask, test_mask, kind, d_dim, STEPS, rng)
                if result is None:
                    continue
                auc, brier, n_test = result
                weights_all[f"{bench}_bayes_{kind}_d{d_dim}"] = {
                    "theta": theta_fit.numpy(), "b": b_fit.numpy(), "log_a": loga_fit.numpy()
                }
                rows.append({"benchmark_id": bench, "model": f"bayes_{kind}_d{d_dim}", "seed": seed,
                            "n_subjects": len(subs), "n_items": len(its),
                            "n_subjects_fit": m2.shape[0], "n_items_fit": m2.shape[1],
                            "n_test_cells": n_test, "auc": auc, "brier": brier})
        print(f"{bench:<20}subs={len(subs):<6}items={len(its):<8}fit={m2.shape}", flush=True)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"{OUT.stem}-seed{seed}{OUT.suffix}")
    frame.to_parquet(out, index=False)
    import torch
    weights_out = OUT.parent / f"bayesian-mirt-d1-d2-d4-filtered-weights-seed{seed}.pt"
    torch.save(weights_all, weights_out)
    volume.commit()
    return {"rows": len(frame), "benchmarks": frame["benchmark_id"].nunique(), "seed": seed, "weights_saved": len(weights_all)}


SEEDS = (20261018, 20261019, 20261020)


@app.local_entrypoint()
def main() -> None:
    handles = [run.spawn(seed) for seed in SEEDS]
    for h in handles:
        print(h.get())
