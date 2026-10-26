"""Row-holdout and column-holdout generalization, classic IRT, n>=30 filter.

Section 5.1 of the Reliable-and-Amortized paper: for a held-out subject, split
its responses into two random disjoint halves, fit ability on half 1 using the
already-calibrated item bank, score on half 2. This script reproduces that for
subjects (rows) exactly, and applies the identical split-half mechanic to
items (columns) -- not described in the paper, which only tests item
generalization via a text-based difficulty predictor (their Section 5.2, not
applicable here since this is a no-text sanity check).

Per benchmark (n>=30 filtered matrix, same filters as
modal_classic_irt_filter_test.py):
  1. Hold out 20% of subjects (rows) and, separately, 20% of items (columns).
  2. Fit (theta, a, b) by joint MLE on the CORE block: training subjects x
     training items only. Held-out rows/columns contribute nothing to this fit.
  3. Subject generalization: for each held-out subject, split its responses
     (on training items) into two random halves. Estimate its theta from half
     1 with item params frozen, score AUC/Brier on half 2.
  4. Item generalization: for each held-out item, split its responses (from
     training subjects) into two random halves. Estimate its (a, b) from half
     1 with subject thetas frozen, score AUC/Brier on half 2.

Run:  modal run --detach modal_classic_irt_generalization.py::main
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-classic-irt-generalization"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT = Path("/data/results/data-analysis/classic-irt-generalization.parquet")

MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HELD_OUT_FRAC = 0.20
STEPS = 400
MIN_MATRIX = 20
MIN_HALF = 4  # need at least this many responses in each half to fit/score

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")


def _bce(z, y, c=None):
    """Binary cross-entropy on logits, or on the 3PL guessing mixture."""
    import torch

    if c is None:
        return torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    p = (c + (1.0 - c) * torch.sigmoid(z)).clamp(1e-6, 1 - 1e-6)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def _guess(logit_c, kind):
    """3PL guessing probability, capped at 0.5; None for 1PL/2PL."""
    import torch

    return torch.sigmoid(logit_c) * 0.5 if kind == "3pl" else None


def _fit_core(mat, kind, steps, seed):
    """Joint MLE on every non-NaN cell of ``mat``. Returns theta, b, log_a."""
    import numpy as np
    import torch

    s_idx, i_idx = np.nonzero(~np.isnan(mat))
    y = torch.as_tensor(mat[s_idx, i_idx], dtype=torch.float32)
    s_t = torch.as_tensor(s_idx, dtype=torch.long)
    i_t = torch.as_tensor(i_idx, dtype=torch.long)
    n_s, n_i = mat.shape

    torch.manual_seed(seed)
    theta = torch.zeros(n_s, requires_grad=True)
    b = torch.zeros(n_i, requires_grad=True)
    log_a = torch.zeros(n_i, requires_grad=True)
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
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i)
        z = a[i_t] * (theta[s_t] - b[i_t])
        c = _guess(logit_c, kind)
        loss = _bce(z, y, None if c is None else c[i_t])
        loss = loss + 0.125 * (theta.square().mean() + b.square().mean())
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu().numpy()))
    return theta.detach(), b.detach(), log_a.detach(), logit_c.detach(), loss_history


def _fit_one_theta(a_known, b_known, y_half, c_known=None, steps=200):
    """New subject: fit a single theta against frozen (a, b) on revealed cells."""
    import torch

    theta = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        z = a_known * (theta - b_known)
        loss = _bce(z, y_half, c_known)
        loss = loss + 0.25 * theta.square().mean()
        loss.backward()
        opt.step()
    return theta.detach()


def _fit_one_item(theta_known, y_half, kind, steps=200):
    """New item: fit (a, b) against frozen theta on revealed cells."""
    import torch

    b = torch.zeros(1, requires_grad=True)
    log_a = torch.zeros(1, requires_grad=True)
    logit_c = torch.full((1,), -2.0, requires_grad=True)
    params = [b]
    if kind in ("2pl", "3pl"):
        params.append(log_a)
    if kind == "3pl":
        params.append(logit_c)
    opt = torch.optim.AdamW(params, lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(1)
        z = a * (theta_known - b)
        loss = _bce(z, y_half, _guess(logit_c, kind))
        loss = loss + 0.25 * b.square().mean()
        loss.backward()
        opt.step()
    a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(1)
    return a.detach(), b.detach(), logit_c.detach()


@app.function(image=image, cpu=16, memory=131_072, volumes={"/data": volume},
              timeout=6 * 60 * 60)
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
    print(f"filtered: {len(d):,} rows, {d.benchmark_id.nunique()} benchmarks", flush=True)

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
            theta_core, b_core, loga_core, logitc_core, loss_history = _fit_core(
                core, kind, STEPS, int(rng_global.integers(0, 2**31)))
            a_core = torch.exp(loga_core) if kind in ("2pl", "3pl") else torch.ones(len(train_i))
            c_core = _guess(logitc_core, kind)
            weights_all[f"{bench}_{kind}"] = {
                "theta": theta_core.numpy(), "b": b_core.numpy(), "log_a": loga_core.numpy(),
                "logit_c": logitc_core.numpy(), "loss_history": loss_history,
                "train_subjects": train_s, "train_items": train_i, "held_subjects": held_s, "held_items": held_i
            }

            # --- subject generalization: held-out rows, split-half on train items ---
            for row in held_s:
                cols = np.flatnonzero(~np.isnan(mat[row, train_i]))
                if len(cols) < 2 * MIN_HALF:
                    continue
                perm = rng.permutation(cols)
                half1, half2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
                y1 = torch.as_tensor(mat[row, train_i][half1], dtype=torch.float32)
                theta_new = _fit_one_theta(
                    a_core[half1], b_core[half1], y1,
                    None if c_core is None else c_core[half1])
                with torch.no_grad():
                    z2 = a_core[half2] * (theta_new - b_core[half2])
                    p2 = torch.sigmoid(z2)
                    if c_core is not None:
                        p2 = c_core[half2] + (1 - c_core[half2]) * p2
                    p2 = p2.numpy()
                y2 = mat[row, train_i][half2]
                if len(np.unique(y2)) < 2:
                    continue
                rows.append({"benchmark_id": bench, "model": kind, "seed": seed, "axis": "subject",
                            "n_fit": len(half1), "n_score": len(half2),
                            "label": y2, "pred_prob": p2,
                            "auc": float(roc_auc_score(y2, p2)),
                            "brier": float(np.mean((p2 - y2) ** 2))})

            # --- item generalization: held-out columns, split-half on train subjects ---
            for col in held_i:
                rows_idx = np.flatnonzero(~np.isnan(mat[train_s, col]))
                if len(rows_idx) < 2 * MIN_HALF:
                    continue
                perm = rng.permutation(rows_idx)
                half1, half2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
                y1 = torch.as_tensor(mat[train_s, col][half1], dtype=torch.float32)
                a_new, b_new, logitc_new = _fit_one_item(theta_core[half1], y1, kind)
                with torch.no_grad():
                    z2 = a_new * (theta_core[half2] - b_new)
                    p2 = torch.sigmoid(z2)
                    c_new = _guess(logitc_new, kind)
                    if c_new is not None:
                        p2 = c_new + (1 - c_new) * p2
                    p2 = p2.numpy()
                y2 = mat[train_s, col][half2]
                if len(np.unique(y2)) < 2:
                    continue
                rows.append({"benchmark_id": bench, "model": kind, "seed": seed, "axis": "item",
                            "n_fit": len(half1), "n_score": len(half2),
                            "label": y2, "pred_prob": p2,
                            "auc": float(roc_auc_score(y2, p2)),
                            "brier": float(np.mean((p2 - y2) ** 2))})
        print(f"{bench:<20}subs={len(subs):<5}items={len(its):<6}"
              f"held_s={len(held_s)}  held_i={len(held_i)}", flush=True)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"{OUT.stem}-seed{seed}{OUT.suffix}")
    frame.to_parquet(out, index=False)
    weights_out = OUT.parent / f"classic-irt-generalization-weights-seed{seed}.pt"
    torch.save(weights_all, weights_out)
    volume.commit()
    return {"rows": len(frame), "benchmarks": frame["benchmark_id"].nunique(), "seed": seed, "weights_saved": len(weights_all)}


SEEDS = (20261018, 20261019, 20261020)


@app.local_entrypoint()
def main() -> None:
    handles = [run.spawn(seed) for seed in SEEDS]
    for h in handles:
        print(h.get())
