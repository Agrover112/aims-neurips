"""Classic 1PL/2PL/3PL (no text) before vs after filtering, per benchmark.

Filters (applied in order):
  1. Drop duplicate questions (exact content_hash duplicates within a benchmark).
  2. Drop questions with identical responses (zero-variance items: everyone right
     or everyone wrong -- no discriminative power).
  3. Drop items answered by fewer than MIN_SUBJECTS (15) distinct subjects.
  4. Keep only subjects who answered >=MIN_ITEMS_PER_TASK (80) items in that
     benchmark, post item-filtering (this is the competition's own pair-eligibility
     rule, applied here directly on the raw response matrix).

Evaluation protocol follows the Reliable-and-Amortized paper's classic-IRT setup:
mask a random 20% of answered cells as held-out, fit free-parameter theta/item
params by MLE on the remaining 80%, then drop any row or column that is now
all-identical in the training cells (numerical stability), refit, and score AUC
+ Brier on the held-out 20%. No text/embeddings used -- every item and subject
parameter is a free scalar, fit directly from the response matrix.

Run:  modal run --detach modal_classic_irt_filter_test.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-classic-irt-filter"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT = Path("/data/results/data-analysis/classic-irt-filter-test.parquet")

MIN_SUBJECTS = 15
MIN_ITEMS_PER_TASK = 80
HOLDOUT_FRAC = 0.20
SEED = 20261026
STEPS = 400
MIN_MATRIX = 20  # skip benchmarks too small to fit at all

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")


def _fit_and_score(mat, mask_train, mask_test, kind, steps, rng):
    """mat: (S, I) float array of 0/1 (NaN elsewhere). Returns (auc, brier) or None."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    s_idx, i_idx = np.nonzero(mask_train)
    y_train = torch.as_tensor(mat[s_idx, i_idx], dtype=torch.float32)
    s_train = torch.as_tensor(s_idx, dtype=torch.long)
    i_train = torch.as_tensor(i_idx, dtype=torch.long)
    n_s, n_i = mat.shape

    torch.manual_seed(int(rng.integers(0, 2**31)))
    theta = torch.zeros(n_s, requires_grad=True)
    b = torch.zeros(n_i, requires_grad=True)
    params = [theta, b]
    log_a = torch.zeros(n_i, requires_grad=True)
    logit_c = torch.full((n_i,), -2.0, requires_grad=True)
    if kind in ("2pl", "3pl"):
        params.append(log_a)
    if kind == "3pl":
        params.append(logit_c)

    opt = torch.optim.AdamW(params, lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i)
        z = a[i_train] * (theta[s_train] - b[i_train])
        if kind == "3pl":
            c = torch.sigmoid(logit_c) * 0.5
            p = c[i_train] + (1 - c[i_train]) * torch.sigmoid(z)
            p = p.clamp(1e-6, 1 - 1e-6)
            loss = -(y_train * torch.log(p) + (1 - y_train) * torch.log(1 - p)).mean()
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_train)
        # weak priors for numerical stability, matching the repo's tau^2=4 convention
        loss = loss + 0.125 * (theta.square().mean() + b.square().mean())
        loss.backward()
        opt.step()

    s_idx, i_idx = np.nonzero(mask_test)
    if len(s_idx) < 10 or len(np.unique(mat[s_idx, i_idx])) < 2:
        return None
    with torch.no_grad():
        a = torch.exp(log_a) if kind in ("2pl", "3pl") else torch.ones(n_i)
        z = a[i_idx] * (theta[torch.as_tensor(s_idx)] - b[i_idx])
        if kind == "3pl":
            c = torch.sigmoid(logit_c) * 0.5
            p = (c[i_idx] + (1 - c[i_idx]) * torch.sigmoid(z)).numpy()
        else:
            p = torch.sigmoid(z).numpy()
    y_test = mat[s_idx, i_idx]
    try:
        auc = float(roc_auc_score(y_test, p))
    except ValueError:
        auc = float("nan")
    brier = float(((p - y_test) ** 2).mean())
    return auc, brier, len(y_test)


def _mask_and_prune(mat, rng):
    """Random 20% cell holdout, then drop rows/cols all-identical in train."""
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


@app.function(image=image, cpu=16, memory=131_072, volumes={"/data": volume},
              timeout=6 * 60 * 60)
def run(seed: int = SEED, min_subjects: int = MIN_SUBJECTS) -> dict:
    import numpy as np
    import pandas as pd

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    items_meta = pd.read_parquet(PROCESSED / "items.parquet",
                                 columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")
    print(f"observations (deduped subject x item): {len(obs):,}", flush=True)

    rows = []
    for bench, d in obs.groupby("benchmark_id"):
        for stage in ("before", "after"):
            dd = d.copy()
            if stage == "after":
                dd = dd.merge(items_meta, on="item_uid", how="left")
                dd = dd.drop_duplicates(["content_hash", "subject_uid"])  # dedupe questions
                var = dd.groupby("item_uid")["label"].agg(["mean", "size"])
                keep_items = var[(var["mean"] > 0) & (var["mean"] < 1)].index  # drop identical-response items
                dd = dd[dd["item_uid"].isin(keep_items)]
                n_sub_per_item = dd.groupby("item_uid")["subject_uid"].nunique()
                dd = dd[dd["item_uid"].isin(n_sub_per_item[n_sub_per_item >= min_subjects].index)]
                n_item_per_task = dd.groupby("subject_uid")["item_uid"].nunique()
                dd = dd[dd["subject_uid"].isin(n_item_per_task[n_item_per_task >= MIN_ITEMS_PER_TASK].index)]
            if len(dd) < MIN_MATRIX:
                continue
            subs = sorted(dd["subject_uid"].unique())
            its = sorted(dd["item_uid"].unique())
            if len(subs) < 5 or len(its) < 5:
                continue
            s_map = {s: i for i, s in enumerate(subs)}
            i_map = {s: i for i, s in enumerate(its)}
            mat = np.full((len(subs), len(its)), np.nan, dtype=np.float32)
            mat[dd["subject_uid"].map(s_map).to_numpy(), dd["item_uid"].map(i_map).to_numpy()] = \
                dd["label"].to_numpy()

            rng = np.random.default_rng(seed)
            pruned = _mask_and_prune(mat, rng)
            if pruned is None:
                continue
            m2, train_mask, test_mask = pruned
            for kind in ("1pl", "2pl", "3pl"):
                result = _fit_and_score(m2, train_mask, test_mask, kind, STEPS, rng)
                if result is None:
                    continue
                auc, brier, n_test = result
                stage_tag = stage if stage == "before" else f"after_n{min_subjects}"
                rows.append({"benchmark_id": bench, "stage": stage_tag, "model": kind, "seed": seed,
                            "min_subjects": min_subjects,
                            "n_subjects": len(subs), "n_items": len(its),
                            "n_subjects_fit": m2.shape[0], "n_items_fit": m2.shape[1],
                            "n_test_cells": n_test, "auc": auc, "brier": brier})
            print(f"{bench:<20}{stage:<8}subs={len(subs):<6}items={len(its):<8}"
                  f"fit={m2.shape}", flush=True)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"{OUT.stem}-seed{seed}-n{min_subjects}{OUT.suffix}")
    frame.to_parquet(out, index=False)
    volume.commit()
    return {"rows": len(frame), "benchmarks": frame["benchmark_id"].nunique(), "seed": seed,
            "min_subjects": min_subjects}


SEEDS = (20261018, 20261019, 20261020)


@app.local_entrypoint()
def main(min_subjects: int = MIN_SUBJECTS) -> None:
    handles = [run.spawn(seed, min_subjects) for seed in SEEDS]
    for h in handles:
        print(h.get())
