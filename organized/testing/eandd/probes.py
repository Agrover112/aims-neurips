"""WP7: reproduce the historical item-embedding probe before matched controls."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ALPHAS = [.01, .1, 1., 10., 100.]
RANDOM_MODELS_SEEN = {"afrieval", "ai2d_test", "alignment_faking", "bird_sql",
                      "bridging_gap", "editbench", "hallusionbench"}


def run(responses, embeddings, item_metadata, output, score):
    rows = pd.read_parquet(responses, columns=["item_uid", "benchmark_id", "benchmark_split", "label"])
    t = rows.groupby(["item_uid", "benchmark_id", "benchmark_split"], as_index=False).label.agg(
        observed="mean", n="size")
    t = t.loc[t.n >= 5].reset_index(drop=True)
    if t.item_uid.duplicated().any() or t.groupby("benchmark_split").benchmark_id.nunique().to_dict() != {
        "train": 11, "validation": 3, "test": 3
    }:
        raise ValueError("Input differs from the canonical benchmark-holdout item set")
    ids = pd.read_parquet(embeddings / "items.parquet", columns=["item_uid"]).item_uid.astype(str)
    pos = {u: i for i, u in enumerate(ids)}
    store = np.load(embeddings / "item_embeddings.npy", mmap_mode="r")
    positions = np.array([pos[u] for u in t.item_uid.astype(str)])
    y = (t["observed"] - t.groupby("benchmark_id")["observed"].transform("mean")).to_numpy()
    train = t.benchmark_split.eq("train")
    rng = np.random.default_rng(20261102)

    def consume_pair_draws(frame):
        for _, g in frame.groupby("benchmark_id"):
            rng.integers(0, len(g), 200_000)
            rng.integers(0, len(g), 200_000)

    # Legacy RNG was shared with eight ANP metrics. Its draws depend on the
    # row counts/order, not predictions; advance it without loading ANPs.
    for j in range(8):
        seen = train if j < 6 else t.benchmark_id.isin(RANDOM_MODELS_SEEN)
        consume_pair_draws(t.loc[seen])
        consume_pair_draws(t.loc[~seen])
    id_splits = []
    for _, g in t.loc[train].groupby("benchmark_id"):
        idx = rng.permutation(g.index.to_numpy())
        cut = int(.8 * len(idx))
        id_splits.append((idx[:cut], idx[cut:]))
        consume_pair_draws(t.loc[idx[cut:]])
    fit_idx = np.concatenate([rng.choice(g.index.to_numpy(), min(800, len(g)), replace=False)
                              for _, g in t.loc[train].groupby("benchmark_id")])
    unseen = t.loc[~train].copy()

    def vectors(idx):
        return np.asarray(store[positions[idx]], dtype=np.float32)

    model = RidgeCV(alphas=ALPHAS).fit(vectors(fit_idx), y[fit_idx])
    unseen["embedding"] = model.predict(vectors(unseen.index.to_numpy()))
    legacy_rng = copy.deepcopy(rng)
    reproduced = score(unseen, "embedding", copy.deepcopy(legacy_rng))
    legacy = reproduced[["rho", "pairwise_acc"]].mean().to_dict()
    output.mkdir(parents=True, exist_ok=False)
    (output / "reproduction.json").write_text(json.dumps(legacy, indent=2))
    if [round(legacy[k], 2) for k in ("rho", "pairwise_acc")] != [.09, .53]:
        raise ValueError(f"Legacy probe did not reproduce; controls not run: {legacy}")

    length = pd.read_parquet(item_metadata, columns=["item_uid", "content_chars"])
    length = length.drop_duplicates("item_uid").set_index("item_uid").content_chars
    chars = length.reindex(t.item_uid).to_numpy(dtype=float)
    if not np.isfinite(chars).all() or np.any(chars < 0):
        raise ValueError("Missing/invalid character counts; do not change the matched item set")
    xlen = np.log1p(chars)[:, None]
    random = np.random.default_rng(20261103).random(len(t))
    length_model = RidgeCV(alphas=ALPHAS).fit(xlen[fit_idx], y[fit_idx])
    unseen["length"] = length_model.predict(xlen[unseen.index])
    unseen["random"] = random[unseen.index]
    all_rows = []
    for feature in ("random", "length", "embedding"):
        out = score(unseen, feature, copy.deepcopy(legacy_rng)).assign(feature=feature, setting="OOD")
        role = unseen.groupby("benchmark_id").benchmark_split.first()
        all_rows.append(out.assign(role=out.benchmark_id.map(role)))
    for fit, hold in id_splits:
        eval_frame = t.loc[hold].copy()
        for feature, design in (("length", xlen), ("embedding", None), ("random", None)):
            if feature == "random":
                eval_frame[feature] = random[hold]
            else:
                x, z = (vectors(fit), vectors(hold)) if design is None else (design[fit], design[hold])
                eval_frame[feature] = RidgeCV(alphas=ALPHAS).fit(x, y[fit]).predict(z)
            all_rows.append(score(eval_frame, feature, np.random.default_rng(20261104)).assign(
                feature=feature, setting="ID", role="train"))
    per_benchmark = pd.concat(all_rows, ignore_index=True)
    summary = per_benchmark.groupby(["setting", "role", "feature"], as_index=False).agg(
        benchmarks=("benchmark_id", "size"), rho=("rho", "mean"), pairwise_acc=("pairwise_acc", "mean"))
    per_benchmark.to_csv(output / "per-benchmark.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    config = dict(legacy=legacy, embedding_alpha=float(model.alpha_), length_alpha=float(length_model.alpha_),
                  train_sample_items=len(fit_idx), item_count=len(t), ridge_alphas=ALPHAS,
                  inputs={str(p): p.stat().st_size for p in
                          (responses, embeddings / "item_embeddings.npy", embeddings / "items.parquet", item_metadata)})
    with responses.open("rb") as f:
        config["response_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    return {"legacy": legacy, "summary": summary.to_dict("records")}
