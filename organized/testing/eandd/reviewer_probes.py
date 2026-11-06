"""R9: benchmark-grouped text probes on the matched WP7 OOD items."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from probes import ALPHAS, RANDOM_MODELS_SEEN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
from difficulty_rank_metrics import score


def wp7_sample(t):
    rng = np.random.default_rng(20261102)

    def advance(frame):
        for _, group in frame.groupby("benchmark_id"):
            for _ in range(2):
                rng.integers(0, len(group), 200_000)

    train = t.benchmark_split.eq("train")
    for j in range(8):
        seen = train if j < 6 else t.benchmark_id.isin(RANDOM_MODELS_SEEN)
        advance(t.loc[seen])
        advance(t.loc[~seen])
    for _, g in t.loc[train].groupby("benchmark_id"):
        idx = rng.permutation(g.index.to_numpy())
        advance(t.loc[idx[int(.8 * len(idx)):]])
    fit = np.concatenate([rng.choice(g.index.to_numpy(), min(800, len(g)), replace=False)
                          for _, g in t.loc[train].groupby("benchmark_id")])
    return fit, rng


def choose(x, y, groups, make, options):
    totals = {alpha: [] for alpha in options}
    for fit, val in GroupKFold(5).split(x, y, groups):
        scaler = StandardScaler().fit(x[fit])
        tx, vx = scaler.transform(x[fit]), scaler.transform(x[val])
        for alpha in options:
            pred = make(alpha).fit(tx, y[fit]).predict(vx)
            for b in np.unique(groups[val]):
                mask = groups[val] == b
                rho = spearmanr(y[val][mask], pred[mask]).statistic
                totals[alpha].append(float(rho) if np.isfinite(rho) else 0.)
    return max(options, key=lambda a: np.mean(totals[a]))


def population(responses):
    rows = pd.read_parquet(responses, columns=["item_uid", "benchmark_id", "benchmark_split", "label"])
    t = rows.groupby(["item_uid", "benchmark_id", "benchmark_split"], as_index=False).label.agg(
        observed="mean", n="size")
    t = t.loc[t.n >= 5].reset_index(drop=True)
    if t.item_uid.duplicated().any() or t.groupby("benchmark_split").benchmark_id.nunique().to_dict() != {
        "train": 11, "validation": 3, "test": 3
    }:
        raise ValueError("Expected the WP7 item population")
    return t


def export_keys(responses, panel):
    ood = pd.read_parquet(responses, columns=["item_uid", "benchmark_id", "benchmark_split", "content_hash"])
    ood = ood.loc[ood.benchmark_split.isin(["validation", "test"])].drop_duplicates()
    if panel:
        keys = pd.read_parquet(panel, columns=["benchmark_id", "content_hash"]).drop_duplicates()
        if keys.merge(ood, how="left", on=["benchmark_id", "content_hash"]).item_uid.isna().any():
            raise ValueError("Panel contains keys outside the canonical validation/test responses")
        ood = ood.merge(keys, on=["benchmark_id", "content_hash"], validate="many_to_one")
    return ood


def fetch_matched_embeddings(responses, embeddings, volume, remote_file, export, panel):
    """Stream the public volume file; retain only the exact WP7 rows needed here."""
    import io
    import modal

    t = population(responses)
    fit, _ = wp7_sample(t)
    keep = set(t.item_uid.iloc[fit].astype(str)) | set(t.loc[t.benchmark_split.ne("train"), "item_uid"].astype(str))
    if export:
        keep.update(export_keys(responses, panel).item_uid.astype(str))
    ids = pd.read_parquet(embeddings / "items.parquet", columns=["item_uid"])
    positions = np.flatnonzero(ids.item_uid.astype(str).isin(keep))
    if len(positions) != len(keep):
        raise ValueError("WP7 item IDs did not match the embedding index")
    path = embeddings / "matched_embeddings.npy"
    if path.exists() or (embeddings / "matched_items.parquet").exists():
        raise FileExistsError("Matched embedding output already exists")
    chunks = iter(modal.Volume.from_name(volume).read_file(remote_file))
    head = io.BytesIO(next(chunks))
    if np.lib.format.read_magic(head) != (1, 0):
        raise ValueError("Unexpected NumPy file version")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(head)
    if fortran or shape[0] != len(ids):
        raise ValueError("Embedding file does not match its index")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(positions), shape[1]))
    row_bytes = shape[1] * dtype.itemsize
    data = head.read()
    start, copied = 0, 0
    while True:
        n, tail = divmod(len(data), row_bytes)
        stop = np.searchsorted(positions, start + n)
        if stop > copied:
            block = np.frombuffer(data, dtype=dtype, count=n*shape[1]).reshape(n, shape[1])
            matrix[copied:stop] = block[positions[copied:stop] - start]
            copied = stop
        start += n
        if copied == len(positions):
            break
        data = (data[-tail:] if tail else b"") + next(chunks)
    matrix.flush()
    ids.iloc[positions].to_parquet(embeddings / "matched_items.parquet", index=False)
    print(f"Cached {copied:,} exact vectors without storing the full embedding file", flush=True)


def run(responses, embeddings, rasch_fit, output, export=False, panel=None):
    t = population(responses)
    fit, rng = wp7_sample(t)
    unseen = t.loc[t.benchmark_split.ne("train")].copy()
    unseen["random"] = np.random.default_rng(20261103).random(len(t))[unseen.index]
    matched = (embeddings / "matched_items.parquet").exists()
    ids = pd.read_parquet(embeddings / ("matched_items.parquet" if matched else "items.parquet"),
                          columns=["item_uid"]).item_uid.astype(str)
    lookup = pd.Index(ids)
    pos = lookup.get_indexer(t.item_uid.astype(str))
    if (pos[fit] < 0).any() or (pos[unseen.index] < 0).any():
        raise ValueError("Missing WP7 embeddings")
    vec = np.load(embeddings / ("matched_embeddings.npy" if matched else "item_embeddings.npy"), mmap_mode="r")
    x, z = np.asarray(vec[pos[fit]], np.float32), np.asarray(vec[pos[unseen.index]], np.float32)
    groups = t.benchmark_id.to_numpy()[fit]
    targets = {"pass_rate": t.observed.to_numpy()[fit]}
    if rasch_fit:
        rasch = pd.read_parquet(rasch_fit)
        difficulty = rasch.loc[rasch.kind.eq("b")].set_index("key").value
        targets["rasch_easiness"] = -difficulty.reindex(t.item_uid.iloc[fit]).to_numpy()
    models = {
        "ridge": (lambda a: Ridge(alpha=a), ALPHAS),
        "mlp": (lambda a: MLPRegressor(hidden_layer_sizes=(32,), alpha=a, random_state=20261106,
                                     batch_size=256, max_iter=100), [.001, .01, .1]),
    }
    selected, scores = {}, [score(unseen, "random", copy.deepcopy(rng)).assign(arm="random")]
    export_models = {}
    for target, raw in targets.items():
        valid = np.isfinite(raw)
        tx, yy, gg = x[valid], raw[valid], groups[valid]
        if len(np.unique(gg)) != 11:
            raise ValueError(f"Target {target} missing a training benchmark")
        centered = yy - pd.Series(yy).groupby(gg).transform("mean").to_numpy()
        scaler = StandardScaler().fit(tx)
        for name, (make, options) in models.items():
            arm = f"{target}_{name}"
            best = choose(tx, centered, gg, make, options)
            model = make(best).fit(scaler.transform(tx), centered)
            unseen[arm] = model.predict(scaler.transform(z))
            if export and target == "rasch_easiness":
                export_models[arm] = (scaler, model)
            scores.append(score(unseen, arm, copy.deepcopy(rng)).assign(arm=arm))
            selected[arm] = {"alpha": best, "fit_items": int(valid.sum()), "wp7_sample": len(fit)}
            print(arm, selected[arm], flush=True)
    by = pd.concat(scores, ignore_index=True)
    by["role"] = by.benchmark_id.map(unseen.groupby("benchmark_id").benchmark_split.first())
    summary = by.groupby(["role", "arm"], as_index=False)[["rho", "pairwise_acc"]].mean()
    random_rho = by.loc[by.arm.eq("random"), "rho"].mean()
    criterion = []
    for arm, g in by.loc[by.arm.ne("random")].groupby("arm"):
        gap = g.rho.mean() - random_rho
        positive = int(g.rho.gt(0).sum())
        criterion.append(dict(arm=arm, macro_rho=g.rho.mean(), random_rho=random_rho,
                              gap=gap, positive=positive, passes=bool(gap >= .05 and positive >= 4)))
    output.mkdir(parents=True, exist_ok=False)
    by.to_csv(output / "wp13-probe-per-benchmark.csv", index=False)
    summary.to_csv(output / "wp13-probe-summary.csv", index=False)
    (output / "wp13-probe-criterion.json").write_text(json.dumps(dict(selection=selected, criterion=criterion), indent=2))
    for row in criterion:
        arm = row["arm"]
        if not row["passes"] or arm not in export_models:
            continue
        scaler, model = export_models[arm]
        keys = export_keys(responses, panel)
        indices = lookup.get_indexer(keys.item_uid.astype(str))
        if (indices < 0).any():
            raise ValueError("Embedding cache is missing export keys; fetch with --export")
        # Training predicts negative b (easiness) for ranking; R5 consumes b.
        keys["difficulty"] = np.concatenate([
            -model.predict(scaler.transform(np.asarray(vec[indices[i:i+2048]], np.float32)))
            for i in range(0, len(indices), 2048)])
        table = keys.groupby(["benchmark_id", "content_hash"], as_index=False).difficulty.mean()
        if not np.isfinite(table.difficulty).all() or table.empty:
            raise ValueError("Invalid exported difficulties")
        table.to_parquet(output / f"r5-{arm}-item-difficulties.parquet", index=False)
        print(f"{arm}: exported {len(table):,} unique benchmark/hash pairs", flush=True)
    print(pd.DataFrame(criterion).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", required=True, type=Path)
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--rasch-fit", type=Path)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--fetch-volume")
    p.add_argument("--remote-file", default="/datasets/full/allbench-embed-bge_m3_long-v2/item_embeddings.npy")
    p.add_argument("--export", action="store_true", help="Export each passing Rasch arm")
    p.add_argument("--panel", type=Path, help="Restrict export to exact validation/test panel keys")
    a = p.parse_args()
    if a.fetch_volume:
        fetch_matched_embeddings(a.responses, a.embeddings, a.fetch_volume, a.remote_file, a.export, a.panel)
    else:
        with threadpool_limits(limits=2):
            run(a.responses, a.embeddings, a.rasch_fit, a.output, a.export, a.panel)
