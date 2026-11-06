"""Freeze a subject holdout, then build replay-compatible ID pools from TRAIN only."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def rank(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()


def split_subjects(pairs, seed):
    held = set()
    for b, group in pairs.groupby("benchmark_id", sort=True):
        ids = sorted(group.subject_uid.unique(), key=lambda s: (rank("subject", seed, b, s), s))
        if len(ids) < 2:
            raise ValueError(f"{b}: need at least two subjects")
        held.update(ids[:max(1, min(len(ids) - 1, int(len(ids) * .2 + .5)))])
    # Remove a held-out ID from every training benchmark, even if it recurs.
    mask = pairs.subject_uid.isin(held)
    fit = pairs.loc[~mask].copy()
    if set(fit.benchmark_id) != set(pairs.benchmark_id):
        raise ValueError("Subject overlap leaves a benchmark without fit subjects")
    return pairs.loc[mask].copy(), fit


def build_panel(responses, held, fit, seed, cap, seeds):
    keys = ["benchmark_id", "subject_uid"]
    if set(held.subject_uid) & set(fit.subject_uid):
        raise ValueError("Held-out subjects also appear in offline fit")
    if responses.duplicated(keys + ["item_uid"]).any():
        raise ValueError("Expected one canonical binary response per task/item")
    known = responses.merge(fit, on=keys)[["benchmark_id", "item_uid"]].drop_duplicates()
    eligible = responses.merge(held, on=keys).merge(known, on=["benchmark_id", "item_uid"])
    tasks, items, orders = [], [], []
    for b, group in eligible.groupby("benchmark_id", sort=True):
        groups = [(s, g) for s, g in group.groupby("subject_uid") if len(g) >= 80]
        for s, g in sorted(groups, key=lambda pair: (rank("panel", seed, b, pair[0]), pair[0]))[:cap]:
            task = f"{s}::{b}"
            g = g.copy()
            g["task_id"] = task
            g["_rank"] = g.item_uid.map(lambda i: rank("pool", seed, task, i))
            g = g.sort_values(["_rank", "item_uid"]).drop(columns="_rank")
            n = (len(g) + 1) // 2
            g["pool"] = ["acquisition"] * n + ["evaluation"] * (len(g) - n)
            items.append(g)
            tasks.append(dict(task_id=task, subject_uid=s, benchmark_id=b, benchmark_split="train",
                              n_source_items=len(g), n_acquisition_items=n, n_evaluation_items=len(g)-n))
            for order_seed in seeds:
                stream = sorted(g.item_uid.iloc[:n], key=lambda i: (rank("stream", seed, order_seed, task, i), i))
                orders.extend(dict(task_id=task, item_uid=i, order_seed=order_seed, stream_position=j)
                              for j, i in enumerate(stream))
    missing = set(held.benchmark_id) - {t["benchmark_id"] for t in tasks}
    if missing:
        raise ValueError(f"No eligible ID tasks with >=80 offline-known items: {sorted(missing)}")
    return pd.DataFrame(tasks), pd.concat(items, ignore_index=True), pd.DataFrame(orders)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--cap-per-benchmark", type=int)
    parser.add_argument("--subjects-only", action="store_true")
    args = parser.parse_args()
    if not args.subjects_only and not (args.cap_per_benchmark and args.cap_per_benchmark > 0):
        parser.error("Provide --cap-per-benchmark for panel construction")
    cols = ["benchmark_id", "subject_uid"]
    columns = cols if args.subjects_only else cols + ["item_uid", "content_hash", "label", "benchmark_split"]
    rows = pd.read_parquet(args.responses, columns=columns, filters=[("benchmark_split", "==", "train")])
    if rows.empty or rows[cols].isna().any().any() or rows.benchmark_id.nunique() != 11:
        raise ValueError("Expected nonempty IDs across exactly 11 training benchmarks")
    pairs = rows[cols].drop_duplicates().sort_values(cols)
    held, fit = split_subjects(pairs, args.seed)
    files = {"heldout_subjects": held, "fit_subjects": fit}
    if not args.subjects_only:
        if not rows.label.isin([0, 1]).all():
            raise ValueError("Expected binary outcomes")
        files.update(zip(("tasks", "items", "candidate_orders"),
                         build_panel(rows, held, fit, args.seed, args.cap_per_benchmark, range(10))))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, frame in files.items():
        frame.to_parquet(args.output / f"{name}.parquet", index=False)
    with args.responses.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = dict(seed=args.seed, response_sha256=sha, input_bytes=args.responses.stat().st_size,
                 cap_per_benchmark=args.cap_per_benchmark, heldout_by_benchmark=held.groupby("benchmark_id").size().to_dict(),
                 fit_by_benchmark=fit.groupby("benchmark_id").size().to_dict(), rows={k: len(v) for k, v in files.items()})
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
