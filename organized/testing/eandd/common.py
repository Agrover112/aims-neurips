#!/usr/bin/env python3
"""Shared contract for the offline-vs-online paper: paths, frozen draws, and how every run is saved.

    python organized/testing/eandd/common.py        # freeze draws.parquet (refuses to overwrite)

Large outputs live in $AIMS_EANDD_DATA (default ~/aims-replay-cache/paper-eandd), not in git:

    draws.parquet          shape, role, draw, task_id, order_seed
    scores/<run>.parquet   run, model, shape, role, draw, task_id, order_seed, budget, brier
    preds/<run>/*.parquet  model, draw, task_id, order_seed, budget, content_hash, p (float32), y (int8)
    fits/<run>.parquet     kind (mu, theta, b, hyper), key, value
    runs/<run>.json        config, seeds, git commit, input sizes, runtime

An item is identified by content_hash, the key the online simulation matches items on.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HOME = Path.home()
DATA = Path(os.environ.get("AIMS_EANDD_DATA", HOME / "aims-replay-cache" / "paper-eandd"))
CACHE = Path(os.environ.get("AIMS_REPLAY_CACHE", HOME / "aims-replay-cache"))
EMB = Path(os.environ.get("AIMS_EMBEDDINGS", HOME / "aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2"))
RESPONSES = Path(os.environ.get("AIMS_RESPONSES", HOME / "aims-noitem-run/filtered-responses.parquet"))
SUMMARY = ROOT / "paper" / "eandd" / "data"     # small CSVs, committed

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
SHAPES = ("single-pair", "multi-pair", "competition")
ROLES = ("validation", "test")
ORDERS_PER_TASK = {"validation": 3, "test": 10}
DRAW_COLUMNS = ["shape", "role", "draw", "task_id", "order_seed"]


def one_per_benchmark(role: str, n: int, rng, items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Competition-shaped draws: exactly one subject per benchmark. Same recipe as codabench_shape_replay."""
    tasks = items[items.benchmark_split == role].drop_duplicates("task_id")
    seeds = sorted(orders.order_seed.unique())[:ORDERS_PER_TASK[role]]
    by_bench = {b: g.task_id.tolist() for b, g in tasks.groupby("benchmark_id")}
    rows = [(d, str(rng.choice(ts)), int(rng.choice(seeds))) for d in range(n) for ts in by_bench.values()]
    return pd.DataFrame(rows, columns=["draw", "task_id", "order_seed"]).assign(shape="competition", role=role)


def build_draws() -> pd.DataFrame:
    items = pd.read_parquet(CACHE / "panel" / "items.parquet")
    orders = pd.read_parquet(CACHE / "panel" / "candidate_orders.parquet")
    frames = []
    # Multi-pair: the draws behind the existing numbers, so every row stays paired with them.
    curves = pd.read_parquet(CACHE / "curves.parquet")
    curves = curves[curves.model == "jeffreys"].drop_duplicates(["role", "draw", "task_id"])
    frames.append(curves[["role", "draw", "task_id", "order_seed"]].assign(shape="multi-pair"))
    # Competition-shaped: validation before test on this generator reproduces the existing draws exactly.
    rng = np.random.default_rng(20261101)
    frames.append(one_per_benchmark("validation", 120, rng, items, orders))
    frames.append(one_per_benchmark("test", 150, rng, items, orders))
    # Single-pair: every task under every frozen order, each its own draw.
    seeds = sorted(orders.order_seed.unique())
    for role in ROLES:
        tasks = sorted(items[items.benchmark_split == role].task_id.unique())
        pairs = [(t, int(s)) for t in tasks for s in seeds[:ORDERS_PER_TASK[role]]]
        frames.append(pd.DataFrame([(i, t, s) for i, (t, s) in enumerate(pairs)],
                                   columns=["draw", "task_id", "order_seed"]).assign(shape="single-pair", role=role))
    d = pd.concat(frames, ignore_index=True)[DRAW_COLUMNS]
    d["draw"] = d.draw.astype(int)
    d["order_seed"] = d.order_seed.astype(int)
    return d


def load_draws(shape: str, role: str) -> dict:
    d = pd.read_parquet(DATA / "draws.parquet")
    d = d[(d["shape"] == shape) & (d.role == role)]
    return {int(i): list(zip(g.task_id, g.order_seed.astype(int))) for i, g in d.groupby("draw")}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_run(run: str, config: dict, started: float, scores: pd.DataFrame | None = None, fits: pd.DataFrame | None = None) -> None:
    """Write scores/fits and the run record. Refuses to overwrite: a new result gets a new run name."""
    for sub in ("scores", "fits", "runs", "preds"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    record = DATA / "runs" / f"{run}.json"
    if record.exists():
        raise FileExistsError(f"{record} exists; use a new run name")
    if scores is not None:
        scores.assign(run=run).to_parquet(DATA / "scores" / f"{run}.parquet", index=False)
    if fits is not None:
        fits.to_parquet(DATA / "fits" / f"{run}.parquet", index=False)
    inputs = {str(p): p.stat().st_size for p in (RESPONSES, CACHE / "panel" / "items.parquet",
                                                 CACHE / "panel" / "candidate_orders.parquet", DATA / "draws.parquet") if p.exists()}
    record.write_text(json.dumps({"run": run, "config": config, "git_commit": git_commit(), "inputs": inputs,
                                  "runtime_seconds": round(time.time() - started, 1),
                                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2, default=str))


def main() -> int:
    path = DATA / "draws.parquet"
    if path.exists():
        print(f"{path} exists; draws are frozen")
    else:
        DATA.mkdir(parents=True, exist_ok=True)
        build_draws().to_parquet(path, index=False)
        print("wrote", path)
    d = pd.read_parquet(path)
    print(d.groupby(["shape", "role"]).agg(draws=("draw", "nunique"), tasks=("task_id", "nunique"), rows=("draw", "size")).to_string())
    # The competition-shaped draws must match the ones behind the existing numbers.
    old_path = ROOT / "organized/testing/online/results/codabench-shape-test.parquet"
    if not old_path.exists():   # a gitignored result: only the person who made the draws has it
        print("existing competition-shaped results not found; skipping the draw-equality check")
        return 0
    old = pd.read_parquet(old_path)
    old = old[(old["shape"] == "one subject per benchmark") & (old.model == "beta") & (old.budget == 0)]
    new = d[(d["shape"] == "competition") & (d.role == "test")]
    a = set(zip(old.draw, old.task_id, old.order_seed)); b = set(zip(new.draw, new.task_id, new.order_seed))
    print("competition test draws identical to existing:", a == b, len(a), len(b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
