"""Dump what the best models predict, to compare with the raw data.

Two products per model (all CPU, inference only, seed 20261018, v4.1 store):

1. ``fig11``: for every validation/test task, acquisition order and budget, the
   revealed-label average (the classical test theory, CTT, estimate of the
   subject's accuracy), the model's implied accuracy (mean predicted
   probability over the evaluation pool), and the true evaluation-pool
   accuracy. This is the quantity behind Figure 11 of the amortized-evaluation
   paper: does an estimate from a random subset recover the true level?
2. ``items``: the model's predicted probability for individual items, at budget
   0 (item information only) and budget 31 (order 0), for train, validation and
   test items, so the distribution of predicted item pass rate can be put next
   to the observed one.

Probabilities are RAW (uncalibrated); calibration is a separate post-hoc step.

Run:  modal run --detach modal_prediction_dump.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-prediction-dump"
VOLUME_NAME = "aims-neurips-data"
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_ROOT = Path("/data/datasets/full")
ABLATION_ROOT = Path("/data/results/r-c-ablation")
OUT_ROOT = Path("/data/results/model-vs-data")
CORPUS = "allbench-train-me5_instruct-cap128-v2"
SEED = 20261018
RUNGS = ("cnp_ln_gelu", "cnp_gelu", "bcirt_d4_gelu_all")
BUDGETS = (0, 1, 3, 7, 15, 31)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "modal>=0.64")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .add_local_file(PROJECT_ROOT / "modal_r_c_ablation.py", "/opt/aims/modal_r_c_ablation.py",
                    copy=True)
    .env({"PYTHONPATH": "/opt/aims/src:/opt/aims"})
)


@app.function(image=image, cpu=8, memory=65_536, volumes={"/data": volume},
              timeout=6 * 60 * 60)
def run(rung: str) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    import modal_r_c_ablation as M

    root = EMBEDDING_ROOT / CORPUS
    item_emb = np.load(root / "item_embeddings.npy")
    subj_emb = np.load(root / "subject_embeddings.npy")
    item_uid = pd.read_parquet(root / "items.parquet")["item_uid"].astype(str).to_numpy()
    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")

    item_dim = int(item_emb.shape[1])
    prior_slice = (item_dim, item_dim + int(subj_emb.shape[1]))
    model = M._make_model(rung, item_dim + int(subj_emb.shape[1]), prior_slice, prior_slice)
    ckpt = ABLATION_ROOT / f"seed-{SEED}-masked-{CORPUS}-v41e1" / f"{rung}-checkpoints" / "model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    def feats(rows):
        return torch.as_tensor(M._features(rows, item_emb, subj_emb))

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    fig11, items = [], []

    # ---- validation / test: orders x budgets ---------------------------------
    for split in ("validation", "test"):
        part = evaluation[evaluation["split"] == split]
        for task_id in sorted(part["task_id"].astype(str).unique()):
            rows = part[part["task_id"].astype(str) == task_id]
            bench = str(rows["benchmark_id"].iloc[0])
            tgt = rows[rows["pool"] == "evaluation"].reset_index(drop=True)
            y_t = tgt["label"].to_numpy("float32")
            x_t = feats(tgt)
            lookup = rows.set_index("item_index")
            for order_seed in range(10):
                o = (order_frame[(order_frame["task_id"].astype(str) == task_id)
                                 & (order_frame["order_seed"] == order_seed)]
                     .sort_values("acquisition_rank").reset_index(drop=True))
                if o.empty:
                    continue
                c = lookup.loc[o["item_index"].to_numpy()].reset_index()
                x_c = feats(c)
                y_c = c["label"].to_numpy("float32")
                for budget in BUDGETS:
                    with torch.no_grad():
                        z = model(x_c[:budget], torch.as_tensor(y_c[:budget]), x_t).numpy()
                    p = sig(z.astype(float))
                    fig11.append({
                        "split": split, "benchmark_id": bench, "task_id": task_id,
                        "order_seed": order_seed, "budget": budget,
                        "ctt": float(y_c[:budget].mean()) if budget else float("nan"),
                        "model": float(p.mean()), "truth": float(y_t.mean()),
                        "n_eval": len(y_t)})
                    if order_seed == 0 and budget in (0, 31):
                        items.append(pd.DataFrame({
                            "split": split, "benchmark_id": bench, "task_id": task_id,
                            "budget": budget,
                            "item_uid": item_uid[tgt["item_index"].to_numpy()],
                            "p": p.astype("float32"), "y": y_t}))
        print(f"{rung}: {split} done", flush=True)

    # ---- train: budget 0 only (item information only) -----------------------
    for task_id, rows in train.groupby("task_id", sort=True):
        with torch.no_grad():
            z = model(feats(rows)[:0], torch.zeros(0), feats(rows)).numpy()
        items.append(pd.DataFrame({
            "split": "train", "benchmark_id": str(task_id).split("::")[-1],
            "task_id": str(task_id), "budget": 0,
            "item_uid": item_uid[rows["item_index"].to_numpy()],
            "p": sig(z.astype(float)).astype("float32"),
            "y": rows["label"].to_numpy("float32")}))
    print(f"{rung}: train done", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fig11).to_parquet(OUT_ROOT / f"{rung}-fig11.parquet", index=False)
    pd.concat(items).to_parquet(OUT_ROOT / f"{rung}-items.parquet", index=False)
    volume.commit()
    return {"rung": rung, "fig11_rows": len(fig11)}


@app.local_entrypoint()
def main() -> None:
    handles = [run.spawn(r) for r in RUNGS]
    for h in handles:
        print(h.get())
