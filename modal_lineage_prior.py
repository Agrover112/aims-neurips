"""Does a subject-lineage ability prior help the multidimensional IRT model?

Fits ``LineagePrior`` on per-training-task ability vectors, then rescores the
104 validation tasks with the prior mean set from each subject's family and
provider instead of zero. Inference-only, so it re-uses a trained checkpoint
(``mirt_scalar_d4`` on the me5 store) and runs on CPU.

Leakage control: the prior is fitted on training tasks only. A validation
subject is matched to training subjects by ``normalized_name`` / ``provider``
text, never by ``subject_uid`` (uids never span benchmarks), and its own
labels are used only as the revealed context at each budget.

Run:  modal run --detach modal_lineage_prior.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-lineage-prior"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
PROCESSED_ROOT = EMBEDDING_ROOT / "processed"
ABLATION_ROOT = REMOTE_ROOT / "results" / "r-c-ablation"
RESULT_ROOT = REMOTE_ROOT / "results" / "lineage-prior"
CORPUS = "allbench-train-me5_instruct-cap128-v2"
UNIFORM_FOLDS = 5
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
STRENGTHS = (1.0, 4.0, 16.0)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src"})
)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=float)))


def _context_rate(labels):
    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _features(rows, item_embeddings, subject_embeddings):
    return np.concatenate(
        [item_embeddings[rows["item_index"].to_numpy()],
         subject_embeddings[rows["subject_index"].to_numpy()]], axis=1,
    ).astype("float32")


def _fit_calibrator(samples, rng):
    from aims_starter.calibration import AnchorCalibrator

    logits = np.concatenate([s["logits"] for s in samples])
    labels = np.concatenate([s["labels"] for s in samples])
    rates = np.concatenate(
        [np.full(len(s["labels"]), _context_rate(s["context_labels"])) for s in samples])
    budgets = np.concatenate(
        [np.full(len(s["labels"]), s["budget"], dtype=int) for s in samples])
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _alc(records, arm):
    """Raw and out-of-fold calibrated per-task ALC for one prior setting."""

    import pandas as pd

    rng = np.random.default_rng(20261020)
    mine = [r for r in records if r["arm"] == arm]
    folds = sorted({int(r["fold"]) for r in mine})
    calibrators = {f: _fit_calibrator([r for r in mine if int(r["fold"]) != f], rng)
                   for f in folds}
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    rows = []
    for r in mine:
        raw = _sigmoid(r["logits"])
        cal = calibrators[int(r["fold"])].transform(
            r["logits"], int(r["budget"]), r["context_labels"])
        for name, p in (("raw", raw), ("oof", cal)):
            rows.append({"calibration": name, "task_id": r["task_id"],
                         "order_seed": r["order_seed"], "budget": r["budget"],
                         "w": weight[r["budget"]] * float(np.mean((p - r["labels"]) ** 2))})
    frame = pd.DataFrame(rows)
    per = (frame.groupby(["calibration", "task_id", "order_seed"])["w"].sum().reset_index()
           .groupby(["calibration", "task_id"])["w"].mean().reset_index())
    return {c: dict(zip(g["task_id"], g["w"], strict=True)) for c, g in per.groupby("calibration")}


@app.function(image=image, cpu=8, memory=65_536, volumes={str(REMOTE_ROOT): volume},
              timeout=6 * 60 * 60)
def run(rung: str = "mirt_scalar_d4", seed: int = 20261018, orders: int = 10) -> dict[str, Any]:
    import pandas as pd
    import torch

    from aims_starter.lineage_prior import LineagePrior
    from aims_starter.r_c_attention import MultidimContextIRT
    from aims_starter.uniform_eval import validation_fold_map

    root = EMBEDDING_ROOT / CORPUS
    item_emb = np.load(root / "item_embeddings.npy")
    subj_emb = np.load(root / "subject_embeddings.npy")
    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)

    # subject text, joined by uid (compact store order -> processed table)
    uids = pd.read_parquet(root / "subjects.parquet")["subject_uid"].astype(str)
    meta = pd.read_parquet(PROCESSED_ROOT / "subjects.parquet",
                           columns=["subject_uid", "normalized_name", "provider"])
    meta["subject_uid"] = meta["subject_uid"].astype(str)
    meta = meta.drop_duplicates("subject_uid").set_index("subject_uid")
    names = meta.reindex(uids.to_numpy())["normalized_name"].fillna("").astype(str).to_numpy()
    provs = meta.reindex(uids.to_numpy())["provider"].fillna("").astype(str).to_numpy()

    ability_dim = int(rung.rsplit("_d", 1)[-1])
    item_dim, x_dim = int(item_emb.shape[1]), int(item_emb.shape[1] + subj_emb.shape[1])
    model = MultidimContextIRT(
        x_dim, mode="scalar", ability_dim=ability_dim, loops=0, d_model=128, n_heads=8,
        subject_feature_slice=(item_dim, x_dim))
    ckpt = (ABLATION_ROOT / f"seed-{seed}-masked-{CORPUS}-v41e1"
            / f"{rung}-checkpoints" / "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    tau = float(model.prior_variance)

    # 1. per-training-task ability vector under the trained model (zero prior)
    thetas, t_fam, t_prov = [], [], []
    for task_id, rows in train.groupby("task_id", sort=True):
        x = torch.as_tensor(_features(rows, item_emb, subj_emb))
        y = torch.as_tensor(rows["label"].to_numpy("float32"))
        with torch.no_grad():
            h = model.encode_item(x)
            a, d = model._head_parameters(h, h.new_zeros((h.shape[-1],)))
            thetas.append(model.map_theta(a, d, y, prior_variance=tau).numpy())
        s = int(rows["subject_index"].iloc[0])
        t_fam.append(names[s]); t_prov.append(provs[s])
    thetas = np.stack(thetas)
    print(f"train tasks {len(thetas)}  theta sd per dim {thetas.std(axis=0).round(2)}",
          flush=True)

    priors = {k: LineagePrior(strength=k).fit(t_fam, t_prov, thetas) for k in STRENGTHS}

    # 2. score validation
    val = evaluation[evaluation["split"] == "validation"]
    records, coverage = [], {"family": 0, "provider": 0, "global": 0}
    arms = ["zero"] + [f"lineage_k{int(k)}" for k in STRENGTHS]
    for task_id in sorted(val["task_id"].astype(str).unique()):
        rows = val[val["task_id"].astype(str) == task_id]
        fold = int(fold_map.get(task_id, -1))
        s = int(rows["subject_index"].iloc[0])
        coverage[priors[4.0].level(names[s], provs[s])] += 1
        tgt = rows[rows["pool"] == "evaluation"].reset_index(drop=True)
        y_t = tgt["label"].to_numpy("float32")
        x_t = torch.as_tensor(_features(tgt, item_emb, subj_emb))
        lookup = rows.set_index("item_index")
        pm = {f"lineage_k{int(k)}": torch.as_tensor(
                priors[k].predict(names[s], provs[s]), dtype=torch.float32)
              for k in STRENGTHS}
        pm["zero"] = None
        for order_seed in range(orders):
            o = (order_frame[(order_frame["task_id"].astype(str) == task_id)
                             & (order_frame["order_seed"] == order_seed)]
                 .sort_values("acquisition_rank").reset_index(drop=True))
            if o.empty:
                continue
            c = lookup.loc[o["item_index"].to_numpy()].reset_index()
            x_c = torch.as_tensor(_features(c, item_emb, subj_emb))
            y_c = c["label"].to_numpy("float32")
            for budget in BUDGETS:
                for arm in arms:
                    with torch.no_grad():
                        z = model(x_c[:budget], torch.as_tensor(y_c[:budget]), x_t,
                                  prior_mean=pm[arm]).numpy().astype(float)
                    records.append({"arm": arm, "task_id": task_id, "fold": fold,
                                    "order_seed": order_seed, "budget": budget,
                                    "labels": y_t.astype(float),
                                    "context_labels": y_c[:budget].astype(float),
                                    "logits": z})
        print(f"scored {task_id}", flush=True)

    per_arm = {arm: _alc(records, arm) for arm in arms}
    summary = []
    for arm in arms:
        for calib in ("raw", "oof"):
            v = np.array(list(per_arm[arm][calib].values()))
            summary.append({"arm": arm, "calibration": calib,
                            "brier_alc": float(v.mean()),
                            "se": float(v.std(ddof=1) / math.sqrt(len(v)))})
    result = {"status": "lineage_prior", "rung": rung, "seed": seed, "corpus": CORPUS,
              "strengths": STRENGTHS, "validation_prior_level": coverage,
              "train_tasks": len(thetas), "summary": summary,
              "task_alc": {a: per_arm[a] for a in arms}}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / f"{rung}-{seed}.json").write_text(json.dumps(result, default=str))
    volume.commit()
    return result


@app.local_entrypoint()
def main(rung: str = "mirt_scalar_d4", seed: int = 20261018) -> None:
    r = run.remote(rung, seed)
    print("validation subjects by prior level:", r["validation_prior_level"])
    for row in sorted(r["summary"], key=lambda x: (x["calibration"], x["brier_alc"])):
        print(f"{row['arm']:<14}{row['calibration']:<5}{row['brier_alc']:>10.6f}{row['se']:>10.6f}")
