"""Does a heavier-tailed ability prior fix subjects that almost always fail?

Some validation subjects pass almost nothing (one cmmlu run scores 0.8%). A
Gaussian ability prior with a +-2 sigma clamp cannot represent such a subject:
the prior pulls it back toward zero and the clamp puts a floor under its
predicted success rate, so the model is still near 0.3 after 31 failures. Two
causes are confounded, so they are separated here:

    gauss_box2    the published behaviour (Gaussian prior, +-2 sigma clamp)
    gauss_box6    same prior, clamp widened to +-6 sigma     (clamp only)
    t3_box6       Student-t, 3 degrees of freedom, wide clamp (tails + clamp)
    cauchy_box6   Student-t, 1 degree of freedom, wide clamp  (heaviest tails)

Inference-only: it re-uses the trained ``mirt_scalar_d4`` checkpoint (me5
store) and runs on CPU. Scores are written per task together with each task's
evaluation base rate so the analysis can split by base-rate group -- a fix has
to help the near-zero tasks without giving back the gain on the easy ones.

Run:  modal run --detach modal_heavytail_prior.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-heavytail-prior"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
PROCESSED_ROOT = EMBEDDING_ROOT / "processed"
ABLATION_ROOT = REMOTE_ROOT / "results" / "r-c-ablation"
RESULT_ROOT = REMOTE_ROOT / "results" / "heavytail-prior"
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

    from aims_starter.r_c_attention import ContextConditionedIRT, MultidimContextIRT
    from aims_starter.uniform_eval import validation_fold_map

    root = EMBEDDING_ROOT / CORPUS
    item_emb = np.load(root / "item_embeddings.npy")
    subj_emb = np.load(root / "subject_embeddings.npy")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)

    item_dim, x_dim = int(item_emb.shape[1]), int(item_emb.shape[1] + subj_emb.shape[1])
    if rung.startswith("mirt_"):
        ability_dim = int(rung.rsplit("_d", 1)[-1])
        model = MultidimContextIRT(
            x_dim, mode="scalar", ability_dim=ability_dim, loops=0, d_model=128, n_heads=8,
            subject_feature_slice=(item_dim, x_dim))
    else:
        # scalar IRT rung, e.g. "2pl_tau1" (trained with prior variance 1)
        mode, _, tau_tag = rung.partition("_tau")
        model = ContextConditionedIRT(
            x_dim, mode=mode, d_model=128, subject_feature_slice=(item_dim, x_dim),
            prior_variance=float(tau_tag or 4.0))
    ckpt = (ABLATION_ROOT / f"seed-{seed}-masked-{CORPUS}-v41e1"
            / f"{rung}-checkpoints" / "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    # (prior_nu, box_sigmas). gauss_box2 is the published behaviour.
    arms = {"gauss_box2": (None, 2.0), "gauss_box6": (None, 6.0),
            "t3_box6": (3.0, 6.0), "cauchy_box6": (1.0, 6.0)}

    val = evaluation[evaluation["split"] == "validation"]
    records, base_rate = [], {}
    for task_id in sorted(val["task_id"].astype(str).unique()):
        rows = val[val["task_id"].astype(str) == task_id]
        fold = int(fold_map.get(task_id, -1))
        tgt = rows[rows["pool"] == "evaluation"].reset_index(drop=True)
        y_t = tgt["label"].to_numpy("float32")
        base_rate[task_id] = float(y_t.mean())
        x_t = torch.as_tensor(_features(tgt, item_emb, subj_emb))
        lookup = rows.set_index("item_index")
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
                for arm, (nu, box) in arms.items():
                    model.prior_nu, model.box_sigmas = nu, box
                    with torch.no_grad():
                        z = model(x_c[:budget], torch.as_tensor(y_c[:budget]),
                                  x_t).numpy().astype(float)
                    records.append({"arm": arm, "task_id": task_id, "fold": fold,
                                    "order_seed": order_seed, "budget": budget,
                                    "labels": y_t.astype(float),
                                    "context_labels": y_c[:budget].astype(float),
                                    "logits": z})
        print(f"scored {task_id}", flush=True)

    # Persist the scored logits before the (slow) calibration step so a
    # cancelled call cannot lose the expensive part; ALC can be recomputed
    # from these offline.
    import pickle

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RESULT_ROOT / f"heavytail-records-{rung}-{seed}.pkl", "wb") as handle:
        pickle.dump({"records": records, "base_rate": base_rate}, handle)
    volume.commit()
    print("records saved", flush=True)

    per_arm = {arm: _alc(records, arm) for arm in arms}
    result = {"status": "heavytail_prior", "rung": rung, "seed": seed, "corpus": CORPUS,
              "arms": {a: list(v) for a, v in arms.items()}, "base_rate": base_rate,
              "task_alc": {a: per_arm[a] for a in arms}}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / f"heavytail-{rung}-{seed}.json").write_text(json.dumps(result, default=str))
    volume.commit()
    return result


@app.local_entrypoint()
def main(rung: str = "mirt_scalar_d4", seed: int = 20261018) -> None:
    r = run.remote(rung, seed)
    print("done", list(r["arms"]))
# Weight-saving placeholder for modal_heavytail_prior.py
