"""Train the CNP/NP family on the response matrix, with W&B curves.

Architecture lives in ``cnp_blocks.py`` so it is unit-testable without Modal;
this file is data, training loop, validation and logging.

    --encoder   linear | mhsa | set_transformer | sgformer
    --pooling   mean | learned
    --use-latent            latent NP (ELBO) instead of deterministic CNP
    --width 64              working width (see cnp_blocks docstring: 64, not 256)

Validation is a held-out **subject** split, not a held-out slice of the same
subjects' responses: the quantity we care about is generalising to a new
test-taker, so the validation loss has to be measured that way or it will look
healthy while the thing we need degrades.

W&B is optional. With a ``wandb`` Modal secret present the run streams live
curves; without one it still trains and writes the curve to the volume, so a
missing secret degrades to "no live plot" rather than "no run".

Run:  modal run --detach modal_cnp_train.py::main --encoder sgformer
      modal run --detach modal_cnp_train.py::main --encoder linear --use-latent
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-cnp-train"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
OUT = Path("/data/results/cnp/cnp.parquet")

ARM = "qwen3_8b"
MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HELD_OUT_FRAC = 0.20
VAL_FRAC = 0.15
MIN_HALF = 4
MIN_MATRIX = 20
MAX_CONTEXT = 64
STEPS = 3000
EPISODES_PER_STEP = 8
LR = 3e-4
VAL_EVERY = 50
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (0, 1, 3, 7, 15, 31)

WANDB_ENTITY = "agrgnn-kth-royal-institute-of-technology"
WANDB_PROJECT = "aims-predictive-evaluation"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2",
    "tqdm>=4.66,<5", "wandb>=0.17,<1")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu118")

# Ship the architecture module. The container copy of this file lives at /root,
# so guard on BOTH existence and not-already-in-the-container: a bare
# .exists() check passes inside the container after shipping and re-adds the
# layer during Modal's re-import, and a bare parents[] index raises IndexError
# there. Both of those crash-looped jobs earlier in this project.
_HERE = Path(__file__).resolve().parent
_BLOCKS = _HERE / "cnp_blocks.py"
if str(_HERE) != "/root" and _BLOCKS.exists():
    image = image.add_local_file(_BLOCKS, "/root/cnp_blocks.py")

# Optional: live W&B curves need a secret holding WANDB_API_KEY.
try:
    _SECRETS = [modal.Secret.from_name("wandb")]
except Exception:
    _SECRETS = []


def _load_filtered(min_subjects):
    import pandas as pd

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    meta = pd.read_parquet(PROCESSED / "items.parquet",
                           columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")
    d = obs.merge(meta, on="item_uid", how="left")
    d = d.drop_duplicates(["content_hash", "subject_uid"])
    var = d.groupby("item_uid")["label"].agg(["mean"])
    d = d[d["item_uid"].isin(var[(var["mean"] > 0) & (var["mean"] < 1)].index)]
    n_sub = d.groupby("item_uid")["subject_uid"].nunique()
    d = d[d["item_uid"].isin(n_sub[n_sub >= min_subjects].index)]
    n_item = d.groupby("subject_uid")["item_uid"].nunique()
    d = d[d["subject_uid"].isin(n_item[n_item >= MIN_ITEMS_PER_TASK].index)]
    return d


def _episode(task, rng, device, emb, max_context=MAX_CONTEXT):
    """Random context size per episode (CNP paper §2.3: N ~ uniform[0, n-1]).

    Varying N during training is what lets one model serve the whole 0..31
    budget ladder at test time without retraining per budget.
    """
    import torch

    items, labels = task["items"], task["labels"]
    n = len(items)
    k = int(rng.integers(0, min(max_context, n - 1) + 1))
    perm = rng.permutation(n)
    c, t = perm[:k], perm[k:]
    if len(t) == 0:
        t = perm[-1:]
    gi = lambda idx: emb[torch.as_tensor(items[idx], dtype=torch.long, device=device)]
    gl = lambda idx: torch.as_tensor(labels[idx], dtype=torch.float32, device=device)
    return gi(c), gl(c), gi(t), gl(t)


def _score(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return auc, float(np.mean((p - y) ** 2))


@app.function(image=image, gpu="L4", memory=65_536, volumes={"/data": volume},
              secrets=_SECRETS, timeout=6 * 60 * 60)
def run(seed: int = SEEDS[0], arm: str = ARM, encoder: str = "linear",
        pooling: str = "mean", use_latent: bool = False, width: int = 64,
        n_blocks: int = 2, steps: int = STEPS, residual_proj: bool = False,
        wandb_mode: str = "online") -> dict:
    import os
    import sys
    import time

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    import cnp_blocks as cb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device: {device}", flush=True)
    torch.manual_seed(seed)

    cfg = {"seed": seed, "arm": arm, "encoder": encoder, "pooling": pooling,
           "use_latent": use_latent, "width": width, "n_blocks": n_blocks,
           "steps": steps, "lr": LR, "residual_proj": residual_proj,
           "episodes_per_step": EPISODES_PER_STEP, "max_context": MAX_CONTEXT,
           "min_subjects": MIN_SUBJECTS, "val_frac": VAL_FRAC}

    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            tag = f"{encoder}-{pooling}{'-latent' if use_latent else ''}-w{width}"
            wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                            name=f"cnp-{tag}-{arm}-seed{seed}", job_type="cnp-train",
                            config=cfg, tags=["cnp", encoder, pooling, arm],
                            mode=wandb_mode, reinit=True)
            print("wandb: live", flush=True)
        except Exception as exc:                      # never let logging kill a run
            print(f"wandb disabled ({exc})", flush=True)
            wb = None
    else:
        print("wandb: no WANDB_API_KEY secret, logging to volume only", flush=True)

    store = DATASETS / f"allbench-embed-{arm}-v2"
    emb_ids = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
    emb_all = np.load(store / "item_embeddings.npy", mmap_mode="r")
    pos = {u: i for i, u in enumerate(emb_ids)}
    print(f"embedding store {arm}: {emb_all.shape}", flush=True)

    d = _load_filtered(MIN_SUBJECTS)
    print(f"filtered (n>={MIN_SUBJECTS}): {len(d):,} rows, "
          f"{d.benchmark_id.nunique()} benchmarks", flush=True)

    rows, curves, rng = [], [], np.random.default_rng(seed)
    gstep = 0

    for bench, dd in tqdm(list(d.groupby("benchmark_id")), desc=f"seed={seed}",
                          unit="bench", mininterval=15.0):
        subs = sorted(dd["subject_uid"].unique())
        its = sorted(dd["item_uid"].unique())
        if len(subs) < MIN_MATRIX or len(its) < MIN_MATRIX or not all(u in pos for u in its):
            continue
        i_map = {u: i for i, u in enumerate(its)}
        emb = torch.as_tensor(
            np.asarray(emb_all[[pos[u] for u in its]], dtype=np.float32)).to(device)

        tasks = [{"items": g["item_uid"].map(i_map).to_numpy(),
                  "labels": g["label"].to_numpy().astype(np.float32)}
                 for _, g in dd.groupby("subject_uid")]

        n_s = len(tasks)
        perm = rng.permutation(n_s)
        n_test = max(1, int(n_s * HELD_OUT_FRAC))
        n_val = max(1, int(n_s * VAL_FRAC))
        test_ix = set(perm[:n_test].tolist())
        val_ix = set(perm[n_test:n_test + n_val].tolist())
        held_i = set(rng.choice(len(its), max(1, int(len(its) * HELD_OUT_FRAC)),
                                replace=False).tolist())

        def _warm(t):
            keep = ~np.isin(t["items"], list(held_i))
            return {"items": t["items"][keep], "labels": t["labels"][keep]}

        train_tasks = [_warm(t) for i, t in enumerate(tasks)
                       if i not in test_ix and i not in val_ix]
        val_tasks = [_warm(t) for i, t in enumerate(tasks) if i in val_ix]
        train_tasks = [t for t in train_tasks if len(t["items"]) > 2 * MIN_HALF]
        val_tasks = [t for t in val_tasks if len(t["items"]) > 2 * MIN_HALF]
        if len(train_tasks) < MIN_MATRIX or not val_tasks:
            continue

        model = cb.NeuralProcess(emb.shape[1], width=width, n_blocks=n_blocks,
                                 encoder=encoder, pooling=pooling,
                                 use_latent=use_latent,
                                 residual_proj=residual_proj).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t0 = time.time()

        for step in tqdm(range(steps), desc=f"{bench[:14]}", unit="step",
                         mininterval=20.0, leave=False):
            model.train()
            opt.zero_grad()
            losses = [model.loss(*_episode(train_tasks[int(rng.integers(len(train_tasks)))],
                                           rng, device, emb))[0]
                      for _ in range(EPISODES_PER_STEP)]
            tr = torch.stack(losses).mean()
            tr.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            if step % VAL_EVERY == 0 or step == steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = torch.stack(
                        [model.loss(*_episode(t, rng, device, emb))[0]
                         for t in val_tasks[:16]]).mean()
                rec = {"benchmark_id": bench, "seed": seed, "step": step,
                       "train_loss": float(tr), "val_loss": float(vl)}
                curves.append(rec)
                if wb is not None:
                    wb.log({f"{bench}/train_loss": float(tr),
                            f"{bench}/val_loss": float(vl),
                            "train_loss": float(tr), "val_loss": float(vl),
                            "benchmark": bench, "step_in_bench": step}, step=gstep)
                gstep += 1
            if step == 49:
                rate = (time.time() - t0) / 50
                print(f"{bench}: {rate:.3f} s/step -> {rate * steps / 60:.1f} min/bench",
                      flush=True)

        # ---- evaluation on held-out subjects, across the budget ladder ----
        model.eval()
        with torch.no_grad():
            for i in sorted(test_ix):
                t = tasks[i]
                keep = ~np.isin(t["items"], list(held_i))
                w_it, w_lb = t["items"][keep], t["labels"][keep]
                c_it, c_lb = t["items"][~keep], t["labels"][~keep]
                if len(w_it) < 2 * MIN_HALF:
                    continue
                order = rng.permutation(len(w_it))
                for k in BUDGETS:
                    if k >= len(order):
                        continue
                    ci, ti = order[:k], order[k:]
                    x_c = emb[torch.as_tensor(w_it[ci], dtype=torch.long, device=device)]
                    y_c = torch.as_tensor(w_lb[ci], dtype=torch.float32, device=device)
                    x_t = emb[torch.as_tensor(w_it[ti], dtype=torch.long, device=device)]
                    p = torch.sigmoid(model.logits(x_c, y_c, x_t)).cpu().numpy()
                    if len(np.unique(w_lb[ti])) > 1:
                        a, b = _score(w_lb[ti], p)
                        rows.append({"benchmark_id": bench, "seed": seed, "arm": arm,
                                     "encoder": encoder, "pooling": pooling,
                                     "use_latent": use_latent, "width": width,
                                     "axis": "subject", "budget": k,
                                     "n_score": int(len(ti)), "auc": a, "brier": b})
                    if len(c_it) >= 2 and len(np.unique(c_lb)) > 1:
                        x_ci = emb[torch.as_tensor(c_it, dtype=torch.long, device=device)]
                        pc = torch.sigmoid(model.logits(x_c, y_c, x_ci)).cpu().numpy()
                        a2, b2 = _score(c_lb, pc)
                        rows.append({"benchmark_id": bench, "seed": seed, "arm": arm,
                                     "encoder": encoder, "pooling": pooling,
                                     "use_latent": use_latent, "width": width,
                                     "axis": "item_coldstart", "budget": k,
                                     "n_score": int(len(c_lb)), "auc": a2, "brier": b2})

    frame = pd.DataFrame(rows)
    curve = pd.DataFrame(curves)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{encoder}-{pooling}{'-latent' if use_latent else ''}-w{width}"
    frame.to_parquet(OUT.with_name(f"{OUT.stem}-{tag}-{arm}-seed{seed}{OUT.suffix}"), index=False)
    curve.to_parquet(OUT.with_name(f"{OUT.stem}-curve-{tag}-{arm}-seed{seed}{OUT.suffix}"), index=False)
    volume.commit()
    if wb is not None:
        if len(frame):
            for axis, g in frame.groupby("axis"):
                for k, gg in g.groupby("budget"):
                    wb.summary[f"final/{axis}/auc@{k}"] = float(gg.auc.mean())
        wb.finish()
    return {"rows": len(frame), "curve_points": len(curve), "seed": seed,
            "encoder": encoder, "pooling": pooling, "use_latent": use_latent,
            "benchmarks": int(frame["benchmark_id"].nunique()) if len(frame) else 0}


@app.local_entrypoint()
def main(arm: str = ARM, encoder: str = "linear", pooling: str = "mean",
         use_latent: bool = False, width: int = 64, n_blocks: int = 2,
         steps: int = STEPS, residual_proj: bool = False) -> None:
    handles = [run.spawn(s, arm, encoder, pooling, use_latent, width, n_blocks,
                         steps, residual_proj) for s in SEEDS]
    for h in handles:
        print(h.get())
