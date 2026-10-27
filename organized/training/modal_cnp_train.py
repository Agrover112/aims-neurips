"""Train the CNP/NP family on the response matrix, with W&B curves.

Architecture lives in ``cnp_blocks.py`` so it is unit-testable without Modal;
this file is data, training loop, validation and logging.

    --encoder   linear | mhsa | set_transformer | sgformer
    --pooling   mean | mhsa
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
WEIGHT_DECAY = 1e-2   # AdamW default; decoupled, so it does not scale with LR
VAL_EVERY = 50
VAL_EPISODES = 32      # MC estimate of val loss; cheap and low-variance enough
PATIENCE = 8           # val checks without improvement before stopping
MAX_EVAL_TARGETS = 512
CURVE_SUBJECTS = 24    # subsample for the periodic per-axis curves
MAX_EPISODE_TARGETS = 512  # targets include context; cap for cost only
                       # per budget is the same estimator as scoring a sample
SEEDS = (20261018, 20261019, 20261020)
# Budget ladder lives in organized/testing/online/, not here.

WANDB_ENTITY = "agrgnn-kth-royal-institute-of-technology"
WANDB_PROJECT = "aims-predictive-evaluation"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2",
    "tqdm>=4.66,<5", "wandb>=0.17,<1")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")

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
    """One task, random context size, targets INCLUDING the context (CNP §2.3).

    The paper: "N ~ uniform[0,...,n-1]. We condition on the subset O_N ... Thus,
    the targets it scores Q_theta on include both the observed and unobserved
    values." So targets are the task's full observation set, not the complement.

    Disjoint context/target is the NP/ANP convention and is what this used to
    do; it is not what the original CNP specifies. Scoring the context points
    too is also what makes the model honest at budget 0 -- it has to explain
    what it was just told, not only what it was not.

    This is the "cell" training style: random subset of ONE task's responses as
    context, score that task's responses. Subject- and item-holdout are test
    protocols, not training recipes -- the paper draws a fresh f ~ P each step
    and has no notion of holding out x locations at all.
    """
    import torch

    items, labels = task["items"], task["labels"]
    n = len(items)
    k = int(rng.integers(0, min(max_context, n - 1) + 1))
    perm = rng.permutation(n)
    c = perm[:k]
    # Disjoint targets, following "Active Evaluation Acquisition" sec B.2:
    # "we randomly sample two non-overlapping subsets from each model as
    # Y^(o) and Y^(u)". That paper solves this exact task. Garnelo's CNP
    # instead scores targets that include the context; we take the
    # Active-Evaluation convention here.
    t = perm[k:k + MAX_EPISODE_TARGETS]
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



def _axis_scores(model, tasks, *, held_i_arr, emb, device, rng, n_subjects,
                 max_targets, min_half, holdout_frac, max_context):
    """cell / subject_splithalf / item_coldstart on a subsample of tasks.

    Used both for the periodic curves during training (on VALIDATION subjects,
    never test -- early stopping reads these, so test here would leak into
    model selection) and for the final report (on test subjects).

    All three are the same CNP operation: encode a context set, decode targets.
    Only the carve differs.
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    acc = {}
    # no_grad: this is called mid-training, so without it the tensors still
    # require grad and .numpy() raises -- and we would build a useless graph.
    torch.set_grad_enabled(False)
    picks = rng.choice(len(tasks), min(n_subjects, len(tasks)), replace=False)
    for ix in picks:
        t = tasks[ix]
        keep = ~np.isin(t["items"], held_i_arr)
        w_it, w_lb = t["items"][keep], t["labels"][keep]
        c_it, c_lb = t["items"][~keep], t["labels"][~keep]
        if len(w_it) < 2 * min_half:
            continue
        order = rng.permutation(len(w_it))

        def ev(axis, ctx, tgt_items, tgt_labels):
            if len(tgt_items) < 2 or len(np.unique(tgt_labels)) < 2:
                return
            x_c = emb[torch.as_tensor(w_it[ctx], dtype=torch.long, device=device)]
            y_c = torch.as_tensor(w_lb[ctx], dtype=torch.float32, device=device)
            x_t = emb[torch.as_tensor(tgt_items, dtype=torch.long, device=device)]
            p = torch.sigmoid(model.logits(x_c, y_c, x_t)).cpu().numpy()
            try:
                auc = float(roc_auc_score(tgt_labels, p))
            except ValueError:
                return
            acc.setdefault(axis, []).append(
                (auc, float(np.mean((p - tgt_labels) ** 2))))

        cut = int(len(order) * (1.0 - holdout_frac))
        ti = order[cut:cut + max_targets]
        ev("cell", order[:cut], w_it[ti], w_lb[ti])

        half = len(order) // 2
        ti = order[half:half + max_targets]
        ev("subject_splithalf", order[:half], w_it[ti], w_lb[ti])

        if len(c_it) >= 2:
            ev("item_coldstart", order[:min(len(order), max_context)],
               c_it[:max_targets], c_lb[:max_targets])

    torch.set_grad_enabled(True)
    out = {}
    for axis, vals in acc.items():
        a = np.array(vals)
        out[f"{axis}/auc"] = float(a[:, 0].mean())
        out[f"{axis}/brier"] = float(a[:, 1].mean())
    return out


# CPU: measured 0.036 s/step on 4 threads (3000 steps ~ 2 min). 171k
# params, <=64 context and <=512 target items per episode -- the matmuls
# are far too small for a GPU to help, and L4 capacity was queueing.
@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
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
           "steps": steps, "lr": LR, "weight_decay": WEIGHT_DECAY,
           "optimizer": "AdamW", "residual_proj": residual_proj,
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

    # ---- POOL every benchmark into one task set -------------------------------
    # Subjects never span benchmarks (checked: 6,264 subjects, all in exactly
    # one), so pooling cannot merge two different abilities into one task.
    # Training per benchmark was wrong twice over: `editbench` has 28 training
    # subjects and would see each ~857 times, and a per-benchmark model cannot
    # be evaluated under the grouped-benchmark-fold protocol at all, since it
    # has no parameters for a benchmark it never saw. The model never receives
    # a benchmark id -- an item is only its embedding -- so nothing about the
    # architecture needs to change.
    d = d[d["item_uid"].isin(pos.keys())]
    items_all = sorted(d["item_uid"].unique())
    i_map = {u: i for i, u in enumerate(items_all)}
    emb = torch.as_tensor(
        np.asarray(emb_all[[pos[u] for u in items_all]], dtype=np.float32)).to(device)
    print(f"pooled item matrix: {tuple(emb.shape)}", flush=True)

    tasks = []
    for (sub, bench), g in d.groupby(["subject_uid", "benchmark_id"]):
        tasks.append({"subject": sub, "benchmark": bench,
                      "items": g["item_uid"].map(i_map).to_numpy(),
                      "labels": g["label"].to_numpy().astype(np.float32)})
    print(f"pooled tasks (subject x benchmark): {len(tasks):,}", flush=True)

    n_s = len(tasks)
    perm = rng.permutation(n_s)
    n_test = max(1, int(n_s * HELD_OUT_FRAC))
    n_val = max(1, int(n_s * VAL_FRAC))
    test_ix = set(perm[:n_test].tolist())
    val_ix = set(perm[n_test:n_test + n_val].tolist())
    held_i = set(rng.choice(len(items_all), max(1, int(len(items_all) * HELD_OUT_FRAC)),
                            replace=False).tolist())
    held_i_arr = np.fromiter(held_i, dtype=np.int64)

    def _warm(t):
        keep = ~np.isin(t["items"], held_i_arr)
        return {**t, "items": t["items"][keep], "labels": t["labels"][keep]}

    train_tasks = [_warm(t) for i, t in enumerate(tasks)
                   if i not in test_ix and i not in val_ix]
    val_tasks = [_warm(t) for i, t in enumerate(tasks) if i in val_ix]
    train_tasks = [t for t in train_tasks if len(t["items"]) > 2 * MIN_HALF]
    val_tasks = [t for t in val_tasks if len(t["items"]) > 2 * MIN_HALF]
    print(f"train/val/test subjects: {len(train_tasks):,}/{len(val_tasks):,}/{len(test_ix):,}"
          f"   held-out items: {len(held_i):,}", flush=True)
    if not train_tasks or not val_tasks:
        raise RuntimeError("pooled split left no training or validation tasks")

    model = cb.NeuralProcess(emb.shape[1], width=width, n_blocks=n_blocks,
                             encoder=encoder, pooling=pooling,
                             use_latent=use_latent,
                             residual_proj=residual_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"params: {sum(q.numel() for q in model.parameters()):,}", flush=True)

    epoch_order: list[int] = []
    best_val, best_state, since_best = float("inf"), None, 0
    t0 = time.time()
    for step in tqdm(range(steps), desc=f"seed={seed}", unit="step", mininterval=20.0):
        model.train()
        opt.zero_grad()
        # Shuffled epochs, not sampling with replacement: every subject is
        # seen once per pass. With replacement and early stopping at ~step 400
        # only ~54% of the 4,071 train subjects would ever be visited.
        batch = []
        for _ in range(EPISODES_PER_STEP):
            if not epoch_order:
                epoch_order.extend(rng.permutation(len(train_tasks)).tolist())
            batch.append(train_tasks[epoch_order.pop()])
        losses = [model.loss(*_episode(t, rng, device, emb))[0] for t in batch]
        tr = torch.stack(losses).mean()
        tr.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % VAL_EVERY == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = torch.stack([
                    model.loss(*_episode(val_tasks[int(rng.integers(len(val_tasks)))],
                                         rng, device, emb))[0]
                    for _ in range(VAL_EPISODES)]).mean()
            v = float(vl)
            curves.append({"seed": seed, "step": step,
                           "train_loss": float(tr), "val_loss": v})
            # Same three carves on BOTH pools: the train/val gap per axis is
            # the generalisation gap for that axis. One model, one objective --
            # these are evaluation carves, not separate training heads.
            kw = dict(held_i_arr=held_i_arr, emb=emb, device=device,
                      n_subjects=CURVE_SUBJECTS, max_targets=MAX_EVAL_TARGETS,
                      min_half=MIN_HALF, holdout_frac=HELD_OUT_FRAC,
                      max_context=MAX_CONTEXT)
            val_axis = _axis_scores(model, val_tasks,
                                    rng=np.random.default_rng(seed), **kw)
            tr_axis = _axis_scores(model, train_tasks,
                                   rng=np.random.default_rng(seed + 1), **kw)
            curves[-1].update({f"val_{k}": x for k, x in val_axis.items()})
            curves[-1].update({f"train_{k}": x for k, x in tr_axis.items()})
            if wb is not None:
                wb.log({"train_loss": float(tr), "val_loss": v,
                        "best_val": min(best_val, v),
                        **{f"val/{k}": x for k, x in val_axis.items()},
                        **{f"train/{k}": x for k, x in tr_axis.items()}}, step=step)
            # early stopping on held-out SUBJECT loss -- no paper publishes a
            # step count, and the right one depends on our data, not theirs.
            if v < best_val - 1e-4:
                best_val, since_best = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= PATIENCE:
                    print(f"early stop at step {step} (best val {best_val:.4f})", flush=True)
                    break
        if step == 49:
            rate = (time.time() - t0) / 50
            print(f"{rate:.3f} s/step -> {rate * steps / 60:.1f} min for {steps} steps",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"restored best val {best_val:.4f}", flush=True)

    # ---- OFFLINE evaluation: the same three protocols as the IRT jobs -------
    # The budget ladder is deliberately NOT here. A ladder without an
    # acquisition policy is just the random-acquisition arm of the online
    # evaluation, and scoring it inside the training job would quietly pass it
    # off as an offline result. It belongs in organized/testing/online/.
    #
    # For a CNP every protocol below is the same operation -- encode a context
    # set, decode targets -- differing only in how context and target are
    # carved. Nothing is fitted at test time.
    model.eval()
    with torch.no_grad():
        for i in tqdm(sorted(test_ix), desc="eval", unit="subj", mininterval=20.0):
            t = tasks[i]
            keep = ~np.isin(t["items"], held_i_arr)
            w_it, w_lb = t["items"][keep], t["labels"][keep]
            c_it, c_lb = t["items"][~keep], t["labels"][~keep]
            if len(w_it) < 2 * MIN_HALF:
                continue
            base = {"benchmark_id": t["benchmark"], "seed": seed, "arm": arm,
                    "encoder": encoder, "pooling": pooling,
                    "use_latent": use_latent, "width": width}
            order = rng.permutation(len(w_it))

            def _emit(axis, ctx_idx, tgt_idx):
                if len(tgt_idx) < 2 or len(np.unique(w_lb[tgt_idx])) < 2:
                    return
                x_c = emb[torch.as_tensor(w_it[ctx_idx], dtype=torch.long, device=device)]
                y_c = torch.as_tensor(w_lb[ctx_idx], dtype=torch.float32, device=device)
                x_t = emb[torch.as_tensor(w_it[tgt_idx], dtype=torch.long, device=device)]
                p = torch.sigmoid(model.logits(x_c, y_c, x_t)).cpu().numpy()
                auc, brier = _score(w_lb[tgt_idx], p)
                rows.append({**base, "axis": axis, "n_score": int(len(tgt_idx)),
                             "auc": auc, "brier": brier})

            # cell: 80% of this subject's responses as context, 20% as target.
            # The IRT counterpart gets to fit this subject's own theta; the CNP
            # fits nothing, so matching it here is the stronger result.
            cut = int(len(order) * (1.0 - HELD_OUT_FRAC))
            _emit("cell", order[:cut], order[cut:cut + MAX_EVAL_TARGETS])

            # subject split-half: matches the IRT generalization protocol.
            half = len(order) // 2
            _emit("subject_splithalf", order[:half], order[half:half + MAX_EVAL_TARGETS])

            # item cold start: target items the model never trained on, with a
            # normal context from this subject's warm items.
            if len(c_it) >= 2 and len(np.unique(c_lb)) > 1:
                ctx = order[:min(len(order), MAX_CONTEXT)]
                x_c = emb[torch.as_tensor(w_it[ctx], dtype=torch.long, device=device)]
                y_c = torch.as_tensor(w_lb[ctx], dtype=torch.float32, device=device)
                cc = c_it[:MAX_EVAL_TARGETS]
                x_ci = emb[torch.as_tensor(cc, dtype=torch.long, device=device)]
                pc = torch.sigmoid(model.logits(x_c, y_c, x_ci)).cpu().numpy()
                a2, b2 = _score(c_lb[:MAX_EVAL_TARGETS], pc)
                rows.append({**base, "axis": "item_coldstart", "n_score": int(len(cc)),
                             "auc": a2, "brier": b2})

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
                wb.summary[f"final/{axis}/auc"] = float(g.auc.mean())
                wb.summary[f"final/{axis}/brier"] = float(g.brier.mean())
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
