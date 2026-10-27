"""Amortized IRT: item difficulty predicted from question-text embeddings.

After *Reliable and Efficient Amortized Model-based Evaluation* §4.1. Ordinary
IRT gives every item its own free difficulty scalar, fit only from that item's
responses -- so an item needs ~30 responses before it is identified at all, and
a brand new item cannot be scored. Amortized calibration replaces the free
scalar with a function of the question's content:

    b_j = f_phi(e_j)

trained **through** the IRT likelihood (the paper's M step maximises
log p(Y | f_phi(e_j)), not a regression onto pre-fit difficulties -- that
two-stage variant already exists as a diagnostic in modal_item_difficulty_probe).

Design decisions (settled with the user, do not drift):
  * unidimensional theta -- multidimensional work belongs in the MIRT jobs;
  * d_j is also amortized for 2PL/3PL (separate head, same embedding), scalar
    per item, never a vector;
  * GELU, not ReLU;
  * n>=30 filtered data only.

The paper only ever amortized Rasch/1PL with a *linear* head, having dropped
2PL/3PL under traditional calibration (their Fig. 9) "likely due to the limited
number of test takers". Amortization changes that calculus -- b_j stops being a
free per-item parameter and starts borrowing strength across items -- so the
2PL/3PL rungs here are new, not a reproduction.

Two protocols, matching the classic-IRT jobs so numbers are comparable:
  cell  -- random 20% cell holdout.
  gen   -- hold out 20% of subjects and 20% of items; for held-out subjects,
           split-half (fit theta on half 1, score half 2). For held-out items
           the amortized model needs **no responses at all**: b and d come
           straight from the embedding, so it is scored cold on every response.
           That cold-start column is the entire point of amortizing and has no
           classic-IRT counterpart.

Run:  modal run --detach modal_amortized_irt.py::main
      modal run --detach modal_amortized_irt.py::main --arm bge_m3_long
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-amortized-irt"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
OUT = Path("/data/results/data-analysis/amortized-irt.parquet")

MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HOLDOUT_FRAC = 0.20
HELD_OUT_FRAC = 0.20
MIN_HALF = 4
MIN_MATRIX = 20
STEPS = 600
LR = 1e-3
HIDDEN = 256
SEEDS = (20261018, 20261019, 20261020)
DEFAULT_ARM = "me5_instruct"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2",
    "tqdm>=4.66,<5")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")


def _head(in_dim, hidden, out_dim=1):
    """GELU MLP mapping an item embedding to an IRT parameter."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden // 2), nn.GELU(),
        nn.Linear(hidden // 2, out_dim),
    )


class AmortizedIRT:
    """theta free per subject; b (and d) predicted from the item embedding.

    ``c`` for 3PL stays a free per-item scalar: guessing is a property of the
    answer format rather than something we expect the question text to predict,
    and tying it to the same embedding as b and d would make the three even
    harder to separate.
    """

    def __init__(self, n_subjects, emb_dim, kind, n_items, seed, device=None):
        import torch

        torch.manual_seed(seed)
        self.kind = kind
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta = torch.zeros(n_subjects, device=self.device, requires_grad=True)
        self.b_head = _head(emb_dim, HIDDEN).to(self.device)
        self.d_head = (_head(emb_dim, HIDDEN).to(self.device)
                       if kind in ("2pl", "3pl") else None)
        self.logit_c = (torch.full((n_items,), -2.0, device=self.device, requires_grad=True)
                        if kind == "3pl" else None)

    def parameters(self):
        import itertools

        ps = [self.theta] + list(self.b_head.parameters())
        if self.d_head is not None:
            ps += list(self.d_head.parameters())
        if self.logit_c is not None:
            ps.append(self.logit_c)
        return list(itertools.chain(ps))

    def item_params(self, emb):
        """(b, d) for the given item embeddings. d is softplus-positive."""
        import torch
        import torch.nn.functional as F

        b = self.b_head(emb).squeeze(-1)
        if self.d_head is None:
            d = torch.ones_like(b)
        else:
            # softplus keeps discrimination positive without exp's blow-up
            d = F.softplus(self.d_head(emb).squeeze(-1)) + 1e-3
        return b, d

    def probs(self, theta_vals, emb, item_idx=None):
        import torch

        b, d = self.item_params(emb)
        p = torch.sigmoid(d * (theta_vals - b))
        if self.logit_c is not None and item_idx is not None:
            c = torch.sigmoid(self.logit_c[item_idx]) * 0.5
            p = c + (1 - c) * p
        return p.clamp(1e-6, 1 - 1e-6)

    def probs_gathered(self, theta_vals, b_all, d_all, item_idx):
        """Same curve, but from item params computed ONCE over the unique items.

        ``probs`` runs the MLP on ``emb[item_idx]`` -- one forward pass per
        *response*. There are far fewer unique items than responses (often by
        10-100x), so the heads should be evaluated on the (n_items, dim)
        embedding matrix once per step and the results gathered. Same maths,
        a fraction of the compute.
        """
        import torch

        b = b_all[item_idx]
        d = d_all[item_idx]
        p = torch.sigmoid(d * (theta_vals - b))
        if self.logit_c is not None:
            c = torch.sigmoid(self.logit_c[item_idx]) * 0.5
            p = c + (1 - c) * p
        return p.clamp(1e-6, 1 - 1e-6)

    def train_mode(self, flag=True):
        self.b_head.train(flag)
        if self.d_head is not None:
            self.d_head.train(flag)

    def state(self):
        return {
            "theta": self.theta.detach().cpu().numpy(),
            "b_head": {k: v.detach().cpu().numpy() for k, v in self.b_head.state_dict().items()},
            "d_head": ({k: v.detach().cpu().numpy()
                        for k, v in self.d_head.state_dict().items()}
                       if self.d_head is not None else None),
            "logit_c": (self.logit_c.detach().cpu().numpy()
                        if self.logit_c is not None else None),
        }


def _fit(model, emb, s_idx, i_idx, y, steps=STEPS, lr=LR, desc=""):
    """Maximise the IRT likelihood through the amortization heads."""
    import torch
    from tqdm import tqdm

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    s_t = torch.as_tensor(s_idx, dtype=torch.long, device=model.device)
    i_t = torch.as_tensor(i_idx, dtype=torch.long, device=model.device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=model.device)
    loss_history = []
    model.train_mode(True)
    for _ in tqdm(range(steps), desc=desc, unit="step", mininterval=10.0, leave=False):
        opt.zero_grad()
        # Heads run once over the unique items, then gather per response.
        b_all, d_all = model.item_params(emb)
        p = model.probs_gathered(model.theta[s_t], b_all, d_all, i_t)
        loss = -(y_t * torch.log(p) + (1 - y_t) * torch.log(1 - p)).mean()
        loss = loss + 0.125 * model.theta.square().mean()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu().numpy()))
    model.train_mode(False)
    return loss_history


def _fit_theta_only(model, emb, i_idx, y, steps=200, lr=0.1):
    """Cold-start a new subject: item params frozen, fit a single theta."""
    import torch

    theta = torch.zeros(1, device=model.device, requires_grad=True)
    i_t = torch.as_tensor(i_idx, dtype=torch.long, device=model.device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        b, d = model.item_params(emb[i_t])
        c = (torch.sigmoid(model.logit_c[i_t]) * 0.5
             if model.logit_c is not None else None)
    opt = torch.optim.AdamW([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p = torch.sigmoid(d * (theta - b))
        if c is not None:
            p = c + (1 - c) * p
        p = p.clamp(1e-6, 1 - 1e-6)
        loss = -(y_t * torch.log(p) + (1 - y_t) * torch.log(1 - p)).mean()
        loss = loss + 0.25 * theta.square().mean()
        loss.backward()
        opt.step()
    return theta.detach()


def _score(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return auc, float(np.mean((p - y) ** 2))


def _load_filtered(min_subjects):
    import pandas as pd

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    items_meta = pd.read_parquet(PROCESSED / "items.parquet",
                                 columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")
    d = obs.merge(items_meta, on="item_uid", how="left")
    d = d.drop_duplicates(["content_hash", "subject_uid"])
    var = d.groupby("item_uid")["label"].agg(["mean"])
    d = d[d["item_uid"].isin(var[(var["mean"] > 0) & (var["mean"] < 1)].index)]
    n_sub = d.groupby("item_uid")["subject_uid"].nunique()
    d = d[d["item_uid"].isin(n_sub[n_sub >= min_subjects].index)]
    n_item = d.groupby("subject_uid")["item_uid"].nunique()
    d = d[d["subject_uid"].isin(n_item[n_item >= MIN_ITEMS_PER_TASK].index)]
    return d


# CPU only. The heads are evaluated once over the unique items rather than
# once per response, which made this ~17x cheaper and removed any need for a
# GPU -- ~0.1 s/step on CPU. Modal CPU is far cheaper than an L4.
@app.function(image=image, cpu=16, memory=65_536, volumes={"/data": volume},
              timeout=6 * 60 * 60)
def run(seed: int = SEEDS[0], arm: str = DEFAULT_ARM) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    store = DATASETS / f"allbench-embed-{arm}-v2"
    emb_items = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
    emb_all = np.load(store / "item_embeddings.npy", mmap_mode="r")
    emb_pos = {u: i for i, u in enumerate(emb_items)}
    print(f"embedding store {arm}: {emb_all.shape}", flush=True)

    d = _load_filtered(MIN_SUBJECTS)
    print(f"filtered (n>={MIN_SUBJECTS}): {len(d):,} rows, "
          f"{d.benchmark_id.nunique()} benchmarks", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device: {device}", flush=True)

    rows, weights_all = [], {}
    rng_global = np.random.default_rng(seed)
    benches = list(d.groupby("benchmark_id"))

    for bench, dd in tqdm(benches, desc=f"seed={seed} arm={arm}", unit="bench",
                          mininterval=5.0):
        subs = sorted(dd["subject_uid"].unique())
        its = sorted(dd["item_uid"].unique())
        if len(subs) < MIN_MATRIX or len(its) < MIN_MATRIX:
            continue
        if not all(u in emb_pos for u in its):
            missing = sum(u not in emb_pos for u in its)
            print(f"{bench}: {missing}/{len(its)} items lack embeddings, skipping", flush=True)
            continue

        s_map = {s: i for i, s in enumerate(subs)}
        i_map = {s: i for i, s in enumerate(its)}
        emb = torch.as_tensor(
            np.asarray(emb_all[[emb_pos[u] for u in its]], dtype=np.float32)
        ).to(device)
        emb_dim = emb.shape[1]

        si = dd["subject_uid"].map(s_map).to_numpy()
        ii = dd["item_uid"].map(i_map).to_numpy()
        yy = dd["label"].to_numpy().astype(np.float32)

        rng = np.random.default_rng([seed, hash(bench) % (2**31)])

        for kind in ("1pl", "2pl", "3pl"):
            # ---------- protocol A: random 20% cell holdout ----------
            perm = rng.permutation(len(yy))
            n_test = int(len(yy) * HOLDOUT_FRAC)
            te, tr = perm[:n_test], perm[n_test:]
            model = AmortizedIRT(len(subs), emb_dim, kind, len(its),
                                 int(rng_global.integers(0, 2**31)), device=device)
            hist = _fit(model, emb, si[tr], ii[tr], yy[tr],
                        desc=f"{bench[:14]} {kind} cell")
            with torch.no_grad():
                i_te = torch.as_tensor(ii[te], dtype=torch.long, device=device)
                s_te = torch.as_tensor(si[te], dtype=torch.long, device=device)
                p = model.probs(model.theta[s_te], emb[i_te], i_te).cpu().numpy()
            auc, brier = _score(yy[te], p)
            rows.append({"benchmark_id": bench, "model": f"amort_{kind}", "arm": arm,
                         "seed": seed, "axis": "cell", "n_score": len(te),
                         "label": yy[te], "pred_prob": p, "auc": auc, "brier": brier})
            weights_all[f"{bench}_amort_{kind}"] = {
                **model.state(), "loss_history": hist,
                "item_ids": np.array(its), "subject_ids": np.array(subs)}

            # ---------- protocol B: held-out subjects and items ----------
            held_s = rng.choice(len(subs), max(1, int(len(subs) * HELD_OUT_FRAC)), replace=False)
            held_i = rng.choice(len(its), max(1, int(len(its) * HELD_OUT_FRAC)), replace=False)
            core = ~(np.isin(si, held_s) | np.isin(ii, held_i))
            if core.sum() < MIN_MATRIX:
                continue
            gmodel = AmortizedIRT(len(subs), emb_dim, kind, len(its),
                                  int(rng_global.integers(0, 2**31)), device=device)
            _fit(gmodel, emb, si[core], ii[core], yy[core],
                 desc=f"{bench[:14]} {kind} gen")

            # subject axis: split-half, theta fit on half 1
            for row in held_s:
                sel = np.flatnonzero((si == row) & ~np.isin(ii, held_i))
                if len(sel) < 2 * MIN_HALF:
                    continue
                sh = rng.permutation(sel)
                h1, h2 = sh[: len(sh) // 2], sh[len(sh) // 2:]
                th = _fit_theta_only(gmodel, emb, ii[h1], yy[h1])
                with torch.no_grad():
                    i2 = torch.as_tensor(ii[h2], dtype=torch.long, device=device)
                    p2 = gmodel.probs(th, emb[i2], i2).cpu().numpy()
                if len(np.unique(yy[h2])) < 2:
                    continue
                auc, brier = _score(yy[h2], p2)
                rows.append({"benchmark_id": bench, "model": f"amort_{kind}", "arm": arm,
                             "seed": seed, "axis": "subject", "n_score": len(h2),
                             "label": yy[h2], "pred_prob": p2, "auc": auc, "brier": brier})

            # item axis: COLD START -- b and d straight from the embedding, the
            # held-out item contributes no responses to the fit at all.
            for col in held_i:
                sel = np.flatnonzero((ii == col) & ~np.isin(si, held_s))
                if len(sel) < 2 * MIN_HALF or len(np.unique(yy[sel])) < 2:
                    continue
                with torch.no_grad():
                    ic = torch.as_tensor(ii[sel], dtype=torch.long, device=device)
                    sc = torch.as_tensor(si[sel], dtype=torch.long, device=device)
                    pc = gmodel.probs(gmodel.theta[sc], emb[ic], ic).cpu().numpy()
                auc, brier = _score(yy[sel], pc)
                rows.append({"benchmark_id": bench, "model": f"amort_{kind}", "arm": arm,
                             "seed": seed, "axis": "item_coldstart", "n_score": len(sel),
                             "label": yy[sel], "pred_prob": pc, "auc": auc, "brier": brier})

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT.with_name(f"{OUT.stem}-{arm}-seed{seed}{OUT.suffix}"), index=False)
    torch.save(weights_all, OUT.parent / f"amortized-irt-{arm}-weights-seed{seed}.pt")
    volume.commit()
    return {"rows": len(frame), "benchmarks": int(frame["benchmark_id"].nunique()) if len(frame) else 0,
            "seed": seed, "arm": arm, "weights_saved": len(weights_all)}


@app.local_entrypoint()
def main(arm: str = DEFAULT_ARM) -> None:
    handles = [run.spawn(seed, arm) for seed in SEEDS]
    for h in handles:
        print(h.get())
