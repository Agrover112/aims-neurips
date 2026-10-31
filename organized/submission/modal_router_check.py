"""Offline test of per-budget routing and blending, with an honest split.

``modal_per_budget_brier.py`` showed the two benchmark-holdout checkpoints are
mirror images: the uniform-budget run wins every budget except 1, and the
ladder run wins budget 1 by enough to matter. Taking the best cell from each
gave an ALC of 0.1875 against the shipped baseline's 0.1921.

That number is not trustworthy as stated, because the choice of which model
wins each budget was made on the same episodes it was then scored on. Six
binary choices over 390 subjects is little capacity to overfit, but "little"
is not "none", and the whole reason this session distrusts the fixed bank is
selection on the thing being reported.

So the validation subjects are split in half. Routing weights are fitted on
the first half only and every reported number comes from the second half,
which the fit never saw. Candidates:

``uniform`` / ``ladder`` / ``baseline``
    The three predictors alone, for reference.
``hard_route``
    Per budget, whichever single predictor is best on the fit half.
``blend``
    Per budget, the convex combination of all three that minimises Brier on
    the fit half, found by a simplex grid. Brier is a proper scoring rule and
    convex in the prediction, so the combination is well behaved and can only
    beat the best single member on the data it was fitted to -- which is
    exactly why it has to be reported out of sample.

Run::

    modal run --profile aims-test organized/submission/modal_router_check.py::main
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-np-router-check"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
CHECKPOINTS = Path("/data/results/cnp/checkpoints")
RESULT_PATH = Path("/data/results/submission/router-check.json")

UNIFORM_LABEL = "anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029"
LADDER_LABEL = "anp-sa1-w64-bgem3-bhold-budgetladder-n15-20261029"

BUDGETS = (0, 1, 3, 7, 15, 31)
BUDGET_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
MAX_TARGETS = 256
MIN_TARGETS = 8
GRID_STEP = 20          # simplex grid resolution, 1/20 = 0.05 increments

# Copied from organized/submission/legacy/submission/aims_predictor/predictor.py, not re-derived.
PRIOR_MEAN = 0.5
PRIOR_STRENGTH = 4.0
SUBJECT_STRENGTH = 8.0

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "torch==2.5.1"
)

_HERE = Path(__file__).resolve().parent
_TRAINING = (_HERE.parent / "training").resolve()
for _name in ("cnp_blocks.py", "anp_crossattn_blocks.py", "benchmark_holdout_filters.py"):
    _path = _TRAINING / _name
    if _path.exists():
        image = image.add_local_file(_path, f"/root/{_name}")


def _baseline_probability(successes: int, count: int) -> float:
    subject_mean = (PRIOR_STRENGTH * PRIOR_MEAN + successes) / (PRIOR_STRENGTH + count)
    estimate = (SUBJECT_STRENGTH * subject_mean + successes) / (SUBJECT_STRENGTH + count)
    return float(min(1.0, max(0.0, estimate)))


def _simplex(step: int):
    """Every convex weight triple on a ``1/step`` grid."""
    for a in range(step + 1):
        for b in range(step + 1 - a):
            yield (a / step, b / step, (step - a - b) / step)


@app.function(
    image=image,
    cpu=4,
    memory=32_768,
    volumes={"/data": volume},
    timeout=2 * 60 * 60,
)
def check(arm: str = "bge_m3_long", seed: int = 20261018,
          min_item_responders: int = 15, min_items_per_subject: int = 80) -> dict:
    import sys

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/root")
    import anp_crossattn_blocks as anp_cb
    import benchmark_holdout_filters as benchmark_filters

    device = torch.device("cpu")
    torch.set_num_threads(4)

    def _load_model(label, emb_dim):
        matches = sorted(p for p in CHECKPOINTS.glob("*.pt")
                         if label in p.name and not p.name.endswith(".resume.pt"))
        if len(matches) != 1:
            raise RuntimeError(f"{label!r} matched {[p.name for p in matches]}")
        checkpoint = torch.load(matches[0], map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        if cfg["arm"] != arm:
            raise RuntimeError(f"{label} was trained on {cfg['arm']!r}, not {arm!r}")
        model = anp_cb.AttentiveNeuralProcess(
            emb_dim, width=int(cfg["width"]),
            self_attention_blocks=int(cfg["n_blocks"]),
            cross_attention_heads=int(cfg["cross_attention_heads"]),
            use_latent=bool(cfg["use_latent"]), latent_dim=int(cfg["latent_dim"]),
            residual_proj=bool(cfg["residual_proj"]), dropout=float(cfg["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        print(f"  {label}: step {checkpoint.get('selection', {}).get('step')} "
              f"brier {checkpoint.get('selection', {}).get('value')}", flush=True)
        return model

    d, audit = benchmark_filters.load_benchmark_holdout_filtered(
        PROCESSED, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject, split_seed=seed,
    )
    store = DATASETS / f"allbench-embed-{arm}-v2"
    emb_ids = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
    emb_all = np.load(store / "item_embeddings.npy", mmap_mode="r")
    pos = {u: i for i, u in enumerate(emb_ids)}
    d = d[d["item_uid"].isin(pos.keys())]
    items_all = sorted(d["item_uid"].unique())
    i_map = {u: i for i, u in enumerate(items_all)}
    emb = torch.as_tensor(
        np.asarray(emb_all[[pos[u] for u in items_all]], dtype=np.float32)).to(device)
    print(f"pooled item matrix: {tuple(emb.shape)}", flush=True)

    tasks = []
    for (subject, benchmark), g in d.groupby(["subject_uid", "benchmark_id"]):
        if str(g["benchmark_split"].iloc[0]) != "validation":
            continue
        tasks.append({"items": g["item_uid"].map(i_map).to_numpy(),
                      "labels": g["label"].to_numpy().astype(np.float32)})
    print(f"validation tasks: {len(tasks):,}", flush=True)

    models = {"uniform": _load_model(UNIFORM_LABEL, emb.shape[1]),
              "ladder": _load_model(LADDER_LABEL, emb.shape[1])}

    # ---- collect predictions once; blending is then free --------------------
    top = max(BUDGETS)
    rng = np.random.default_rng(seed + 777)
    episodes = []
    with torch.no_grad():
        for task in tasks:
            items, labels = task["items"], task["labels"]
            if len(items) < top + MIN_TARGETS:
                continue
            perm = rng.permutation(len(items))
            target = perm[top:top + MAX_TARGETS]
            y = labels[target]
            x_t = emb[torch.as_tensor(items[target], dtype=torch.long)]
            record = {"y": y, "p": {}}
            for k in BUDGETS:
                context = perm[:k]
                y_c = labels[context]
                x_c = emb[torch.as_tensor(items[context], dtype=torch.long)]
                y_c_t = torch.as_tensor(y_c, dtype=torch.float32)
                row = [torch.sigmoid(models[name].logits(x_c, y_c_t, x_t)).numpy()
                       for name in ("uniform", "ladder")]
                row.append(np.full_like(y, _baseline_probability(int(y_c.sum()), int(k))))
                record["p"][k] = row
            episodes.append(record)
    print(f"episodes scored: {len(episodes):,}", flush=True)

    # ---- honest split: fit routing on half, report on the other half -------
    split = np.arange(len(episodes)) % 2 == 0
    fit_ix = np.flatnonzero(split)
    report_ix = np.flatnonzero(~split)
    print(f"fit subjects {len(fit_ix)} / report subjects {len(report_ix)}", flush=True)

    def brier(indices, k, weights):
        total = 0.0
        for i in indices:
            episode = episodes[i]
            u, l, b = episode["p"][k]
            p = weights[0] * u + weights[1] * l + weights[2] * b
            total += float(((p - episode["y"]) ** 2).mean())
        return total / max(len(indices), 1)

    pure = {"uniform": (1.0, 0.0, 0.0), "ladder": (0.0, 1.0, 0.0),
            "baseline": (0.0, 0.0, 1.0)}
    grid = list(_simplex(GRID_STEP))

    rows, chosen_route, chosen_blend = {}, {}, {}
    for k in BUDGETS:
        row = {name: brier(report_ix, k, w) for name, w in pure.items()}

        route_name = min(pure, key=lambda n: brier(fit_ix, k, pure[n]))
        chosen_route[k] = route_name
        row["hard_route"] = brier(report_ix, k, pure[route_name])

        best = min(grid, key=lambda w: brier(fit_ix, k, w))
        chosen_blend[k] = best
        row["blend"] = brier(report_ix, k, best)
        rows[k] = row

    names = ("uniform", "ladder", "baseline", "hard_route", "blend")
    alc = {n: float(sum(w * rows[k][n] for k, w in zip(BUDGETS, BUDGET_WEIGHTS)))
           for n in names}

    print("\nreport half only -- routing never saw these subjects", flush=True)
    print(f"{'budget':>7} " + "".join(f"{n:>12}" for n in names)
          + f"{'route':>10}{'blend w (u,l,b)':>22}", flush=True)
    for k in BUDGETS:
        w = chosen_blend[k]
        print(f"{k:>7} " + "".join(f"{rows[k][n]:>12.5f}" for n in names)
              + f"{chosen_route[k]:>10}"
              + f"{f'{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}':>22}", flush=True)
    print(f"{'ALC':>7} " + "".join(f"{alc[n]:>12.5f}" for n in names), flush=True)
    print("", flush=True)
    for n in ("uniform", "ladder", "hard_route", "blend"):
        delta = alc[n] - alc["baseline"]
        verdict = "BEATS baseline" if delta < 0 else "loses to baseline"
        print(f"  {n:>11}: {delta:+.5f} vs baseline -> {verdict}", flush=True)

    summary = {
        "arm": arm, "seed": seed,
        "uniform_label": UNIFORM_LABEL, "ladder_label": LADDER_LABEL,
        "n_episodes": len(episodes),
        "n_fit_subjects": int(len(fit_ix)), "n_report_subjects": int(len(report_ix)),
        "budgets": list(BUDGETS), "budget_weights": list(BUDGET_WEIGHTS),
        "report_half_brier": {str(k): rows[k] for k in BUDGETS},
        "alc_report_half": alc,
        "hard_route_choice": {str(k): chosen_route[k] for k in BUDGETS},
        "blend_weights": {str(k): list(chosen_blend[k]) for k in BUDGETS},
        "blend_weight_order": ["uniform", "ladder", "baseline"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(arm: str = "bge_m3_long", seed: int = 20261018,
         min_item_responders: int = 15, min_items_per_subject: int = 80) -> None:
    check.remote(arm=arm, seed=seed, min_item_responders=min_item_responders,
                 min_items_per_subject=min_items_per_subject)
