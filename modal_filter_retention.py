"""How much data survives filtering, and does classic IRT recover item pass rate.

Two things in one pass per benchmark, matching modal_classic_irt_filter_test.py's
filter order exactly (dedupe content -> drop identical-response items -> drop
items with <15 subjects -> drop subjects with <80 items):

1. Retention counts: rows/items/subjects/benchmarks before vs after each filter
   step, so we can see where data is actually lost.
2. A lightweight 2PL fit (no text) on the AFTER-filtering matrix only, with no
   train/test split -- every cell is used, since the goal here is "does the
   fitted item difficulty recover the observed item pass rate", not held-out
   prediction. Dumps per-item (predicted pass rate under the fitted item
   parameters averaged over that benchmark's fitted subjects, observed pass
   rate) for the model-vs-data visualization.

Run:  modal run --detach modal_filter_retention.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-filter-retention"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT_COUNTS = Path("/data/results/data-analysis/filter-retention.parquet")
OUT_ITEMS = Path("/data/results/data-analysis/filter-2pl-items.parquet")

MIN_SUBJECTS = 15
MIN_ITEMS_PER_TASK = 80
STEPS = 300

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")


@app.function(image=image, cpu=16, memory=131_072, volumes={"/data": volume},
              timeout=6 * 60 * 60)
def run(seed: int = 20261018) -> dict:
    import pandas as pd
    import torch

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    items_meta = pd.read_parquet(PROCESSED / "items.parquet",
                                 columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")

    def snapshot(d, step):
        return {"step": step, "rows": len(d), "items": d["item_uid"].nunique(),
                "subjects": d["subject_uid"].nunique(), "benchmarks": d["benchmark_id"].nunique()}

    overall = [snapshot(obs, "0_raw")]
    d = obs.merge(items_meta, on="item_uid", how="left")
    d = d.drop_duplicates(["content_hash", "subject_uid"])
    overall.append(snapshot(d, "1_dedupe_content"))
    var = d.groupby("item_uid")["label"].agg(["mean", "size"])
    keep_items = var[(var["mean"] > 0) & (var["mean"] < 1)].index
    d = d[d["item_uid"].isin(keep_items)]
    overall.append(snapshot(d, "2_drop_identical_response"))
    n_sub_per_item = d.groupby("item_uid")["subject_uid"].nunique()
    d = d[d["item_uid"].isin(n_sub_per_item[n_sub_per_item >= MIN_SUBJECTS].index)]
    overall.append(snapshot(d, f"3_min_{MIN_SUBJECTS}_subjects_per_item"))
    n_item_per_task = d.groupby("subject_uid")["item_uid"].nunique()
    d = d[d["subject_uid"].isin(n_item_per_task[n_item_per_task >= MIN_ITEMS_PER_TASK].index)]
    overall.append(snapshot(d, f"4_min_{MIN_ITEMS_PER_TASK}_items_per_task"))
    print("overall retention:", overall, flush=True)

    per_bench = []
    for bench, before in obs.groupby("benchmark_id"):
        after = d[d["benchmark_id"] == bench]
        per_bench.append({
            "benchmark_id": bench,
            "rows_before": len(before), "rows_after": len(after),
            "items_before": before["item_uid"].nunique(), "items_after": after["item_uid"].nunique(),
            "subjects_before": before["subject_uid"].nunique(),
            "subjects_after": after["subject_uid"].nunique(),
        })

    # ---- 2PL fit on the filtered matrix, item predictions vs observed ----
    item_rows = []
    weights_all = {}
    for bench, dd in d.groupby("benchmark_id"):
        subs = sorted(dd["subject_uid"].unique())
        its = sorted(dd["item_uid"].unique())
        if len(subs) < 5 or len(its) < 5:
            continue
        s_map = {s: i for i, s in enumerate(subs)}
        i_map = {s: i for i, s in enumerate(its)}
        s_idx = dd["subject_uid"].map(s_map).to_numpy()
        i_idx = dd["item_uid"].map(i_map).to_numpy()
        y = torch.as_tensor(dd["label"].to_numpy(dtype="float32"))
        s_t = torch.as_tensor(s_idx, dtype=torch.long)
        i_t = torch.as_tensor(i_idx, dtype=torch.long)

        torch.manual_seed(seed)
        theta = torch.zeros(len(subs), requires_grad=True)
        b = torch.zeros(len(its), requires_grad=True)
        log_a = torch.zeros(len(its), requires_grad=True)
        opt = torch.optim.AdamW([theta, b, log_a], lr=0.1)
        for _ in range(STEPS):
            opt.zero_grad()
            a = torch.exp(log_a)
            z = a[i_t] * (theta[s_t] - b[i_t])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss = loss + 0.125 * (theta.square().mean() + b.square().mean())
            loss.backward()
            opt.step()

        with torch.no_grad():
            a = torch.exp(log_a)
            # predicted pass rate per item = mean over the benchmark's fitted
            # subjects (same quantity as the earlier "pass rate" analysis,
            # marginalized over who answered rather than conditioned on it).
            z_all = a[None, :] * (theta[:, None] - b[None, :])
            pred_item = torch.sigmoid(z_all).mean(dim=0).numpy()
        obs_item = dd.groupby("item_uid")["label"].agg(["mean", "size"])
        obs_item = obs_item.reindex(its)
        item_rows.append(pd.DataFrame({
            "benchmark_id": bench, "item_uid": its, "seed": seed,
            "pred_pass_rate": pred_item, "obs_pass_rate": obs_item["mean"].to_numpy(),
            "n_responses": obs_item["size"].to_numpy(),
        }))
        weights_all[f"{bench}_2pl"] = {
            "theta": theta.detach().numpy(), "b": b.detach().numpy(), "log_a": log_a.detach().numpy(),
            "subject_ids": np.array(subs), "item_ids": np.array(its)
        }
        print(f"{bench:<20}2PL fit  subs={len(subs):<6}items={len(its):<8}", flush=True)

    OUT_COUNTS.parent.mkdir(parents=True, exist_ok=True)
    counts_out = OUT_COUNTS.with_name(f"{OUT_COUNTS.stem}-seed{seed}{OUT_COUNTS.suffix}")
    items_out = OUT_ITEMS.with_name(f"{OUT_ITEMS.stem}-seed{seed}{OUT_ITEMS.suffix}")
    pd.DataFrame(per_bench).to_parquet(counts_out, index=False)
    pd.concat(item_rows).to_parquet(items_out, index=False)
    weights_out = OUT_COUNTS.parent / f"filter-retention-weights-seed{seed}.pt"
    torch.save(weights_all, weights_out)
    volume.commit()
    return {"overall": overall, "benchmarks_fit": len(item_rows), "seed": seed, "weights_saved": len(weights_all)}


SEEDS = (20261018, 20261019, 20261020)


@app.local_entrypoint()
def main() -> None:
    handles = [run.spawn(seed) for seed in SEEDS]
    for h in handles:
        print(h.get())
