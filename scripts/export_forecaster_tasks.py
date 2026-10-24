"""Export the five validation tasks used by the AI Evaluation Forecaster demo.

Run from the repository root:

    .venv/bin/modal run scripts/export_forecaster_tasks.py
    .venv/bin/modal volume get aims-neurips-data \
        /results/demo/tasks.json demo/forecaster/assets/tasks.json

The export keeps 31 acquisition items and 60 evaluation items per task.  The
evaluation labels are stored in the asset file for the local scorer, but the
server never returns them until the user explicitly asks to score a run.
"""

import modal
app = modal.App("aims-demo-export")
vol = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = modal.Image.debian_slim().pip_install("pandas", "pyarrow")

SELECTED = [
    "subject_ba6ce5739b390727fcc5::afrimedqa",
    "subject_b47e893a3087d8777292::appworld",
    "subject_2b559ae1200f7ebc1fa8::cmmlu",
    "subject_c007455acf50cfee436d::cruxeval",
    "subject_34628ce85da1bf6e31c0::devbench",
]

@app.function(volumes={"/data": vol}, image=image, timeout=900)
def run():
    import json, pandas as pd, numpy as np
    from pathlib import Path
    ev = pd.read_parquet("/data/datasets/full/allbench-features-v1/eval_items.parquet")
    ev = ev[ev.split == "validation"].copy()
    subjects = pd.read_parquet("/data/datasets/full/processed/subjects.parquet")
    benchmarks = pd.read_parquet("/data/datasets/full/processed/benchmarks.parquet")
    items = pd.read_parquet(
        "/data/datasets/full/processed/items.parquet",
        columns=["item_id", "raw_item_id", "benchmark_id", "content", "item_features", "content_chars"],
    ).reset_index(names="item_index")
    bench_lookup = benchmarks.set_index("benchmark_id").to_dict("index")
    rng = np.random.default_rng(20261021)

    payload = {"generated_for": "AI Evaluation Forecaster demo", "protocol": "validation-only, 31 acquisition / 60 evaluation", "tasks": []}
    for task_id in SELECTED:
        rows = ev[ev.task_id == task_id].copy()
        if rows.empty:
            raise RuntimeError(f"missing task {task_id}")
        first = rows.iloc[0]
        subject = subjects.iloc[int(first.subject_index)]
        benchmark_id = str(first.benchmark_id)
        benchmark = bench_lookup[benchmark_id]
        acquisition = rows[rows.pool == "acquisition"].sort_values("acquisition_rank").head(31).copy()
        evaluation = rows[rows.pool == "evaluation"].copy()
        if len(evaluation) > 60:
            evaluation = evaluation.iloc[rng.permutation(len(evaluation))[:60]]
        evaluation = evaluation.sort_index()

        def item_records(frame):
            out = []
            for row in frame.itertuples():
                item = items.iloc[int(row.item_index)]
                out.append(
                    {
                        "item_id": str(item.item_id),
                        "raw_item_id": str(item.raw_item_id),
                        "benchmark_id": str(item.benchmark_id),
                        "content": str(item.content or ""),
                        "item_features": str(item.item_features or ""),
                        "content_chars": int(item.content_chars or 0),
                        "label": int(row.label),
                        "acquisition_rank": None if pd.isna(row.acquisition_rank) else int(row.acquisition_rank),
                    }
                )
            return out

        payload["tasks"].append(
            {
                "task_id": task_id,
                "benchmark_id": benchmark_id,
                "benchmark": {
                    "name": str(benchmark.get("name") or benchmark_id),
                    "description": str(benchmark.get("one_line_description") or benchmark.get("description") or ""),
                    "domain": str(benchmark.get("domain") or ""),
                    "response_type": str(benchmark.get("response_type") or ""),
                },
                "subject": {
                    "display_name": str(subject.get("display_name") or subject.get("normalized_name") or "Unknown subject"),
                    "normalized_name": str(subject.get("normalized_name") or ""),
                    "provider": str(subject.get("provider") or ""),
                    "release_date": str(subject.get("release_date") or ""),
                    "access_date": str(subject.get("access_date") or ""),
                    "harness": str(subject.get("harness") or ""),
                    "reasoning_effort": str(subject.get("reasoning_effort") or ""),
                    "harness_version": str(subject.get("harness_version") or ""),
                    "subject_features_extra": str(subject.get("subject_features_extra") or ""),
                },
                "acquisition": item_records(acquisition),
                "evaluation": item_records(evaluation),
            }
        )
        print(task_id, "acq", len(acquisition), "eval", len(evaluation), "eval mean", float(evaluation.label.mean()))

    out = Path("/data/results/demo/tasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    vol.commit()
    print("wrote", out, out.stat().st_size)
    return str(out)

if __name__ == "__main__":
    with app.run():
        run.remote()
