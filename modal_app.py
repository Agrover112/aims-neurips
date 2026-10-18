"""Modal control plane for PAIEC data, experiments, and artifacts.

Examples
--------
modal run modal_app.py --action health
modal run modal_app.py --action inspect
modal run modal_app.py --action evaluate-pilot
modal run modal_app.py --action evaluate-full
modal run modal_ingest.py --dataset-name pilot --limit 2
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub>=1.0,<2",
        "duckdb>=1.4,<2",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "wandb>=0.17,<1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .add_local_dir(PROJECT_ROOT / "scripts", "/opt/aims/scripts", copy=True)
    .add_local_dir(PROJECT_ROOT / "configs", "/opt/aims/configs", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src:/opt/aims/scripts",
            "HF_HOME": "/tmp/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

neural_image = image.pip_install(
    "torch==2.5.1",
    index_url="https://download.pytorch.org/whl/cpu",
)


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
        raise ValueError("names must contain only lowercase letters, digits, '.', '_' or '-'")
    return value


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_report(name: str) -> dict[str, Any]:
    root = REMOTE_ROOT / "datasets" / name
    raw = root / "raw"
    processed = root / "processed"
    episodes = root / "episodes"
    return {
        "name": name,
        "raw_ready": (raw / "download_manifest.json").exists(),
        "processed_ready": (processed / "manifest.json").exists(),
        "raw_bytes": _directory_bytes(raw),
        "processed_bytes": _directory_bytes(processed),
        "episodes_ready": (episodes / "manifest.json").exists(),
        "episodes_bytes": _directory_bytes(episodes),
        "download_manifest": _read_json(raw / "download_manifest.json"),
        "processed_manifest": _read_json(processed / "manifest.json"),
        "episodes_manifest": _read_json(episodes / "manifest.json"),
    }


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, timeout=10 * 60)
def health() -> dict[str, Any]:
    """Verify the code image and persistent Volume without requiring HF access."""

    from aims_starter.metrics import DEFAULT_BUDGETS

    REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
    return {
        "app": APP_NAME,
        "volume": VOLUME_NAME,
        "volume_root": str(REMOTE_ROOT),
        "budgets": list(DEFAULT_BUDGETS),
        "utc": datetime.now(UTC).isoformat(),
    }


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    cpu=4,
    memory=16_384,
    timeout=2 * 60 * 60,
)
def evaluate_beta_binomial(
    dataset_name: str = "full",
    benchmark_split: str = "test",
    minimum_items: int = 80,
    seed: int = 20261017,
) -> dict[str, Any]:
    """Run the first competition-shaped baseline and persist its result."""

    import pandas as pd

    from aims_starter.evaluation import simulate_beta_binomial

    dataset_name = _safe_name(dataset_name)
    if benchmark_split not in {"all", "train", "validation", "test"}:
        raise ValueError("benchmark_split must be all, train, validation, or test")
    processed = REMOTE_ROOT / "datasets" / dataset_name / "processed"
    observations_path = processed / "observations.parquet"
    if not observations_path.exists():
        raise FileNotFoundError(f"Prepared observations not found: {observations_path}")

    observations = pd.read_parquet(
        observations_path,
        filters=(
            [("split_by_benchmark", "==", benchmark_split)]
            if benchmark_split != "all"
            else None
        ),
    )
    episode_path = REMOTE_ROOT / "datasets" / dataset_name / "episodes" / "episode_items.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(
            f"Frozen episode index not found: {episode_path}. Run --action prepare-episodes."
        )
    episode_index = pd.read_parquet(episode_path)
    result = simulate_beta_binomial(
        observations,
        seed=seed,
        minimum_items=minimum_items,
        benchmark_split=None if benchmark_split == "all" else benchmark_split,
        episode_index=episode_index,
    )
    run_name = _safe_name(
        f"beta-binomial-{dataset_name}-{benchmark_split}-episodes-v1-"
        f"min{minimum_items}-seed{seed}"
    )
    output = REMOTE_ROOT / "results" / run_name / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stored = _read_json(output)
        if stored != result:
            raise RuntimeError(f"Existing result differs from reproducible run: {output}")
    else:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        volume.commit()
    return {
        "run_name": run_name,
        "result_path": str(output),
        "n_tasks": result["n_tasks"],
        "brier_alc": result["brier_alc"],
        "by_budget": result["by_budget"],
    }


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    cpu=4,
    memory=16_384,
    timeout=2 * 60 * 60,
)
def prepare_episodes(
    dataset_name: str = "full",
    minimum_items: int = 80,
    seed: int = 20261017,
) -> dict[str, Any]:
    """Create the immutable shared acquisition/evaluation assignment artifact."""

    import shutil
    import tempfile

    import pandas as pd

    from aims_starter.data import prepare_episode_index

    dataset_name = _safe_name(dataset_name)
    dataset_root = REMOTE_ROOT / "datasets" / dataset_name
    observations_path = dataset_root / "processed" / "observations.parquet"
    if not observations_path.exists():
        raise FileNotFoundError(f"Prepared observations not found: {observations_path}")
    destination = dataset_root / "episodes"
    existing = _read_json(destination / "manifest.json")
    if existing is not None:
        if existing.get("seed") != seed or existing.get("minimum_items") != minimum_items:
            raise RuntimeError(
                "An episode index already exists with different settings; use another "
                "dataset snapshot instead of overwriting it."
            )
        return {"status": "reused", **existing, "output": str(destination)}
    if destination.exists():
        raise RuntimeError(f"Incomplete episode directory exists: {destination}")

    print(f"Reading observations: {observations_path}", flush=True)
    observations = pd.read_parquet(observations_path)
    print(f"Loaded {len(observations):,} response rows", flush=True)
    with tempfile.TemporaryDirectory(prefix=f"aims-episodes-{dataset_name}-") as temporary:
        stage = Path(temporary) / "episodes"
        print("Building grouped episode assignments", flush=True)
        prepared = prepare_episode_index(
            observations,
            stage,
            seed=seed,
            minimum_items=minimum_items,
        )
        print(f"Built {prepared.manifest['rows']['episode_items']:,} episode items", flush=True)
        shutil.copytree(stage, destination)
        volume.commit()
    print("Committed episode index to the Modal Volume", flush=True)
    return {"status": "created", **prepared.manifest, "output": str(destination)}


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    cpu=16,
    memory=32_768,
    timeout=30 * 60,
)
def benchmark_duckdb(dataset_name: str = "full") -> dict[str, Any]:
    """Time the expensive task-item aggregation with parallel DuckDB."""

    import time

    import duckdb

    dataset_name = _safe_name(dataset_name)
    observations_path = (
        REMOTE_ROOT / "datasets" / dataset_name / "processed" / "observations.parquet"
    )
    if not observations_path.exists():
        raise FileNotFoundError(f"Prepared observations not found: {observations_path}")

    print(f"DuckDB benchmark reading {observations_path}", flush=True)
    started = time.perf_counter()
    connection = duckdb.connect()
    connection.execute("SET threads = 16")
    connection.execute("SET preserve_insertion_order = false")
    item_rows, task_count = connection.execute(
        """
        WITH item_summary AS (
            SELECT task_id, item_uid,
                   min(observation_id) AS label_observation_id,
                   count(*) AS n_response_rows
            FROM read_parquet(?)
            GROUP BY task_id, item_uid
        )
        SELECT count(*) AS item_rows, count(DISTINCT task_id) AS task_count
        FROM item_summary
        """,
        [str(observations_path)],
    ).fetchone()
    elapsed = time.perf_counter() - started
    result = {
        "dataset_name": dataset_name,
        "item_rows": int(item_rows),
        "task_count": int(task_count),
        "elapsed_seconds": elapsed,
        "cpu_cores": 16,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    cpu=4,
    memory=16_384,
    timeout=20 * 60,
)
def smoke_one_benchmark(
    dataset_name: str = "full",
    minimum_tasks: int = 4,
) -> dict[str, Any]:
    """Validate one compact multi-subject benchmark through an amortized 2PL pass."""

    import pandas as pd

    from aims_starter.smoke import run_benchmark_smoke

    dataset_name = _safe_name(dataset_name)
    dataset_root = REMOTE_ROOT / "datasets" / dataset_name
    processed = dataset_root / "processed"
    episodes = dataset_root / "episodes"
    required = {
        "tasks": processed / "tasks.parquet",
        "observations": processed / "observations.parquet",
        "items": processed / "items.parquet",
        "subjects": processed / "subjects.parquet",
        "episode_items": episodes / "episode_items.parquet",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Smoke-test inputs are missing: {missing}")

    tasks = pd.read_parquet(required["tasks"])
    candidates = (
        tasks[(tasks["n_items"] >= 80) & (tasks["split_by_benchmark"] == "train")]
        .groupby("benchmark_id", as_index=False)
        .agg(
            n_tasks=("task_id", "size"),
            n_observations=("n_observations", "sum"),
            median_items=("n_items", "median"),
        )
    )
    candidates = candidates[candidates["n_tasks"] >= minimum_tasks].sort_values(
        ["n_observations", "benchmark_id"]
    )
    if candidates.empty:
        raise ValueError(f"No training benchmark has at least {minimum_tasks} eligible tasks")
    benchmark_id = str(candidates.iloc[0]["benchmark_id"])
    print(f"Selected smoke benchmark: {benchmark_id}", flush=True)

    filters = [("benchmark_id", "==", benchmark_id)]
    observations = pd.read_parquet(required["observations"], filters=filters)
    episode_items = pd.read_parquet(required["episode_items"], filters=filters)
    items = pd.read_parquet(required["items"], filters=filters)
    subjects = pd.read_parquet(required["subjects"], filters=filters)
    print(
        f"Loaded {len(observations):,} responses, {len(episode_items):,} assignments, "
        f"{len(items):,} items, {len(subjects):,} subject records",
        flush=True,
    )
    result = run_benchmark_smoke(observations, episode_items, items, subjects)
    output = REMOTE_ROOT / "results" / f"smoke-{benchmark_id}-episodes-v1" / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stored = _read_json(output)
        if stored != result:
            raise RuntimeError(f"Existing smoke result differs: {output}")
    else:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        volume.commit()
    return {**result, "result_path": str(output)}


@app.function(
    image=neural_image,
    volumes={str(REMOTE_ROOT): volume},
    cpu=8,
    memory=16_384,
    timeout=2 * 60 * 60,
)
def neural_pilot_one_benchmark(
    dataset_name: str = "full",
    benchmark_id: str = "beavertails",
    seed: int = 20261018,
) -> dict[str, Any]:
    """Compare Online-AIRT, NP-AIRT and ANP-AIRT on one held-out-subject pilot."""

    import pandas as pd

    from aims_starter.neural_pilot import run_neural_pilot

    dataset_name = _safe_name(dataset_name)
    benchmark_id = _safe_name(benchmark_id)
    run_name = f"neural-pilot-{benchmark_id}-loso-stacked-mhsa-v2-seed{seed}"
    output = REMOTE_ROOT / "results" / run_name
    stored = _read_json(output / "metrics.json")
    if stored is not None:
        return {"status": "reused", "result_path": str(output / "metrics.json"), **stored}
    processed = REMOTE_ROOT / "datasets" / dataset_name / "processed"
    episodes = REMOTE_ROOT / "datasets" / dataset_name / "episodes"
    filters = [("benchmark_id", "==", benchmark_id)]
    observations = pd.read_parquet(processed / "observations.parquet", filters=filters)
    episode_items = pd.read_parquet(episodes / "episode_items.parquet", filters=filters)
    items = pd.read_parquet(processed / "items.parquet", filters=filters)
    print(
        f"Neural pilot {benchmark_id}: {len(observations):,} observations, "
        f"{episode_items['task_id'].nunique()} tasks, {items['item_uid'].nunique()} items",
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    result = run_neural_pilot(
        observations,
        episode_items,
        items,
        output_dir=output,
        seed=seed,
    )
    volume.commit()
    return {"result_path": str(output / "metrics.json"), **result}


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, timeout=10 * 60)
def inspect_storage() -> dict[str, Any]:
    """Return manifests and byte counts without downloading stored artifacts."""

    datasets_root = REMOTE_ROOT / "datasets"
    dataset_names = (
        sorted(path.name for path in datasets_root.iterdir() if path.is_dir())
        if datasets_root.exists()
        else []
    )
    results_root = REMOTE_ROOT / "results"
    results = (
        sorted(str(path.relative_to(REMOTE_ROOT)) for path in results_root.rglob("metrics.json"))
        if results_root.exists()
        else []
    )
    return {
        "volume": VOLUME_NAME,
        "total_bytes": _directory_bytes(REMOTE_ROOT),
        "datasets": [_dataset_report(name) for name in dataset_names],
        "results": results,
    }


@app.function(image=image, timeout=60 * 60)
def score_prediction_record(predictions_text: str, config_text: str) -> str:
    """Preserve the original lightweight JSON scoring utility."""

    import tomllib

    config = tomllib.loads(config_text)
    record = json.loads(predictions_text)
    weights = config["evaluation"]["alc_weights"]
    truth = record["truth"]
    scores = []
    for budget in config["evaluation"]["budgets"]:
        probabilities = record["probabilities"][str(budget)]
        scores.append(
            sum(
                (float(probability) - float(outcome)) ** 2
                for outcome, probability in zip(truth, probabilities, strict=True)
            )
            / len(truth)
        )
    return json.dumps(
        {
            "brier_alc": sum(
                score * weight for score, weight in zip(scores, weights, strict=True)
            )
        }
    )


@app.local_entrypoint()
def main(
    action: str = "health",
    predictions: str = "examples/predictions.json",
    config: str = "configs/default.toml",
) -> None:
    """Run one explicit pipeline action from the local CLI."""

    if action == "health":
        result = health.remote()
    elif action == "inspect":
        result = inspect_storage.remote()
    elif action == "evaluate-pilot":
        result = evaluate_beta_binomial.remote(dataset_name="pilot", benchmark_split="all")
    elif action == "evaluate-full":
        result = evaluate_beta_binomial.remote(dataset_name="full", benchmark_split="test")
    elif action == "prepare-episodes":
        result = prepare_episodes.remote(dataset_name="full")
    elif action == "benchmark-duckdb":
        result = benchmark_duckdb.remote(dataset_name="full")
    elif action == "smoke-one":
        result = smoke_one_benchmark.remote(dataset_name="full")
    elif action == "neural-pilot":
        result = neural_pilot_one_benchmark.remote(dataset_name="full")
    elif action == "score":
        result = score_prediction_record.remote(
            Path(predictions).read_text(encoding="utf-8"),
            Path(config).read_text(encoding="utf-8"),
        )
    else:
        raise ValueError(
            "action must be health, inspect, prepare-episodes, benchmark-duckdb, "
            "smoke-one, neural-pilot, evaluate-pilot, evaluate-full, or score"
        )
    print(json.dumps(result, indent=2, sort_keys=True) if not isinstance(result, str) else result)
