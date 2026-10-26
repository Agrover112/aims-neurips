"""Evaluate off-the-shelf TabPFN-v2 on the frozen BeaverTails protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-tabpfn"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "faithful-pilot-beavertails-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "tabpfn-v2-beavertails-3seed-v1"
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface-secret")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "tabpfn==9.2.0",
    )
    .env(
        {
            "HF_HOME": "/data/huggingface-cache",
            "TABPFN_MODEL_CACHE_DIR": "/data/tabpfn-cache",
            "TABPFN_NO_BROWSER": "1",
        }
    )
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_tasks() -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    embeddings = np.load(CORPUS_ROOT / "embeddings.npy")
    item_to_index = {
        item_uid: index for index, item_uid in enumerate(items["item_uid"].astype(str))
    }
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    result = []
    for task_id in sorted(beaver["task_id"].astype(str).unique()):
        assignment = assignments[assignments["task_id"].astype(str) == task_id]
        acquisition_rows = assignment[assignment["pool"] == "acquisition"].sort_values(
            "acquisition_rank"
        )
        label_rows = observation_lookup.loc[
            acquisition_rows["label_observation_id"].astype(str).tolist()
        ]
        acquisition_indices = np.asarray(
            [item_to_index[str(value)] for value in label_rows["item_uid"]]
        )
        evaluation_items = set(
            assignment.loc[assignment["pool"] == "evaluation", "item_uid"].astype(str)
        )
        targets = beaver[
            (beaver["task_id"].astype(str) == task_id)
            & (beaver["item_uid"].astype(str).isin(evaluation_items))
        ]
        target_indices = np.asarray(
            [item_to_index[str(value)] for value in targets["item_uid"]]
        )
        result.append(
            {
                "task_id": task_id,
                "x_C_pool": embeddings[acquisition_indices],
                "y_C_pool": label_rows["label"].to_numpy(dtype="int64"),
                "x_T": embeddings[target_indices],
                "y_T": targets["label"].to_numpy(dtype="float64"),
            }
        )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    secrets=[hf_secret],
    timeout=30 * 60,
)
def smoke() -> dict[str, Any]:
    """Verify checkpoint access and the one-class behavior before the full run."""

    import inspect

    import numpy as np
    import tabpfn
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cuda", random_state=20261018
    )
    two_class = None
    one_class = None
    try:
        model.fit(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]))
        two_class = model.predict_proba(np.asarray([[0.5]])).tolist()
    except Exception as error:  # noqa: BLE001  # pragma: no cover - dependency probe
        two_class = f"{type(error).__name__}: {error}"
    try:
        model.fit(np.asarray([[0.0]]), np.asarray([1]))
        one_class = model.predict_proba(np.asarray([[0.5]])).tolist()
    except Exception as error:  # noqa: BLE001  # pragma: no cover - expected behavior probe
        one_class = f"{type(error).__name__}: {error}"
    volume.commit()
    return {
        "tabpfn_version": tabpfn.__version__,
        "factory_signature": str(
            inspect.signature(TabPFNClassifier.create_default_for_version)
        ),
        "two_class": two_class,
        "one_class": one_class,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
)
def run_seed(seed: int) -> dict[str, Any]:
    """Evaluate one TabPFN ensemble seed on all frozen folds and budgets."""

    import numpy as np
    import tabpfn
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    output = RESULT_ROOT / f"seed-{seed}.json"
    existing = _read_json(output)
    if existing is not None:
        return {"cache_status": "reused", **existing}
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device="cuda",
        random_state=seed,
    )
    folds = []
    for fold, task in enumerate(_evaluation_tasks()):
        y_T = task["y_T"]
        results = {}
        for budget in BUDGETS:
            y_C = task["y_C_pool"][:budget]
            if budget == 0 or len(np.unique(y_C)) < 2:
                # A classifier cannot infer a class mapping from zero or one observed class.
                # The symmetric Beta(2,2) posterior is an explicit, recorded fallback.
                probability = (2.0 + float(y_C.sum())) / (4.0 + len(y_C))
                p_y_T = np.full(len(y_T), probability, dtype="float64")
                mode = "beta_binomial_fallback"
            else:
                model.fit(task["x_C_pool"][:budget], y_C)
                class_one = int(np.flatnonzero(model.classes_ == 1)[0])
                p_y_T = model.predict_proba(task["x_T"])[:, class_one]
                mode = "tabpfn_v2"
            results[str(budget)] = {
                "brier": float(np.mean((p_y_T - y_T) ** 2)),
                "mean_probability": float(np.mean(p_y_T)),
                "prediction_std": float(np.std(p_y_T)),
                "mode": mode,
                "context_positives": int(y_C.sum()),
                "context_size": len(y_C),
            }
        results["brier_alc"] = float(
            sum(
                weight * results[str(budget)]["brier"]
                for budget, weight in zip(BUDGETS, WEIGHTS, strict=True)
            )
        )
        folds.append({"fold": fold, "heldout_task": task["task_id"], "models": {"tabpfn_v2": results}})
    result = {
        "status": "off_the_shelf_tabpfn_v2",
        "seed": seed,
        "tabpfn_version": tabpfn.__version__,
        "checkpoint": "TabPFN v2 default classifier",
        "features": "384-dimensional normalized all-MiniLM-L6-v2 item embeddings",
        "protocol": {
            "outer_split": "frozen four-fold leave-one-BeaverTails-subject-out",
            "budgets": list(BUDGETS),
            "no_task_specific_training": True,
            "fallback": "symmetric Beta(2,2) posterior when context has fewer than two classes",
        },
        "folds": folds,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={str(REMOTE_ROOT): volume},
    timeout=10 * 60,
)
def aggregate() -> dict[str, Any]:
    import numpy as np

    seed_results = [_read_json(RESULT_ROOT / f"seed-{seed}.json") for seed in SEEDS]
    if any(result is None for result in seed_results):
        raise FileNotFoundError("one or more TabPFN seed results are missing")
    by_budget = {}
    for budget in BUDGETS:
        rows = [
            fold["models"]["tabpfn_v2"][str(budget)]
            for result in seed_results
            for fold in result["folds"]
        ]
        values = [row["brier"] for row in rows]
        by_budget[str(budget)] = {
            "brier": float(np.mean(values)),
            "brier_std": float(np.std(values, ddof=1)),
            "brier_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "n": len(values),
            "tabpfn_evaluations": sum(row["mode"] == "tabpfn_v2" for row in rows),
            "fallback_evaluations": sum(row["mode"] != "tabpfn_v2" for row in rows),
        }
    seed_alc = [
        float(np.mean([fold["models"]["tabpfn_v2"]["brier_alc"] for fold in result["folds"]]))
        for result in seed_results
    ]
    result = {
        "status": "off_the_shelf_tabpfn_v2_three_seed_evaluation",
        "seeds": list(SEEDS),
        "folds_per_seed": 4,
        "aggregate": {
            "tabpfn_v2": {
                "brier_alc": float(np.mean(seed_alc)),
                "brier_alc_seed_std": float(np.std(seed_alc, ddof=1)),
                "seed_brier_alc": seed_alc,
                "by_budget": by_budget,
            }
        },
    }
    (RESULT_ROOT / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(smoke_only: bool = False) -> None:
    print(json.dumps(smoke.remote(), indent=2, sort_keys=True))
    if smoke_only:
        return
    completed = list(run_seed.map(SEEDS, order_outputs=True))
    print(json.dumps({"completed_seeds": [row["seed"] for row in completed]}, indent=2))
    print(json.dumps(aggregate.remote(), indent=2, sort_keys=True))
