"""Controlled embedding ablation for the benchmark-held-out PAIEC protocol.

The experiment separates two questions that the current report conflates:

1. Does a stronger frozen sentence encoder improve the item signal?
2. Does task-specific adaptation of the item representation help, and does it
   help the AIRT and CNP-AIRT predictors equally?

Every arm uses the same 41 training benchmarks, the same validation tasks, the
same acquisition orders, the same budgets, and the same calibration procedure.
The only changed object is the item/subject embedding table.  Test labels are
never used here; this is a validation-only model-selection experiment.

Typical commands
----------------
modal run modal_embedding_ablation.py --action prepare --arm bge_base
modal run modal_embedding_ablation.py --action run --arm bge_base --seed 20261018
modal run modal_embedding_ablation.py --action adapt --arm bge_base --seed 20261018
modal run modal_embedding_ablation.py --action run --arm bge_base_2pladapt --seed 20261018
modal run modal_embedding_ablation.py --action prepare --arm llama31_8b
modal run modal_embedding_ablation.py --action run --arm llama31_8b --seed 20261018

The Llama arm follows the encoder used in Truong et al. (2025): the last
non-padding token of Llama-3.1-8B-Instruct, L2-normalized, matching vLLM's
``task="embed"`` default.  It is loaded from an ungated mirror because the
official Meta repo is gated.  The 4096-d table is stored as float16; the
predictor code casts each minibatch to float32.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-embedding-ablation"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = REMOTE_ROOT / "datasets" / "full" / "processed"
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-embedding-ablation"
HF_HOME = REMOTE_ROOT / "huggingface-cache"
UNIFORM_FOLDS = 5
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SEEDS = (20261018, 20261019, 20261020)
MODES = ("masked", "visible")
MAX_SEQUENCE_LENGTH = 512
FIT_SAMPLE_LIMIT = 1_000_000

MODELS = {
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "source": "existing",
        "max_sequence_length": 256,
        "description": "Current 384-d MiniLM feature store.",
    },
    "minilm512": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "source": "encode",
        "max_sequence_length": 512,
        "description": "MiniLM 384-d, 512-token cap, mean pooling.",
    },
    "bge_base": {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "source": "encode",
        "max_sequence_length": 512,
        "description": "Strong English encoder, 768-d.",
    },
    "bge_m3": {
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "source": "encode",
        "max_sequence_length": 512,
        "description": "Strong multilingual encoder, 1024-d.",
    },
    "llama31_8b": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "dim": 4096,
        "source": "llama",
        "max_sequence_length": 512,
        "description": (
            "4096-d Llama-3.1-8B-Instruct embedding used in Truong et al. (2025), "
            "loaded from an ungated mirror of the same Meta weights."
        ),
    },
}

SUBJECT_COLUMNS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "reasoning_effort",
    "harness_version",
    "subject_features_extra",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "scipy>=1.13,<2",
        "torch==2.5.1",
        "transformers==4.48.3",
        "sentence-transformers==3.4.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src",
            "HF_HOME": str(HF_HOME),
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _compact_root(arm: str) -> Path:
    return EMBEDDING_ROOT / f"allbench-embed-{arm}-v1"


def _features(rows, item_embeddings, subject_embeddings):
    # Preserve the stored dtype.  Large decoder embeddings are stored as
    # float16 to keep the 8192-d Llama table within CPU memory; the training
    # code casts each minibatch to float32.
    return np.concatenate(
        [
            item_embeddings[rows["item_index"].to_numpy()],
            subject_embeddings[rows["subject_index"].to_numpy()],
        ],
        axis=1,
    )


def _meta_tasks(train, item_embeddings, subject_embeddings):
    tasks = []
    for task_id, rows in train.groupby("task_id", sort=True):
        tasks.append(
            {
                "task_id": str(task_id),
                "x": _features(rows, item_embeddings, subject_embeddings),
                "y": rows["label"].to_numpy(dtype="float32"),
            }
        )
    return tasks


def _flatten(samples: list[dict[str, Any]]):
    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["labels"] for sample in samples])
    rates = np.concatenate(
        [
            np.full(len(sample["labels"]), _context_rate(sample["context_labels"]))
            for sample in samples
        ]
    )
    budgets = np.concatenate(
        [np.full(len(sample["labels"]), sample["budget"], dtype=int) for sample in samples]
    )
    return logits, labels, rates, budgets


def _context_rate(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _fit_one_calibrator(samples: list[dict[str, Any]], rng):
    from aims_starter.calibration import AnchorCalibrator

    logits, labels, rates, budgets = _flatten(samples)
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = (
            logits[index],
            labels[index],
            rates[index],
            budgets[index],
        )
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _fit_calibrators(records: list[dict[str, Any]]):
    rng = np.random.default_rng(20261020)
    output: dict[str, Any] = {}
    arms = sorted({record["arm"] for record in records if record["arm"] != "beta"})
    for arm in arms:
        samples = [
            record
            for record in records
            if record["arm"] == arm and record["split"] == "validation"
        ]
        folds = sorted({int(record["fold"]) for record in samples})
        output[arm] = {
            "full": _fit_one_calibrator(samples, rng),
            "folds": {
                fold: _fit_one_calibrator(
                    [record for record in samples if int(record["fold"]) != fold], rng
                )
                for fold in folds
            },
        }
    return output


def _metric_rows(records: list[dict[str, Any]], calibrators) -> list[dict[str, Any]]:
    from aims_starter.faithful_np import binary_auc
    from aims_starter.metrics import calibration_ece

    rows = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        probability = _sigmoid(logits)
        arm = record["arm"]
        if arm == "beta":
            probability_calibrated = probability
        else:
            key = int(record["fold"]) if record["split"] == "validation" else "full"
            calibrator = (
                calibrators[arm]["folds"][key] if key != "full" else calibrators[arm]["full"]
            )
            probability_calibrated = calibrator.transform(
                logits, record["budget"], record["context_labels"]
            )
        rows.append(
            {
                "arm": arm,
                "split": record["split"],
                "fold": int(record["fold"]),
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": int(record["order_seed"]),
                "budget": int(record["budget"]),
                "brier_raw": float(np.mean((probability - labels) ** 2)),
                "brier_cal": float(np.mean((probability_calibrated - labels) ** 2)),
                "ece_raw": float(calibration_ece(labels.tolist(), probability.tolist())),
                "ece_cal": float(
                    calibration_ece(labels.tolist(), probability_calibrated.tolist())
                ),
                "auc_raw": float(binary_auc(labels, probability)),
                "auc_cal": float(binary_auc(labels, probability_calibrated)),
            }
        )
    return rows


def _summarize(metric_rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(metric_rows)
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted_raw"] = frame["budget"].map(weight) * frame["brier_raw"]
    frame["weighted_cal"] = frame["budget"].map(weight) * frame["brier_cal"]

    by_budget = []
    for (arm, split, budget), group in frame.groupby(["arm", "split", "budget"], sort=True):
        task_means = group.groupby("task_id", sort=False)[
            ["brier_raw", "brier_cal", "ece_raw", "ece_cal", "auc_raw", "auc_cal"]
        ].mean()
        by_budget.append(
            {
                "arm": arm,
                "split": split,
                "budget": int(budget),
                "brier_raw": float(task_means["brier_raw"].mean()),
                "brier_cal": float(task_means["brier_cal"].mean()),
                "ece_raw": float(task_means["ece_raw"].mean()),
                "ece_cal": float(task_means["ece_cal"].mean()),
                "auc_raw": float(task_means["auc_raw"].mean()),
                "auc_cal": float(task_means["auc_cal"].mean()),
                "tasks": len(task_means),
            }
        )

    task_alc = (
        frame.groupby(["arm", "split", "task_id", "order_seed"], sort=False)[
            ["weighted_raw", "weighted_cal"]
        ]
        .sum()
        .reset_index()
        .groupby(["arm", "split", "task_id"], sort=False)[["weighted_raw", "weighted_cal"]]
        .mean()
        .reset_index()
    )
    alc = []
    for (arm, split), group in task_alc.groupby(["arm", "split"], sort=True):
        alc.append(
            {
                "arm": arm,
                "split": split,
                "brier_alc_raw": float(group["weighted_raw"].mean()),
                "brier_alc_cal": float(group["weighted_cal"].mean()),
                "brier_alc_raw_se": float(
                    group["weighted_raw"].std(ddof=1) / np.sqrt(len(group))
                ),
                "brier_alc_cal_se": float(
                    group["weighted_cal"].std(ddof=1) / np.sqrt(len(group))
                ),
                "tasks": len(group),
            }
        )
    return by_budget, alc, task_alc.to_dict(orient="records")


def _paired(task_alc: list[dict[str, Any]], split: str, first: str, second: str):
    from scipy import stats

    rows = [row for row in task_alc if row["split"] == split]
    first_rows = {row["task_id"]: row for row in rows if row["arm"] == first}
    second_rows = {row["task_id"]: row for row in rows if row["arm"] == second}
    tasks = sorted(set(first_rows) & set(second_rows))
    output = {"split": split, "first": first, "second": second, "tasks": len(tasks)}
    for metric in ("weighted_raw", "weighted_cal"):
        delta = np.asarray(
            [first_rows[task][metric] - second_rows[task][metric] for task in tasks], dtype=float
        )
        mean = float(delta.mean())
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        critical = float(stats.t.ppf(0.975, len(delta) - 1))
        output[metric] = {
            "mean": mean,
            "se": se,
            "ci_low": mean - critical * se,
            "ci_high": mean + critical * se,
            "wins": int((delta < 0.0).sum()),
            "tasks": len(delta),
        }
    return output


# ---------------------------------------------------------------------------
# Feature-store construction
# ---------------------------------------------------------------------------


def _encode_texts(encoder, texts: list[str]) -> np.ndarray:
    return encoder.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")


def _encode_llama(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int = 32,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Last non-padding token embedding from a decoder-only model."""

    import torch
    from tqdm import tqdm

    vectors = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="encode"):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            lengths = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=device), lengths]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype("float32")


def _write_compact_corpus(
    arm: str,
    train,
    evaluation,
    orders,
    item_embeddings: np.ndarray,
    subject_embeddings: np.ndarray,
    item_uids,
    subject_uids,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    import pandas as pd

    root = _compact_root(arm)
    root.mkdir(parents=True, exist_ok=False)
    train.to_parquet(root / "train_tasks.parquet", index=False)
    evaluation.to_parquet(root / "eval_items.parquet", index=False)
    orders.to_parquet(root / "eval_orders.parquet", index=False)
    pd.DataFrame({"item_uid": item_uids}).to_parquet(root / "items.parquet", index=False)
    pd.DataFrame({"subject_uid": subject_uids}).to_parquet(root / "subjects.parquet", index=False)
    np.save(root / "item_embeddings.npy", item_embeddings)
    np.save(root / "subject_embeddings.npy", subject_embeddings)
    manifest = {
        "format_version": 1,
        "arm": arm,
        "item_embedding_dim": int(item_embeddings.shape[1]),
        "subject_embedding_dim": int(subject_embeddings.shape[1]),
        "combined_x_dimension": int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
        "feature_slices": {
            "item": [0, int(item_embeddings.shape[1])],
            "subject": [
                int(item_embeddings.shape[1]),
                int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
            ],
        },
        "unique_items": int(item_embeddings.shape[0]),
        "unique_subjects": int(subject_embeddings.shape[0]),
        "train_tasks": int(train["task_id"].nunique()),
        "train_rows": len(train),
        "evaluation_tasks": int(evaluation["task_id"].nunique()),
        "evaluation_rows": len(evaluation),
        "order_rows": len(orders),
        **metadata,
    }
    _write_json(root / "manifest.json", manifest)
    volume.commit()
    return manifest


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    secrets=[hf_secret],
    timeout=12 * 60 * 60,
)
def prepare(arm: str) -> dict[str, Any]:
    """Build a compact feature store for one embedding arm.

    The compact store contains only the 218k distinct items and 6.8k distinct
    subjects touched by the train/validation/test protocol, not all 1.35M items.
    """

    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    if arm not in MODELS:
        raise ValueError(f"unknown arm: {arm}; choose from {sorted(MODELS)}")
    existing = _read_json(_compact_root(arm) / "manifest.json")
    if existing is not None:
        return {"status": "reused", **existing}

    spec = MODELS[arm]
    max_sequence_length = int(spec.get("max_sequence_length", MAX_SEQUENCE_LENGTH))
    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    evaluation = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")

    item_indices = np.unique(
        np.concatenate(
            [
                train["item_index"].to_numpy(dtype=np.int64),
                evaluation["item_index"].to_numpy(dtype=np.int64),
            ]
        )
    )
    subject_indices = np.unique(
        np.concatenate(
            [
                train["subject_index"].to_numpy(dtype=np.int64),
                evaluation["subject_index"].to_numpy(dtype=np.int64),
            ]
        )
    )
    item_map = {int(old): int(new) for new, old in enumerate(item_indices)}
    subject_map = {int(old): int(new) for new, old in enumerate(subject_indices)}
    train = train.copy()
    evaluation = evaluation.copy()
    orders = orders.copy()
    train["item_index"] = train["item_index"].map(item_map).astype("int32")
    train["subject_index"] = train["subject_index"].map(subject_map).astype("int32")
    evaluation["item_index"] = evaluation["item_index"].map(item_map).astype("int32")
    evaluation["subject_index"] = evaluation["subject_index"].map(subject_map).astype("int32")
    orders["item_index"] = orders["item_index"].map(item_map).astype("int32")

    items = pd.read_parquet(
        PROCESSED_ROOT / "items.parquet", columns=["item_uid", "content", "item_features"]
    )
    items["item_uid"] = items["item_uid"].astype(str)
    items = items.drop_duplicates("item_uid").sort_values("item_uid").reset_index(drop=True)
    subjects = pd.read_parquet(PROCESSED_ROOT / "subjects.parquet")
    subjects["subject_uid"] = subjects["subject_uid"].astype(str)
    subjects = (
        subjects.drop_duplicates("subject_uid").sort_values("subject_uid").reset_index(drop=True)
    )
    if len(items) <= int(item_indices.max()) or len(subjects) <= int(subject_indices.max()):
        raise RuntimeError("protocol indices exceed processed feature tables")

    compact_items = items.iloc[item_indices].reset_index(drop=True)
    compact_subjects = subjects.iloc[subject_indices].reset_index(drop=True)

    item_texts = [
        f"item: {content}\nmetadata: {features}"
        for content, features in zip(
            compact_items["content"].fillna(""), compact_items["item_features"].fillna(""),
            strict=True,
        )
    ]
    subject_texts = [
        "\n".join(f"{column}: {getattr(row, column)}" for column in SUBJECT_COLUMNS)
        for row in compact_subjects.fillna("").itertuples(index=False)
    ]

    if spec["source"] == "existing":
        item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")[item_indices]
        subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")[subject_indices]
        encoding_metadata = {
            "embedding_model": spec["model"],
            "embedding_source": "reused_allbench_features_v1",
            "max_sequence_length": 256,
        }
    elif spec["source"] == "llama":
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(spec["model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModel.from_pretrained(
            spec["model"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        print(f"encoding {len(item_texts):,} items with {spec['model']}", flush=True)
        item_embeddings = _encode_llama(
            model, tokenizer, item_texts, max_length=max_sequence_length
        ).astype("float16")
        print(f"encoding {len(subject_texts):,} subjects", flush=True)
        subject_embeddings = _encode_llama(
            model, tokenizer, subject_texts, max_length=max_sequence_length
        ).astype("float16")
        encoding_metadata = {
            "embedding_model": spec["model"],
            "embedding_model_reference": "meta-llama/Llama-3.1-8B-Instruct",
            "embedding_model_note": "ungated mirror of the same Meta weights; gated repo was inaccessible",
            "embedding_source": "decoder_last_token_reencode",
            "max_sequence_length": max_sequence_length,
            "pooling": "last_non_padding_token",
            "normalized": True,
            "forward_dtype": "bfloat16",
            "storage_dtype": "float16",
            "item_text": "item: {content}\\nmetadata: {item_features}",
            "subject_text": "newline-joined eight subject fields",
        }
    elif spec["source"] == "encode":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = SentenceTransformer(spec["model"], device=device)
        encoder.max_seq_length = max_sequence_length
        print(f"encoding {len(item_texts):,} items with {spec['model']}", flush=True)
        item_embeddings = _encode_texts(encoder, item_texts)
        print(f"encoding {len(subject_texts):,} subjects", flush=True)
        subject_embeddings = _encode_texts(encoder, subject_texts)
        encoding_metadata = {
            "embedding_model": spec["model"],
            "embedding_source": "raw_text_reencode",
            "max_sequence_length": max_sequence_length,
            "item_text": "item: {content}\\nmetadata: {item_features}",
            "subject_text": "newline-joined eight subject fields",
        }
    else:
        raise ValueError(f"unknown embedding source: {spec['source']}")

    if item_embeddings.shape[1] != int(spec["dim"]):
        raise RuntimeError(
            f"{arm} produced {item_embeddings.shape[1]} dims, expected {spec['dim']}"
        )
    manifest = _write_compact_corpus(
        arm,
        train,
        evaluation,
        orders,
        item_embeddings,
        subject_embeddings,
        compact_items["item_uid"].to_numpy(),
        compact_subjects["subject_uid"].to_numpy(),
        metadata={
            **encoding_metadata,
            "description": spec["description"],
            "protocol": str(CORPUS_ROOT),
            "source_item_indices": len(item_indices),
            "source_subject_indices": len(subject_indices),
        },
    )
    return manifest


# ---------------------------------------------------------------------------
# Predictor training and validation evaluation
# ---------------------------------------------------------------------------


def _collect_validation(arm: str, seed: int, mode: str, orders: int):
    import pandas as pd
    import torch

    from aims_starter.faithful_np import (
        map_theta_2pl,
        train_amortized_2pl,
        train_cnp,
    )
    from aims_starter.uniform_eval import validation_fold_map

    root = _compact_root(arm)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing for {arm}: {root}")
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    item_dim = int(item_embeddings.shape[1])
    subject_slice = (
        (item_dim, item_dim + int(subject_embeddings.shape[1])) if mode == "masked" else None
    )
    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)

    tasks = _meta_tasks(train, item_embeddings, subject_embeddings)
    print(f"{arm} seed={seed} mode={mode}: training CNP-AIRT", flush=True)
    cnp, cnp_history = train_cnp(
        tasks,
        seed=seed,
        steps=800,
        episodes_per_step=4,
        subject_feature_slice=subject_slice,
    )
    print(f"{arm} seed={seed} mode={mode}: training Online-AIRT", flush=True)
    online, online_history = train_amortized_2pl(
        tasks,
        seed=seed,
        steps=800,
        subject_feature_slice=subject_slice,
    )
    checkpoint_root = RESULT_ROOT / f"seed-{seed}-{mode}" / f"{arm}-checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(cnp.state_dict(), checkpoint_root / "cnp_airt.pt")
    torch.save(online.state_dict(), checkpoint_root / "online_airt_2pl.pt")
    _write_json(checkpoint_root / "cnp-history.json", {"history": cnp_history})
    _write_json(checkpoint_root / "online-history.json", {"history": online_history})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, Any]] = []
    validation = evaluation[evaluation["split"] == "validation"]
    task_ids = sorted(validation["task_id"].astype(str).unique())
    for task_id in task_ids:
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark = str(task_rows["benchmark_id"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        x_T_np = _features(target_rows, item_embeddings, subject_embeddings)
        y_T = target_rows["label"].to_numpy(dtype="float32")
        lookup = task_rows.set_index("item_index")
        with torch.no_grad():
            x_T_t = torch.as_tensor(x_T_np, dtype=torch.float32, device=device)
            a_T, b_T = online.item_parameters(x_T_t)
            cnp_x_T = cnp.encode_item_x(x_T_t)
            cnp_raw_T = cnp.irt_parameter_net(cnp_x_T)
            cnp_a_T = torch.nn.functional.softplus(cnp_raw_T[:, 0]) + 0.05
            cnp_b_T = cnp_raw_T[:, 1]
        for order_seed in range(orders):
            order_rows = (
                order_frame[
                    (order_frame["task_id"].astype(str) == task_id)
                    & (order_frame["order_seed"] == order_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            x_C_np = _features(context_rows, item_embeddings, subject_embeddings)
            y_C = context_rows["label"].to_numpy(dtype="float32")
            x_C_t = torch.as_tensor(x_C_np, dtype=torch.float32, device=device)
            y_C_t = torch.as_tensor(y_C, dtype=torch.float32, device=device)
            with torch.no_grad():
                cnp_x_C = cnp.encode_item_x(x_C_t)
            for budget in BUDGETS:
                beta_probability = (2.0 + float(y_C[:budget].sum())) / (4.0 + budget)
                records.append(
                    {
                        "arm": "beta",
                        "seed": seed,
                        "mode": mode,
                        "split": "validation",
                        "benchmark_id": benchmark,
                        "task_id": task_id,
                        "order_seed": int(order_seed),
                        "fold": int(fold_map.get(task_id, -1)),
                        "budget": int(budget),
                        "logits": np.full(
                            len(y_T), _logit(beta_probability), dtype=float
                        ),
                        "labels": np.asarray(y_T, dtype=float),
                        "context_labels": np.asarray(y_C[:budget], dtype=float),
                    }
                )
                with torch.no_grad():
                    theta = map_theta_2pl(online, x_C_t[:budget], y_C_t[:budget])
                    airt_logits = (a_T * (theta - b_T)).cpu().numpy()
                    cnp_theta = cnp.map_theta(cnp_x_C[:budget], y_C_t[:budget])
                    cnp_theta = cnp_theta + cnp.delta_theta(cnp_x_C[:budget], y_C_t[:budget])
                    cnp_logits = (cnp_a_T * (cnp_theta - cnp_b_T)).cpu().numpy()
                for arm_name, logits in (
                    ("airt", airt_logits),
                    ("cnp", cnp_logits),
                ):
                    records.append(
                        {
                            "arm": arm_name,
                            "seed": seed,
                            "mode": mode,
                            "split": "validation",
                            "benchmark_id": benchmark,
                            "task_id": task_id,
                            "order_seed": int(order_seed),
                            "fold": int(fold_map.get(task_id, -1)),
                            "budget": int(budget),
                            "logits": np.asarray(logits, dtype=float),
                            "labels": np.asarray(y_T, dtype=float),
                            "context_labels": np.asarray(y_C[:budget], dtype=float),
                        }
                    )
        print(f"{arm} validation {task_id} collected", flush=True)
    return records, manifest, {"cnp": cnp_history, "airt": online_history}


def _score_validation(arm: str, seed: int, mode: str, orders: int, cnp, online):
    """Score validation tasks from already-loaded predictors, without training."""

    import pandas as pd
    import torch

    from aims_starter.faithful_np import map_theta_2pl
    from aims_starter.uniform_eval import validation_fold_map

    root = _compact_root(arm)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing for {arm}: {root}")
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, Any]] = []
    validation = evaluation[evaluation["split"] == "validation"]
    task_ids = sorted(validation["task_id"].astype(str).unique())
    for task_id in task_ids:
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark = str(task_rows["benchmark_id"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        x_T_np = _features(target_rows, item_embeddings, subject_embeddings)
        y_T = target_rows["label"].to_numpy(dtype="float32")
        lookup = task_rows.set_index("item_index")
        with torch.no_grad():
            x_T_t = torch.as_tensor(x_T_np, dtype=torch.float32, device=device)
            a_T, b_T = online.item_parameters(x_T_t)
            cnp_x_T = cnp.encode_item_x(x_T_t)
            cnp_raw_T = cnp.irt_parameter_net(cnp_x_T)
            cnp_a_T = torch.nn.functional.softplus(cnp_raw_T[:, 0]) + 0.05
            cnp_b_T = cnp_raw_T[:, 1]
        for order_seed in range(orders):
            order_rows = (
                order_frame[
                    (order_frame["task_id"].astype(str) == task_id)
                    & (order_frame["order_seed"] == order_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            x_C_np = _features(context_rows, item_embeddings, subject_embeddings)
            y_C = context_rows["label"].to_numpy(dtype="float32")
            x_C_t = torch.as_tensor(x_C_np, dtype=torch.float32, device=device)
            y_C_t = torch.as_tensor(y_C, dtype=torch.float32, device=device)
            with torch.no_grad():
                cnp_x_C = cnp.encode_item_x(x_C_t)
            for budget in BUDGETS:
                beta_probability = (2.0 + float(y_C[:budget].sum())) / (4.0 + budget)
                records.append(
                    {
                        "arm": "beta",
                        "seed": seed,
                        "mode": mode,
                        "split": "validation",
                        "benchmark_id": benchmark,
                        "task_id": task_id,
                        "order_seed": int(order_seed),
                        "fold": int(fold_map.get(task_id, -1)),
                        "budget": int(budget),
                        "logits": np.full(len(y_T), _logit(beta_probability), dtype=float),
                        "labels": np.asarray(y_T, dtype=float),
                        "context_labels": np.asarray(y_C[:budget], dtype=float),
                    }
                )
                with torch.no_grad():
                    theta = map_theta_2pl(online, x_C_t[:budget], y_C_t[:budget])
                    airt_logits = (a_T * (theta - b_T)).cpu().numpy()
                    cnp_theta = cnp.map_theta(cnp_x_C[:budget], y_C_t[:budget])
                    cnp_theta = cnp_theta + cnp.delta_theta(cnp_x_C[:budget], y_C_t[:budget])
                    cnp_logits = (cnp_a_T * (cnp_theta - cnp_b_T)).cpu().numpy()
                for arm_name, logits in (("airt", airt_logits), ("cnp", cnp_logits)):
                    records.append(
                        {
                            "arm": arm_name,
                            "seed": seed,
                            "mode": mode,
                            "split": "validation",
                            "benchmark_id": benchmark,
                            "task_id": task_id,
                            "order_seed": int(order_seed),
                            "fold": int(fold_map.get(task_id, -1)),
                            "budget": int(budget),
                            "logits": np.asarray(logits, dtype=float),
                            "labels": np.asarray(y_T, dtype=float),
                            "context_labels": np.asarray(y_C[:budget], dtype=float),
                        }
                    )
        print(f"{arm} rescored {task_id}", flush=True)
    return records, manifest


def _load_rescore_models(arm: str, seed: int, mode: str):
    import torch

    from aims_starter.faithful_np import Amortized2PL, CNPAIRT

    root = _compact_root(arm)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing for {arm}: {root}")
    item_dim = int(manifest["item_embedding_dim"])
    subject_dim = int(manifest["subject_embedding_dim"])
    x_dim = item_dim + subject_dim
    subject_slice = (item_dim, x_dim) if mode == "masked" else None
    checkpoint_root = RESULT_ROOT / f"seed-{seed}-{mode}" / f"{arm}-checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnp = CNPAIRT(x_dim, subject_feature_slice=subject_slice).to(device)
    cnp.load_state_dict(torch.load(checkpoint_root / "cnp_airt.pt", map_location=device))
    online_state = torch.load(checkpoint_root / "online_airt_2pl.pt", map_location=device)
    online = Amortized2PL(
        x_dim, int(online_state["theta"].shape[0]), subject_feature_slice=subject_slice
    ).to(device)
    online.load_state_dict(online_state)
    cnp.eval()
    online.eval()
    return cnp, online


def _calibration_variants(records: list[dict[str, Any]]) -> dict[str, Any]:
    from aims_starter.calibration import DEFAULT_ANCHORS, AnchorCalibrator

    anchor_grid = DEFAULT_ANCHORS
    folds = sorted({int(record["fold"]) for record in records})
    variants = ("raw", "platt", "anchor")
    fit_modes = ("oof", "full")
    output: dict[str, Any] = {}

    for predictor in ("cnp", "airt"):
        samples = [record for record in records if record["arm"] == predictor]
        output[predictor] = {}
        for variant in variants:
            output[predictor][variant] = {}
            for fit_mode in fit_modes:
                if variant == "raw":
                    calibrators = None
                else:
                    anchors = (0.0,) if variant == "platt" else anchor_grid

                    def fit_calibrator(subset):
                        return AnchorCalibrator.fit(
                            np.concatenate([r["logits"] for r in subset]),
                            np.concatenate([r["labels"] for r in subset]),
                            np.concatenate(
                                [
                                    np.full(
                                        len(r["labels"]),
                                        _context_rate(r["context_labels"]),
                                    )
                                    for r in subset
                                ]
                            ),
                            np.concatenate(
                                [
                                    np.full(len(r["labels"]), int(r["budget"]), dtype=int)
                                    for r in subset
                                ]
                            ),
                            anchors=anchors,
                        )

                    if fit_mode == "oof":
                        calibrators = {
                            fold: fit_calibrator(
                                [r for r in samples if int(r["fold"]) != fold]
                            )
                            for fold in folds
                        }
                    else:
                        calibrators = fit_calibrator(samples)

                task_order: dict[tuple[str, int], float] = {}
                for record in samples:
                    logits = np.asarray(record["logits"], dtype=float)
                    if variant == "raw":
                        probability = _sigmoid(logits)
                    else:
                        calibrator = (
                            calibrators[int(record["fold"])]
                            if fit_mode == "oof"
                            else calibrators
                        )
                        probability = calibrator.transform(
                            logits,
                            int(record["budget"]),
                            np.asarray(record["context_labels"], dtype=float),
                        )
                    labels = np.asarray(record["labels"], dtype=float)
                    brier = float(np.mean((probability - labels) ** 2))
                    key = (str(record["task_id"]), int(record["order_seed"]))
                    task_order[key] = task_order.get(key, 0.0) + (
                        WEIGHTS[BUDGETS.index(int(record["budget"]))] * brier
                    )

                task_values: dict[str, list[float]] = {}
                for (task_id, _), value in task_order.items():
                    task_values.setdefault(task_id, []).append(value)
                per_task = np.asarray([np.mean(values) for values in task_values.values()])
                output[predictor][variant][fit_mode] = {
                    "brier_alc": float(per_task.mean()),
                    "se": float(per_task.std(ddof=1) / np.sqrt(len(per_task))),
                    "tasks": int(len(per_task)),
                }
    return output


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def rescore_calibration(arm: str, seed: int = 20261018, mode: str = "masked") -> dict[str, Any]:
    """Refit calibration variants on saved validation logits, without retraining."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    output_path = RESULT_ROOT / f"calibration-variants-{arm}-{seed}-{mode}.json"
    existing = _read_json(output_path)
    if existing is not None:
        return existing
    cnp, online = _load_rescore_models(arm, seed, mode)
    records, manifest = _score_validation(arm, seed, mode, 10, cnp, online)
    import pickle

    records_path = RESULT_ROOT / f"calibration-records-{arm}-{seed}-{mode}.pkl"
    with records_path.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    result = {
        "status": "calibration_variants",
        "arm": arm,
        "seed": seed,
        "mode": mode,
        "split": "validation",
        "orders": 10,
        "manifest": manifest,
        "variants": _calibration_variants(records),
    }
    _write_json(output_path, result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=8 * 60 * 60,
)
def run(arm: str, seed: int = 20261018, mode: str = "masked", orders: int = 10) -> dict[str, Any]:
    """Train both predictors on one embedding arm and score validation only."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if orders != 10:
        raise ValueError("this ablation is fixed at ten orders")
    output = RESULT_ROOT / f"embedding-{arm}-{seed}-{mode}.json"
    existing = _read_json(output)
    if existing is not None:
        return existing

    records, manifest, histories = _collect_validation(arm, seed, mode, orders)
    calibrators = _fit_calibrators(records)
    metric_rows = _metric_rows(records, calibrators)
    by_budget, alc, task_alc = _summarize(metric_rows)
    paired = []
    for first, second in (("cnp", "airt"), ("cnp", "beta"), ("airt", "beta")):
        paired.append(_paired(task_alc, "validation", first, second))
    result = {
        "status": "embedding_ablation",
        "arm": arm,
        "seed": seed,
        "mode": mode,
        "orders": orders,
        "split": "validation",
        "manifest": manifest,
        "fits": {
            key: {
                "full": value["full"].summary(),
                "folds": {
                    str(fold): calibrator.summary()
                    for fold, calibrator in value["folds"].items()
                },
            }
            for key, value in calibrators.items()
        },
        "by_budget": by_budget,
        "alc": alc,
        "paired": paired,
        "task_alc": task_alc,
        "training": histories,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    import pandas as pd

    pd.DataFrame(metric_rows).to_parquet(output.with_suffix(".rows.parquet"), index=False)
    volume.commit()
    return result


# ---------------------------------------------------------------------------
# Task-adapted residual projection, trained with a 2PL objective
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def adapt(arm: str, seed: int = 20261018, steps: int = 800) -> dict[str, Any]:
    """Train a residual item projection with a 2PL response objective.

    The base encoder is frozen.  The learned projection is the embedding
    transform; it sees only item features and is trained with per-task
    abilities.  After fitting, the projection is frozen and a fresh AIRT and
    CNP head are trained on the adapted table, so the embedding effect is not
    confounded with a jointly trained predictor.
    """

    import pandas as pd
    import torch
    import torch.nn.functional as F

    from aims_starter.faithful_np import Amortized2PL

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    base_root = _compact_root(arm)
    base_manifest = _read_json(base_root / "manifest.json")
    if base_manifest is None:
        raise FileNotFoundError(f"prepare {arm} first")
    output_arm = f"{arm}_2pladapt"
    output_root = _compact_root(output_arm)
    existing = _read_json(output_root / "manifest.json")
    if existing is not None:
        return {"status": "reused", **existing}

    item_embeddings = np.load(base_root / "item_embeddings.npy")
    subject_embeddings = np.load(base_root / "subject_embeddings.npy")
    train = pd.read_parquet(base_root / "train_tasks.parquet")
    evaluation = pd.read_parquet(base_root / "eval_items.parquet")
    orders = pd.read_parquet(base_root / "eval_orders.parquet")
    items = pd.read_parquet(base_root / "items.parquet")
    subjects = pd.read_parquet(base_root / "subjects.parquet")

    class ResidualAdapter(torch.nn.Module):
        def __init__(self, dim: int, hidden: int = 256):
            super().__init__()
            self.norm = torch.nn.LayerNorm(dim)
            self.fc1 = torch.nn.Linear(dim, hidden)
            self.fc2 = torch.nn.Linear(hidden, dim)
            torch.nn.init.zeros_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc2.bias)

        def forward(self, x):
            return x + self.fc2(F.gelu(self.fc1(self.norm(x))))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_dim = int(item_embeddings.shape[1])
    subject_dim = int(subject_embeddings.shape[1])
    adapter = ResidualAdapter(item_dim).to(device)
    head = Amortized2PL(
        item_dim + subject_dim,
        int(train["task_id"].nunique()),
        subject_feature_slice=(item_dim, item_dim + subject_dim),
    ).to(device)
    task_ids = sorted(train["task_id"].astype(str).unique())
    task_map = {task_id: index for index, task_id in enumerate(task_ids)}
    item_t = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    subject_t = torch.as_tensor(subject_embeddings, dtype=torch.float32, device=device)
    train_item = torch.as_tensor(
        train["item_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device
    )
    train_subject = torch.as_tensor(
        train["subject_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device
    )
    train_task = torch.as_tensor(
        [task_map[str(task_id)] for task_id in train["task_id"]],
        dtype=torch.long,
        device=device,
    )
    train_y = torch.as_tensor(train["label"].to_numpy(dtype="float32"), device=device)
    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        indices = torch.as_tensor(
            rng.integers(0, len(train), size=min(2048, len(train))), device=device
        )
        adapted = adapter(item_t[train_item[indices]])
        x = torch.cat([adapted, subject_t[train_subject[indices]]], dim=-1)
        logits = head.logits(x, train_task[indices])
        loss = F.binary_cross_entropy_with_logits(logits, train_y[indices])
        loss = loss + 1e-3 * head.theta.square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()
        with torch.no_grad():
            head.theta -= head.theta.mean()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    adapter.eval()
    with torch.no_grad():
        adapted_items = adapter(item_t).cpu().numpy().astype("float32")
    manifest = _write_compact_corpus(
        output_arm,
        train,
        evaluation,
        orders,
        adapted_items,
        subject_embeddings,
        items["item_uid"].to_numpy(),
        subjects["subject_uid"].to_numpy(),
        metadata={
            "embedding_model": f"{base_manifest['embedding_model']} + residual 2PL adapter",
            "embedding_source": "task_adapted_projection",
            "base_arm": arm,
            "base_embedding_model": base_manifest["embedding_model"],
            "max_sequence_length": base_manifest.get("max_sequence_length"),
            "adapter_hidden": 256,
            "adapter_steps": int(steps),
            "adapter_seed": int(seed),
            "adapter_objective": "2PL BCE with one ability per training task",
            "training_history": history,
        },
    )
    checkpoint_root = RESULT_ROOT / f"adapter-{arm}-{seed}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), checkpoint_root / "adapter.pt")
    _write_json(checkpoint_root / "history.json", {"history": history})
    volume.commit()
    return manifest


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    action: str = "probe",
    arm: str = "bge_base",
    seed: int = 20261018,
    mode: str = "masked",
    orders: int = 10,
) -> None:
    if action == "probe":
        raise ValueError(
            "probe was moved to modal_data_probe.py; use prepare, run, or adapt here"
        )
    if action == "prepare":
        print(json.dumps(prepare.remote(arm), indent=2, sort_keys=True))
        return
    if action == "run":
        print(json.dumps(run.remote(arm, seed, mode, orders), indent=2, sort_keys=True))
        return
    if action == "rescore":
        print(
            json.dumps(
                rescore_calibration.remote(arm, seed, mode), indent=2, sort_keys=True
            )
        )
        return
    if action == "adapt":
        print(json.dumps(adapt.remote(arm, seed), indent=2, sort_keys=True))
        return
    raise ValueError("action must be prepare, run, rescore, or adapt")
