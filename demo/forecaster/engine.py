"""Inference engine for the AI Evaluation Forecaster demo.

The app never trains.  It loads the released CNP-AIRT and Online-AIRT
checkpoints, freezes MiniLM, and scores the same held-out evaluation items at
every acquisition budget.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aims_starter.calibration import AnchorCalibrator, BudgetCalibrator  # noqa: E402
from aims_starter.faithful_np import Amortized2PL, CNPAIRT, map_theta_2pl  # noqa: E402

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 256
BUDGETS = (0, 1, 3, 7, 15, 31)
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


def _item_text(item: dict[str, Any]) -> str:
    content = str(item.get("content", "") or "")
    features = str(item.get("item_features", "") or "")
    return f"item: {content}\nmetadata: {features}"


def _subject_text(subject: dict[str, Any]) -> str:
    return "\n".join(f"{column}: {subject.get(column, '') or ''}" for column in SUBJECT_COLUMNS)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_calibrator(entries: list[dict[str, Any]] | None) -> AnchorCalibrator | None:
    if not entries:
        return None
    fitted = {
        int(entry["budget"]): BudgetCalibrator(
            scale=float(entry["scale"]),
            bias=float(entry["bias"]),
            anchor=float(entry["anchor"]),
            samples=int(entry.get("samples", 0)),
            fitted=bool(entry.get("fitted", True)),
        )
        for entry in entries
    }
    return AnchorCalibrator(budgets=tuple(sorted(fitted)), budgets_map=fitted)


def _budget_key(calibrator: AnchorCalibrator | None, count: int) -> int | None:
    if calibrator is None:
        return None
    candidates = [budget for budget in calibrator.budgets if budget <= count]
    return max(candidates) if candidates else None


@dataclass
class Prediction:
    beta: float
    cnp: np.ndarray
    airt: np.ndarray
    cnp_curve: list[float]
    airt_curve: list[float]
    beta_curve: list[float]
    evaluation_count: int


@dataclass
class Reveal:
    index: int
    outcome: int
    cnp: float
    airt: float


class ForecastEngine:
    def __init__(self, asset_root: Path = ASSET_ROOT) -> None:
        self.asset_root = asset_root
        self.tasks = self._load_tasks(asset_root / "tasks.json")
        self.cnp_meta = _read_json(asset_root / "metadata_cnp.json")
        self.airt_meta = _read_json(asset_root / "metadata_airt.json")
        self.cnp = self._load_cnp()
        self.airt = self._load_airt()
        self.cnp_calibrator = _load_calibrator(self.cnp_meta.get("calibration"))
        self.airt_calibrator = _load_calibrator(self.airt_meta.get("calibration"))
        self.tokenizer, self.encoder = self._load_encoder()
        self._cache: dict[str, np.ndarray] = {}
        self._all_item_embeddings = None
        self._prepare_tasks()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tasks(path: Path) -> dict[str, dict[str, Any]]:
        payload = _read_json(path)
        return {task["task_id"]: task for task in payload["tasks"]}

    def _load_cnp(self) -> CNPAIRT:
        meta = self.cnp_meta
        model = CNPAIRT(
            int(meta["x_dim"]),
            subject_feature_slice=tuple(meta["subject_slice"])
            if meta.get("mode") == "masked"
            else None,
        )
        state = torch.load(self.asset_root / meta["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def _load_airt(self) -> Amortized2PL:
        meta = self.airt_meta
        state = torch.load(
            self.asset_root / meta["checkpoint"], map_location="cpu", weights_only=True
        )
        model = Amortized2PL(
            int(meta["x_dim"]),
            int(state["theta"].shape[0]),
            subject_feature_slice=tuple(meta["subject_slice"])
            if meta.get("mode") == "masked"
            else None,
        )
        model.load_state_dict(state)
        model.eval()
        return model

    @staticmethod
    def _load_encoder():
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        encoder = AutoModel.from_pretrained(EMBEDDING_MODEL)
        encoder.eval()
        return tokenizer, encoder

    @torch.no_grad()
    def _embed_many(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True,
            )
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
        if not vectors:
            return np.zeros((0, 384), dtype="float32")
        return np.concatenate(vectors, axis=0)

    def _embed(self, text: str) -> np.ndarray:
        vector = self._cache.get(text)
        if vector is None:
            vector = self._embed_many([text])[0]
            self._cache[text] = vector
        return vector

    def _prepare_tasks(self) -> None:
        all_items: list[np.ndarray] = []
        for task in self.tasks.values():
            subject_text = _subject_text(task["subject"])
            subject_embedding = self._embed(subject_text)
            acquisition_texts = [_item_text(item) for item in task["acquisition"]]
            evaluation_texts = [_item_text(item) for item in task["evaluation"]]
            acquisition_embeddings = self._embed_many(acquisition_texts)
            evaluation_embeddings = self._embed_many(evaluation_texts)
            task["_subject_embedding"] = subject_embedding
            task["_acquisition_x"] = np.concatenate(
                [acquisition_embeddings, np.tile(subject_embedding, (len(acquisition_embeddings), 1))],
                axis=1,
            ).astype("float32")
            task["_evaluation_x"] = np.concatenate(
                [evaluation_embeddings, np.tile(subject_embedding, (len(evaluation_embeddings), 1))],
                axis=1,
            ).astype("float32")
            all_items.extend(acquisition_embeddings)
            all_items.extend(evaluation_embeddings)
        self._all_item_embeddings = np.stack(all_items) if all_items else None

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _cnp_logits(self, x_c: np.ndarray, y_c: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        x_c_t = torch.as_tensor(x_c, dtype=torch.float32)
        y_c_t = torch.as_tensor(y_c, dtype=torch.float32)
        x_t_t = torch.as_tensor(x_t, dtype=torch.float32)
        encoded_c = self.cnp.encode_item_x(x_c_t)
        theta = self.cnp.map_theta(encoded_c, y_c_t)
        theta = theta + self.cnp.delta_theta(encoded_c, y_c_t)
        raw = self.cnp.irt_parameter_net(self.cnp.encode_item_x(x_t_t))
        discrimination = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        difficulty = raw[:, 1]
        return (discrimination * (theta - difficulty)).cpu().numpy().astype(float)

    @torch.no_grad()
    def _airt_logits(self, x_c: np.ndarray, y_c: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        x_c_t = torch.as_tensor(x_c, dtype=torch.float32)
        y_c_t = torch.as_tensor(y_c, dtype=torch.float32)
        x_t_t = torch.as_tensor(x_t, dtype=torch.float32)
        theta = map_theta_2pl(self.airt, x_c_t, y_c_t)
        discrimination, difficulty = self.airt.item_parameters(x_t_t)
        return (discrimination * (theta - difficulty)).cpu().numpy().astype(float)

    def _calibrate(self, logits: np.ndarray, count: int, y_c: np.ndarray, calibrator) -> np.ndarray:
        budget = _budget_key(calibrator, count)
        if calibrator is None or budget is None:
            return 1.0 / (1.0 + np.exp(-logits))
        return calibrator.transform(np.asarray(logits, dtype=float), budget, y_c)

    def _context(self, task: dict[str, Any], revealed: list[Reveal]):
        if not revealed:
            return (
                np.zeros((0, int(task["_acquisition_x"].shape[1])), dtype="float32"),
                np.zeros((0,), dtype="float32"),
            )
        indices = [entry.index for entry in revealed]
        return task["_acquisition_x"][indices], np.asarray(
            [entry.outcome for entry in revealed], dtype="float32"
        )

    def predict_targets(self, task: dict[str, Any], revealed: list[Reveal]):
        x_c, y_c = self._context(task, revealed)
        x_t = task["_evaluation_x"]
        cnp_logits = self._cnp_logits(x_c, y_c, x_t)
        airt_logits = self._airt_logits(x_c, y_c, x_t)
        cnp = self._calibrate(cnp_logits, len(revealed), y_c, self.cnp_calibrator)
        airt = self._calibrate(airt_logits, len(revealed), y_c, self.airt_calibrator)
        beta = float((2.0 + y_c.sum()) / (4.0 + len(y_c)))
        return beta, cnp, airt

    def forecast(self, task: dict[str, Any], revealed: list[Reveal]) -> Prediction:
        beta, cnp, airt = self.predict_targets(task, revealed)
        cnp_curve, airt_curve, beta_curve = [], [], []
        for count in range(len(revealed) + 1):
            prefix = revealed[:count]
            x_c, y_c = self._context(task, prefix)
            cnp_logits = self._cnp_logits(x_c, y_c, task["_evaluation_x"])
            airt_logits = self._airt_logits(x_c, y_c, task["_evaluation_x"])
            cnp_curve.append(
                float(
                    np.mean(
                        self._calibrate(
                            cnp_logits, count, y_c, self.cnp_calibrator
                        )
                    )
                )
            )
            airt_curve.append(
                float(
                    np.mean(
                        self._calibrate(
                            airt_logits, count, y_c, self.airt_calibrator
                        )
                    )
                )
            )
            beta_curve.append(float((2.0 + y_c.sum()) / (4.0 + len(y_c))))
        return Prediction(
            beta=beta,
            cnp=cnp,
            airt=airt,
            cnp_curve=cnp_curve,
            airt_curve=airt_curve,
            beta_curve=beta_curve,
            evaluation_count=len(task["evaluation"]),
        )

    def next_item(
        self,
        task: dict[str, Any],
        revealed: list[Reveal],
        policy: str,
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        chosen = {entry.index for entry in revealed}
        candidates = [index for index in range(len(task["acquisition"])) if index not in chosen]
        if not candidates:
            return None
        x_c, y_c = self._context(task, revealed)
        x_candidates = task["_acquisition_x"][candidates]
        cnp_logits = self._cnp_logits(x_c, y_c, x_candidates)
        airt_logits = self._airt_logits(x_c, y_c, x_candidates)
        cnp = self._calibrate(cnp_logits, len(revealed), y_c, self.cnp_calibrator)
        airt = self._calibrate(airt_logits, len(revealed), y_c, self.airt_calibrator)
        if policy == "random":
            position = int(rng.integers(len(candidates)))
        elif policy == "hardest":
            position = int(np.argmin(cnp))
        else:
            position = int(np.argmin(np.abs(cnp - 0.5)))
        item_index = candidates[position]
        return {
            "index": item_index,
            "item": task["acquisition"][item_index],
            "cnp": float(cnp[position]),
            "airt": float(airt[position]),
        }

    def custom_predict(self, subject_text: str, item_text: str) -> dict[str, Any]:
        subject = {
            "normalized_name": subject_text.strip()[:200],
            "provider": "",
            "release_date": "",
            "access_date": "",
            "harness": "",
            "reasoning_effort": "",
            "harness_version": "",
            "subject_features_extra": subject_text.strip(),
        }
        item = {
            "content": item_text.strip(),
            "item_features": "",
            "benchmark_id": "custom",
        }
        subject_embedding = self._embed(_subject_text(subject))
        item_embedding = self._embed(_item_text(item))
        x = np.concatenate([item_embedding, subject_embedding]).astype("float32")[None, :]
        empty_x = np.zeros((0, x.shape[1]), dtype="float32")
        empty_y = np.zeros((0,), dtype="float32")
        cnp_logit = self._cnp_logits(empty_x, empty_y, x)
        airt_logit = self._airt_logits(empty_x, empty_y, x)
        cnp = float(self._calibrate(cnp_logit, 0, empty_y, self.cnp_calibrator)[0])
        airt = float(self._calibrate(airt_logit, 0, empty_y, self.airt_calibrator)[0])
        support = 0.0
        if self._all_item_embeddings is not None:
            support = float(np.max(self._all_item_embeddings @ item_embedding))
        return {
            "cnp": cnp,
            "airt": airt,
            "support": support,
            "support_label": "near" if support >= 0.55 else "moderate" if support >= 0.40 else "far",
            "subject_masked": True,
        }

    def score(self, task: dict[str, Any], revealed: list[Reveal]) -> dict[str, Any]:
        beta, cnp, airt = self.predict_targets(task, revealed)
        labels = np.asarray([item["label"] for item in task["evaluation"]], dtype=float)
        return {
            "actual_success": float(labels.mean()),
            "brier": {
                "beta": float(np.mean((beta - labels) ** 2)),
                "cnp": float(np.mean((cnp - labels) ** 2)),
                "airt": float(np.mean((airt - labels) ** 2)),
            },
            "items": [
                {
                    "item_id": item["item_id"],
                    "content": item["content"][:900],
                    "label": int(item["label"]),
                    "cnp": float(cnp[index]),
                    "airt": float(airt[index]),
                    "beta": float(beta),
                }
                for index, item in enumerate(task["evaluation"])
            ],
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def task_summary(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task["task_id"],
            "benchmark": task["benchmark"],
            "subject": {
                "display_name": task["subject"]["display_name"],
                "provider": task["subject"].get("provider", ""),
                "normalized_name": task["subject"].get("normalized_name", ""),
                "harness": task["subject"].get("harness", ""),
            },
            "n_acquisition": len(task["acquisition"]),
            "n_evaluation": len(task["evaluation"]),
            "acquisition_preview": task["acquisition"][0]["content"][:240],
        }

    def checkpoint_summary(self) -> dict[str, Any]:
        cnp_path = self.asset_root / self.cnp_meta["checkpoint"]
        airt_path = self.asset_root / self.airt_meta["checkpoint"]
        return {
            "cnp": {
                "file": cnp_path.name,
                "sha256": _sha256(cnp_path)[:16],
                "bytes": cnp_path.stat().st_size,
                "architecture": self.cnp_meta.get("architecture", "cnp_airt"),
                "mode": self.cnp_meta.get("mode", "masked"),
                "embedding": EMBEDDING_MODEL,
            },
            "airt": {
                "file": airt_path.name,
                "sha256": _sha256(airt_path)[:16],
                "bytes": airt_path.stat().st_size,
                "architecture": self.airt_meta.get("architecture", "online_airt_2pl"),
                "mode": self.airt_meta.get("mode", "masked"),
                "embedding": EMBEDDING_MODEL,
            },
        }

    def validation_summary(self) -> list[dict[str, Any]]:
        return [
            {"model": "CNP-AIRT", "brier_alc": 0.222576, "tasks": 104},
            {"model": "Beta–Binomial", "brier_alc": 0.222838, "tasks": 104},
            {"model": "Online-AIRT", "brier_alc": 0.223312, "tasks": 104},
        ]
