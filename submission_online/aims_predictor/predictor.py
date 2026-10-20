"""Episodic Online-AIRT predictor: MiniLM features and a MAP 2PL ability.

The checkpoint is a semantic item network trained on benchmark-held-out
episodes. Each item gets a discrimination ``a(x)`` and difficulty ``b(x)``.
A task is one subject-benchmark pair: the context is the revealed outcomes for
that pair, and the target is the current item. The task ability ``theta`` is
solved exactly from the context labels with a one-dimensional Newton step, so
inference is stateless and everything the model conditions on comes from the
current ``labeled`` argument.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import AnchorCalibrator, BudgetCalibrator
from .faithful_np import Amortized2PL, map_theta_2pl

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 256
MAX_CONTEXT = 31
CACHE_LIMIT = 30_000
ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"

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
    content = str(item.get("item_content", "") or "")
    features = str(item.get("item_features", "") or "")
    return f"item: {content}\nmetadata: {features}"


def _subject_text(subject: dict[str, Any]) -> str:
    return "\n".join(f"{column}: {subject.get(column, '') or ''}" for column in SUBJECT_COLUMNS)


def _subject_key(subject: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(subject.get(column, "") or "") for column in SUBJECT_COLUMNS)


def _evidence(labeled: list | None):
    """Yield well-formed (subject, item, outcome) entries in the given order."""

    for entry in labeled or ():
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        pair, outcome = entry
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        subject, item = pair
        if not isinstance(subject, dict) or not isinstance(item, dict):
            continue
        if outcome in (0, 1, False, True):
            yield subject, item, int(outcome)


class OnlinePredictor:
    def __init__(self, asset_root: Path = ASSET_ROOT) -> None:
        metadata = json.loads((asset_root / "metadata.json").read_text(encoding="utf-8"))
        self.x_dim = int(metadata["x_dim"])
        self.subject_slice = (
            tuple(metadata["subject_slice"]) if metadata["mode"] == "masked" else None
        )
        self.calibrator = self._load_calibrator(metadata.get("calibration"))

        state = self._load_state(asset_root / metadata["checkpoint"])
        model = Amortized2PL(
            self.x_dim,
            int(state["theta"].shape[0]),
            subject_feature_slice=self.subject_slice,
        )
        model.load_state_dict(state)
        model.eval()
        self.model = model

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.encoder = AutoModel.from_pretrained(EMBEDDING_MODEL)
        self.encoder.eval()
        self._item_cache: dict[str, np.ndarray] = {}
        self._subject_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _load_state(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without the weights_only keyword
            return torch.load(path, map_location="cpu")

    @staticmethod
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

    @torch.no_grad()
    def _embed(self, text: str) -> np.ndarray:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().numpy().astype("float32")

    def _item_vector(self, item: dict[str, Any]) -> np.ndarray:
        text = _item_text(item)
        vector = self._item_cache.get(text)
        if vector is None:
            vector = self._embed(text)
            if len(self._item_cache) >= CACHE_LIMIT:
                self._item_cache.clear()
            self._item_cache[text] = vector
        return vector

    def _subject_vector(self, subject: dict[str, Any]) -> np.ndarray:
        text = _subject_text(subject)
        vector = self._subject_cache.get(text)
        if vector is None:
            vector = self._embed(text)
            if len(self._subject_cache) >= CACHE_LIMIT:
                self._subject_cache.clear()
            self._subject_cache[text] = vector
        return vector

    def _features(self, subject: dict[str, Any], item: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [self._item_vector(item), self._subject_vector(subject)], axis=0
        ).astype("float32")

    @torch.no_grad()
    def predict(self, input_pair: list, labeled: list | None = None) -> float:
        if not isinstance(input_pair, list) or len(input_pair) != 2:
            raise ValueError("input must be [subject, item]")
        subject, item = input_pair
        if not isinstance(subject, dict) or not isinstance(item, dict):
            raise TypeError("subject and item must be dictionaries")

        key = _subject_key(subject)
        benchmark = str(item.get("benchmark_id", ""))
        context = [
            (seen_subject, seen_item, outcome)
            for seen_subject, seen_item, outcome in _evidence(labeled)
            if _subject_key(seen_subject) == key
            and str(seen_item.get("benchmark_id", "")) == benchmark
        ][-MAX_CONTEXT:]

        x_T = torch.from_numpy(self._features(subject, item)).unsqueeze(0)
        if context:
            x_C = torch.from_numpy(
                np.stack(
                    [
                        self._features(seen_subject, seen_item)
                        for seen_subject, seen_item, _ in context
                    ]
                )
            )
            y_C = torch.tensor([outcome for _, _, outcome in context], dtype=torch.float32)
        else:
            x_C = torch.zeros((0, self.x_dim), dtype=torch.float32)
            y_C = torch.zeros((0,), dtype=torch.float32)

        a_T, b_T = self.model.item_parameters(x_T)
        theta = map_theta_2pl(self.model, x_C, y_C)
        logit = float((a_T * (theta - b_T)).item())

        probability = float(1.0 / (1.0 + np.exp(-logit)))
        if self.calibrator is not None:
            budget = self._budget_key(len(context))
            if budget is not None:
                probability = float(
                    self.calibrator.transform(np.asarray([logit]), budget, y_C.numpy())[0]
                )
        return min(1.0, max(0.0, probability))

    def _budget_key(self, count: int) -> int | None:
        if self.calibrator is None:
            return None
        candidates = [budget for budget in self.calibrator.budgets if budget <= count]
        return max(candidates) if candidates else None


_PREDICTOR: OnlinePredictor | None = None


def _get_predictor() -> OnlinePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = OnlinePredictor()
    return _PREDICTOR


def predict_probability(input_pair: list, labeled: list | None = None) -> float:
    """Return the probability that the subject answers this item correctly."""

    return _get_predictor().predict(input_pair, labeled)
