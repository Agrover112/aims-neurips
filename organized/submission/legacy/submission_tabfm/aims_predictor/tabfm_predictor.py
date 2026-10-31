"""Stateless TabFM predictor for the competition interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import _ItemEmbedder, _evidence, _item_text, _subject_key

ROOT = Path(__file__).resolve().parents[1]
PCA = np.load(ROOT / "assets" / "pca.npz")
PCA_MEAN = PCA["mean"].astype("float32")
PCA_COMPONENTS = PCA["components"].astype("float32")


def _context(input_pair, labeled):
    subject, item = input_pair
    key = _subject_key(subject)
    benchmark = str(item.get("benchmark_id", "") or "")
    return [
        (seen_item, outcome)
        for seen_subject, seen_item, outcome in _evidence(labeled)
        if _subject_key(seen_subject) == key
        and str(seen_item.get("benchmark_id", "") or "") == benchmark
    ][-100:]


class _Predictor:
    def __init__(self):
        import tabfm

        self.embedder = _ItemEmbedder()
        model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")
        self.classifier = tabfm.TabFMClassifier(model=model)

    def predict(self, input_pair, labeled):
        context = _context(input_pair, labeled)
        if len(context) < 2 or len({int(y) for _, y in context}) < 2:
            return 0.5
        texts = [_item_text(item) for item, _ in context] + [_item_text(input_pair[1])]
        vectors = self.embedder.encode_many(texts)
        features = (vectors - PCA_MEAN) @ PCA_COMPONENTS.T
        columns = [f"pc_{i}" for i in range(features.shape[1])]
        train = pd.DataFrame(features[:-1], columns=columns)
        target = pd.DataFrame(features[-1:], columns=columns)
        labels = np.asarray([int(y) for _, y in context])
        self.classifier.fit(train, labels)
        probability = self.classifier.predict_proba(target)[0]
        classes = list(self.classifier.classes_)
        return float(probability[classes.index(1)])


_PREDICTOR = None


def predict_probability(input_pair, labeled=None):
    global _PREDICTOR
    try:
        if _PREDICTOR is None:
            _PREDICTOR = _Predictor()
        return float(np.clip(_PREDICTOR.predict(input_pair, labeled), 0.0, 1.0))
    except Exception:
        return 0.5
