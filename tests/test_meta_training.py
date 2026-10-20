"""Protocol tests for the episodic meta-training skeleton."""

from __future__ import annotations

import unittest

import numpy as np

from aims_starter.meta_training import (
    budget_contexts,
    evaluate_task,
    sample_episode,
    train_meta,
)

try:
    import torch
except ModuleNotFoundError:  # Torch is an experiment/Modal dependency.
    torch = None


def _task(n: int = 100, dimensions: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(n * dimensions, dtype=np.float32).reshape(n, dimensions)
    y = (rng.random(n) < 0.5).astype(np.float64)
    return {"x": x, "y": y}


def _constant_predict(
    x_context: np.ndarray, y_context: np.ndarray, x_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return np.full(len(x_target), 0.5), np.zeros(len(x_target))


class EpisodeSamplingTests(unittest.TestCase):
    def test_disjoint_context_and_target_share_no_rows(self) -> None:
        task = _task()
        rng = np.random.default_rng(11)
        for _ in range(200):
            episode = sample_episode(task, rng)
            context_rows = {tuple(row) for row in episode.x_context}
            target_rows = {tuple(row) for row in episode.x_target}
            self.assertFalse(context_rows & target_rows)
            self.assertEqual(len(episode.y_context), episode.budget)

    def test_same_rng_state_gives_same_episode(self) -> None:
        task = _task()
        first = sample_episode(task, np.random.default_rng(5))
        second = sample_episode(task, np.random.default_rng(5))
        self.assertTrue(np.array_equal(first.x_context, second.x_context))
        self.assertTrue(np.array_equal(first.x_target, second.x_target))
        self.assertEqual(first.budget, second.budget)

    def test_target_floor_is_respected_without_overlap(self) -> None:
        task = _task(n=40)
        rng = np.random.default_rng(3)
        for _ in range(50):
            episode = sample_episode(task, rng, min_targets=32)
            self.assertGreaterEqual(len(episode.y_target), 32)
            context_rows = {tuple(row) for row in episode.x_context}
            target_rows = {tuple(row) for row in episode.x_target}
            self.assertFalse(context_rows & target_rows)

    def test_max_targets_caps_the_target_set(self) -> None:
        task = _task()
        episode = sample_episode(task, np.random.default_rng(7), max_targets=16)
        self.assertEqual(len(episode.y_target), 16)


class BudgetPrefixTests(unittest.TestCase):
    def test_prefixes_are_empty_at_budget_zero_and_nested(self) -> None:
        task = _task()
        order = np.arange(len(task["y"]))
        prefixes = budget_contexts(task, order)
        self.assertEqual(len(prefixes[0][1]), 0)
        for (_, x_small, _), (_, x_large, _) in zip(prefixes, prefixes[1:], strict=False):
            self.assertLessEqual(len(x_small), len(x_large))
            self.assertTrue(np.array_equal(x_large[: len(x_small)], x_small))


class EvaluateTaskTests(unittest.TestCase):
    def test_constant_model_matches_closed_form(self) -> None:
        task = _task()
        result = evaluate_task(None, task, predict_fn=_constant_predict)
        for budget in ("0", "1", "3", "7", "15", "31"):
            self.assertAlmostEqual(result["by_budget"][budget]["brier"], 0.25)
        self.assertAlmostEqual(result["brier_alc"], 0.25)

    def test_frozen_order_is_reused(self) -> None:
        task = _task()
        order = np.arange(len(task["y"]))
        first = evaluate_task(None, task, order=order, predict_fn=_constant_predict)
        second = evaluate_task(None, task, order=order, predict_fn=_constant_predict)
        self.assertEqual(first, second)


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class TrainingLoopTests(unittest.TestCase):
    def test_loop_runs_and_reduces_loss(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self, dimensions: int):
                super().__init__()
                self.linear = torch.nn.Linear(dimensions, 1)

            def training_loss(self, x_c, y_c, x_t, y_t):
                logits = self.linear(x_t).squeeze(-1)
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)

        tasks = [_task(seed=1), _task(seed=2)]
        model = TinyModel(4)
        before = model.linear.weight.detach().clone()
        history = train_meta(model, tasks, seed=1, steps=6, episodes_per_step=2, log_every=1)
        self.assertEqual(len(history), 6)
        self.assertTrue(all(np.isfinite(row["loss"]) for row in history))
        self.assertFalse(torch.equal(before, model.linear.weight.detach()))


if __name__ == "__main__":
    unittest.main()
