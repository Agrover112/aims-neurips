import unittest

import numpy as np

try:
    import torch

    from aims_starter.faithful_ladder import (
        ALC_PAIR_WEIGHTS,
        AnchorCorrectionIRT,
        build_pair_epoch_batches,
        collate_episodes,
        deterministic_batch_loss,
        deterministic_episode_loss,
        latent_episode_loss,
        latent_predict_probability,
        make_task,
        sample_episode,
        sample_pair_episode,
        split_task_indices,
    )
    from aims_starter.faithful_np import FaithfulNPAIRT
except ModuleNotFoundError:  # Torch is an experiment dependency, not a submission dependency.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class FaithfulLadderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.x_c = torch.randn(5, 8)
        self.y_c = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        self.x_t = torch.randn(7, 8)
        self.y_t = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    def test_task_split_is_fixed_disjoint_and_complete(self):
        first = split_task_indices(101, "task-a")
        second = split_task_indices(101, "task-a")
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(set(first[0]) & set(first[1]), set())
        self.assertEqual(set(first[0]) | set(first[1]), set(range(101)))
        self.assertEqual(len(first[0]), 50)
        self.assertEqual(len(first[1]), 51)

    def test_episode_targets_are_only_the_evaluation_half(self):
        x = np.arange(40, dtype=np.float32)[:, None]
        y = (np.arange(40) % 2).astype(np.float32)
        task = make_task("task-b", x, y)
        rng = np.random.default_rng(0)
        x_c, _y_c, x_t, y_t, budget = sample_episode(
            task, rng, torch.device("cpu"), budget=3
        )
        self.assertEqual(budget, 3)
        self.assertEqual(tuple(x_c.shape), (3, 1))
        self.assertEqual(tuple(x_t.shape), (len(task["evaluation"]), 1))
        self.assertEqual(set(x_t[:, 0].tolist()), set(task["evaluation"].tolist()))
        self.assertEqual(set(x_c[:, 0].tolist()) <= set(task["acquisition"].tolist()), True)
        self.assertTrue(set(y_t.tolist()) <= {0.0, 1.0})

    def test_anchor_correction_starts_at_anchor_and_uses_context(self):
        model = AnchorCorrectionIRT(
            8,
            use_gp=True,
            use_anp=True,
            d_model=16,
            basis_size=4,
            n_heads=4,
        )
        logits = model(self.x_c, self.y_c, self.x_t)
        self.assertEqual(tuple(logits.shape), (7,))
        self.assertTrue(torch.isfinite(logits).all())
        with torch.no_grad():
            model.gamma.zero_()
        without_correction = model(self.x_c, self.y_c, self.x_t)
        self.assertTrue(torch.allclose(logits, without_correction, atol=1e-6))

        with torch.no_grad():
            model.gamma.fill_(1.0)
            model.delta_head[-1].bias.fill_(0.75)
            model.delta_head[-1].weight.normal_()
        first = model(self.x_c, self.y_c, self.x_t)
        second = model(self.x_c, self.y_c.flip(0), self.x_t)
        self.assertFalse(torch.allclose(first, second))
        self.assertGreater(float(model.aux_loss()), 0.0)

    def test_gp_penalty_is_active(self):
        model = AnchorCorrectionIRT(
            8,
            use_gp=True,
            d_model=16,
            basis_size=4,
            n_heads=4,
        )
        with torch.no_grad():
            model.coefficient_head[-1].bias.fill_(0.1)
        model(self.x_c, self.y_c, self.x_t)
        self.assertGreater(float(model.gp_penalty()), 0.0)

    def test_item_path_excludes_subject_block(self):
        model = AnchorCorrectionIRT(
            8,
            use_gp=False,
            use_anp=False,
            d_model=16,
            subject_feature_slice=(3, 5),
        )
        changed_c = self.x_c.clone()
        changed_t = self.x_t.clone()
        changed_c[:, 3:5] += 100.0
        changed_t[:, 3:5] += 100.0
        first = model(self.x_c, self.y_c, self.x_t)
        second = model(changed_c, self.y_c, changed_t)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_collate_episodes_pads_and_masks(self):
        episodes = [
            {
                "task_index": 0,
                "budget": 0,
                "x_c": np.zeros((0, 3), dtype=np.float32),
                "y_c": np.zeros((0,), dtype=np.float32),
                "x_t": np.ones((2, 3), dtype=np.float32),
                "y_t": np.array([0.0, 1.0], dtype=np.float32),
            },
            {
                "task_index": 1,
                "budget": 3,
                "x_c": np.ones((4, 3), dtype=np.float32),
                "y_c": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
                "x_t": np.full((5, 3), 2.0, dtype=np.float32),
                "y_t": np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            },
        ]
        batch = collate_episodes(episodes, torch.device("cpu"))
        self.assertEqual(tuple(batch.x_c.shape), (2, 4, 3))
        self.assertEqual(tuple(batch.x_t.shape), (2, 5, 3))
        self.assertTrue(torch.equal(batch.context_mask.sum(1), torch.tensor([0, 4])))
        self.assertTrue(torch.equal(batch.target_mask.sum(1), torch.tensor([2, 5])))
        self.assertTrue(torch.equal(batch.budgets, torch.tensor([0, 3])))
        self.assertTrue(torch.equal(batch.task_indices, torch.tensor([0, 1])))
        self.assertTrue(torch.all(batch.x_c[0] == 0.0))
        self.assertTrue(torch.all(batch.x_t[0, 2:] == 0.0))

    def test_batched_forward_matches_single_task_forward(self):
        model = AnchorCorrectionIRT(
            8,
            use_gp=True,
            use_anp=True,
            d_model=16,
            basis_size=4,
            n_heads=4,
        )
        with torch.no_grad():
            model.coefficient_head[-1].bias.fill_(0.1)
            model.delta_head[-1].weight.normal_()
            model.delta_head[-1].bias.fill_(0.2)
            model.gamma.fill_(1.0)
        model.eval()
        episodes = []
        single_logits = []
        for n_context, n_target in ((0, 3), (5, 7), (2, 4)):
            x_c = torch.randn(n_context, 8)
            y_c = torch.randint(0, 2, (n_context,), dtype=torch.float32)
            x_t = torch.randn(n_target, 8)
            y_t = torch.randint(0, 2, (n_target,), dtype=torch.float32)
            single_logits.append(model(x_c, y_c, x_t))
            episodes.append(
                {
                    "task_index": len(episodes),
                    "budget": n_context,
                    "x_c": x_c.numpy(),
                    "y_c": y_c.numpy(),
                    "x_t": x_t.numpy(),
                    "y_t": y_t.numpy(),
                }
            )
        batch = collate_episodes(episodes, torch.device("cpu"))
        batched, mask = model.forward_batch(
            batch.x_c,
            batch.y_c,
            batch.x_t,
            batch.context_mask,
            batch.target_mask,
        )
        self.assertTrue(torch.equal(mask, batch.target_mask))
        for index, expected in enumerate(single_logits):
            actual = batched[index, : expected.shape[0]]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
        self.assertGreater(float(model.gp_penalty()), 0.0)
        self.assertGreater(float(model.aux_loss()), 0.0)

    def test_pair_epoch_has_one_visit_per_budget_and_distinct_batches(self):
        tasks = [{"task_id": f"task-{index}"} for index in range(6649)]
        batches = build_pair_epoch_batches(tasks, seed=7, batch_size=16)
        self.assertEqual(len(batches), 2494)
        self.assertEqual(sum(len(batch) for batch in batches), 39894)
        self.assertEqual([len(batch) for batch in batches[-2:]], [16, 6])
        visits: dict[int, list[int]] = {index: [] for index in range(len(tasks))}
        for batch in batches:
            self.assertEqual(
                len({visit.task_index for visit in batch}), len(batch)
            )
            for visit in batch:
                visits[visit.task_index].append(visit.budget)
        for budgets in visits.values():
            self.assertEqual(sorted(budgets), sorted((0, 1, 3, 7, 15, 31)))
        final = batches[-1]
        self.assertEqual(sorted(visit.budget for visit in final), [0, 1, 3, 7, 15, 31])

    def test_batch_loss_size_weights_and_brier_alc_weights(self):
        model = AnchorCorrectionIRT(8, use_gp=False, use_anp=False, d_model=16)
        model.eval()
        task = make_task(
            "weights",
            np.random.default_rng(0).normal(size=(8, 8)).astype(np.float32),
            (np.arange(8) % 2).astype(np.float32),
        )
        rng = np.random.default_rng(1)
        six = [
            sample_pair_episode(
                task,
                budget=(0, 1, 3, 7, 15, 31)[index],
                rng=rng,
                visit_key=f"e0:b{(0, 1, 3, 7, 15, 31)[index]}",
            )
            for index in range(6)
        ]
        for index, episode in enumerate(six):
            episode["task_index"] = index
        batch = collate_episodes(six, torch.device("cpu"))
        _loss, diagnostic = deterministic_batch_loss(
            model, batch, objective="bce", batch_size=16
        )
        self.assertAlmostEqual(
            diagnostic["prediction"], diagnostic["bce"] * 6.0 / 16.0, places=6
        )
        self.assertAlmostEqual(ALC_PAIR_WEIGHTS[0], 0.6)
        self.assertAlmostEqual(ALC_PAIR_WEIGHTS[31], 0.6)
        self.assertAlmostEqual(ALC_PAIR_WEIGHTS[7], 1.2)

    def test_sample_pair_episode_rotates_disjoint_splits(self):
        x = np.arange(80, dtype=np.float32)[:, None]
        y = (np.arange(80) % 2).astype(np.float32)
        task = make_task("rotating", x, y)
        rng = np.random.default_rng(3)
        first = sample_pair_episode(
            task, budget=3, rng=rng, visit_key="e0:b0"
        )
        second = sample_pair_episode(
            task, budget=3, rng=rng, visit_key="e0:b1"
        )
        first_context = set(first["x_c"][:, 0].tolist())
        first_target = set(first["x_t"][:, 0].tolist())
        second_context = set(second["x_c"][:, 0].tolist())
        second_target = set(second["x_t"][:, 0].tolist())
        self.assertEqual(first_context & first_target, set())
        self.assertEqual(second_context & second_target, set())
        self.assertNotEqual(first_context, second_context)

    def test_latent_loss_divides_kl_by_target_count(self):
        model = FaithfulNPAIRT(
            8,
            variant="np",
            d_model=8,
            d_z=4,
            n_heads=2,
            anchor_scale=0.0,
        )
        zero_kl, zero_diag = latent_episode_loss(
            model,
            self.x_c,
            self.y_c,
            self.x_t,
            self.y_t,
            objective="bce",
            beta=0.0,
            n_z_samples=1,
        )
        with_kl, with_diag = latent_episode_loss(
            model,
            self.x_c,
            self.y_c,
            self.x_t,
            self.y_t,
            objective="bce",
            beta=1.0,
            n_z_samples=1,
        )
        self.assertTrue(torch.isfinite(zero_kl))
        self.assertTrue(torch.isfinite(with_kl))
        self.assertAlmostEqual(float(zero_kl), float(zero_diag["bce"]), places=6)
        self.assertGreater(float(with_diag["kl"]), 0.0)
        self.assertAlmostEqual(
            float(with_kl),
            float(with_diag["bce"]) + float(with_diag["kl"]),
            places=6,
        )
        self.assertAlmostEqual(with_diag["kl"], zero_diag["kl"], places=7)

    def test_latent_prediction_and_deterministic_objectives(self):
        model = FaithfulNPAIRT(
            8,
            variant="anp",
            d_model=8,
            d_z=4,
            n_heads=2,
            anchor_scale=0.0,
            residual_scale=0.1,
        )
        probability = latent_predict_probability(
            model,
            self.x_c,
            self.y_c,
            self.x_t,
            n_z_samples=3,
        )
        self.assertEqual(tuple(probability.shape), (7,))
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue(((probability >= 0.0) & (probability <= 1.0)).all())

        anchor = AnchorCorrectionIRT(8, d_model=16)
        for objective in ("bce", "brier", "bce_brier"):
            loss, diagnostic = deterministic_episode_loss(
                anchor,
                self.x_c,
                self.y_c,
                self.x_t,
                self.y_t,
                objective=objective,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("bce", diagnostic)
            self.assertIn("brier", diagnostic)


if __name__ == "__main__":
    unittest.main()
