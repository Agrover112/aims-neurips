import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from organized.training import tabpfn_lora_core as core
from organized.training.benchmark_holdout_filters import prepare_benchmark_holdout

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_projection = nn.Linear(8, 8)
        self.out_projection = nn.Linear(8, 2)
        self.other = nn.Linear(8, 8)
    def forward(self, x):
        return self.out_projection(self.q_projection(x).relu())

class LoRATests(unittest.TestCase):
    def test_zero_init_freeze_and_adapter_roundtrip(self):
        torch.manual_seed(9)
        model = Tiny()
        x = torch.randn(7, 8)
        expected = model(x).detach().clone()
        frozen = {n: p.detach().clone() for n, p in model.named_parameters()}
        targets = core.attach_lora(model, 2, 4.)
        self.assertEqual(len(targets), 2)
        torch.testing.assert_close(model(x), expected)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(x).square().mean().backward()
        opt.step()
        for name, original in frozen.items():
            module_name, param = name.rsplit(".", 1)
            layer = model.get_submodule(module_name)
            actual = layer.parametrizations.weight.original if param == "weight" and module_name in targets else getattr(layer, param)
            torch.testing.assert_close(actual, original)
        after = model(x).detach().clone()
        self.assertFalse(torch.allclose(after, expected))
        saved = core.adapter_state(model)
        for p in model.parameters():
            if p.requires_grad:
                p.data.zero_()
        core.load_adapters(model, saved)
        torch.testing.assert_close(model(x), after)
        with self.assertRaises(ValueError):
            core.load_adapters(model, {})

    def test_checkpoint_restores_rng_and_optimizer(self):
        model = Tiny()
        core.attach_lora(model, 2, 4.)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        rng = np.random.default_rng(10)
        state = core.capture_rng(rng)
        expected_np, expected_torch = rng.random(5), torch.rand(5)
        expected_global = np.random.random(5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            core.atomic_save({"rng": state, "optimizer": opt.state_dict(), "adapter": core.adapter_state(model)}, path)
            loaded = torch.load(path, weights_only=False)
            core.restore_rng(loaded["rng"], rng)
            np.testing.assert_array_equal(rng.random(5), expected_np)
            torch.testing.assert_close(torch.rand(5), expected_torch)
            np.testing.assert_array_equal(np.random.random(5), expected_global)
            opt.load_state_dict(loaded["optimizer"])
            self.assertTrue(opt.state)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_zero_and_single_class_fallback(self):
        xt = torch.zeros(3, 8)
        for labels in ([], [1], [0, 0, 0]):
            yc = torch.tensor(labels)
            result = core.predict(None, torch.zeros(len(labels), 8), yc, xt)
            torch.testing.assert_close(result, torch.full((3,), (2 + sum(labels)) / (4 + len(labels))))

    def test_frozen_bank_nested_and_disjoint(self):
        tasks = [{"role": "validation", "benchmark": f"b{i % 3}", "items": np.arange(80)} for i in range(9)]
        bank = core.make_bank(tasks, "validation", 42, limit=6)
        self.assertEqual(len(bank), 6)
        self.assertEqual({tasks[e["task"]]["benchmark"] for e in bank}, {"b0", "b1", "b2"})
        for e in bank:
            self.assertFalse(set(e["acquisition"]) & set(e["targets"]))
            for low, high in zip(core.BUDGETS, core.BUDGETS[1:]):
                self.assertTrue(set(e["acquisition"][:low]) <= set(e["acquisition"][:high]))

    def test_reuses_canonical_eleven_three_three_split(self):
        import pandas as pd
        observations, items = [], []
        for b in range(17):
            for i in range(80):
                uid = f"b{b}-i{i}"
                items.append({"item_uid": uid, "content_hash": uid})
                for s in range(15):
                    observations.append({"benchmark_id": f"b{b}", "subject_uid": f"s{s}",
                                         "item_uid": uid, "label": (s + i) % 2})
        frame, audit = prepare_benchmark_holdout(pd.DataFrame(observations), pd.DataFrame(items))
        roles = audit["benchmark_split_map"]
        self.assertEqual({r: list(roles.values()).count(r) for r in set(roles.values())},
                         {"train": 11, "validation": 3, "test": 3})
        self.assertEqual(frame.groupby("content_hash")["benchmark_split"].nunique().max(), 1)
