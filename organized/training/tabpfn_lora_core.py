"""LoRA and episodic binary inference for pinned TabPFN 9.2 / architecture v2."""
from __future__ import annotations
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)

class LoRAWeight(nn.Module):
    def __init__(self, weight, rank=8, alpha=16.):
        super().__init__()
        if weight.ndim != 2 or not 0 < rank <= min(weight.shape):
            raise ValueError("LoRA requires a matrix and a valid rank")
        self.scale = alpha / rank
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        return weight + (self.B @ self.A) * self.scale

def attach_lora(model, rank=8, alpha=16.):
    model.requires_grad_(False)
    targets = []
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.endswith(
            ("q_projection", "k_projection", "v_projection", "out_projection")
        ):
            parametrize.register_parametrization(layer, "weight", LoRAWeight(layer.weight, rank, alpha))
            targets.append(name)
    if not targets:
        raise RuntimeError("No supported TabPFN attention projections; refusing a no-op fine-tune")
    return targets

def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

def load_adapters(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Adapter checkpoint has incompatible parameter names")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or expected.intersection(result.missing_keys):
        raise ValueError("Incomplete adapter state")

def logits(model, xc, yc, xt):
    if len(yc) < 2 or len(torch.unique(yc)) < 2:
        raise ValueError("Raw TabPFN inference requires both binary classes in context")
    result = model(torch.cat([xc, xt]).unsqueeze(1), yc.float().unsqueeze(1), task_type="classification")
    if result.ndim != 3 or result.shape[:2] != (len(xt), 1) or result.shape[2] < 2:
        raise RuntimeError(f"Unexpected TabPFN output shape: {tuple(result.shape)}")
    return result[:, 0, :2]

def predict(model, xc, yc, xt):
    # Label-densification in the pretrained multiclass encoder cannot infer a
    # missing binary class. Use an explicit, identical fallback before/after FT.
    if len(yc) < 2 or len(torch.unique(yc)) < 2:
        return xt.new_full((len(xt),), (2. + float(yc.sum())) / (4. + len(yc)))
    return logits(model, xc, yc, xt).softmax(-1)[:, 1]

def make_bank(tasks, role, seed, limit=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, task in enumerate(tasks):
        if task["role"] == role:
            groups.setdefault(task["benchmark"], []).append(i)
    if not groups:
        raise ValueError(f"Empty {role} split")
    if limit and limit < len(groups):
        raise ValueError("Validation bank must represent every validation benchmark")
    bank = []
    for group_index, (benchmark, indices) in enumerate(sorted(groups.items())):
        n = min(len(indices), limit // len(groups) + (group_index < limit % len(groups))) if limit else len(indices)
        selected = rng.choice(indices, n, replace=False)
        for i in selected:
            task = tasks[int(i)]
            if len(task["items"]) < 80:
                raise ValueError("Task below competition eligibility threshold")
            order = rng.permutation(len(task["items"]))
            half = len(order) // 2
            bank.append({"task": int(i), "acquisition": order[:half], "targets": order[half:],
                         "weight": len(indices) / n})
    return bank

def evaluate(model, tasks, bank, features, device, query_chunk=256):
    model.eval()
    rows = []
    with torch.inference_mode():
        for episode in bank:
            task = tasks[episode["task"]]
            ids, labels = task["items"], task["labels"]
            scores, beta_scores = [], []
            for budget in BUDGETS:
                ci = episode["acquisition"][:budget]
                xc = torch.tensor(features[ids[ci]], device=device)
                yc = torch.tensor(labels[ci], device=device)
                total, beta_total, count = 0., 0., 0
                beta = (2. + float(yc.sum())) / (4. + budget)
                for start in range(0, len(episode["targets"]), query_chunk):
                    ti = episode["targets"][start:start + query_chunk]
                    xt = torch.tensor(features[ids[ti]], device=device)
                    yt = torch.tensor(labels[ti], device=device)
                    p = predict(model, xc, yc, xt)
                    if not torch.isfinite(p).all():
                        raise RuntimeError("Nonfinite validation predictions")
                    total += float(((p - yt) ** 2).sum())
                    beta_total += float(((beta - yt) ** 2).sum())
                    count += len(ti)
                scores.append(total / count)
                beta_scores.append(beta_total / count)
            rows.append({"subject": task["subject"], "benchmark": task["benchmark"],
                         "weight": episode["weight"], "by_budget": dict(zip(map(str, BUDGETS), scores)),
                         "beta_by_budget": dict(zip(map(str, BUDGETS), beta_scores)),
                         "brier_alc": sum(w * v for w, v in zip(WEIGHTS, scores)),
                         "beta_brier_alc": sum(w * v for w, v in zip(WEIGHTS, beta_scores))})
    weights = [r["weight"] for r in rows]
    by_bench = {b: float(np.mean([r["brier_alc"] for r in rows if r["benchmark"] == b]))
                for b in sorted({r["benchmark"] for r in rows})}
    return {"brier_alc": float(np.average([r["brier_alc"] for r in rows], weights=weights)),
            "beta_brier_alc": float(np.average([r["beta_brier_alc"] for r in rows], weights=weights)),
            "by_budget": {str(b): float(np.average([r["by_budget"][str(b)] for r in rows], weights=weights)) for b in BUDGETS},
            "by_benchmark": by_bench, "benchmark_macro_alc": float(np.mean(list(by_bench.values()))),
            "tasks": rows}

def capture_rng(rng):
    return {"numpy": rng.bit_generator.state, "numpy_global": np.random.get_state(),
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state, rng):
    rng.bit_generator.state = state["numpy"]
    np.random.set_state(state["numpy_global"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])

def atomic_save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
