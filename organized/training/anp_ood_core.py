"""Matched ANP sampling, robust objectives, SAM and weight averaging."""
import copy
import numpy as np
import torch

VARIANTS = ("balanced_erm", "groupdro", "swa", "sam")
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)

def training_groups(tasks):
    groups = {}
    for i, t in enumerate(tasks):
        if t["role"] == "train":
            groups.setdefault(t["benchmark"], []).append(i)
    return dict(sorted(groups.items()))

def sample_batch(tasks, groups, features, rng, max_context=64, max_targets=128):
    batch, metadata = [], []
    for benchmark, candidates in groups.items():
        index = int(rng.choice(candidates))
        task = tasks[index]
        order = rng.permutation(len(task["items"]))
        k = int(rng.integers(0, min(max_context, len(order)-1) + 1))
        c, t = order[:k], order[k:k + max_targets]
        ci, ti = task["items"][c], task["items"][t]
        batch.append((features[ci], torch.tensor(task["labels"][c], dtype=torch.float32),
                      features[ti], torch.tensor(task["labels"][t], dtype=torch.float32)))
        metadata.append((index, benchmark, ci, ti))
    return batch, metadata

def losses(model, batch):
    return torch.stack([model.loss(*episode)[0] for episode in batch])

def dro_weights(state, group_losses, eta=.01, smoothing=.9):
    detached = group_losses.detach()
    ema = detached.clone() if state is None else smoothing * state["ema"] + (1-smoothing) * detached
    logq = torch.zeros_like(ema) if state is None else state["logq"]
    logq = logq + eta * ema
    logq = logq - torch.logsumexp(logq, dim=0)
    return {"ema": ema, "logq": logq}, logq.exp()

def sam_step(model, optimizer, closure, rho=.05):
    optimizer.zero_grad(set_to_none=True)
    before = torch.get_rng_state()
    first = closure()
    first.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite SAM gradient")
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad * (rho / (norm + 1e-12)))
        optimizer.zero_grad(set_to_none=True)
        # Identical latent samples in the two passes: no stochastic-noise confound.
        torch.set_rng_state(before)
        second = closure()
        second.backward()
        if not torch.isfinite(second):
            raise RuntimeError("Nonfinite SAM objective")
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()
    return float(first.detach()), float(second.detach())

def average_state(average, count, current):
    if average is None:
        return {k: v.detach().clone() for k,v in current.items()}, 1
    with torch.no_grad():
        for key, value in current.items():
            if torch.is_floating_point(value):
                average[key].add_((value.detach() - average[key]) / (count + 1))
            else:
                average[key].copy_(value)
    return average, count + 1

def evaluate(model, tasks, bank, features, chunk=256):
    model.eval()
    rows = []
    with torch.inference_mode():
        for ep in bank:
            task = tasks[ep["task"]]
            t = ep["targets"]
            scores, beta_scores = [], []
            for budget in BUDGETS:
                c = ep["acquisition"][:budget]
                xc = features[task["items"][c]]
                yc = torch.tensor(task["labels"][c], dtype=torch.float32)
                total, count = 0., 0
                beta = (2. + float(yc.sum())) / (4. + budget)
                beta_total = 0.
                for start in range(0, len(t), chunk):
                    selected = t[start:start + chunk]
                    xt = features[task["items"][selected]]
                    yt = torch.tensor(task["labels"][selected], dtype=torch.float32)
                    p = model.logits(xc, yc, xt).sigmoid()
                    if not torch.isfinite(p).all():
                        raise RuntimeError("Nonfinite predictions")
                    total += float(((p-yt)**2).sum())
                    beta_total += float(((beta-yt)**2).sum())
                    count += len(selected)
                scores.append(total / count)
                beta_scores.append(beta_total / count)
            rows.append({"subject": task["subject"], "benchmark": task["benchmark"], "weight": ep["weight"],
                         "by_budget": dict(zip(map(str, BUDGETS), scores)),
                         "brier_alc": sum(w*s for w,s in zip(WEIGHTS,scores)),
                         "beta_brier_alc": sum(w*s for w,s in zip(WEIGHTS,beta_scores))})
    weights = [r["weight"] for r in rows]
    macro = {b: float(np.mean([r["brier_alc"] for r in rows if r["benchmark"] == b]))
             for b in sorted({r["benchmark"] for r in rows})}
    return {"brier_alc": float(np.average([r["brier_alc"] for r in rows], weights=weights)),
            "beta_brier_alc": float(np.average([r["beta_brier_alc"] for r in rows], weights=weights)),
            "by_budget": {str(b): float(np.average([r["by_budget"][str(b)] for r in rows], weights=weights)) for b in BUDGETS},
            "by_benchmark": macro, "benchmark_macro_alc": float(np.mean(list(macro.values()))), "tasks": rows}

def diagnostics(model, tasks, bank, features, population, max_context=64, max_targets=512):
    from sklearn.metrics import roc_auc_score
    values = {axis: [] for axis in ("same_subject_split_half", "same_subject_response_holdout")}
    model.eval()
    with torch.inference_mode():
        for ep in bank:
            task = tasks[ep["task"]]
            order = np.r_[ep["acquisition"], ep["targets"]]
            for axis, cut in (("same_subject_split_half", len(ep["acquisition"])),
                              ("same_subject_response_holdout", int(.8 * len(order)))):
                c, t = order[:cut][:max_context], order[cut:][:max_targets]
                yc = torch.tensor(task["labels"][c], dtype=torch.float32)
                y = task["labels"][t]
                p = model.logits(features[task["items"][c]], yc, features[task["items"][t]]).sigmoid().numpy()
                if not np.isfinite(p).all():
                    raise RuntimeError("Nonfinite diagnostic predictions")
                values[axis].append((float(np.mean((p-y)**2)), float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None, ep["weight"]))
    result = {}
    for axis, rows in values.items():
        prefix = f"diagnostic/{population}/{axis}"
        result[prefix + "/brier_score"] = float(np.average([r[0] for r in rows], weights=[r[2] for r in rows]))
        valid = [r for r in rows if r[1] is not None]
        if valid:
            result[prefix + "/roc_auc"] = float(np.average([r[1] for r in valid], weights=[r[2] for r in valid]))
        result[prefix + "/roc_auc_tasks"] = len(valid)
    return result
