"""Frozen ANP scores and causal streaming evaluation; no pool lookahead."""
import time
import numpy as np
import torch

POLICIES = ('random', 'uncertainty', 'bald', 'expected_brier')
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)


@torch.inference_mode()
def score(model, xc, yc, candidates, policy, reference, noise):
    p = model.logits(xc, yc, candidates).sigmoid()
    if policy == 'uncertainty':
        return p * (1-p)
    if policy == 'bald':
        mu, lv = model.latent_params(xc, yc)
        samples = torch.stack([model.logits(xc, yc, candidates, z=mu + e * (.5*lv).exp()).sigmoid()
                               for e in noise])
        def entropy(q):
            q = q.clamp(1e-7, 1-1e-7)
            return -q*q.log() - (1-q)*(1-q).log()
        return (entropy(samples.mean(0)) - entropy(samples).mean(0)).clamp_min(0)
    if policy == 'expected_brier':
        baseline = model.logits(xc, yc, reference).sigmoid()
        result = []
        for x, probability in zip(candidates, p):
            xnew = torch.cat([xc, x[None]], 0)
            changes = []
            for y in (0., 1.):
                ynew = torch.cat([yc, yc.new_tensor([y])])
                updated = model.logits(xnew, ynew, reference).sigmoid()
                changes.append(((updated-baseline)**2).mean())
            result.append((1-probability)*changes[0] + probability*changes[1])
        return torch.stack(result)
    raise ValueError(policy)


def accept(score_value, calibration, b, r, uniform):
    if b >= r:
        return True
    if calibration is None:
        return uniform < b/r
    # Smoothed upper-tail rank; random tie breaking prevents collapsed BALD
    # or an item-independent predictor from buying all labels immediately.
    a = np.asarray(calibration)
    tail = (np.count_nonzero(a > score_value) + uniform*(1+np.count_nonzero(a == score_value))) / (len(a)+1)
    return tail <= b/r


@torch.inference_mode()
def trajectory(model, features, item_ids, stream, targets, reveal, target_labels,
               policy, reference, probes, seed):
    """Only reveal(index) can read acquisition outcomes, after acceptance.

    Calibration probes and references are train-only item features. Scores are
    recomputed after each purchase, without using future stream features.
    target_labels is used solely by the scoring snapshots.
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(16, model.latent_dim, generator=gen)
    xc, yc = features.new_empty((0, features.shape[1])), features.new_empty(0)
    scores, positions, forced, score_values, latent_spread = {}, [], [], [], []
    calibration = None
    started = time.monotonic()
    def snapshot(k):
        total = 0.
        for start in range(0, len(targets), 256):
            t = targets[start:start+256]
            p = model.logits(xc, yc, features[item_ids[t]]).sigmoid().numpy()
            total += float(np.square(p-target_labels[start:start+256]).sum())
        scores[str(k)] = total / len(targets)
    snapshot(0)
    for position, index in enumerate(stream):
        b, r = 31-len(positions), len(stream)-position
        if b == 0:
            break
        uniform = rng.random()
        if policy == 'random':
            value = 0.
        else:
            if calibration is None:
                calibration = score(model, xc, yc, probes, policy, reference, noise).numpy()
            candidate = features[item_ids[[index]]]
            value = float(score(model, xc, yc, candidate, policy, reference, noise)[0])
            if not np.isfinite(value) or not np.isfinite(calibration).all():
                raise RuntimeError('Nonfinite acquisition score')
            score_values.append(value)
        if accept(value, calibration, b, r, uniform):
            label = float(reveal(int(index)))
            xc = torch.cat([xc, features[item_ids[[index]]]], 0)
            yc = torch.cat([yc, yc.new_tensor([label])])
            positions.append(position+1)
            forced.append(b >= r)
            calibration = None
            if len(positions) in BUDGETS:
                snapshot(len(positions))
    if len(positions) != 31:
        raise RuntimeError('Budget not filled')
    return {'by_budget': scores, 'brier_alc': sum(w*scores[str(k)] for w,k in zip(WEIGHTS,BUDGETS)),
            'purchase_positions': positions, 'forced_purchases': sum(forced),
            'candidates_seen': positions[-1], 'pool_size': len(stream),
            'runtime_seconds': time.monotonic()-started,
            'score_mean': float(np.mean(score_values)) if score_values else None,
            'score_max': max(score_values) if score_values else None}


def aggregate(rows):
    weights = [r['weight'] for r in rows]
    benchmarks = sorted({r['benchmark'] for r in rows})
    macro = {b: float(np.mean([r['brier_alc'] for r in rows if r['benchmark']==b])) for b in benchmarks}
    return {'brier_alc': float(np.average([r['brier_alc'] for r in rows], weights=weights)),
            'by_budget': {str(k): float(np.average([r['by_budget'][str(k)] for r in rows], weights=weights)) for k in BUDGETS},
            'by_benchmark': macro, 'benchmark_macro_alc': float(np.mean(list(macro.values()))),
            'mean_forced_purchases': float(np.mean([r['forced_purchases'] for r in rows])),
            'total_runtime_seconds': sum(r['runtime_seconds'] for r in rows), 'tasks': rows}
