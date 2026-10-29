"""CPU statistical predictors, fitted only to observed training-role responses."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from scipy.special import expit, betaln, digamma, logsumexp
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, block_diag

MODELS = ("beta_binomial", "rasch", "feature_rasch", "linear_regression", "bayesian_rbf")
FRACTIONS = (.05, .10, .25, .50, 1.)
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)

@lru_cache(maxsize=4)
def quadrature(n):
    return np.polynomial.hermite.hermgauss(n)

def training_cells(tasks, fraction, seed):
    rng = np.random.default_rng(seed + 4000)
    ti, ii, yy = [], [], []
    for task_id, task in enumerate(tasks):
        if task["role"] != "train":
            continue
        order = rng.permutation(len(task["items"]))
        chosen = order[:max(1, int(np.ceil(fraction * len(order))))]
        ti.append(np.full(len(chosen), task_id, dtype=np.int32))
        ii.append(task["items"][chosen])
        yy.append(task["labels"][chosen])
    return np.concatenate(ti), np.concatenate(ii), np.concatenate(yy).astype(float)

def feature_map(features, train_items, kind, seed, kernel_dim=128):
    mean = np.mean(features[train_items], axis=0)
    scale = np.maximum(np.std(features[train_items], axis=0), 1e-6)
    z = (np.asarray(features, dtype=np.float64) - mean) / scale
    metadata = {"mean": mean, "scale": scale}
    if kind == "bayesian_rbf":
        rng = np.random.default_rng(seed + 7000)
        # RBF length scale sqrt(input dimension), fixed before validation.
        omega = rng.normal(size=(z.shape[1], kernel_dim)) / np.sqrt(z.shape[1])
        phase = rng.uniform(0, 2 * np.pi, kernel_dim)
        z = np.sqrt(2. / kernel_dim) * np.cos(z @ omega + phase)
        metadata.update(omega=omega, phase=phase, kernel="RBF random Fourier approximation")
    return np.column_stack([np.ones(len(z)), z]), metadata

def fit(kind, cells, phi, n_tasks, n_items, iterations=30):
    task, item, y = cells
    nt = np.bincount(task, minlength=n_tasks).astype(float)
    st = np.bincount(task, weights=y, minlength=n_tasks)
    ni = np.bincount(item, minlength=n_items).astype(float)
    si = np.bincount(item, weights=y, minlength=n_items)
    active_t, active_i = nt > 0, ni > 0
    if kind == "beta_binomial":
        n, s = nt[active_t], st[active_t]
        def objective(log_ab):
            a, b = np.exp(log_ab)
            value = -np.sum(betaln(a + s, b + n - s) - betaln(a, b))
            common = digamma(a + b + n) - digamma(a + b)
            gradient = -np.array([a * np.sum(digamma(a + s) - digamma(a) - common),
                                  b * np.sum(digamma(b + n - s) - digamma(b) - common)])
            return value, gradient
        result = minimize(objective, np.log([2., 2.]), jac=True, method="L-BFGS-B",
                          bounds=[(np.log(.05), np.log(1e4))] * 2)
        if not np.isfinite(result.fun):
            raise RuntimeError("Beta prior fit failed")
        return {"kind": kind, "alpha": float(np.exp(result.x[0])), "beta": float(np.exp(result.x[1])),
                "fit_objective": float(result.fun), "optimizer_success": bool(result.success)}
    if kind == "rasch":
        theta = np.zeros(n_tasks)
        theta[active_t] = np.log((st[active_t] + .5) / (nt[active_t] - st[active_t] + .5))
        b = np.zeros(n_items)
        mu, ability_sd, difficulty_sd = float(theta[active_t].mean()), 1., 1.
        for _ in range(iterations):
            p = expit(theta[task] - b[item])
            g = np.bincount(task, weights=y - p, minlength=n_tasks) - (theta - mu) / ability_sd ** 2
            h = np.bincount(task, weights=p * (1-p), minlength=n_tasks) + 1 / ability_sd ** 2
            theta[active_t] += .5 * np.clip((g / h)[active_t], -2, 2)
            p = expit(theta[task] - b[item])
            g = np.bincount(item, weights=p - y, minlength=n_items) - b / difficulty_sd ** 2
            h = np.bincount(item, weights=p * (1-p), minlength=n_items) + 1 / difficulty_sd ** 2
            b[active_i] += .5 * np.clip((g / h)[active_i], -2, 2)
            shift = float(b[active_i].mean())
            b[active_i] -= shift
            theta[active_t] -= shift
            mu = float(theta[active_t].mean())
            ability_sd = float(np.clip(np.std(theta[active_t]), .2, 3.))
            difficulty_sd = float(np.clip(np.std(b[active_i]), .2, 3.))
        return {"kind": kind, "difficulty": b, "known_items": active_i,
                "ability_mean": mu, "ability_sd": ability_sd, "difficulty_sd": difficulty_sd}
    if kind == "feature_rasch":
        theta = np.zeros(n_tasks)
        w = np.zeros(phi.shape[1])
        w[0] = np.log((y.mean() + 1e-5) / (1 - y.mean() + 1e-5))
        penalty = np.full(len(w), 10.)
        penalty[0] = 1e-6
        fi = phi[active_i]
        for _ in range(iterations):
            ease = phi @ w
            p = expit(theta[task] + ease[item])
            g = np.bincount(task, weights=y-p, minlength=n_tasks) - theta
            h = np.bincount(task, weights=p*(1-p), minlength=n_tasks) + 1.
            theta[active_t] += .5 * np.clip((g / h)[active_t], -2, 2)
            p = expit(theta[task] + ease[item])
            gi = np.bincount(item, weights=y-p, minlength=n_items)
            hi = np.bincount(item, weights=p*(1-p), minlength=n_items)
            gradient = phi.T @ gi - penalty * w
            hessian = fi.T @ (hi[active_i, None] * fi) + np.diag(penalty)
            w += .5 * np.linalg.solve(hessian, gradient)
            shift = float(theta[active_t].mean())
            theta[active_t] -= shift
            w[0] += shift
        return {"kind": kind, "w": w, "item_easiness": phi @ w, "ability_mean": 0., "ability_sd": 1.}
    if kind not in ("linear_regression", "bayesian_rbf"):
        raise ValueError(kind)
    if kind == "linear_regression":
        # Ordinary least squares on item features plus a subject-context rate.
        # Leave the target response OUT of its training context-rate feature.
        rate = (2. + st[task] - y) / (3. + nt[task])
        fi = phi[active_i]
        gram = fi.T @ (ni[active_i, None] * fi)
        cross = phi.T @ np.bincount(item, weights=rate, minlength=n_items)
        gram = np.block([[gram, cross[:, None]], [cross[None, :], np.array([[rate @ rate]])]])
        rhs = np.r_[phi.T @ si, rate @ y]
        coefficient = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
        return {"kind": kind, "w": coefficient[:-1], "context_coefficient": float(coefficient[-1]),
                "interpretation": "ordinary linear probability regression; leave-one-out context-rate feature"}
    # Gaussian subject intercepts are integrated out using a Schur complement.
    noise_var, subject_var = .25, .25
    ratio = noise_var / subject_var
    sums = np.zeros((n_tasks, phi.shape[1]))
    for j in range(phi.shape[1]):
        sums[:, j] = np.bincount(task, weights=phi[item, j], minlength=n_tasks)
    regularization = 1.
    prior = np.eye(phi.shape[1]) * regularization
    prior[0, 0] = 1e-8  # diffuse global intercept; numerical stabilization for OLS
    fi = phi[active_i]
    v = sums[active_t]
    denominator = ratio + nt[active_t]
    precision = (fi.T @ (ni[active_i, None] * fi) - v.T @ (v / denominator[:, None])) / noise_var + prior
    rhs = (phi.T @ si - v.T @ (st[active_t] / denominator)) / noise_var
    precision = (precision + precision.T) / 2
    factor = cho_factor(precision)
    covariance = cho_solve(factor, np.eye(len(rhs)))
    w = cho_solve(factor, rhs)
    return {"kind": kind, "w": w, "covariance": covariance, "precision": precision,
            "noise_var": noise_var, "subject_var": subject_var,
            "interpretation": "finite-feature RBF Gaussian process regression"}

def predict(state, ci, cy, ti, phi):
    cy = np.asarray(cy, dtype=float)
    kind = state["kind"]
    if kind == "beta_binomial":
        return np.full(len(ti), (state["alpha"] + cy.sum()) / (state["alpha"] + state["beta"] + len(cy)))
    if kind in ("rasch", "feature_rasch"):
        nodes, weights = quadrature(81)
        theta = state["ability_mean"] + np.sqrt(2.) * state["ability_sd"] * nodes
        if kind == "feature_rasch":
            ease = state["item_easiness"]
            pc = expit(theta[:, None] + ease[ci][None, :])
            pt = expit(theta[:, None] + ease[ti][None, :])
        else:
            dn, dw = quadrature(21)
            unknown = expit(theta[:, None] - np.sqrt(2.) * state["difficulty_sd"] * dn) @ (dw / np.sqrt(np.pi))
            pc = np.broadcast_to(unknown[:, None], (len(theta), len(ci))).copy()
            pt = np.broadcast_to(unknown[:, None], (len(theta), len(ti))).copy()
            known_c = state["known_items"][ci]
            known_t = state["known_items"][ti]
            pc[:, known_c] = expit(theta[:, None] - state["difficulty"][ci[known_c]])
            pt[:, known_t] = expit(theta[:, None] - state["difficulty"][ti[known_t]])
        pc = np.clip(pc, 1e-12, 1-1e-12)
        log_weight = np.log(weights / np.sqrt(np.pi)) + (cy * np.log(pc) + (1-cy) * np.log1p(-pc)).sum(axis=1)
        posterior = np.exp(log_weight - logsumexp(log_weight))
        return posterior @ pt
    xc, xt = phi[ci], phi[ti]
    if kind == "linear_regression":
        rate = (2. + cy.sum()) / (4. + len(cy))
        return np.clip(xt @ state["w"] + state["context_coefficient"] * rate, 0., 1.)
    m = np.r_[state["w"], 0.]
    if len(ci):
        dc = np.column_stack([xc, np.ones(len(ci))])
        covariance = block_diag(state["covariance"], state["subject_var"])
        if len(ci) <= len(m):
            gain_rhs = covariance @ dc.T
            gram = dc @ gain_rhs + np.eye(len(ci)) * state["noise_var"]
            m += gain_rhs @ cho_solve(cho_factor(gram), cy - dc @ m)
        else:
            precision = block_diag(state["precision"], 1 / state["subject_var"])
            m = cho_solve(cho_factor(precision + dc.T @ dc / state["noise_var"]),
                          precision @ m + dc.T @ cy / state["noise_var"])
    return np.clip(xt @ m[:-1] + m[-1], 0., 1.)

def metrics(y, p):
    from sklearn.metrics import roc_auc_score
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError("Invalid probability predictions")
    return {"brier_score": float(np.mean((y-p)**2)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}

def diagnostics(state, tasks, bank, phi, population):
    rows = {"same_subject_split_half": [], "same_subject_response_holdout": []}
    for ep in bank:
        task = tasks[ep["task"]]
        order = np.r_[ep["acquisition"], ep["targets"]]
        for axis, cut in (("same_subject_split_half", len(ep["acquisition"])),
                          ("same_subject_response_holdout", int(.8 * len(order)))):
            c, t = order[:cut], order[cut:]
            p = predict(state, task["items"][c], task["labels"][c], task["items"][t], phi)
            rows[axis].append((metrics(task["labels"][t], p), ep["weight"]))
    result = {}
    for axis, values in rows.items():
        for metric in ("brier_score", "roc_auc"):
            valid = [(m[metric], w) for m, w in values if m[metric] is not None]
            if valid:
                result[f"diagnostic/{population}/{axis}/{metric}"] = float(np.average([v for v,w in valid], weights=[w for v,w in valid]))
            result[f"diagnostic/{population}/{axis}/{metric}_tasks"] = len(valid)
    return result

def evaluate(state, tasks, bank, phi):
    rows = []
    for ep in bank:
        task = tasks[ep["task"]]
        scores, aucs = [], []
        t = ep["targets"]
        for budget in BUDGETS:
            c = ep["acquisition"][:budget]
            p = predict(state, task["items"][c], task["labels"][c], task["items"][t], phi)
            m = metrics(task["labels"][t], p)
            scores.append(m["brier_score"])
            aucs.append(m["roc_auc"])
        rows.append({"subject": task["subject"], "benchmark": task["benchmark"], "weight": ep["weight"],
                     "brier_alc": sum(w*s for w,s in zip(WEIGHTS, scores)),
                     "by_budget": dict(zip(map(str, BUDGETS), scores)),
                     "auc_by_budget": dict(zip(map(str, BUDGETS), aucs))})
    weights = [r["weight"] for r in rows]
    by_bench = {b: float(np.mean([r["brier_alc"] for r in rows if r["benchmark"] == b])) for b in sorted({r["benchmark"] for r in rows})}
    return {"brier_alc": float(np.average([r["brier_alc"] for r in rows], weights=weights)),
            "by_budget": {str(b): float(np.average([r["by_budget"][str(b)] for r in rows], weights=weights)) for b in BUDGETS},
            "by_benchmark": by_bench, "benchmark_macro_alc": float(np.mean(list(by_bench.values()))), "tasks": rows}
