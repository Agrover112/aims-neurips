"""R10: compare scoring rules using saved predictions and frozen draws."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import BUDGETS, WEIGHTS
from wp9 import PRED_KEYS, verify_draws

WEIGHTINGS = {"official": list(WEIGHTS.values()), "uniform": [1/6]*6,
              "late": [0, 0, 0, 0, .5, .5]}


def brier_weight_sensitivity(handoff, runs, output):
    tables = []
    for run_name in runs:
        score = pd.read_parquet(handoff / "scores" / f"{run_name}.parquet")
        verify_draws(score, handoff)
        curve = (score.groupby(["model", "task_id", "budget"]).brier.mean()
                 .groupby(["model", "budget"]).mean().unstack("budget").reindex(columns=BUDGETS))
        if curve.isna().any().any():
            raise ValueError(f"Missing score budgets in {run_name}")
        summary = pd.DataFrame({name: curve @ weights for name, weights in WEIGHTINGS.items()})
        tables.append(summary.rename_axis(index="model", columns="weighting").stack().rename("brier")
                      .reset_index().assign(run=run_name))
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(tables, ignore_index=True).to_csv(output / "wp13-brier-weight-sensitivity.csv", index=False)


def run(handoff, runs, output):
    metrics, reliability, weighting = [], [], []
    for run_name in runs:
        score = pd.read_parquet(handoff / "scores" / f"{run_name}.parquet")
        verify_draws(score, handoff)
        cols = PRED_KEYS + ["content_hash", "p", "y"]
        records, bins = [], []
        files = sorted((handoff / "preds" / run_name).glob("draw-*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing saved predictions for {run_name}")
        for path in files:
            rows = pd.read_parquet(path, columns=cols)
            if rows.duplicated(cols[:-2]).any() or not rows.p.between(0, 1).all() or not rows.y.isin([0, 1]).all():
                raise ValueError(f"Invalid predictions in {path}")
            for key, group in rows.groupby(PRED_KEYS, observed=True, sort=False):
                p, y = group.p.to_numpy(float), group.y.to_numpy(float)
                q = p.clip(1e-7, 1-1e-7)
                rankable = y.min() != y.max() and p.min() != p.max()
                bucket = np.minimum((p*10).astype(int), 9)
                mass = np.bincount(bucket, minlength=10) / len(p)
                pp = np.bincount(bucket, weights=p, minlength=10) / len(p)
                yy = np.bincount(bucket, weights=y, minlength=10) / len(p)
                records.append((*key, np.mean((p-y)**2), -np.mean(y*np.log(q)+(1-y)*np.log1p(-q)),
                                abs(pp-yy).sum(), roc_auc_score(y, p) if rankable else np.nan,
                                spearmanr(y, p).statistic if rankable else np.nan))
                bins.extend((*key, i, mass[i], pp[i], yy[i]) for i in range(10))
        m = pd.DataFrame(records, columns=PRED_KEYS + ["brier_pred", "log_loss", "ece10", "auc", "spearman"])
        matched = score.merge(m, on=PRED_KEYS, how="outer", validate="one_to_one")
        err = abs(matched.brier - matched.brier_pred)
        if err.isna().any() or err.max() > 1e-7:
            raise ValueError(f"Prediction scores did not reproduce for {run_name}")
        cols = ["brier", "log_loss", "ece10", "auc", "spearman"]
        tasks = matched.groupby(["model", "task_id", "budget"])[cols].mean()
        macro = tasks.groupby(["model", "budget"]).mean().join(
            tasks.auc.groupby(["model", "budget"]).count().rename("rankable_tasks"))
        metrics.append(macro.reset_index().assign(run=run_name))
        for model, g in macro.groupby("model"):
            curve = g.droplevel("model").reindex(BUDGETS)
            for name, weights in WEIGHTINGS.items():
                row = (curve[["brier", "log_loss", "ece10"]].T @ np.asarray(weights)).to_dict()
                weighting.append(dict(run=run_name, model=model, weighting=name, **row))
        b = pd.DataFrame(bins, columns=PRED_KEYS + ["bin", "mass", "sum_p", "sum_y"])
        b = b.groupby(["model", "task_id", "budget", "bin"])[["mass", "sum_p", "sum_y"]].mean()
        b = b.groupby(["model", "budget", "bin"]).mean().reset_index().query("mass > 0").copy()
        b["probability"], b["observed"] = b.sum_p/b.mass, b.sum_y/b.mass
        reliability.append(b.drop(columns=["sum_p", "sum_y"]).assign(run=run_name))
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(metrics).to_csv(output / "wp13-metrics.csv", index=False)
    pd.DataFrame(weighting).to_csv(output / "wp13-metric-alc.csv", index=False)
    pd.concat(reliability).to_csv(output / "wp13-reliability.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff", required=True, type=Path)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scores-only", action="store_true")
    args = p.parse_args()
    if args.scores_only:
        brier_weight_sensitivity(args.handoff, args.runs, args.output)
    else:
        run(args.handoff, args.runs, args.output)
