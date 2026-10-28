#!/usr/bin/env python3
"""Print compact W&B histories for the current NP-family ablations."""

from __future__ import annotations

import json
import sys

import wandb


ENTITY = "agrgnn-kth-royal-institute-of-technology"
PROJECT = "aims-predictive-evaluation"
BASE_URL = "https://api.forge.coreweave.com"
DEFAULT_RUN_IDS = (
    "np-237d5d384f46dd28cb65",  # latent CNP
    "np-426061726576447014f7",  # ANP, one SA layer
    "np-0dbbdf8c1dc54da9bda1",  # stacked ANP, two SA layers
)
KEYS = (
    "_step",
    "loss/negative_elbo_per_target/training",
    "loss/negative_elbo_per_target/validation_resampled_episodes",
    "loss/context_reconstruction_nll/validation_fixed_bank",
    "loss/unseen_target_nll/validation_fixed_bank",
    "brier_score/unseen_target/validation_fixed_bank",
    "loss/posterior_target_nll/training",
    "loss/posterior_target_nll/validation_resampled_episodes",
    "loss/latent_kl_total/training",
    "loss/latent_kl_total/validation_resampled_episodes",
    "latent/prior/variance_mean",
    "latent/posterior/variance_mean",
    "latent/posterior_minus_prior/mu_abs_mean",
    "latent/kl/active_dimensions_above_0.01_nat",
)


def main() -> None:
    api = wandb.Api(overrides={"base_url": BASE_URL})
    run_ids = tuple(sys.argv[1:]) or DEFAULT_RUN_IDS
    for run_id in run_ids:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        rows = list(run.scan_history(keys=list(KEYS), page_size=1000))
        rows = [{key: row.get(key) for key in KEYS if row.get(key) is not None}
                for row in rows]
        rows = [row for row in rows if len(row) > 1]
        print(json.dumps({
            "id": run_id,
            "name": run.name,
            "state": run.state,
            "config": {
                key: run.config.get(key)
                for key in ("model_family", "width", "n_blocks", "latent_dim",
                            "steps", "requested_accelerator")
            },
            "history_rows": len(rows),
            "first": rows[0] if rows else None,
            "tail": rows[-8:],
        }, allow_nan=False))


if __name__ == "__main__":
    main()
