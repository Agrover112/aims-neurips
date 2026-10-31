#!/usr/bin/env python3
"""Run the Modal ANP trainer on this machine, with the same resume behaviour.

The trainer (``codex_es_fixed_modal_cnp_train.run``) reads and writes under
``/data`` and commits a Modal volume. This points those paths at local folders
and replaces the volume with a no-op, then calls the unchanged function through
``run.local``. Checkpoints are atomic and written every 500 steps; start the
same command again and it resumes from the latest ``.resume.pt`` in
``--checkpoints``.

    python organized/training/local_anp_train.py \\
        --datasets <dir holding allbench-embed-bge_m3_long-v2> \\
        --responses filtered-responses.parquet --checkpoints <dir> \\
        --run-label anp-sa1-w64-bgem3-bhold-denseprefix-noitem-n15-cpu-20261031 \\
        --dense-prefix-training --item-independent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codex_es_fixed_modal_cnp_train as trainer  # noqa: E402


class _LocalVolume:
    """Stands in for the Modal volume: files are already on local disk."""

    def commit(self) -> None:
        return None

    def reload(self) -> None:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-label", required=True)
    p.add_argument("--arm", default="bge_m3_long")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--self-attention-blocks", type=int, default=1)
    p.add_argument("--cross-attention-heads", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--max-context", type=int, default=31)
    p.add_argument("--dense-prefix-training", action="store_true")
    p.add_argument("--item-independent", action="store_true")
    p.add_argument("--no-alc-weighted", action="store_true")
    p.add_argument("--wandb", action="store_true", help="log to W&B (needs WANDB_API_KEY)")
    a = p.parse_args()

    a.checkpoints.mkdir(parents=True, exist_ok=True)
    trainer.DATASETS = a.datasets
    trainer.PROCESSED = a.datasets / "processed"
    trainer.CHECKPOINTS = a.checkpoints
    trainer.OUT = a.out or a.checkpoints / "local-run.parquet"
    trainer.volume = _LocalVolume()

    result = trainer.run.local(
        seed=a.seed, arm=a.arm, use_latent=True, width=a.width, n_blocks=a.self_attention_blocks,
        steps=a.steps, wandb_mode="online" if a.wandb else "disabled", run_label=a.run_label,
        model_family="anp_cross_attention", max_context=a.max_context,
        cross_attention_heads=a.cross_attention_heads, latent_dim=a.latent_dim,
        requested_accelerator="local-cpu", benchmark_holdout=True,
        benchmark_min_item_responders=15, benchmark_min_items_per_subject=80,
        prefiltered_responses=str(a.responses), alc_weighted=not a.no_alc_weighted,
        dense_prefix_training=a.dense_prefix_training, item_independent=a.item_independent,
    )
    print("done:", result if not isinstance(result, dict) else {k: result[k] for k in list(result)[:6]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
