#!/usr/bin/env python3
"""Monitor the current NP jobs without ever launching or restarting one.

It reports each Modal app's state, whether its atomic best/resume checkpoint
still exists, and whether the final result file arrived. If a job stops before
writing its result, it prints recent Modal logs and exits non-zero. That makes
credit exhaustion/preemption visible while preserving the latest checkpoint.

Usage:
  .venv/bin/python scripts/watch_modal_np_runs.py
  .venv/bin/python scripts/watch_modal_np_runs.py --once
  .venv/bin/python scripts/watch_modal_np_runs.py --interval 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv/bin/modal"
PROFILE = "aims-neurips"
VOLUME = "aims-neurips-data"


@dataclass(frozen=True)
class Run:
    name: str
    app_id: str
    best_checkpoint: str
    resume_checkpoint: str
    result_file: str


# These are the currently running apps. Do not replace their IDs with a shared
# launcher description: CA-only and SA=1 ANP deliberately share that name.
RUNS = (
    Run(
        "latent CNP-128",
        "ap-lYf2PWjgHFl45rTCgLePco",
        "results/cnp/checkpoints/linear-mean-latent-w128-es-fixed-"
        "latent-cnp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.pt",
        "results/cnp/checkpoints/linear-mean-latent-w128-es-fixed-"
        "latent-cnp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.resume.pt",
        "results/cnp/cnp-codex-es-fixed-linear-mean-latent-w128-es-fixed-"
        "latent-cnp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.parquet",
    ),
    Run(
        "ANP CA-only-128",
        "ap-HQrhnbM15uFl8a29XwGLvt",
        "results/cnp/checkpoints/anp-crossattn-latent-w128-sa0-h4-z128-es-fixed-"
        "anp-ca-only-w128-l4-100k-s20261018-qwen3_8b-seed20261018.pt",
        "results/cnp/checkpoints/anp-crossattn-latent-w128-sa0-h4-z128-es-fixed-"
        "anp-ca-only-w128-l4-100k-s20261018-qwen3_8b-seed20261018.resume.pt",
        "results/cnp/cnp-codex-es-fixed-anp-crossattn-latent-w128-sa0-h4-z128-es-fixed-"
        "anp-ca-only-w128-l4-100k-s20261018-qwen3_8b-seed20261018.parquet",
    ),
    Run(
        "ANP SA=1-128",
        "ap-2uIV01jCnhVy1MrhkDx81x",
        "results/cnp/checkpoints/anp-crossattn-latent-w128-sa1-h4-z128-es-fixed-"
        "anp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.pt",
        "results/cnp/checkpoints/anp-crossattn-latent-w128-sa1-h4-z128-es-fixed-"
        "anp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.resume.pt",
        "results/cnp/cnp-codex-es-fixed-anp-crossattn-latent-w128-sa1-h4-z128-es-fixed-"
        "anp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.parquet",
    ),
    Run(
        "stacked ANP SA=2-128",
        "ap-ixnPRsfRh54hBNbI2w7jPO",
        "results/cnp/checkpoints/anp-crossattn-latent-w128-sa2-h4-z128-es-fixed-"
        "stacked-anp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.pt",
        "results/cnp/checkpoints/anp-crossattn-latent-w128-sa2-h4-z128-es-fixed-"
        "stacked-anp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.resume.pt",
        "results/cnp/cnp-codex-es-fixed-anp-crossattn-latent-w128-sa2-h4-z128-es-fixed-"
        "stacked-anp128-cpu10-diagnostics-100k-s20261018-qwen3_8b-seed20261018.parquet",
    ),
)


def _modal(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (str(MODAL), *args), cwd=ROOT, text=True, capture_output=True, check=False
    )


def _volume_paths(directory: str) -> set[str]:
    """List one directory once; polling must not make 12 remote calls."""
    response = _modal("volume", "ls", VOLUME, "/" + directory, "--profile", PROFILE)
    if response.returncode:
        raise RuntimeError(response.stderr.strip() or f"could not list {directory}")
    return {line.strip().lstrip("/") for line in response.stdout.splitlines() if line.strip()}


def _apps() -> dict[str, dict]:
    response = _modal("app", "list", "--profile", PROFILE, "--json")
    if response.returncode:
        raise RuntimeError(response.stderr.strip() or "Modal app status query failed")
    return {app["app_id"]: app for app in json.loads(response.stdout)}


def _recent_logs(app_id: str) -> str:
    response = _modal(
        "app", "logs", app_id, "--tail", "30", "--timestamps", "--profile", PROFILE
    )
    text = response.stdout.strip() or response.stderr.strip()
    return text or "(no Modal logs returned)"


def _is_stopped(app: dict | None) -> bool:
    return app is None or str(app.get("state", "")).lower() == "stopped"


def poll_once(apps: dict[str, dict], *, show_logs: bool) -> bool:
    """Print one health snapshot and return True only on an incomplete stop."""
    failed = False
    checkpoint_paths = _volume_paths("results/cnp/checkpoints")
    result_paths = _volume_paths("results/cnp")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"\n[{stamp}]", flush=True)
    for run in RUNS:
        app = apps.get(run.app_id)
        state = "missing" if app is None else str(app.get("state", "unknown"))
        tasks = "?" if app is None else str(app.get("tasks", "?"))
        best = run.best_checkpoint in checkpoint_paths
        resume = run.resume_checkpoint in checkpoint_paths
        finished = run.result_file in result_paths
        print(
            f"{run.name}: state={state}; tasks={tasks}; "
            f"best_checkpoint={'yes' if best else 'NO'}; "
            f"resume_checkpoint={'yes' if resume else 'NO'}; "
            f"final_result={'yes' if finished else 'no'}",
            flush=True,
        )
        if not finished and _is_stopped(app):
            failed = True
            print(f"ALERT: {run.name} stopped before writing its final result.", flush=True)
            print("Latest checkpoint is preserved; this watcher will not restart it.", flush=True)
            print(_recent_logs(run.app_id), flush=True)
        elif show_logs:
            print(_recent_logs(run.app_id), flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60, help="polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--logs", action="store_true", help="show the last 30 logs every poll")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")

    while True:
        try:
            failed = poll_once(_apps(), show_logs=args.logs)
        except RuntimeError as error:
            print(f"monitor query failed: {error}", file=sys.stderr, flush=True)
            return 3
        if failed:
            return 2
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
