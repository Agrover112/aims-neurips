#!/usr/bin/env python3
"""Live progress for the detached replay runs, written to a status file.

The replay runners are single long container calls with no checkpointing, so
the only trace of progress is the ``task i/N`` line each one prints. This reads
those lines, turns them into seconds per task and a finish estimate, and flags
any cell whose estimate lands after the deadline -- the one fact that matters
on the last day, and one a bare progress bar hides.

Run it once in the background and watch the file from any terminal::

    nohup python organized/testing/online/watch_replay.py --profile iclrgram &
    watch -n 30 cat organized/testing/online/results/progress.txt

or print to the terminal instead with ``--stdout``.

Estimates divide elapsed time by tasks completed, so they assume every task
costs the same. Tasks differ in item count, so treat an ETA as a rough bound
that tightens as more tasks finish, not as a promise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODAL = str(ROOT / ".venv" / "bin" / "modal")
STATUS = HERE / "results" / "progress.txt"
VOLUME = "aims-neurips-data"
DEFAULT_PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"

# 23:59 Anywhere-on-Earth on 2026-10-30 is 11:59 UTC on the 31st.
DEADLINE = datetime(2026, 10, 31, 11, 59, tzinfo=timezone.utc)

# run_sweep.py dispatches a model's policies in this order, so apps created
# together can be matched to cells by creation order. Scored policies also print
# their own name, which overrides the guess and exposes a mismatch.
POLICY_ORDER = ["random_prefix", "uncertainty", "confidence_extremes",
                "bernoulli_variance", "expected_brier_reduction"]
TASK_LINE = re.compile(r"^task (\d+)/(\d+) t=(\d+)s fits=(\d+) predicts=(\d+)", re.M)
POLICY_LINE = re.compile(r"policy (bernoulli_variance|expected_brier_reduction):")
TRACEBACK = re.compile(r"Traceback|ValueError|ExecutionError|hit its timeout")


def _run(command: list[str], timeout: int) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return done.stdout + done.stderr
    except Exception:  # a flaky CLI call must not kill the watcher
        return ""


def list_apps(profile: str, since: datetime) -> list[dict]:
    raw = _run([MODAL, "app", "list", "--profile", profile, "--json"], 90)
    try:
        apps = json.loads(raw[raw.index("["):])
    except ValueError:
        return []
    kept = []
    for app in apps:
        if "online-replay" not in app.get("description", ""):
            continue
        created = datetime.fromisoformat(app["created_at"])
        if created >= since:
            app["created"] = created
            kept.append(app)
    return sorted(kept, key=lambda item: item["created"])


def inspect(profile: str, app: dict) -> dict:
    logs = _run([MODAL, "app", "logs", app["app_id"], "--profile", profile,
                 "--tail", "400"], 120)
    tasks = TASK_LINE.findall(logs)
    policy = POLICY_LINE.search(logs)
    info = {"app": app, "policy_seen": policy.group(1) if policy else None,
            "failed": bool(TRACEBACK.search(logs))}
    if tasks:
        index, total, elapsed, fits, predicts = (int(v) for v in tasks[-1])
        info.update(index=index, total=total, elapsed=elapsed, fits=fits,
                    predicts=predicts)
    return info


def finished_cells(profile: str, panel: str) -> list[str]:
    out = _run([MODAL, "volume", "ls", VOLUME,
                f"results/online-replay/runs/{panel}", "--profile", profile], 90)
    return sorted(line.strip().rstrip("/").split("/")[-1]
                  for line in out.splitlines() if "/" in line)


def render(profile: str, panel: str, since: datetime, interval: int) -> str:
    now = datetime.now(timezone.utc)
    apps = list_apps(profile, since)
    with ThreadPoolExecutor(max_workers=6) as pool:
        details = list(pool.map(lambda app: inspect(profile, app), apps))
    # A stopped app that never reached a task line is a launch that crashed in
    # its constructor. It would otherwise occupy a slot in the creation-order
    # match below and shift every later cell onto the wrong policy.
    details = [detail for detail in details
               if detail["app"]["state"].startswith("ephemeral") or "index" in detail]

    # Label cells: group by description, then match creation order to policies.
    seen: dict[str, int] = {}
    for detail in details:
        description = detail["app"]["description"]
        position = seen.get(description, 0)
        seen[description] = position + 1
        guess = POLICY_ORDER[position] if position < len(POLICY_ORDER) else "?"
        policy = detail["policy_seen"] or guess
        detail["conflict"] = bool(detail["policy_seen"] and detail["policy_seen"] != guess)
        kind = description.replace("aims-neurips-", "").replace("-online-replay", "")
        kind = kind.replace("benchmark-holdout", "ANP/beta")
        detail["label"] = f"{kind} x {policy}"

    lines = [
        f"replay progress   profile={profile}   refresh={interval}s   "
        f"updated {now.astimezone().strftime('%a %H:%M:%S')} local",
        f"panel {panel}",
        f"deadline in {str(DEADLINE - now).split('.')[0]}  "
        f"({DEADLINE.astimezone().strftime('%a %H:%M')} local)",
        "",
        f"{'cell':<42}{'state':<10}{'task':>9}{'  %':>5}{'  s/task':>9}"
        f"{'  eta (local)':>14}  note",
        "-" * 100,
    ]
    for detail in details:
        app = detail["app"]
        running = app["state"].startswith("ephemeral")
        state = "running" if running else "stopped"
        note = []
        if detail["conflict"]:
            note.append("policy order guess was wrong")
        if detail["failed"]:
            note.append("ERROR in logs")
        if "index" not in detail:
            lines.append(f"{detail['label']:<42}{state:<10}{'-':>9}{'':>5}{'':>9}"
                         f"{'starting':>14}  {'; '.join(note)}")
            continue
        index, total = detail["index"], detail["total"]
        done = max(index - 1, 0)
        rate = detail["elapsed"] / done if done else None
        if running and rate:
            eta = now + timedelta(seconds=rate * (total - done))
            eta_text = eta.astimezone().strftime("%a %H:%M")
            if eta > DEADLINE:
                note.append("MISSES DEADLINE")
        else:
            eta_text = "-"
        lines.append(
            f"{detail['label']:<42}{state:<10}{f'{index}/{total}':>9}"
            f"{100 * done / total:>5.0f}{(f'{rate:.0f}' if rate else '-'):>9}"
            f"{eta_text:>14}  {'; '.join(note)}")

    names = finished_cells(profile, panel)
    lines += ["", f"cells with results on the volume: {len(names)}"]
    if names:
        lines.append("  " + ", ".join(
            n.replace("validation-", "").replace("-s3-v2", "") for n in names))
    lines += ["", "ETA = elapsed / tasks done, so it assumes equal-cost tasks; "
              "it tightens as more finish."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="iclrgram")
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--interval", type=int, default=120, help="seconds between polls")
    parser.add_argument("--since-hours", type=float, default=6.0,
                        help="only track apps created in this window")
    parser.add_argument("--out", default="", help="status file (default results/progress.txt); "
                        "use a different name per profile when watching two workspaces")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing a file")
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
    status = Path(args.out) if args.out else STATUS
    status.parent.mkdir(parents=True, exist_ok=True)
    while True:
        text = render(args.profile, args.panel, since, args.interval)
        if args.stdout:
            print("\033[2J\033[H" + text, flush=True)
        else:
            temporary = status.with_suffix(".tmp")
            temporary.write_text(text + "\n")
            temporary.replace(status)  # a reader never sees a half-written file
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
