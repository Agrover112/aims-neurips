#!/usr/bin/env python3
"""Poll a Codabench submission until it reaches a terminal status.

Examples:
    python scripts/watch_codabench.py
    python scripts/watch_codabench.py 1005155
    python scripts/watch_codabench.py 1005155 --interval 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://www.codabench.org/api"
PHASE_ID = 29785
TOKEN_FILE = Path(
    os.environ.get("CODABENCH_TOKEN_FILE", "~/.config/paiec/codabench-token")
).expanduser()

TERMINAL = {"Finished", "Failed", "Cancelled", "None"}
NEEDS_ACTION = {"Awaiting validation"}


def load_token() -> str:
    token = os.environ.get("CODABENCH_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.is_file():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "No Codabench token. Run:\n"
        "  .venv/bin/python reference/paiec_baseline/submit_api.py --login"
    )


def request_json(session: requests.Session, path: str, params=None):
    last_error = None
    for attempt in range(4):
        try:
            response = session.get(f"{BASE}/{path}", params=params, timeout=30)
        except requests.RequestException as error:
            last_error = error
        else:
            if response.status_code == 401:
                raise SystemExit("Token rejected. Run submit_api.py --login again.")
            if response.status_code < 500:
                response.raise_for_status()
                return response.json()
            last_error = RuntimeError(f"HTTP {response.status_code}")
        if attempt < 3:
            time.sleep(2**attempt)
    raise SystemExit(f"Request failed: {last_error}")


def parse_time(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def age(created: datetime | None) -> str:
    if created is None:
        return "?"
    seconds = int((datetime.now(timezone.utc) - created).total_seconds())
    hours, remainder = divmod(max(seconds, 0), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def summary(submission: dict) -> str:
    status = submission.get("status") or "None"
    created = parse_time(submission.get("created_when"))
    started = parse_time(submission.get("started_when"))
    finished = parse_time(submission.get("finished_when"))
    parts = [f"id={submission.get('id')}", f"status={status}", f"age={age(created)}"]
    if started:
        parts.append(f"started={started.strftime('%H:%M:%S')}Z")
    if finished:
        parts.append(f"finished={finished.strftime('%H:%M:%S')}Z")
    return " | ".join(parts)


def detail_text(submission: dict, limit: int = 800) -> str | None:
    text = (submission.get("status_details") or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    return text[:limit]


def score_text(submission: dict) -> str:
    values = submission.get("scores") or []
    if not values:
        return "hidden or not reported"
    rendered = []
    for value in values:
        if not isinstance(value, dict):
            rendered.append(str(value))
            continue
        label = value.get("name") or value.get("column") or value.get("key")
        score = value.get("score")
        rendered.append(f"{label}={score}" if label else json.dumps(value))
    return ", ".join(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_id",
        nargs="?",
        type=int,
        help="submission ID; omit to use the latest submission in phase 29785",
    )
    parser.add_argument(
        "--interval", type=float, default=60, help="seconds between polls (default: 60)"
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=9,
        help="stop after this many hours (default: 9)",
    )
    parser.add_argument(
        "--once", action="store_true", help="print one status and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every poll, not only status changes",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers["Authorization"] = f"Token {load_token()}"
    session.headers["Accept"] = "application/json"

    if args.submission_id is None:
        listing = request_json(
            session, "submissions/", {"phase": PHASE_ID, "page_size": 20}
        )
        results = listing.get("results", [])
        if not results:
            raise SystemExit("No submissions found for phase 29785.")
        submission_id = max(results, key=lambda item: item["id"])["id"]
    else:
        submission_id = args.submission_id

    last_status = None
    deadline = time.monotonic() + args.timeout_hours * 3600

    while True:
        submission = request_json(session, f"submissions/{submission_id}/")
        status = submission.get("status") or "None"

        if status != last_status or args.verbose:
            print(summary(submission), flush=True)
            note = detail_text(submission)
            if note and (status in TERMINAL or status in NEEDS_ACTION or args.verbose):
                print(f"  {note}", flush=True)
            last_status = status

        if args.once:
            print(f"Scores: {score_text(submission)}", flush=True)
            return 0
        if status in TERMINAL:
            print(f"Scores: {score_text(submission)}", flush=True)
            return 0 if status == "Finished" else 1
        if status in NEEDS_ACTION:
            print("Stopped: the submission needs organizer action.", flush=True)
            return 3
        if time.monotonic() >= deadline:
            print(
                f"Stopped after {args.timeout_hours:g} hours; last status: {status}",
                file=sys.stderr,
            )
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
