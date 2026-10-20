#!/usr/bin/env python3
"""Print the Codabench logs for a submission.

Examples:
    python scripts/codabench_logs.py 1005187
    python scripts/codabench_logs.py 1005187 --list
    python scripts/codabench_logs.py 1005187 --name prediction_stderr --tail 100
    python scripts/codabench_logs.py 1005187 --follow --interval 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.codabench.org/api"
TOKEN_FILE = Path(
    os.environ.get("CODABENCH_TOKEN_FILE", "~/.config/paiec/codabench-token")
).expanduser()


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


def fetch_details(session: requests.Session, submission_id: int) -> dict:
    response = session.get(
        f"{BASE}/submissions/{submission_id}/get_details/", timeout=30
    )
    if response.status_code == 401:
        raise SystemExit("Token rejected. Run submit_api.py --login again.")
    if response.status_code == 403:
        raise SystemExit("Logs are hidden for this phase.")
    response.raise_for_status()
    return response.json()


def fetch_log(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def tail(text: str, lines: int) -> str:
    if lines <= 0:
        return text
    parts = text.splitlines(keepends=True)
    return "".join(parts[-lines:])


def print_log(name: str, text: str, lines: int | None = None) -> None:
    body = tail(text, lines) if lines is not None else text
    print(f"===== {name} ({len(text)} bytes) =====", flush=True)
    if not body:
        print("(empty)", flush=True)
    else:
        print(body, end="" if body.endswith("\n") else "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id", type=int, help="Codabench submission ID")
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="log name to print; repeat for multiple logs (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list log names and sizes, then exit"
    )
    parser.add_argument(
        "--tail", type=int, metavar="N", help="print only the last N lines"
    )
    parser.add_argument(
        "--follow", action="store_true", help="print new log output as it appears"
    )
    parser.add_argument(
        "--interval", type=float, default=15, help="seconds between checks in follow mode"
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers["Authorization"] = f"Token {load_token()}"
    session.headers["Accept"] = "application/json"

    selected = set(args.name)
    offsets: dict[str, int] = {}
    first = True

    try:
        while True:
            details = fetch_details(session, args.submission_id)
            logs = details.get("logs") or []
            if args.list:
                if not logs:
                    print("No logs available.")
                for log in logs:
                    name = log.get("name", "?")
                    text = fetch_log(log["data_file"])
                    print(f"{name} ({len(text)} bytes)")
                return 0
            if not logs:
                if first:
                    print("No logs available.")
                first = False
            for log in logs:
                name = log.get("name", "?")
                if selected and name not in selected:
                    continue
                text = fetch_log(log["data_file"])
                if args.follow:
                    previous = offsets.get(name, 0)
                    if len(text) < previous:
                        previous = 0
                    new_text = text[previous:]
                    offsets[name] = len(text)
                    if new_text.strip():
                        print_log(name, new_text)
                else:
                    print_log(name, text, args.tail)
            if not args.follow:
                return 0
            first = False
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    except requests.RequestException as error:
        print(f"Log request failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
