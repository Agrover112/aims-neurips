#!/usr/bin/env python3
"""Audit literal ID overlap between CodaBench scoring logs and the public corpus.

The CodaBench scorer prints opaque aliases such as ``subject_120099`` and
``benchmark_869474``.  This script deliberately performs literal matching only;
it does not infer a private-to-public identity mapping.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROW = re.compile(r"^\s*(subject_\S+)\s+(benchmark_\S+)\s+\d+\s+")


def parse_log(path: Path) -> set[tuple[str, str]]:
    """Return the unique subject--benchmark pairs printed by a scoring log."""
    pairs: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW.match(line)
        if match:
            pairs.add((match.group(1), match.group(2)))
    return pairs


def status(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logs = sorted(args.results_root.glob("codabench-*/scoring_stdout.txt"))
    by_submission: dict[str, set[tuple[str, str]]] = {}
    pair_sources: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for path in logs:
        submission_id = path.parent.name.removeprefix("codabench-")
        pairs = parse_log(path)
        if not pairs:
            continue
        by_submission[submission_id] = pairs
        for pair in pairs:
            pair_sources[pair].append(submission_id)

    public = pd.read_parquet(args.public_tasks, columns=["subject_uid", "benchmark_id"])
    public_subjects = set(public["subject_uid"].astype(str))
    public_benchmarks = set(public["benchmark_id"].astype(str))
    public_pairs = set(map(tuple, public[["subject_uid", "benchmark_id"]].astype(str).to_numpy()))

    hidden_pairs = sorted(pair_sources)
    hidden_subjects = sorted({subject for subject, _ in hidden_pairs})
    hidden_benchmarks = sorted({benchmark for _, benchmark in hidden_pairs})

    lines = [
        "# CodaBench ID overlap audit",
        "",
        "## Scope",
        "",
        "This report extracts the subject and benchmark aliases printed in every "
        "available `results/codabench-*/scoring_stdout.txt` file, then performs "
        "literal string matching against the public task table used by the "
        "non-benchmark-held-out experiments.",
        "",
        "**Important limitation:** CodaBench uses opaque numeric aliases, while "
        "the public table uses hashed subject UIDs and named benchmark IDs. A "
        "literal non-match is therefore evidence of different identifier "
        "namespaces, not evidence that the underlying subject or benchmark is "
        "unseen in the public corpus.",
        "",
        "## Summary",
        "",
        f"- Scoring logs parsed: {len(by_submission)} of {len(logs)} available logs.",
        f"- Unique CodaBench subject aliases: {len(hidden_subjects)}.",
        f"- Unique CodaBench benchmark aliases: {len(hidden_benchmarks)}.",
        f"- Unique CodaBench subject--benchmark pairs: {len(hidden_pairs)}.",
        f"- Literal subject matches in public tasks: {sum(subject in public_subjects for subject in hidden_subjects)}.",
        f"- Literal benchmark matches in public tasks: {sum(benchmark in public_benchmarks for benchmark in hidden_benchmarks)}.",
        f"- Literal pair matches in public tasks: {sum(pair in public_pairs for pair in hidden_pairs)}.",
        "",
        "No identity-level overlap conclusion is possible without an organizer "
        "mapping from opaque CodaBench aliases to public IDs.",
        "",
        "## Logs parsed",
        "",
        "| Submission | Unique pairs |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {submission_id} | {len(pairs)} |" for submission_id, pairs in by_submission.items())

    lines.extend([
        "",
        "## Unique subject aliases",
        "",
        "| CodaBench subject alias | Literal match in public task table |",
        "| --- | --- |",
    ])
    lines.extend(f"| {subject} | {status(subject in public_subjects)} |" for subject in hidden_subjects)

    lines.extend([
        "",
        "## Unique benchmark aliases",
        "",
        "| CodaBench benchmark alias | Literal match in public task table |",
        "| --- | --- |",
    ])
    lines.extend(f"| {benchmark} | {status(benchmark in public_benchmarks)} |" for benchmark in hidden_benchmarks)

    lines.extend([
        "",
        "## Unique subject--benchmark pairs",
        "",
        "| CodaBench subject alias | CodaBench benchmark alias | Seen in submissions | Literal pair match in public task table |",
        "| --- | --- | --- | --- |",
    ])
    lines.extend(
        f"| {subject} | {benchmark} | {', '.join(pair_sources[(subject, benchmark)])} | "
        f"{status((subject, benchmark) in public_pairs)} |"
        for subject, benchmark in hidden_pairs
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
