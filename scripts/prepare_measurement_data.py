#!/usr/bin/env python3
"""Prepare downloaded Measurement Data Bank tables for model experiments."""

from __future__ import annotations

import argparse
import json

from aims_starter.data import prepare_measurement_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Root containing per-benchmark tables")
    parser.add_argument("--out", required=True, help="New or empty output directory")
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    prepared = prepare_measurement_data(args.source, args.out, seed=args.seed)
    print(json.dumps(prepared.manifest["rows"], indent=2, sort_keys=True))
    print(f"Prepared data: {prepared.output}")


if __name__ == "__main__":
    main()
