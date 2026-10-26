"""Exhaustive field audit of the processed corpus.

For every column of ``items``, ``subjects``, and ``benchmarks``: how often is it
populated, how many distinct values does it take, and -- the question that
decides whether a field can act as a bridge between benchmarks -- how many of
its values appear under more than one benchmark.

A field is only useful for cross-benchmark transfer if it is both populated and
shared. High fill with no sharing (``harness``) is a benchmark proxy. High fill
with near-unique values (``subject_features_extra``) is a fingerprint. Both look
informative and are not.

Run:

    modal run modal_field_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-field-audit"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROCESSED_ROOT = REMOTE_ROOT / "datasets" / "full" / "processed"
OUTPUT_ROOT = REMOTE_ROOT / "results" / "feature-diagnostics"

MAX_EXAMPLES = 4
EXAMPLE_CHARS = 70

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _audit_frame(frame, name: str, benchmark_column: str | None) -> list[dict[str, Any]]:
    import pandas as pd

    rows = []
    total = len(frame)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            filled_mask = series.notna()
            text = series.astype(str)
        else:
            text = series.fillna("").astype(str).str.strip()
            filled_mask = text != ""

        filled = int(filled_mask.sum())
        populated = text[filled_mask]
        distinct = int(populated.nunique()) if filled else 0

        entry: dict[str, Any] = {
            "table": name,
            "column": column,
            "dtype": str(series.dtype),
            "rows": total,
            "filled": filled,
            "fill_rate": filled / total if total else 0.0,
            "distinct": distinct,
            "uniqueness": distinct / filled if filled else 0.0,
            "examples": [
                value[:EXAMPLE_CHARS] for value in populated.drop_duplicates().head(MAX_EXAMPLES)
            ],
        }

        if benchmark_column and benchmark_column in frame.columns and filled:
            sub = frame.loc[filled_mask, [benchmark_column]].copy()
            sub["value"] = populated.to_numpy()
            spans = sub.groupby("value")[benchmark_column].nunique()
            entry["values_spanning_benchmarks"] = int((spans > 1).sum())
            entry["max_benchmarks_for_one_value"] = int(spans.max())
            entry["rows_bridged"] = int(sub["value"].map(spans).gt(1).sum())
            entry["bridge_coverage"] = entry["rows_bridged"] / total if total else 0.0
        rows.append(entry)
    return rows


@app.function(
    image=image,
    cpu=8,
    memory=131_072,
    volumes={str(REMOTE_ROOT): volume},
    timeout=60 * 60,
)
def audit() -> dict[str, Any]:
    import pandas as pd

    report: dict[str, Any] = {"status": "field_audit", "tables": {}}
    audits: list[dict[str, Any]] = []

    for name, filename, benchmark_column in (
        ("benchmarks", "benchmarks.parquet", "benchmark_id"),
        ("subjects", "subjects.parquet", "benchmark_id"),
        ("items", "items.parquet", "benchmark_id"),
    ):
        path = PROCESSED_ROOT / filename
        if not path.exists():
            print(f"missing {path}", flush=True)
            continue
        frame = pd.read_parquet(path)
        print(f"{name}: {len(frame):,} rows x {len(frame.columns)} columns", flush=True)
        report["tables"][name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
        audits.extend(_audit_frame(frame, name, benchmark_column))
        if name == "benchmarks":
            report["benchmark_sample"] = frame.head(3).astype(str).to_dict(orient="records")
        del frame

    report["fields"] = audits
    _write_json(OUTPUT_ROOT / "field-audit.json", report)
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    result = audit.remote()
    print(f"\n{'table':<12}{'column':<28}{'fill':>7}{'distinct':>10}{'uniq':>7}{'bridges':>9}")
    print("-" * 75)
    for row in result["fields"]:
        print(
            f"{row['table']:<12}{row['column']:<28}{row['fill_rate']:>6.0%}"
            f"{row['distinct']:>10}{row['uniqueness']:>7.2f}"
            f"{row.get('bridge_coverage', float('nan')):>9.0%}"
        )
