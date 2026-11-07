"""Compare white and light-grey panels on Figure 3 without changing defaults."""
import argparse
from pathlib import Path

import pandas as pd

from wp9 import per_benchmark_plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-benchmark", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(a.per_benchmark)
    rows = rows.loc[rows.run.isin(["wp5-competition-val", "wp5-competition-test"])]
    for theme in ("white", "light-grey"):
        per_benchmark_plot(rows, a.output / f"figure3-{theme}", panel_theme=theme)


if __name__ == "__main__":
    main()
