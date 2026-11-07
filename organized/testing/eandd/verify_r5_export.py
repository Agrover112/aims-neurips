"""Check original-fit provenance, training split and exact R5 panel coverage."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from reviewer_probes import population, wp7_sample, export_keys, score


def verify(output, panel, fit, responses):
    metadata = json.loads((output / "MANIFEST.json").read_text())
    criterion = json.loads((output / "wp13-probe-criterion.json").read_text())
    fit_sha = hashlib.sha256(fit.read_bytes()).hexdigest()
    assert metadata["provenance"]["rasch_fit_sha256"] == fit_sha
    assert criterion["provenance"]["rasch_fit_sha256"] == fit_sha
    original = pd.read_parquet(fit)
    assert int(original.kind.eq("b").sum()) == 109987
    hyper = original.loc[original.kind.eq("hyper")].set_index("key").value
    assert abs(float(hyper["tau_i"]) - 1.4913057945180008) < 1e-9
    criterion["provenance"]["original_fit_confirmed"] = True
    (output / "wp13-probe-criterion.json").write_text(json.dumps(criterion, indent=2))
    expected = pd.read_parquet(panel, columns=["benchmark_id", "content_hash"]).drop_duplicates()
    expected_keys = set(map(tuple, expected.to_numpy()))
    train = pd.read_parquet(output / "probe-training-items.parquet")
    source = pd.read_parquet(responses, columns=["benchmark_id", "item_uid", "benchmark_split"]).drop_duplicates()
    joined = train.merge(source, on=["benchmark_id", "item_uid"], how="left", validate="one_to_one")
    assert joined.benchmark_split.eq("train").all() and joined.benchmark_id.nunique() == 11
    assert not set(joined.benchmark_id) & set(expected.benchmark_id)
    assert "Rasch difficulty" in metadata["sign"]
    t = population(responses)
    _, pair_rng = wp7_sample(t)
    eval_items = t.loc[t.benchmark_split.ne("train")].copy()
    item_keys = export_keys(responses, panel).drop(columns="benchmark_split").drop_duplicates()
    for row in criterion["criterion"]:
        arm = row["arm"]
        if not (arm.endswith("_mlp") and row["passes"]):
            continue
        assert arm in metadata["exports"], arm
        record = metadata["exports"][arm]
        path = output / record["file"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        values = pd.read_parquet(path)
        assert list(values) == ["benchmark_id", "content_hash", "difficulty"]
        assert not values.duplicated(["benchmark_id", "content_hash"]).any()
        assert set(map(tuple, values[["benchmark_id", "content_hash"]].to_numpy())) == expected_keys
        assert np.isfinite(values.difficulty).all()
        record["coverage"] = {"expected_keys": len(expected), "exported_keys": len(values),
                              "missing": 0, "extra": 0}
        signed = eval_items.merge(item_keys, on=["benchmark_id", "item_uid"],
                                  how="left", validate="one_to_one").merge(
                                      values, on=["benchmark_id", "content_hash"],
                                      how="left", validate="many_to_one")
        assert signed.difficulty.notna().all()
        signed["exported_easiness"] = -signed.difficulty
        ranked = score(signed, "exported_easiness", copy.deepcopy(pair_rng))
        exported_rho = float(ranked.rho.mean())
        exported_positive = int(ranked.rho.gt(0).sum())
        assert exported_rho - row["random_rho"] >= .05 and exported_positive >= 4
        record["exported_sign_rank_check"] = {"macro_rho": exported_rho,
                                             "gap_to_random": exported_rho-row["random_rho"],
                                             "positive_benchmarks": exported_positive}
    with responses.open("rb") as stream:
        response_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata["verification"] = {"panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
                                "responses_sha256": response_sha,
                                "training_items": len(train), "training_benchmarks": 11,
                                "validation_or_test_items_used_for_fit": 0,
                                "python": platform.python_version(),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    (output / "MANIFEST.json").write_text(json.dumps(metadata, indent=2))
    md = ["# R5 original-fit item difficulty handoff", "", metadata["sign"], "",
          metadata["training"], "", "Each exported file exactly covers every unique benchmark/content_hash key "
          "in Person A's validation/test replay panel, including acquisition and evaluation items.", "",
          f"Original trimmed fit SHA-256: {fit_sha}", "",
          "The pass-rate arm is mapped from a centered probability correction to the logit scale using only "
          "the training-derived baseline; the Rasch-target arm negates the predicted easiness. "
          "Positive values mean harder items in both files.", "",
          "| Arm | Spearman | Random | Gap | Positive benchmarks | Pass | File |",
          "|---|---:|---:|---:|---:|---|---|"]
    for row in criterion["criterion"]:
        arm = row["arm"]
        filename = metadata["exports"].get(arm, {}).get("file", "No export")
        md.append(f"| {arm} | {row['macro_rho']:.6f} | {row['random_rho']:.6f} | "
                  f"{row['gap']:.6f} | {row['positive']}/6 | {row['passes']} | {filename} |")
    (output / "MANIFEST.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "panel", "fit", "responses"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    verify(a.output, a.panel, a.fit, a.responses)
