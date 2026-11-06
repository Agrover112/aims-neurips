"""Build the ID handoff from the canonical-filter cache in arshiaswe."""
from pathlib import Path
import json
import sys

import modal

app = modal.App("eandd-id-panel")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
source = "/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1/filtered.parquet"
root = Path("/data/results/eandd-analysis/id-v1")
image = (modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "pyarrow")
         .add_local_file(Path(__file__).with_name("build_id_panel.py"), "/root/build_id_panel.py"))


@app.function(image=image, cpu=2, memory=8192, timeout=900, volumes={"/data": volume})
def build(cap_per_benchmark=20):
    import build_id_panel
    import pyarrow.parquet as pq

    volume.reload()
    audit = json.loads(Path(source).with_name("split-audit.json").read_text())
    prepared = json.loads(Path(source).with_name("prepared.json").read_text())
    if prepared["identity"]["split_seed"] != 20261018 or prepared["identity"]["filter_sha256"] != "bc6b06fc21899430c9ae8ad7b92b249200d2d698f0cf364393077e98c9c0e4f9":
        raise ValueError("Filtered cache does not match the canonical recipe")
    if not 6_335_000 <= pq.ParquetFile(source).metadata.num_rows < 6_345_000:
        raise ValueError("Source count does not match the plan's rounded 6.34 million")
    for output, extra in ((root / "subjects", ["--subjects-only"]),
                          (root / f"panel-cap{cap_per_benchmark}", ["--cap-per-benchmark", str(cap_per_benchmark)])):
        if not (output / "manifest.json").exists():
            sys.argv = ["build_id_panel", "--responses", source, "--output", str(output), *extra]
            build_id_panel.main()
        volume.commit()
        print(output, (output / "manifest.json").read_text(), flush=True)
    return {"volume_path": str(root), "split": audit["benchmark_split_map"]}


@app.local_entrypoint()
def main(cap_per_benchmark: int = 20):
    print(build.spawn(cap_per_benchmark).object_id)
