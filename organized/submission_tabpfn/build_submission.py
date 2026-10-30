"""Build the TabPFN-v2 CodaBench archive."""

from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_tabpfn"
NP = ROOT / "submission" / "aims_predictor"
PCA = Path("/tmp/tabpfn-assets/pca.npz")
OUT = ROOT / "dist" / "tabpfn-v2-bge-m3-pca500-random.zip"


def main():
    if not PCA.exists():
        raise FileNotFoundError("build PCA asset first: /tmp/tabpfn-assets/pca.npz")
    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tabpfn-submission-") as tmp:
        staged = Path(tmp)
        for name in ("model.py", "requirements.txt"):
            shutil.copy2(SOURCE / name, staged / name)
        package = staged / "aims_predictor"
        package.mkdir()
        shutil.copy2(SOURCE / "aims_predictor" / "__init__.py", package / "__init__.py")
        shutil.copy2(SOURCE / "aims_predictor" / "tabpfn_predictor.py", package / "tabpfn_predictor.py")
        # Reuse the already validated BGE-M3 embedding implementation.
        shutil.copy2(NP / "predictor.py", package / "predictor.py")
        shutil.copy2(ROOT / "training" / "cnp_blocks.py", package / "cnp_blocks.py")
        shutil.copy2(ROOT / "training" / "anp_crossattn_blocks.py", package / "anp_crossattn_blocks.py")
        assets = staged / "assets"
        assets.mkdir()
        shutil.copy2(PCA, assets / "pca.npz")
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staged))
    print(OUT)


if __name__ == "__main__":
    main()
