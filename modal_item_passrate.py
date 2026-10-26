"""Raw per-item pass rate, ignoring every split we built.

Reads ``processed/observations.parquet`` (one row per recorded response, binary
``label``) and reduces it to one row per item: how many responses, from how
many distinct subjects, and the fraction that were correct. Nothing here uses
train/validation/test, the subject-benchmark task split, or the 128/500 item
caps -- it is the data in its raw form.

Each item also gets a language: an explicit ``lang=``/``language=`` tag in
``item_features`` where present, otherwise the dominant non-Latin script of
``content``, otherwise ``English/Latin``. That is a heuristic (Latin-script
non-English text without a tag is labelled English/Latin) and is recorded as
such in the output.

Run:  modal run --detach modal_item_passrate.py
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

APP_NAME = "aims-neurips-item-passrate"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
OUT = Path("/data/results/data-analysis/item-passrate.parquet")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")

LANG_TAG = re.compile(r"(?:^|;)\s*(?:lang|language|lang_code|locale)=([^;]*)")
NAMES = {"kin": "Kinyarwanda", "ibo": "Igbo", "yor": "Yoruba", "swa": "Swahili",
         "pcm": "Nigerian Pidgin", "hau": "Hausa", "lug": "Luganda", "wol": "Wolof",
         "sna": "Shona", "nya": "Chichewa", "zul": "Zulu", "xho": "Xhosa", "mos": "Moore",
         "bam": "Bambara", "fon": "Fon", "twi": "Twi", "ewe": "Ewe", "tsn": "Tswana",
         "bbj": "Ghomala", "amh": "Amharic", "luo": "Luo", "eu": "Basque", "en": "English"}
RANGES = [("Chinese", 0x4E00, 0x9FFF), ("Chinese", 0x3400, 0x4DBF),
          ("Japanese", 0x3040, 0x30FF), ("Korean", 0xAC00, 0xD7AF),
          ("Russian", 0x0400, 0x04FF), ("Arabic", 0x0600, 0x06FF),
          ("Hindi", 0x0900, 0x097F), ("Hebrew", 0x0590, 0x05FF), ("Thai", 0x0E00, 0x0E7F),
          ("Greek", 0x0370, 0x03FF), ("Amharic", 0x1200, 0x137F)]


def _language(feature: str, text: str) -> str:
    m = LANG_TAG.search(feature or "")
    if m and m.group(1).strip():
        code = m.group(1).strip()[:12]
        return NAMES.get(code, code)
    counts: dict[str, int] = {}
    for ch in (text or "")[:1500]:
        o = ord(ch)
        if o < 0x80:
            continue
        for name, lo, hi in RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if counts:
        best = max(counts, key=counts.get)
        if counts[best] >= 8:
            return best
    return "English/Latin"


@app.function(image=image, cpu=8, memory=131_072, volumes={"/data": volume},
              timeout=4 * 60 * 60)
def run() -> dict:
    import pandas as pd

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["item_uid", "subject_uid", "benchmark_id", "label"])
    print(f"observations: {len(obs):,}", flush=True)
    g = obs.groupby("item_uid", sort=False)
    table = pd.DataFrame({
        "n_responses": g["label"].size(),
        "n_subjects": g["subject_uid"].nunique(),
        "pass_rate": g["label"].mean(),
        "benchmark_id": g["benchmark_id"].first(),
    }).reset_index()
    del obs
    print(f"items with responses: {len(table):,}", flush=True)

    items = pd.read_parquet(PROCESSED / "items.parquet",
                            columns=["item_uid", "content", "item_features", "content_chars"])
    items["item_uid"] = items["item_uid"].astype(str)
    table["item_uid"] = table["item_uid"].astype(str)
    table = table.merge(items, on="item_uid", how="left")
    table["language"] = [_language(f, c) for f, c in
                         zip(table["item_features"].fillna(""), table["content"].fillna(""),
                             strict=True)]
    table = table.drop(columns=["content", "item_features"])
    table["pass_rate"] = table["pass_rate"].astype("float32")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    volume.commit()
    return {"items": len(table),
            "mean_pass_rate": float(table["pass_rate"].mean()),
            "single_response_items": int((table["n_responses"] == 1).sum())}


@app.local_entrypoint()
def main() -> None:
    print(run.remote())
