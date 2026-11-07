# Reviewing copy (`reviewing.pdf`)

`reviewing.pdf` is the paper with review marks: light red = text to delete or rewrite, light green = text or analyses to add, green boxes = open questions, red boxes = still to improve. Accepted edits show as plain text.

## Where to edit
- **The paper itself:** `paper/neurips_2026/eandd.tex` (main text, appendix) and `paper/eandd/person-b-sections.tex` (Person-B's sections: Data and evaluation protocol, Probes, Item features, Limitations, Disclosure). Rebuild with `paper/build-eandd.sh`, which regenerates the tables first.
- **The review marks:** `paper/reviewing/make_reviewing.py`. It reads the current `eandd.tex`, wraps exact sentences in red/green boxes and inserts the notes, then writes `paper/neurips_2026/eandd_reviewing.tex` (generated; do not edit by hand).

## Rebuild `reviewing.pdf`
```bash
cd paper/neurips_2026
python3 ../reviewing/make_reviewing.py
../../.tools/tectonic/tectonic -X compile eandd_reviewing.tex --outdir ../build
cp ../build/eandd_reviewing.pdf ../reviewing/reviewing.pdf
```
If a sentence in `eandd.tex` that the script wraps was reworded, the script stops with an assertion naming the sentence; update the string in `make_reviewing.py` (or delete that mark once it is accepted).
