#!/usr/bin/env bash
# Rebuild the E&D paper: regenerate the tables from the committed CSVs, compile, save the next numbered draft.
# usage: paper/build-eandd.sh ["short note about this version"]
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python paper/eandd/assemble.py
mkdir -p paper/build paper/eandd-drafts
(cd paper/neurips_2026 && ../../.tools/tectonic/tectonic -X compile eandd.tex --outdir ../build >/dev/null)
n=$(find paper/eandd-drafts -name "eandd-v*.pdf" | wc -l); n=$(printf "%02d" $((n + 1)))
cp paper/build/eandd.pdf "paper/eandd-drafts/eandd-v$n.pdf"
echo "$(date -Iseconds)  v$n  ${1:-}" >> paper/eandd-drafts/CHANGELOG.txt
echo "wrote paper/eandd-drafts/eandd-v$n.pdf"
