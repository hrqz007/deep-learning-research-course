#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" -m unittest -v test_experiment.py
"$PYTHON" -O -m unittest -v test_experiment.py
"$PYTHON" experiment.py
"$PYTHON" make_figures.py
"$PYTHON" create_notebook.py
"$PYTHON" execute_notebook.py experiment.ipynb
for document in lecture lab answers; do
  "$PYTHON" build_pdf.py "$document.md"
done
