#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output outputs
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
for file in lecture lab answers; do python build_pdf.py "$file.md"; done
