#!/usr/bin/env bash
# setup.sh  (Section 0)  Builds the Chicago LSLR research environment.
# Usage:  bash setup.sh          Safe to re-run; never overwrites existing files.
set -euo pipefail

ROOT="$HOME/Downloads/Chicago_LSLR_Research"
mkdir -p "$ROOT"/{data/raw,data/processed/results,src,paper,figures,bib}
cd "$ROOT"

[ -f requirements.txt ] || cat > requirements.txt <<'EOF'
numpy>=1.26
pandas>=2.1
scipy>=1.11
geopandas>=0.14
shapely>=2.0
pyarrow>=14.0
openpyxl>=3.1
scikit-learn>=1.3
statsmodels>=0.14
pulp==2.9.0
matplotlib>=3.8
EOF

[ -f bib/references.bib ] || printf '%% references.bib\n' > bib/references.bib

[ -f paper/main.tex ] || cat > paper/main.tex <<'EOF'
\documentclass[11pt]{article}
\usepackage{fontspec}\setmainfont{Times New Roman}
\usepackage[authoryear,round]{natbib}
\begin{document}
\title{Algorithmic Equity and Capital Allocation}\maketitle
\bibliographystyle{plainnat}\bibliography{../bib/references}
\end{document}
EOF

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 src/fetch_data.py || echo "Run 'python3 src/fetch_data.py' once you are online."

echo "Environment ready at $ROOT"
echo "Run:   source .venv/bin/activate && python src/run_all.py"
echo "Build: cd paper && latexmk -xelatex main.tex   (requires Times New Roman and TeX Gyre Termes Math)"
