# Chicago LSLR Research

Manuscript, data pipeline and optimization code for *Algorithmic Equity and Capital Allocation:
A Computational Optimization Framework for Chicago's ~400,000 Lead Service Line Replacements*.

## What is real and what is modeled
Observed public data: the 2025 CDWM service line inventory (491,705 records, geocoded by Inside
Climate News / Grist / WBEZ), CDC/ATSDR SVI 2022, ACS 2019-2023, HOLC redlining grades, and CDPH
blood-lead screening (1999-2013). Valuation follows EPA's 2024 LCRI Economic Analysis. Modeled
assumptions (dose increments, unit and mobilization costs, budgets, crews) are listed in
`src/config.py` and varied jointly in the probabilistic sensitivity analysis.

## Layout
```
setup.sh               Section 0: builds folders, virtual environment, installs requirements
src/fetch_data.py      downloads all raw inputs from their sources (stdlib only), writes checksums
src/etl_pipeline.py    inventory + ACS + SVI + HOLC + CDPH -> lines, tracts, zones; imputation with CV
src/config.py          every parameter and its source
src/welfare.py         EPA dose-response and valuation, NPV, financing, consent model, inequality metrics
src/optimize.py        MOMILP (PuLP/CBC), heuristic baselines, common evaluator
src/run_scenarios.py   solves the scenario battery; caches schedules in data/processed/results/
src/analysis.py        tables, decomposition, frontier, PSA, consent econometrics, validation, LaTeX macros
src/figures.py         all figures (Times New Roman, hatched series, keyed)
src/run_all.py         runs everything end to end
paper/main.tex         manuscript (XeLaTeX); results.tex and tab_*.tex are generated
bib/references.bib     45 references, journal DOIs verified against Crossref
```

## Reproduce
```
bash setup.sh
source .venv/bin/activate
python src/run_all.py            # about 1-1.5 hours, mostly MILP solves
cd paper && latexmk -xelatex main.tex
```
`python src/run_all.py --skip-solve` regenerates analysis and figures from cached schedules in minutes.
The manuscript needs Times New Roman (installed on macOS) and TeX Gyre Termes Math (included in TeX Live).
