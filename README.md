# Chicago LSLR Research

Manuscript, data pipeline and optimization code for *Algorithmic Equity and Capital Allocation:
A Computational Optimization Framework for Chicago's ~400,000 Lead Service Line Replacements*.

## Live explorer
**[chicago-lslr.vercel.app](https://chicago-lslr.vercel.app)**: an interactive, no-install view of the results.
Switch between the 14 cached schedules, map replacement timing by community area (or the observed
inputs: SVI, children per line, redlining, income, blood-lead screening), and explore the
efficiency-equity frontier, per-line benefit-cost ratios, financing and the probabilistic
sensitivity analysis. The page is static (`web/`); `web/export.py` turns the cached pipeline
outputs in `data/processed/` into `web/data/data.json` without re-solving anything.

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
web/                   static explorer (index.html, app.js); export.py writes web/data/*.json
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
To refresh the explorer after re-running the pipeline:
```
python web/export.py --geojson <City of Chicago community-area boundaries GeoJSON>   # data portal dataset igwz-8jzy
cd web && vercel --prod
```
The manuscript needs Times New Roman (installed on macOS) and TeX Gyre Termes Math (included in TeX Live).
