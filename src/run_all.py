"""
run_all.py
End-to-end driver.

    python src/run_all.py                 # fetch (if needed) -> ETL -> solve -> analyze -> figures
    python src/run_all.py --skip-solve    # reuse cached schedules in data/processed/results/
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")


def main(skip_solve: bool = False, skip_etl: bool = False):
    from config import DATA_RAW, DATA_PROC
    if not (DATA_RAW / "MANIFEST.txt").exists():
        import fetch_data
        fetch_data.main()
    if not skip_etl or not (DATA_PROC / "zones.csv").exists():
        import etl_pipeline
        etl_pipeline.build_tables()
    if not skip_solve:
        import run_scenarios
        run_scenarios.main()
    import analysis
    import figures
    T, ev, dec, F, D, TOR, cons, refc, val, npv, fin, hm, R = analysis.main()
    figures.make_all(R, ev, T, F, D, TOR, cons, refc, val, npv)
    print(T[["pv_harm_bn", "pv_cost_bn", "pv_social_bn", "completion_year", "q5_gap"]].round(2))
    print("Decomposition ($bn):", {k: round(v, 2) for k, v in dec.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--skip-etl", action="store_true")
    a = ap.parse_args()
    main(a.skip_solve, a.skip_etl)
