"""
run_scenarios.py
Solves the full scenario battery and caches every schedule to
data/processed/results/<key>.pkl so analysis and figures never re-solve.

    python src/run_scenarios.py            # ~60-90 min on 2 cores
    python src/run_scenarios.py --only HW  # a single scenario
"""
from __future__ import annotations

import argparse
import pickle
import time

import pandas as pd

from config import DATA_PROC
from optimize import Scenario, prepare_zones, solve, status_quo, proportional, priority

RES = DATA_PROC / "results"


def zones(tier="central", exposure="uniform"):
    return prepare_zones(pd.read_csv(DATA_PROC / "zones.csv", index_col=0), tier=tier, exposure=exposure)


def battery(z_c):
    """(key, scenario, tier, exposure). Frontier grid is centered on the HW gap."""
    base = [
        ("CM", Scenario("Cost-minimizing", a_H=0.0), "central", "uniform"),
        ("HW", Scenario("Health-welfare"), "central", "uniform"),
        ("HW_oneoff", Scenario("Health-welfare, no block programs", allow_block=False), "central", "uniform"),
        ("EW", Scenario("Equity-weighted (linear SVI)", a_E=1.0), "central", "uniform"),
        ("EWc", Scenario("Equity-weighted (convex SVI)", a_E=1.0, equity_weight="svi_convex"), "central", "uniform"),
        ("DW", Scenario("Distributionally weighted (eta = 1)", harm_weight="income"), "central", "uniform"),
        ("HW_r2", Scenario("Health-welfare, r = 2%", rate=0.02), "central", "uniform"),
        ("HW_r7", Scenario("Health-welfare, r = 7%", rate=0.07), "central", "uniform"),
        ("HW_child", Scenario("Health-welfare, child benefits only"), "child", "uniform"),
        ("HW_high", Scenario("Health-welfare, high CVD function"), "high", "uniform"),
        ("HW_huynh", Scenario("Health-welfare, tap-lead gradient"), "central", "huynh"),
    ]
    return base


def frontier_battery(g_star):
    eff = [(f"EPS_{i}", Scenario(f"eps {g_star - d:.2f}", eps_gap=round(g_star - d, 3)), "central", "uniform")
           for i, d in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))]
    dom = [(f"DOM_{i}", Scenario(f"gap >= {g:.2f}", gap_min=round(g, 3), ej_floor=False), "central", "uniform")
           for i, g in enumerate((g_star + 1.5, g_star + 3.0, 0.0, 2.0))]
    return eff + dom


def run(items, force=False):
    RES.mkdir(parents=True, exist_ok=True)
    for key, sc, tier, exposure in items:
        out = RES / f"{key}.pkl"
        if out.exists() and not force:
            print(f"cached  {key}", flush=True)
            continue
        z = zones(tier, exposure)
        t0 = time.time()
        r = solve(z, sc)
        r |= {"tier": tier, "exposure": exposure, "seconds": time.time() - t0}
        pickle.dump(r, open(out, "wb"))
        print(f"solved  {key:10s} {r['status']:10s} {r['seconds']:6.0f}s  gap={r['solver'].get('gap')}", flush=True)


def main(only=None, force=False):
    z = zones()
    RES.mkdir(parents=True, exist_ok=True)
    for key, r in (("SQ", status_quo(z)), ("SQ_block", status_quo(z, block=True)), ("PROP", proportional(z)), ("PRIO", priority(z))):
        pickle.dump(r | {"tier": "central", "exposure": "uniform"}, open(RES / f"{key}.pkl", "wb"))
    items = battery(z)
    if only:
        items = [i for i in items if i[0] in only]
    run(items, force)
    if not only:
        from optimize import evaluate
        hw = pickle.load(open(RES / "HW.pkl", "rb"))
        g_star = evaluate(hw, z)["q5_gap"]
        run(frontier_battery(g_star), force)
    print("all scenarios done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.only, a.force)
