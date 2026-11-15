"""
web/export.py
Turns the pipeline's cached outputs into compact JSON for the static explorer.
Nothing is re-solved and no new values are modeled: scenario metrics come from
data/processed/scenario_results.csv, schedules from data/processed/results/*.pkl
(scored with optimize.evaluate, the same evaluator the paper uses), and the
remaining views from the CSV/JSON files written by src/analysis.py.

The only new computation is aggregation: zone schedules are rolled up to the
77 community areas by weighting each zone by the action-required lines it
contributes to each area (line -> census tract -> community area).

    python web/export.py --geojson path/to/city_community_areas.geojson

--geojson is the City of Chicago "Boundaries - Community Areas" export
(data.cityofchicago.org, dataset igwz-8jzy). It is simplified and written to
web/data/community_areas.geojson. Without it, only data.json is rewritten.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import COST, FIN, INV, HEALTH, OPS, DATA_PROC  # noqa: E402
from optimize import prepare_zones, evaluate, budget_path  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
RES = DATA_PROC / "results"

SCEN = {  # key: (label, group, plain-language description)
    "SQ": ("Status quo", "baseline",
           "Keeps replacing lines where the city replaced them in 2021-2025 (empirical pattern), priced one line at a time."),
    "SQ_block": ("Status quo, block-priced", "baseline",
                 "Same geography as the status quo, but work is bundled into block programs where batches allow."),
    "PROP": ("Proportional pace", "baseline",
             "Every corridor advances at the same pace, with block pricing."),
    "PRIO": ("Harm priority, one-off", "baseline",
             "Greedy: replace the highest-harm corridors first, one line at a time (no block programs)."),
    "CM": ("Cost-minimizing", "optimized",
           "MILP that minimizes present-value cost only; health harm is ignored."),
    "HW": ("Health-welfare", "optimized",
           "MILP that minimizes health harm + cost (EPA LCRI valuation). The paper's main schedule."),
    "EW": ("Equity-weighted (linear SVI)", "optimized",
           "Health-welfare objective plus a penalty on waiting, weighted by social vulnerability."),
    "EWc": ("Equity-weighted (convex SVI)", "optimized",
            "As above, but the vulnerability weight rises steeply (SVI cubed) for the most vulnerable areas."),
    "DW": ("Distributional weights", "optimized",
           "Health harm is up-weighted in lower-income corridors (inequality aversion eta = 1)."),
    "HW_r2": ("Health-welfare, r = 2%", "robustness", "Health-welfare schedule solved with a 2% discount rate."),
    "HW_r7": ("Health-welfare, r = 7%", "robustness", "Health-welfare schedule solved with a 7% discount rate."),
    "HW_child": ("Health-welfare, child benefits only", "robustness",
                 "Only children's IQ losses are valued; adult cardiovascular deaths are left out."),
    "HW_high": ("Health-welfare, high CVD function", "robustness",
                "Adult cardiovascular mortality valued with the steeper Lanphear et al. (2018) function."),
    "HW_huynh": ("Health-welfare, tap-lead gradient", "robustness",
                 "Exposure per line varies with the tap-lead gradient reported by Huynh et al. (2024)."),
}


def r(x, d=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), d)


def rl(a, d=4):
    return [r(v, d) for v in a]


def main(geojson: str | None):
    OUT.mkdir(parents=True, exist_ok=True)
    zc = prepare_zones(pd.read_csv(DATA_PROC / "zones.csv", index_col=0))
    T = pd.read_csv(DATA_PROC / "scenario_results.csv", index_col=0)

    # ---- line -> zone -> community area weights (action-required lines only)
    lines = pd.read_parquet(DATA_PROC / "lines.parquet", columns=["geoid", "action", "zone"])
    lines = lines[lines.action]
    tr = pd.read_sql("select geoid, area_num_1 from tracts", sqlite3.connect(DATA_PROC / "tracts.gpkg"))
    lines = lines.merge(tr, on="geoid", how="left")
    unmapped = int(lines.area_num_1.isna().sum())
    lines = lines.dropna(subset=["area_num_1"])
    lines["zone"] = lines["zone"].astype(int)
    Wt = lines.groupby(["area_num_1", "zone"]).size().unstack(fill_value=0)      # CA x zone line counts
    Wt = Wt.reindex(columns=zc.index, fill_value=0)
    Wn = Wt.div(Wt.sum(axis=1), axis=0)

    ca = pd.read_csv(DATA_PROC / "community_areas.csv")
    ca["area_num_1"] = ca["area_num_1"].astype(str)
    ca = ca.set_index("area_num_1")
    # modeled damage of one unreplaced line, line-weighted over the zones in each area
    h_ca = Wn @ zc["h"]

    # ---- scenarios
    n_years = max(pickle.load(open(RES / f"{k}.pkl", "rb"))["X"].shape[1]
                  for k in SCEN if (RES / f"{k}.pkl").exists())
    years = list(range(INV.base_year, INV.base_year + n_years))
    scen, ca_sched = {}, {}
    for key, (label, group, desc) in SCEN.items():
        p = RES / f"{key}.pkl"
        if not p.exists() or key not in T.index:
            continue
        res = pickle.load(open(p, "rb"))
        e = evaluate(res, zc, r=FIN.base_rate)
        X = res["X"].reindex(zc.index).fillna(0.0)
        cum = X.cumsum(axis=1).clip(upper=1.0)
        cum = cum.reindex(columns=range(1, len(years) + 1)).ffill(axis=1).fillna(1.0)
        ca_cum = Wn @ cum                                        # share of the area's lines done by end of year t
        ca_W = Wn @ pd.Series(e["W"], index=zc.index)             # expected wait (years from 2026)
        row = T.loc[key]
        scen[key] = {
            "label": label, "group": group, "desc": desc,
            "metrics": {k: r(row[k]) for k in ["pv_harm_bn", "pv_cost_bn", "pv_social_bn", "nominal_cost_bn",
                                               "completion_year", "mean_wait", "q5_gap", "wait_ej", "wait_nonej",
                                               "child_exposure_m", "gini_exposure", "conc_index_svi", "block_share",
                                               "mobilizations"]},
            "lines_year": rl(e["lines_year"], 0), "cost_year_musd": rl(e["cost_year"] / 1e6, 2),
        }
        ca_sched[key] = {"year": {a: r(INV.base_year - 1 + v, 2) for a, v in ca_W.items()},
                         "cum": {a: rl(ca_cum.loc[a].values, 3) for a in ca_cum.index}}

    cas = []
    for a, c in ca.iterrows():
        cas.append({"id": a, "name": c["community"], "n_action": int(c["n_action"]),
                    "lines_total": int(c["lines_total"]), "action_share": r(c["n_action"] / c["lines_total"]),
                    "svi": r(c["svi"]), "kids_per_line": r(c["kids_per_line"]), "holc_D": r(c["holc_D"]),
                    "median_income": r(c["median_income"], 0), "bll_2009_13": r(c["bll_elev_2009_13"], 3),
                    "bll_1999_13": r(c["bll_elev_1999_13"], 3),
                    "screened_2009_13": None if pd.isna(c["screened_2009_13"]) else int(c["screened_2009_13"]),
                    "harm_per_line": r(h_ca.get(a), 0)})

    F = pd.read_csv(DATA_PROC / "frontier.csv")
    frontier = [{"key": f.key, "branch": f.branch, "q5_gap": r(f.q5_gap), "pv_social_bn": r(f.pv_social_bn)}
                for f in F.itertuples()]

    fin = pd.read_csv(DATA_PROC / "financing.csv").round(2).to_dict("records")
    npv = pd.read_csv(DATA_PROC / "npv_sensitivity.csv").round(4).to_dict("records")
    tor = pd.read_csv(DATA_PROC / "tornado.csv")
    tor["param"] = tor["param"].str.replace(r"\\\$", "$", regex=True).str.replace(r"\\%", "%", regex=True) \
        .str.replace(r"\$\\beta\$", "β", regex=True)
    tor = tor.round(4).to_dict("records")

    D = pd.read_csv(DATA_PROC / "psa_draws.csv")
    sav_sq, sav_cm = D["SQ_social"] - D["HW_social"], D["CM_social"] - D["HW_social"]
    psa = {"n": int(len(D)),
           "save_vs_sq": rl(sav_sq.values, 3), "save_vs_cm": rl(sav_cm.values, 3),
           "bcr_31k": {f"{rt:.2f}": rl(np.sort(D.loc[D.rate == rt, "bcr_31k"].values), 3)
                       for rt in sorted(D.rate.unique())},
           "hw_beats_sq": r((D["HW_social"] < D["SQ_social"]).mean()),
           "hw_best": r((D["best"] == "HW").mean()),
           "bcr_below_one": r((D["bcr_31k"] < 1).mean()),
           "save_q": {q: r(sav_sq.quantile(q), 3) for q in (0.05, 0.5, 0.95)},
           "bcr_q": {q: r(D["bcr_31k"].quantile(q), 3) for q in (0.05, 0.5, 0.95)}}

    aux = json.load(open(DATA_PROC / "aux_results.json"))
    n_all = len(pd.read_parquet(DATA_PROC / "lines.parquet", columns=["action"]))

    data = {
        "meta": {"years": years, "base_year": INV.base_year, "horizon": INV.horizon,
                 "n_lines": n_all, "n_action": int(zc["n_lsl"].sum()), "n_zones": int(len(zc)),
                 "n_ca_unmapped_lines": unmapped, "total_budget_bn": r(budget_path(INV.horizon).sum() / 1e9, 2),
                 "budget_musd": rl(budget_path(INV.horizon) / 1e6, 0), "capacity": list(OPS.capacity[:INV.horizon]),
                 "costs": {"block_marginal": COST.c_block_marginal, "oneoff": COST.c_oneoff,
                           "mobilization": COST.mobilization, "epa_national": COST.epa_national},
                 "rates": list(FIN.social_rates), "dwsrf_rate": FIN.dwsrf_rate, "bond_rate": FIN.muni_bond_rate,
                 "loan_term": FIN.loan_term, "benefit_life": HEALTH.benefit_life_yrs},
        "scenarios": scen, "ca": cas, "ca_sched": ca_sched, "frontier": frontier,
        "financing": fin, "npv": npv, "tornado": tor, "psa": psa,
        "decomposition": aux["decomposition"], "refusal": aux["refusal_cost"], "consent": aux["consent"],
        "harm_mean": aux["harm_mean"],
    }
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"data.json: {len(scen)} scenarios, {len(cas)} community areas, {unmapped} lines without an area "
          f"({(OUT / 'data.json').stat().st_size / 1e3:.0f} kB)")

    if geojson:
        from shapely.geometry import shape, mapping
        g = json.load(open(geojson))
        feats = []

        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 5), round(c[1], 5)]
        for f in g["features"]:
            geom = shape(f["geometry"]).simplify(0.00015, preserve_topology=True)
            m = mapping(geom)
            feats.append({"type": "Feature",
                          "properties": {"id": str(f["properties"]["area_num_1"]),
                                         "name": f["properties"]["community"].title()},
                          "geometry": {"type": m["type"], "coordinates": rnd(m["coordinates"])}})
        out = {"type": "FeatureCollection",
               "source": "City of Chicago Data Portal, Boundaries - Community Areas (igwz-8jzy); simplified",
               "features": feats}
        (OUT / "community_areas.geojson").write_text(json.dumps(out, separators=(",", ":")))
        print(f"community_areas.geojson: {len(feats)} areas "
              f"({(OUT / 'community_areas.geojson').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson")
    main(ap.parse_args().geojson)
