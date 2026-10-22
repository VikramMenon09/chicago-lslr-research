"""
analysis.py
Everything downstream of the cached schedules: scenario tables, decomposition,
frontier, probabilistic sensitivity analysis (PSA), consent econometrics,
welfare cost of refusals, external validation, per-line NPV, and LaTeX output.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats

from config import COST, FIN, HEALTH, INV, OPS, DATA_PROC, PAPER_DIR, SEED
from optimize import prepare_zones, evaluate, budget_path
from welfare import (unit_values, harm_per_line_year, annuity_factor, per_line_npv,
                     breakeven_harm, financed_cost_pv)

RES = DATA_PROC / "results"
SCALAR = ["pv_harm_bn", "pv_cost_bn", "pv_social_bn", "nominal_cost_bn", "completion_year",
          "share_done_2046", "mean_wait", "q5_gap", "wait_ej", "wait_nonej", "child_exposure_m",
          "gini_exposure", "atk_1", "atk_2", "conc_index_svi", "block_share", "mobilizations"]
MAIN = ["SQ", "PROP", "PRIO", "CM", "HW", "EW", "EWc", "DW"]
LABEL = {"SQ": "Status quo (empirical)", "SQ_block": "Status quo, block-priced", "PROP": "Proportional pace", "PRIO": "Harm priority, one-off",
         "CM": "Cost-minimizing", "HW_oneoff": "Health-welfare, one-off only", "HW": "Health-welfare",
         "EW": "Equity-weighted, linear SVI", "EWc": "Equity-weighted, convex SVI",
         "DW": "Distributional weights ($\\eta=1$)", "HW_r2": "HW, $r=2\\%$", "HW_r7": "HW, $r=7\\%$",
         "HW_child": "HW, child benefits only", "HW_high": "HW, high CVD function",
         "HW_huynh": "HW, tap-lead gradient"}


def zones(tier="central", exposure="uniform", p=None):
    return prepare_zones(pd.read_csv(DATA_PROC / "zones.csv", index_col=0), tier=tier, exposure=exposure, p=p)


def load_results() -> dict:
    R = {p.stem: pickle.load(open(p, "rb")) for p in sorted(RES.glob("*.pkl"))}
    return {k: v for k, v in R.items() if v["status"] not in ("Infeasible", "Not Solved", "Undefined")}


def scalar(e):
    return {k: float(e[k]) for k in SCALAR}


# ------------------------------------------------------------ scenarios
def scenario_table(R: dict) -> tuple[pd.DataFrame, dict]:
    zc = zones()
    rows, ev = {}, {}
    for k, r in R.items():
        e = evaluate(r, zc, r=FIN.base_rate)                    # common yardstick: central, r = 3%
        ev[k] = e
        s = scalar(e) | {"status": r["status"], "gap": (r.get("solver") or {}).get("gap"),
                         "gap_abs_musd": ((r.get("solver") or {}).get("gap_abs") or np.nan) / 1e6,
                         "seconds": r.get("seconds", np.nan)}
        if r.get("tier", "central") != "central" or r.get("exposure", "uniform") != "uniform" \
                or r.get("rate", FIN.base_rate) != FIN.base_rate:
            zo = zones(r.get("tier", "central"), r.get("exposure", "uniform"))
            own = evaluate(r, zo, r=r.get("rate", FIN.base_rate))
            s |= {f"own_{k2}": float(own[k2]) for k2 in ("pv_harm_bn", "pv_cost_bn", "pv_social_bn", "q5_gap")}
        rows[k] = s
    alt = evaluate(R["SQ"], zc, r=FIN.base_rate, cost={"cO": 31_000.0})       # city-reported average
    rows["SQ_31k"] = scalar(alt) | {"status": "Heuristic"}
    ev["SQ_31k"] = alt
    return pd.DataFrame(rows).T, ev


def decomposition(T: pd.DataFrame) -> dict:
    sq, hwo, hw, cm = (T.loc[k] for k in ("SQ", "PRIO", "HW", "CM"))
    return {"total": sq.pv_social_bn - hw.pv_social_bn,
            "ordering": sq.pv_social_bn - hwo.pv_social_bn,           # risk ordering, one-off delivery
            "clustering": hwo.pv_social_bn - hw.pv_social_bn,          # adding block programs
            "harm_total": sq.pv_harm_bn - hw.pv_harm_bn, "cost_total": sq.pv_cost_bn - hw.pv_cost_bn,
            "cm_vs_sq": sq.pv_social_bn - cm.pv_social_bn}


# ------------------------------------------------------------ PSA
def _fast_eval(X: np.ndarray, Y, Bm, Om, N, h, r, c):
    T = np.arange(1, X.shape[1] + 1, dtype=float)
    disc = (1 + r) ** -(T - 0.5)
    cumprev = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, axis=1)[:, :-1]], axis=1)
    resid = np.clip(1 - cumprev - 0.5 * X, 0, 1)
    harm = (h[:, None] * N[:, None] * resid * disc).sum()
    cost = ((c["F"] * Y + c["cB"] * Bm + c["cO"] * Om).sum(axis=0) * disc).sum()
    return harm, cost


def psa(R: dict, n: int = 2000, keys=("SQ", "PROP", "CM", "HW", "EW", "DW")) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    zc = zones()
    N = zc["n_lsl"].values
    mats = {}
    for k in keys:
        r = R[k]
        cols = r["X"].columns
        f = lambda d: d.reindex(zc.index).fillna(0).reindex(columns=cols, fill_value=0).values
        mats[k] = (f(r["X"]), f(r["Y"]), f(r["B"]), f(r["O"]))
    out = []
    for i in range(n):
        p = {"iq_beta": rng.uniform(HEALTH.iq_beta_low, HEALTH.iq_beta),
             "value_per_iq": rng.uniform(HEALTH.value_per_iq_low, HEALTH.value_per_iq_high),
             "child_dbll": rng.triangular(0.4, HEALTH.child_dbll, 1.5),
             "adult_dbll": rng.triangular(0.1, HEALTH.adult_dbll, 0.4),
             "cvd_beta": rng.choice([rng.uniform(0.05, 0.68), rng.uniform(0.54, 1.37)]),
             "vsl": HEALTH.vsl * rng.uniform(0.75, 1.25)}
        c = {"F": rng.uniform(0.75e6, 3.0e6), "cB": rng.uniform(27_000, 35_000), "cO": rng.uniform(35_000, 45_000)}
        r = float(rng.choice([0.02, 0.03, 0.07]))
        h = harm_per_line_year(zc["kids_per_line"], zc["adults_per_line"], zc["lead_frac"], 1.0, p, "central")
        h = np.asarray(h, float)
        row = {"draw": i, "rate": r} | p | c
        for k in keys:
            hk, ck = _fast_eval(*mats[k], N, h, r, c)
            row[f"{k}_social"] = (hk + ck) / 1e9
        vc, va = unit_values(p, "central")
        hm = np.average(h, weights=N)
        row["bcr_31k"] = hm * annuity_factor(r, HEALTH.benefit_life_yrs) / 31_000
        out.append(row)
    D = pd.DataFrame(out)
    D["best"] = D[[f"{k}_social" for k in keys]].idxmin(axis=1).str.replace("_social", "")
    return D


def tornado(n_grid: int = 2) -> pd.DataFrame:
    """One-way sensitivity of the citywide per-line BCR at $31k, r = 3%."""
    zc = zones()
    N = zc["n_lsl"].values

    def bcr(p, tier="central", r=0.03, cost=31_000):
        h = harm_per_line_year(zc["kids_per_line"], zc["adults_per_line"], zc["lead_frac"], 1.0, p, tier)
        return np.average(h, weights=N) * annuity_factor(r, HEALTH.benefit_life_yrs) / cost
    base = bcr({})
    rows = [("IQ slope $\\beta$ (2.70 to 3.25)", bcr({"iq_beta": 2.70}), bcr({"iq_beta": 3.25})),
            ("Value per IQ point (\\$38.4k to \\$46.2k)", bcr({"value_per_iq": 38_404}), bcr({"value_per_iq": 46_204})),
            ("Child BLL increment (0.4 to 1.5)", bcr({"child_dbll": 0.4}), bcr({"child_dbll": 1.5})),
            ("Adult BLL increment (0.1 to 0.4)", bcr({"adult_dbll": 0.1}), bcr({"adult_dbll": 0.4})),
            ("CVD function (child-only to Lanphear 2018)", bcr({}, "child"), bcr({}, "high")),
            ("Discount rate (7\\% to 2\\%)", bcr({}, r=0.07), bcr({}, r=0.02)),
            ("Unit cost (\\$39k to \\$25k)", bcr({}, cost=39_000), bcr({}, cost=25_000))]
    T = pd.DataFrame(rows, columns=["param", "low", "high"])
    T["base"] = base
    T["range"] = (T["high"] - T["low"]).abs()
    return T.sort_values("range")


# ------------------------------------------------------------ consent
def consent_models() -> dict:
    t = gpd.read_file(DATA_PROC / "tracts.gpkg")
    d = t[t["n_action"] >= 25].copy()
    d["attempts"] = d["n_refused"] + d["n_replaced_2021_25"]
    d = d[d["attempts"] >= 5].copy()
    d["log_income"] = np.log(d["median_household_income"])
    X = pd.DataFrame({
        "Renter share (per 10 pp)": d["renter_share"] * 10,
        "2-4 unit share (per 10 pp)": d["two_to_four_share"] * 10,
        "log median income": d["log_income"],
        "Black share (per 10 pp)": d["pct_black_nonhispanic"] / 10,
        "Hispanic share (per 10 pp)": d["pct_hispanic"] / 10,
        "Limited English (per 10 pp)": d["pct_limited_or_no_english"] / 10,
        "Pre-1940 housing (per 10 pp)": d["pre1940_share"] * 10})
    ok = X.notna().all(axis=1) & d["area_num_1"].notna()
    X, d = sm.add_constant(X[ok]), d[ok]
    grp = d["area_num_1"].astype(int).values
    endog = np.c_[d["n_refused"].values, (d["attempts"] - d["n_refused"]).values]
    m_share = sm.GLM(endog, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": grp})
    m_rate = sm.GLM(d["n_refused"].values, X, family=sm.families.NegativeBinomial(alpha=0.5),
                    offset=np.log(d["n_action"].values)).fit(cov_type="cluster", cov_kwds={"groups": grp})
    Xb = sm.add_constant(X[["Renter share (per 10 pp)"]])
    m_biv = sm.GLM(endog, Xb, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": grp})

    def tab(m, name):
        ci = m.conf_int()
        return pd.DataFrame({"model": name, "coef": m.params, "se": m.bse, "p": m.pvalues,
                             "lo": ci[0], "hi": ci[1]})
    out = pd.concat([tab(m_share, "share"), tab(m_rate, "rate"), tab(m_biv, "bivariate")])
    b0, b1 = m_biv.params
    p_owner, p_renter = 1 / (1 + np.exp(-b0)), 1 / (1 + np.exp(-(b0 + 10 * b1)))
    return {"table": out, "n_tracts": int(len(d)), "attempts": int(d["attempts"].sum()),
            "refusals": int(d["n_refused"].sum()), "p_refuse_owner": float(p_owner),
            "p_refuse_renter": float(p_renter), "irr_renter": float(np.exp(m_rate.params.iloc[1])),
            "irr_renter_p": float(m_rate.pvalues.iloc[1])}


def refusal_cost(R: dict, key: str = "HW") -> dict:
    """Extra PV harm if refused lines in each zone wait until the final year."""
    zc = zones()
    rbar = zc["refused"].sum() / (zc["refused"].sum() + zc["replaced_2021_25"].sum())
    k = 50.0
    rho = (zc["refused"] + k * rbar) / (zc["refused"] + zc["replaced_2021_25"] + k)
    X = R[key]["X"].reindex(zc.index).fillna(0).values
    T = np.arange(1, X.shape[1] + 1, dtype=float)
    disc = (1 + FIN.base_rate) ** -(T - 0.5)
    tail = np.array([disc[t:].sum() for t in range(len(T))])           # years from t to the end
    extra = (rho.values[:, None] * zc["n_lsl"].values[:, None] * zc["h"].values[:, None] * X * tail).sum(axis=1)
    return {"total_bn": float(extra.sum() / 1e9), "ej_share": float(extra[zc["ej"].values].sum() / extra.sum()),
            "rho_mean": float(np.average(rho, weights=zc["n_lsl"])),
            "rho_q5": float(np.average(rho[zc["svi_q"] == 5], weights=zc.loc[zc["svi_q"] == 5, "n_lsl"])),
            "rho_q1": float(np.average(rho[zc["svi_q"] == 1], weights=zc.loc[zc["svi_q"] == 1, "n_lsl"])),
            "by_q": pd.Series(extra, index=zc.index).groupby(zc["svi_q"]).sum().div(1e9).to_dict()}


# ------------------------------------------------------------ validation
def validation() -> dict:
    t = gpd.read_file(DATA_PROC / "tracts.gpkg")
    ca = pd.read_csv(DATA_PROC / "community_areas.csv").dropna(subset=["bll_elev_2009_13"])
    rng = np.random.default_rng(SEED)

    def boot(x, y, n=2000):
        r0 = stats.spearmanr(x, y).statistic
        bs = [stats.spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
        return r0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    ca["action_share"] = ca["n_action"] / ca["lines_total"]
    out = {"n_ca": int(len(ca))}
    for name, col in [("svi", "svi"), ("holc_D", "holc_D"), ("kids", "kids_per_line"), ("action", "action_share")]:
        for yr in ("2009_13", "1999_13"):
            r, lo, hi = boot(ca[col].values, ca[f"bll_elev_{yr}"].values)
            out[f"rho_{name}_{yr}"] = (float(r), float(lo), float(hi))
    t["holc_dom"] = np.where(t["holc_graded"] < 0.5, "Ungraded",
                             t[["holc_A", "holc_B", "holc_C", "holc_D"]].idxmax(axis=1).str[-1])
    g = t.groupby("holc_dom").agg(tracts=("geoid", "size"), lines=("lines_total", "sum"),
                                  action=("n_action", "sum"), svi=("svi", "mean"))
    g["action_share"] = g["action"] / g["lines"]
    out["holc"] = g
    out["imputation"] = json.load(open(DATA_PROC / "imputation_cv.json"))
    return out


# ------------------------------------------------------------ NPV
def npv_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    zc = zones()
    N = zc["n_lsl"].values
    hm = {t: float(np.average(harm_per_line_year(zc["kids_per_line"], zc["adults_per_line"], zc["lead_frac"],
                                                 1.0, None, t), weights=N)) for t in ("child", "central", "high")}
    rows = []
    for tier, h in hm.items():
        for c in (COST.epa_national, 10_000.0, 31_000.0, COST.c_oneoff):
            for r in FIN.social_rates:
                rows.append({"tier": tier, "cost": c, "rate": r, "npv": per_line_npv(c, h, r),
                             "bcr": h * annuity_factor(r, HEALTH.benefit_life_yrs) / c})
    fin = pd.DataFrame([{"rate": r, "cash": 31_000.0, "dwsrf": financed_cost_pv(31_000.0, FIN.dwsrf_rate, r),
                         "bond": financed_cost_pv(31_000.0, FIN.muni_bond_rate, r)} for r in FIN.social_rates])
    return pd.DataFrame(rows), fin, hm


# ------------------------------------------------------------ LaTeX
def fmt(x, d=2):
    return f"{x:,.{d}f}"


def write_latex(T, ev, dec, F, D, TOR, cons, refc, val, npv, fin, hm, R):
    m = lambda k, v: f"\\newcommand{{\\{k}}}{{{v}}}\n"
    zc = zones()
    s = "% auto-generated by src/analysis.py; do not edit\n"
    ab = {"SQ": "SQ", "PROP": "PR", "PRIO": "PO", "CM": "CM", "HW": "HW", "EW": "EW", "EWc": "EC",
          "DW": "DW", "HW_r2": "HRtwo", "HW_r7": "HRseven", "HW_child": "HCh", "HW_high": "HHi", "HW_huynh": "HHu"}
    for k, a in ab.items():
        if k not in T.index:
            continue
        r = T.loc[k]
        for nm, col, d in [("Harm", "pv_harm_bn", 2), ("Cost", "pv_cost_bn", 2), ("Social", "pv_social_bn", 2),
                           ("Nominal", "nominal_cost_bn", 2), ("Gap", "q5_gap", 2), ("Wait", "mean_wait", 1),
                           ("WaitEJ", "wait_ej", 1), ("WaitNonEJ", "wait_nonej", 1), ("Gini", "gini_exposure", 3),
                           ("AtkOne", "atk_1", 3), ("Conc", "conc_index_svi", 3), ("Done", "share_done_2046", 3),
                           ("Exposure", "child_exposure_m", 2)]:
            v = r[col]
            s += m(f"{a}{nm}", (f"{v:+.{d}f}" if nm == "Gap" or nm == "Conc" else fmt(v, d)))
        s += m(f"{a}Completion", f"{int(r.completion_year)}")
        s += m(f"{a}GapAbs", f"{abs(r.q5_gap):.1f}") + m(f"{a}DonePct", f"{100 * r.share_done_2046:.1f}")
        s += m(f"{a}Block", f"{100 * r.block_share:.0f}")
    for k, v in dec.items():
        s += m("Dec" + "".join(w.capitalize() for w in k.split("_")), fmt(v, 2))
    a31 = T.loc["SQ_31k"]
    s += m("SQAltCost", fmt(a31.pv_cost_bn)) + m("SQAltSocial", fmt(a31.pv_social_bn))
    s += m("SavingAlt", fmt(a31.pv_social_bn - T.loc["HW", "pv_social_bn"]))
    s += m("SavingAltPct", f"{100 * (a31.pv_social_bn - T.loc['HW', 'pv_social_bn']) / a31.pv_social_bn:.0f}")
    s += m("RefCostShareGain", f"{100 * refc['total_bn'] / dec['total']:.0f}")
    s += m("DecOrderPct", f"{100 * dec['ordering'] / dec['total']:.0f}")
    s += m("DecClusterPct", f"{100 * dec['clustering'] / dec['total']:.0f}")
    s += m("GainPct", f"{100 * dec['total'] / T.loc['SQ', 'pv_social_bn']:.1f}")
    gaps = T.loc[[k for k in T.index if T.loc[k, "status"] != "Heuristic" and pd.notna(T.loc[k, "gap_abs_musd"])], "gap_abs_musd"]
    s += m("MaxGapMusd", f"{gaps.max():.0f}") + m("MedGapMusd", f"{gaps.median():.0f}")
    # frontier
    hw = T.loc["HW"]
    eff = F[F["branch"] == "efficient"].sort_values("q5_gap")
    s += m("FrontMinGap", f"{eff.q5_gap.iloc[0]:+.2f}") + m("FrontMinGapCost", fmt(eff.pv_social_bn.iloc[0] - hw.pv_social_bn))
    dom = F[F["branch"] == "dominated"]
    if len(dom):
        z0 = dom.iloc[(dom.q5_gap - 0).abs().argmin()]
        s += m("DomZeroCost", fmt(z0.pv_social_bn - hw.pv_social_bn))
    # PSA
    s += m("PsaN", f"{len(D):,}")
    s += m("PsaHWbeatsSQ", f"{100 * (D['HW_social'] < D['SQ_social']).mean():.1f}")
    s += m("PsaHWbest", f"{100 * (D['best'] == 'HW').mean():.1f}")
    sav = D["SQ_social"] - D["HW_social"]
    s += m("PsaSaveMed", fmt(sav.median())) + m("PsaSaveLo", fmt(sav.quantile(0.05))) + m("PsaSaveHi", fmt(sav.quantile(0.95)))
    s += m("PsaBcrMed", fmt(D["bcr_31k"].median())) + m("PsaBcrLo", fmt(D["bcr_31k"].quantile(0.05)))
    s += m("PsaBcrHi", fmt(D["bcr_31k"].quantile(0.95))) + m("PsaBcrBelowOne", f"{100 * (D['bcr_31k'] < 1).mean():.1f}")
    # consent
    ct = cons["table"]
    rr = ct[(ct.model == "rate") & (ct.index == "Renter share (per 10 pp)")].iloc[0]
    s += m("ConsN", f"{cons['n_tracts']}") + m("ConsAttempts", f"{cons['attempts']:,}") + m("ConsRefusals", f"{cons['refusals']:,}")
    s += m("ConsIRR", f"{np.exp(rr.coef):.3f}") + m("ConsIRRp", f"{rr.p:.3f}")
    s += m("ConsIRRlo", f"{np.exp(rr.lo):.3f}") + m("ConsIRRhi", f"{np.exp(rr.hi):.3f}")
    s += m("PRefOwner", f"{100 * cons['p_refuse_owner']:.1f}") + m("PRefRenter", f"{100 * cons['p_refuse_renter']:.1f}")
    s += m("RefCost", fmt(refc["total_bn"])) + m("RefCostEJ", f"{100 * refc['ej_share']:.0f}")
    s += m("RefRho", f"{100 * refc['rho_mean']:.1f}") + m("RefRhoQF", f"{100 * refc['rho_q5']:.1f}") + m("RefRhoQO", f"{100 * refc['rho_q1']:.1f}")
    # validation
    for k, v in val.items():
        if k.startswith("rho_"):
            nm = "Rho" + "".join(w.capitalize() for w in k[4:].replace("_13", "").split("_"))
            nm = nm.replace("2009", "Late").replace("1999", "Long")
            s += m(nm, f"{v[0]:.2f}") + m(nm + "CI", f"[{v[1]:.2f}, {v[2]:.2f}]")
    imp = val["imputation"]
    s += m("ImpAUC", f"{imp['auc']:.3f}") + m("ImpN", f"{imp['n_train']:,}") + m("ImpBase", f"{100 * imp['base_rate']:.1f}")
    s += m("ImpBrier", f"{imp['brier']:.3f}") + m("ImpBrierNull", f"{imp['brier_null']:.3f}")
    s += m("NCa", f"{val['n_ca']}")
    # inventory facts
    lines = pd.read_parquet(DATA_PROC / "lines.parquet", columns=["classification", "action", "lead_prob",
                                                                   "year_replaced", "refused_access", "high_risk"])
    vc = lines["classification"].value_counts()
    s += m("NLines", f"{len(lines):,}") + m("NAction", f"{int(lines['action'].sum()):,}")
    for c in ("L", "GRR", "U", "NL"):
        s += m(f"N{c}", f"{int(vc.get(c, 0)):,}")
    s += m("NLeadExp", f"{lines.loc[lines.action, 'lead_prob'].sum():,.0f}")
    s += m("PULead", f"{100 * lines.loc[lines.classification == 'U', 'lead_prob'].mean():.1f}")
    s += m("NReplaced", f"{int(lines['year_replaced'].between(2021, 2025).sum()):,}")
    s += m("NRefused", f"{int(lines['refused_access'].sum()):,}") + m("NHighRisk", f"{int(lines['high_risk'].sum()):,}")
    t = gpd.read_file(DATA_PROC / "tracts.gpkg")
    s += m("NTracts", f"{len(t)}") + m("NZones", f"{len(zc)}")
    s += m("KidsServed", f"{(t.kids_per_line * t.n_action).sum():,.0f}") + m("AdultsServed", f"{(t.adults_per_line * t.n_action).sum():,.0f}")
    s += m("KidsCity", f"{t.kids_u6.sum():,.0f}")
    qz = zc.groupby("svi_q").apply(lambda g: pd.Series({
        "h": np.average(g.h, weights=g.n_lsl), "repl": 1000 * g.replaced_2021_25.sum() / g.n_lsl.sum(),
        "ref": 1000 * g.refused.sum() / g.n_lsl.sum(), "kids": np.average(g.kids_per_line, weights=g.n_lsl),
        "renter": np.average(g.renter_share, weights=g.n_lsl)}))
    for q, r in qz.iterrows():
        L = "ABCDE"[int(q) - 1]
        s += m(f"HQ{L}", fmt(r.h, 0)) + m(f"ReplQ{L}", fmt(r.repl, 1)) + m(f"RefQ{L}", fmt(r.ref, 1))
        s += m(f"KidsQ{L}", fmt(r.kids, 2)) + m(f"RentQ{L}", f"{100 * r.renter:.0f}")
    for tier, h in hm.items():
        s += m(f"HMean{tier.capitalize()}", fmt(h, 0))
    vcd, vad = unit_values(None, "central")
    _, vah = unit_values(None, "high")
    s += m("VChild", fmt(vcd, 0)) + m("VAdult", fmt(vad, 0)) + m("VAdultHigh", fmt(vah, 0))
    cname = {4700: "Epa", 10000: "Ten", 31000: "ThirtyOne", 39000: "ThirtyNine"}
    rname = {0.02: "Two", 0.03: "Three", 0.07: "Seven"}
    for _, r in npv.iterrows():
        s += m(f"BCR{r.tier.capitalize()}{cname[int(r.cost)]}{rname[round(r.rate, 2)]}", fmt(r.bcr, 2))
    s += m("TotalBudget", fmt(budget_path(INV.horizon).sum() / 1e9, 1))
    s += m("EJShare", f"{100 * zc.loc[zc.ej, 'n_lsl'].sum() / zc.n_lsl.sum():.1f}")
    (PAPER_DIR / "results.tex").write_text(s)

    # ---- tables
    rows = []
    for k in MAIN:
        r = T.loc[k]
        g = "--" if pd.isna(r.gap_abs_musd) else f"{r.gap_abs_musd:.0f}"
        rows.append(f"{LABEL[k]} & {r.pv_harm_bn:.2f} & {r.pv_cost_bn:.2f} & {r.pv_social_bn:.2f} & "
                    f"{int(r.completion_year)} & {r.wait_ej:.1f} & {r.wait_nonej:.1f} & {r.q5_gap:+.1f} & "
                    f"{r.conc_index_svi:+.2f} & {r.gini_exposure:.2f} & {100 * r.block_share:.0f} & {g} \\\\")
    (PAPER_DIR / "tab_scenarios.tex").write_text("\n".join(rows))
    rows = []
    for k in ("HW", "HW_r2", "HW_r7", "HW_child", "HW_high", "HW_huynh"):
        if k not in T.index:
            continue
        r = T.loc[k]
        own = r.get("own_pv_social_bn", r.pv_social_bn)
        own = r.pv_social_bn if pd.isna(own) else own
        rows.append(f"{LABEL[k]} & {own:.2f} & {r.pv_social_bn:.2f} & {r.q5_gap:+.1f} & {r.wait_ej:.1f} & "
                    f"{r.wait_nonej:.1f} & {100 * r.block_share:.0f} \\\\")
    (PAPER_DIR / "tab_robust.tex").write_text("\n".join(rows))
    piv = npv.pivot_table(index=["tier", "cost"], columns="rate", values="bcr")
    tl = {"child": "Child IQ only", "central": "Central (+CVD, Aoki)", "high": "High (+CVD, Lanphear)"}
    rows = []
    for tier in ("child", "central", "high"):
        for j, c in enumerate((4700.0, 10000.0, 31000.0, 39000.0)):
            lab = tl[tier] if j == 0 else ""
            rows.append(f"{lab} & \\${c:,.0f} & " + " & ".join(f"{piv.loc[(tier, c), r]:.2f}" for r in FIN.social_rates) + " \\\\")
        if tier != "high":
            rows.append("\\addlinespace")
    (PAPER_DIR / "tab_npv.tex").write_text("\n".join(rows))
    rows = [f"{r.rate * 100:.0f}\\% & \\${r.cash:,.0f} & \\${r.dwsrf:,.0f} & \\${r.bond:,.0f} \\\\" for _, r in fin.iterrows()]
    (PAPER_DIR / "tab_finance.tex").write_text("\n".join(rows))
    ct = cons["table"]
    rows = []
    for v in ct.index.unique():
        if v == "const":
            continue
        cells = []
        for mod in ("share", "rate"):
            q = ct[(ct.model == mod) & (ct.index == v)]
            if len(q):
                q = q.iloc[0]
                st = "^{**}" if q.p < 0.01 else ("^{*}" if q.p < 0.05 else ("^{\\dagger}" if q.p < 0.10 else ""))
                cells.append(f"${q.coef:.3f}{st}$ & $({q.se:.3f})$")
            else:
                cells.append("& ")
        rows.append(f"{v} & " + " & ".join(cells) + " \\\\")
    (PAPER_DIR / "tab_consent.tex").write_text("\n".join(rows))
    h = val["holc"].reindex(["A", "B", "C", "D", "Ungraded"])
    rows = [f"{g} & {int(r.tracts)} & {int(r.lines):,} & {100 * r.action_share:.1f}\\% & {r.svi:.2f} \\\\" for g, r in h.iterrows()]
    (PAPER_DIR / "tab_holc.tex").write_text("\n".join(rows))
    rows = [f"{r.param} & {r.low:.2f} & {r.high:.2f} \\\\" for _, r in TOR.sort_values("range", ascending=False).iterrows()]
    (PAPER_DIR / "tab_tornado.tex").write_text("\n".join(rows))


def frontier_df(T: pd.DataFrame) -> pd.DataFrame:
    rows = [T.loc["HW"].to_dict() | {"key": "HW", "branch": "both"}]
    for k in T.index:
        if k.startswith("EPS_"):
            rows.append(T.loc[k].to_dict() | {"key": k, "branch": "efficient"})
        if k.startswith("DOM_"):
            rows.append(T.loc[k].to_dict() | {"key": k, "branch": "dominated"})
    F = pd.DataFrame(rows)
    F.loc[F.branch == "both", "branch"] = "efficient"
    F = pd.concat([F, F[F.key == "HW"].assign(branch="dominated")])
    return F


def main():
    R = load_results()
    T, ev = scenario_table(R)
    dec = decomposition(T)
    F = frontier_df(T)
    D = psa(R)
    TOR = tornado()
    cons = consent_models()
    refc = refusal_cost(R)
    val = validation()
    npv, fin, hm = npv_tables()
    DATA_PROC.mkdir(exist_ok=True)
    T.to_csv(DATA_PROC / "scenario_results.csv")
    F.to_csv(DATA_PROC / "frontier.csv", index=False)
    D.to_csv(DATA_PROC / "psa_draws.csv", index=False)
    TOR.to_csv(DATA_PROC / "tornado.csv", index=False)
    cons["table"].to_csv(DATA_PROC / "consent_models.csv")
    npv.to_csv(DATA_PROC / "npv_sensitivity.csv", index=False)
    fin.to_csv(DATA_PROC / "financing.csv", index=False)
    json.dump({"decomposition": dec, "refusal_cost": {k: v for k, v in refc.items()},
               "consent": {k: v for k, v in cons.items() if k != "table"},
               "validation": {k: v for k, v in val.items() if k not in ("holc", "imputation")},
               "harm_mean": hm}, open(DATA_PROC / "aux_results.json", "w"), indent=2, default=float)
    write_latex(T, ev, dec, F, D, TOR, cons, refc, val, npv, fin, hm, R)
    return T, ev, dec, F, D, TOR, cons, refc, val, npv, fin, hm, R


if __name__ == "__main__":
    out = main()
    T = out[0]
    print(T[["pv_harm_bn", "pv_cost_bn", "pv_social_bn", "completion_year", "q5_gap", "conc_index_svi", "status", "gap_abs_musd"]].round(3))
    print("decomposition", {k: round(v, 3) for k, v in out[2].items()})
