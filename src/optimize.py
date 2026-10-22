"""
optimize.py
Multi-objective MILP for multi-year LSL replacement scheduling, heuristic
baselines, and a common evaluator.

Objective (constant-free form; H = H0 - A):
    min  a_C * C  -  a_H * A_w  +  a_E * lambda * E
where A_w is (optionally distributionally weighted) avoided harm.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from config import COST, OPS, FIN, INV, EQ, HEALTH
from welfare import harm_per_line_year, gini, atkinson, concentration_index


@dataclass
class Scenario:
    name: str
    a_H: float = 1.0
    a_C: float = 1.0
    a_E: float = 0.0
    equity_weight: str = "svi"           # 'svi' | 'svi_convex'
    harm_weight: str = "none"            # 'none' | 'income' (Atkinson-type distributional weights)
    eps_gap: float | None = None
    gap_min: float | None = None
    ej_floor: bool = True
    rate: float = FIN.base_rate
    time_limit: int = 200
    gap_rel: float = 0.001
    gap_abs: float = 2.0e7
    threads: int = 2
    allow_block: bool = True


# ------------------------------------------------------------ inputs
def prepare_zones(z: pd.DataFrame, tier: str = "central", p: dict | None = None,
                  exposure: str = "uniform") -> pd.DataFrame:
    z = z.copy()
    z["svi"] = z["svi"].fillna(z["svi"].median())
    if exposure == "huynh":
        # Huynh et al. (2024): +11.2% chance of detectable tap lead per +10 pts Hispanic share
        rho = 1.112 ** ((z["pct_hispanic"] - np.average(z["pct_hispanic"], weights=z["n_lsl"])) / 10)
        rho = rho / np.average(rho, weights=z["n_lsl"])
    else:
        rho = 1.0
    z["h"] = harm_per_line_year(z["kids_per_line"], z["adults_per_line"], z["lead_frac"], rho, p, tier)
    z["ej"] = z["svi"] >= np.quantile(z["svi"], OPS.ej_svi_quantile)
    order = z.sort_values("svi", ascending=False)
    cum = order["n_lsl"].cumsum() / order["n_lsl"].sum()
    z["q5"] = z.index.isin(order.index[(cum - order["n_lsl"] / order["n_lsl"].sum()) < 0.20])
    zq = order["n_lsl"].cumsum() / order["n_lsl"].sum()
    z["svi_q"] = 5 - np.minimum(4, np.floor((zq.reindex(z.index) - 1e-9) * 5)).astype(int)
    z["w_income"] = (np.average(z["median_income"], weights=z["n_lsl"]) / z["median_income"]) ** EQ.eta
    return z


def budget_path(T: int) -> np.ndarray:
    return np.minimum(OPS.budget_start_musd + OPS.budget_step_musd * np.arange(T), OPS.budget_cap_musd) * 1e6


def _parse_cbc_log(path: str) -> dict:
    out = {"bound": None, "obj": None, "gap": None}
    try:
        txt = open(path).read()
    except OSError:
        return out
    m = re.findall(r"Objective value:\s*([-\d.eE+]+)", txt)
    if m:
        out["obj"] = float(m[-1])
    m = re.findall(r"(?:Lower bound|best possible):?\s*([-\d.eE+]+)", txt)
    if m:
        out["bound"] = float(m[-1])
    if out["obj"] is not None and out["bound"] is not None:
        out["gap_abs"] = abs(out["obj"] - out["bound"])
        out["gap"] = out["gap_abs"] / max(1.0, abs(out["obj"]))
    out["optimal"] = "Optimal solution found" in txt
    return out


# ------------------------------------------------------------ the MILP
def solve(z: pd.DataFrame, sc: Scenario, verbose: bool = False) -> dict:
    Z, T = list(z.index), list(range(1, INV.horizon + 1))
    N, h, svi = z["n_lsl"].to_dict(), z["h"].to_dict(), z["svi"].to_dict()
    hw = (z["h"] * (z["w_income"] if sc.harm_weight == "income" else 1.0)).to_dict()
    ew = (z["svi"] ** EQ.convex_power / np.mean(z["svi"] ** EQ.convex_power) * z["svi"].mean()
          if sc.equity_weight == "svi_convex" else z["svi"]).to_dict()
    Ntot = sum(N.values())
    EJ = [i for i in Z if z.at[i, "ej"]]
    Q5 = [i for i in Z if z.at[i, "q5"]]
    N_ej, N_q5 = sum(N[i] for i in EJ), sum(N[i] for i in Q5)
    disc = {t: (1 + sc.rate) ** -(t - 0.5) for t in T}
    K = dict(zip(T, OPS.capacity))
    B = dict(zip(T, budget_path(len(T))))

    m = pulp.LpProblem(re.sub(r"\W", "_", sc.name), pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (Z, T), 0, 1)
    b = pulp.LpVariable.dicts("b", (Z, T), 0)
    o = pulp.LpVariable.dicts("o", (Z, T), 0)
    y = pulp.LpVariable.dicts("y", (Z, T), cat="Binary")
    d = pulp.LpVariable.dicts("d", Z, 0)

    cum = {(i, t): pulp.lpSum(x[i][s] for s in T if s < t) for i in Z for t in T}
    W = {i: pulp.lpSum(t * x[i][t] for t in T) for i in Z}
    Wbar = pulp.lpSum(N[i] * W[i] for i in Z) * (1.0 / Ntot)
    cost_t = {t: pulp.lpSum(COST.mobilization * y[i][t] + COST.c_block_marginal * b[i][t]
                            + COST.c_oneoff * o[i][t] for i in Z) for t in T}
    A_w = pulp.lpSum(disc[t] * hw[i] * N[i] * (cum[i, t] + 0.5 * x[i][t]) for i in Z for t in T)
    C = pulp.lpSum(disc[t] * cost_t[t] for t in T)
    E = pulp.lpSum(ew[i] * N[i] * d[i] for i in Z)
    m += sc.a_C * C - sc.a_H * A_w + sc.a_E * EQ.lambda_equity * E

    for i in Z:
        m += pulp.lpSum(x[i][t] for t in T) == 1
        m += d[i] >= W[i] - Wbar
        mb = min(COST.min_block_batch, N[i])
        for t in T:
            m += N[i] * x[i][t] == b[i][t] + o[i][t]
            m += b[i][t] <= N[i] * y[i][t]
            m += b[i][t] >= mb * y[i][t]
            if not sc.allow_block:
                m += y[i][t] == 0
    for t in T:
        done_t = pulp.lpSum(N[i] * (cum[i, t] + x[i][t]) for i in Z)
        m += pulp.lpSum(N[i] * x[i][t] for i in Z) <= K[t]
        m += pulp.lpSum(y[i][t] for i in Z) <= OPS.max_block_zones_per_year
        m += cost_t[t] <= B[t]
        if t < T[-1]:
            m += done_t >= Ntot * max(0.0, t / len(T) - OPS.pace_slack)
        if sc.ej_floor:
            m += pulp.lpSum(N[i] * (cum[i, t] + x[i][t]) for i in EJ) * (1.0 / N_ej) >= done_t * (1.0 / Ntot)
    G = pulp.lpSum(N[i] * W[i] for i in Q5) * (1.0 / N_q5) - Wbar
    if sc.eps_gap is not None:
        m += G <= sc.eps_gap
    if sc.gap_min is not None:
        m += G >= sc.gap_min

    log = tempfile.NamedTemporaryFile(suffix=".log", delete=False).name
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=sc.time_limit, gapRel=sc.gap_rel,
                               gapAbs=sc.gap_abs, threads=sc.threads, logPath=log)
    status = m.solve(solver)
    info = _parse_cbc_log(log)
    os.unlink(log)

    def frame(v):
        return pd.DataFrame({t: [v[i][t].value() or 0.0 for i in Z] for t in T}, index=Z)
    Yv = frame(y).round()
    return {"scenario": sc.name, "status": pulp.LpStatus[status], "X": frame(x), "B": frame(b),
            "O": frame(o), "Y": Yv, "obj": pulp.value(m.objective), "solver": info, "rate": sc.rate}


# ---------------------------------------------------- heuristic schedules
def _waterfill(remaining: pd.Series, weight: pd.Series, cap: float) -> pd.Series:
    alloc = pd.Series(0.0, index=remaining.index)
    rem = remaining.copy()
    for _ in range(60):
        if cap <= 1e-6 or rem.sum() <= 1e-6:
            break
        w = weight * rem
        if w.sum() <= 0:
            w = rem
        a = np.minimum(rem, cap * w / w.sum())
        alloc += a
        rem -= a
        cap -= a.sum()
    return alloc


def _price(lines: pd.Series, block: bool) -> tuple[float, pd.Series, pd.Series, pd.Series]:
    """Cost of a vector of zone-year line counts; block pricing where batch >= m."""
    if not block:
        return float(COST.c_oneoff * lines.sum()), lines * 0, lines.copy(), lines * 0
    use = lines >= np.minimum(COST.min_block_batch, lines.max() + 1)
    bl, ol = lines.where(use, 0.0), lines.where(~use, 0.0)
    ycount = use.astype(float)
    return float(COST.mobilization * ycount.sum() + COST.c_block_marginal * bl.sum() + COST.c_oneoff * ol.sum()), bl, ol, ycount


def heuristic(z: pd.DataFrame, weight: pd.Series, name: str, block: bool) -> dict:
    """Allocate annual capacity by `weight` x remaining lines until done,
    scaling each year to the capital budget."""
    years = list(range(1, INV.max_years_sq + 1))
    K = dict(zip(years, OPS.capacity))
    Bud = dict(zip(years, budget_path(len(years))))
    rem = z["n_lsl"].astype(float).copy()
    X, Bm, Om, Ym = {}, {}, {}, {}
    for t in years:
        if rem.sum() <= 1e-6:
            break
        alloc = _waterfill(rem, weight, K[t])
        cost, bl, ol, yc = _price(alloc, block)
        if cost > Bud[t]:
            alloc = _waterfill(rem, weight, alloc.sum() * Bud[t] / cost)
            cost, bl, ol, yc = _price(alloc, block)
        X[t], Bm[t], Om[t], Ym[t] = alloc / z["n_lsl"], bl, ol, yc
        rem -= alloc
    f = lambda d: pd.DataFrame(d)
    return {"scenario": name, "status": "Heuristic", "X": f(X), "B": f(Bm), "O": f(Om), "Y": f(Ym),
            "solver": {}, "rate": FIN.base_rate}


def status_quo(z: pd.DataFrame, block: bool = False, kappa: float = 500.0) -> dict:
    """Empirical status quo: allocation follows the 2021-2025 replacement
    propensity of each zone (empirical-Bayes smoothed), one-off pricing."""
    rbar = z["replaced_2021_25"].sum() / z["n_lsl"].sum()
    rate = (z["replaced_2021_25"] + kappa * rbar) / (z["n_lsl"] + kappa)
    nm = "Status quo (block-priced)" if block else "Status quo"
    return heuristic(z, rate / rate.mean(), nm, block)


def priority(z: pd.DataFrame, block: bool = False) -> dict:
    """Greedy harm priority: each year, fill capacity (then budget) in
    descending h_z order. One-off pricing isolates the value of ordering."""
    years = list(range(1, INV.max_years_sq + 1))
    K = dict(zip(years, OPS.capacity))
    Bud = dict(zip(years, budget_path(len(years))))
    rem = z["n_lsl"].astype(float).copy()
    order = z["h"].sort_values(ascending=False).index
    X, Bm, Om, Ym = {}, {}, {}, {}
    for t in years:
        if rem.sum() <= 1e-6:
            break
        cap = min(K[t], Bud[t] / COST.c_oneoff) if not block else K[t]
        alloc = pd.Series(0.0, index=z.index)
        for i in order:
            if cap <= 0:
                break
            a = min(rem[i], cap)
            alloc[i] = a
            cap -= a
        cost, bl, ol, yc = _price(alloc, block)
        X[t], Bm[t], Om[t], Ym[t] = alloc / z["n_lsl"], bl, ol, yc
        rem -= alloc
    f = lambda d: pd.DataFrame(d)
    return {"scenario": "Harm priority, one-off", "status": "Heuristic", "X": f(X), "B": f(Bm), "O": f(Om),
            "Y": f(Ym), "solver": {}, "rate": FIN.base_rate}


def proportional(z: pd.DataFrame) -> dict:
    return heuristic(z, pd.Series(1.0, index=z.index), "Proportional pace", block=True)


# ---------------------------------------------------------- evaluation
def evaluate(res: dict, z: pd.DataFrame, r: float = FIN.base_rate, h: pd.Series | None = None,
             cost: dict | None = None) -> dict:
    X = res["X"].reindex(z.index).fillna(0.0)
    T = np.array(X.columns, float)
    N = z["n_lsl"].values
    hv = (z["h"] if h is None else h).values
    disc = (1 + r) ** -(T - 0.5)
    cumprev = X.cumsum(axis=1).shift(1, axis=1).fillna(0.0).values
    resid = np.clip(1 - cumprev - 0.5 * X.values, 0, 1)
    tail = np.clip(1 - X.values.sum(axis=1), 0, 1)                       # unfinished (should be 0)
    harm = (hv[:, None] * N[:, None] * resid * disc).sum() + (hv * N * tail).sum() * disc[-1] / r
    c = {"F": COST.mobilization, "cB": COST.c_block_marginal, "cO": COST.c_oneoff} | (cost or {})
    Yv, Bv, Ov = (res[k].reindex(z.index).fillna(0.0).reindex(columns=X.columns, fill_value=0.0) for k in "YBO")
    cost_year = (c["F"] * Yv + c["cB"] * Bv + c["cO"] * Ov).sum(axis=0).values
    pv_cost = (cost_year * disc).sum()
    W = (X.values * T).sum(axis=1)
    Wbar = np.average(W, weights=N)
    q5, ej = z["q5"].values, z["ej"].values
    finish = np.array([T[np.argmax(np.cumsum(row) >= 0.999)] if row.sum() >= 0.999 else np.nan
                       for row in X.values])
    kids = z["kids_per_line"].values * N
    exp_child = (resid.sum(axis=1)) * z["kids_per_line"].values * N     # child exposure-years
    exp_pc = exp_child / np.maximum(kids, 1e-9)
    L = np.maximum(INV.horizon + 1 - W, 0.5)
    return {
        "scenario": res["scenario"], "status": res["status"],
        "pv_harm_bn": harm / 1e9, "pv_cost_bn": pv_cost / 1e9, "pv_social_bn": (harm + pv_cost) / 1e9,
        "nominal_cost_bn": cost_year.sum() / 1e9, "completion_year": INV.base_year - 1 + np.nanmax(finish),
        "share_done_2046": float((X.values[:, :INV.horizon] * N[:, None]).sum() / N.sum()),
        "mean_wait": Wbar, "q5_gap": np.average(W[q5], weights=N[q5]) - Wbar,
        "wait_ej": np.average(W[ej], weights=N[ej]), "wait_nonej": np.average(W[~ej], weights=N[~ej]),
        "child_exposure_m": exp_child.sum() / 1e6,
        "gini_exposure": gini(exp_pc, kids), "atk_1": atkinson(L, 1.0, N), "atk_2": atkinson(L, 2.0, N),
        "conc_index_svi": concentration_index(exp_pc, z["svi"].values, kids),
        "block_share": float(Bv.values.sum() / N.sum()), "mobilizations": int(Yv.values.sum()),
        "W": W, "cost_year": cost_year, "lines_year": (X.values * N[:, None]).sum(axis=0),
        "resid": resid, "exp_pc": exp_pc,
    }
