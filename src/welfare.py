"""
welfare.py
Damage functions (EPA LCRI benefit methodology), discounting, per-line NPV,
financing, the structural consent model, and inequality metrics.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from config import HEALTH, FIN, COST

BASE = asdict(HEALTH)


# ---------------------------------------------------------- dose-response
def iq_loss(bll0, dbll, beta=HEALTH.iq_beta):
    """IQ points lost when BLL rises from bll0 to bll0 + dbll (log-linear)."""
    return beta * np.log((bll0 + dbll) / bll0)


def cvd_excess_deaths(beta_log10, bll0, dbll, m0=HEALTH.cvd_mort_rate):
    """EPA EA Eq. 16: avoided deaths per adult-year when BLL falls by dbll."""
    return m0 * (1.0 - np.exp(-beta_log10 * np.log10((bll0 + dbll) / bll0)))


def unit_values(p: dict | None = None, tier: str = "central") -> tuple[float, float]:
    """($ per child-year, $ per adult-year) of LSL exposure.

    tier: 'child'   IQ (+ADHD/special-ed add-on) only
          'central' + adult CVD mortality, Aoki et al. (2016) function
          'high'    + adult CVD mortality, Lanphear et al. (2018) function"""
    p = BASE | (p or {})
    child = (iq_loss(p["child_bll0"], p["child_dbll"], p["iq_beta"]) / p["window_yrs"]
             * p["value_per_iq"] * (1 + p["addon_adhd_sped"]))
    if tier == "child":
        return child, 0.0
    beta = p["cvd_beta_high"] if tier == "high" else p["cvd_beta_low"]
    if "cvd_beta" in p:
        beta = p["cvd_beta"]
    adult = cvd_excess_deaths(beta, p["adult_bll0"], p["adult_dbll"], p["cvd_mort_rate"]) * p["vsl"]
    return child, adult


def harm_per_line_year(kids, adults, lead_frac, exposure_mod=1.0, p=None, tier="central"):
    """Monetized annual damage of one unreplaced line ($/line-yr):
    h = lead_frac * rho * (k * v_child + a * v_adult)."""
    vc, va = unit_values(p, tier)
    return np.asarray(lead_frac) * np.asarray(exposure_mod) * (np.asarray(kids) * vc + np.asarray(adults) * va)


# ------------------------------------------------------------ discounting
def annuity_factor(r: float, n: int) -> float:
    return n if r == 0 else (1 - (1 + r) ** -n) / r


def per_line_npv(cost, h, r, life=HEALTH.benefit_life_yrs):
    return h * annuity_factor(r, life) - cost


def breakeven_harm(cost, r, life=HEALTH.benefit_life_yrs):
    return cost / annuity_factor(r, life)


def financed_cost_pv(cost, i_b, r, term=FIN.loan_term):
    pmt = cost * i_b / (1 - (1 + i_b) ** -term) if i_b > 0 else cost / term
    return pmt * annuity_factor(r, term)


# ---------------------------------------------- split-ownership mechanism
def consent_prob(cost_share, theta, private_benefit, hassle, scale, cost=COST.c_block_marginal):
    """P(owner consents) = Logistic((theta*B - s*c - kappa_bar)/sigma)."""
    u = (theta * private_benefit - cost_share * cost - hassle) / scale
    return 1 / (1 + np.exp(-u))


# --------------------------------------------------- inequality metrics
def gini(x, w=None) -> float:
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.r_[0, np.cumsum(w) / w.sum()]
    cxw = np.r_[0, np.cumsum(x * w) / (x * w).sum()]
    return 1 - np.sum((cw[1:] - cw[:-1]) * (cxw[1:] + cxw[:-1]))


def atkinson(x, eps: float, w=None) -> float:
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    w = w / w.sum()
    mu = np.sum(w * x)
    ede = np.exp(np.sum(w * np.log(x))) if np.isclose(eps, 1.0) else np.sum(w * x ** (1 - eps)) ** (1 / (1 - eps))
    return 1 - ede / mu


def concentration_index(burden, rank_var, w) -> float:
    """Wagstaff-style concentration index of `burden` ranked by `rank_var`
    (ascending). Positive = burden concentrated among high rank_var."""
    o = np.argsort(rank_var)
    b, w = np.asarray(burden, float)[o], np.asarray(w, float)[o]
    cw = np.cumsum(w) / w.sum()
    frac_rank = cw - 0.5 * w / w.sum()
    mu = np.average(b, weights=w)
    return 2 * np.average((b - mu) * (frac_rank - 0.5), weights=w) / mu
