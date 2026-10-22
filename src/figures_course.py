"""
figures_course.py
Re-renders every figure to the CEE 176E figure rules:
  * black Cambria-metric type (Caladea, the metric-compatible open twin of Cambria)
  * no major or minor gridlines; black outline around the data area
  * outer (outward) tick marks
  * no chart titles (the caption carries the description); panel letters only
  * legends inside the data area where they fit
  * axis labels with units
Output: figures/course/<name>.png (600 dpi) and .pdf
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figures as F
from config import FIG_DIR

OUT = FIG_DIR / "course"
FONT = "Caladea"

LABELS = {
    "Mean CDC SVI (line-weighted)": "Mean CDC SVI (percentile rank, 0-1)",
    "Children screened with elevated BLL (%)": "Screened children with elevated BLL (%)",
    "Predicted P(lead)": "Predicted probability of lead (0-1)",
    "Observed lead share": "Observed lead share (0-1)",
    "IQ change relative to BLL = 1": "IQ change vs. 1 µg/dL (IQ points)",
    "Excess CVD deaths per 100,000 adults": "Excess deaths (per 100,000 adults/yr)",
    "SVI quintile (Q5 = most vulnerable)": "SVI quintile (Q5 = most vulnerable)",
    "Harm per line-year (\\$ thousand)": "Harm (\\$ thousand per line-year)",
    "Lines per zone-year mobilization $n$": "Batch size per mobilization, $n$ (lines)",
    "Average cost (\\$ thousand per line)": "Average cost (\\$ thousand per line)",
    "Per 1,000 lines requiring action": "Events (per 1,000 lines requiring action)",
    "SVI quintile": "SVI quintile (Q5 = most vulnerable)",
    "Incidence-rate ratio of refusals (95% CI)": "Incidence-rate ratio (ratio, 95% CI)",
    "PV harm of refusals (\\$ billion)": "PV harm of refusals (\\$ billion)",
    "Lines replaced (thousand)": "Lines replaced (thousand per year)",
    "Cumulative share replaced (%)": "Cumulative share replaced (%)",
    "Q5 wait gap $G$ (years)": "Q5 wait-time gap, $G$ (years)",
    "Cumulative share of children, most to least vulnerable": "Cumulative share of children (fraction)",
    "Cumulative share of exposure-years": "Cumulative share of exposure-years (fraction)",
    "Cost per line (\\$ thousand)": "Cost per line (\\$ thousand)",
    "Benefit-cost ratio ($r$ = 3%)": "Benefit-cost ratio at $r$ = 3% (ratio)",
    "Benefit-cost ratio at \\$31k, $r$ = 3%": "Benefit-cost ratio at \\$31k and $r$ = 3% (ratio)",
    "PV saving of health-welfare schedule (\\$ billion)": "PV saving of health-welfare schedule (\\$ billion)",
    "Draws": "Monte Carlo draws (count)",
    "Per-line benefit-cost ratio at \\$31k": "Benefit-cost ratio at \\$31k (ratio)",
    "Cumulative probability": "Cumulative probability (fraction)",
    "Child blood lead level (µg/dL)": "Child blood lead level (µg/dL)",
    "Adult BLL increment from LSL (µg/dL)": "Adult BLL increment from LSL (µg/dL)",
    "Dominant HOLC grade": "Dominant HOLC grade of tract",
}


def _style():
    plt.rcParams.update({
        "font.family": FONT, "font.serif": [FONT], "mathtext.fontset": "custom",
        "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.titlecolor": "black", "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "axes.grid": False, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.top": False, "ytick.right": False,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "legend.edgecolor": "black", "legend.framealpha": 1.0, "legend.fancybox": False,
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "axes.titleweight": "bold", "axes.titlesize": 10,
    })


def _finalize(fig):
    for ax in fig.axes:
        if not ax.axison:                                   # maps: keep letter, drop descriptive title
            t = ax.get_title(loc="left")
            m = re.match(r"\s*(\([a-z]\))", t)
            ax.set_title(m.group(1) if m else "", loc="left", fontsize=10, fontweight="bold")
            continue
        if not ax.get_xlabel() and not ax.get_ylabel() and not ax.lines and not ax.patches and not ax.collections:
            continue
        ax.grid(False, which="both")
        ax.minorticks_off() if ax.get_xscale() == "linear" and ax.get_yscale() == "linear" else None
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("black")
            s.set_linewidth(0.8)
        ax.tick_params(which="both", direction="out", top=False, right=False, colors="black")
        ax.set_xlabel(LABELS.get(ax.get_xlabel(), ax.get_xlabel()), color="black")
        ax.set_ylabel(LABELS.get(ax.get_ylabel(), ax.get_ylabel()), color="black")
        t = ax.get_title(loc="left")
        m = re.match(r"\s*(\([a-z]\))", t)
        ax.set_title("", loc="left")
        if m:
            ax.text(-0.01, 1.02, m.group(1), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, fontweight="bold", color="black")
        for txt in ax.texts:
            txt.set_color("black")
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(0.6)
            for tx in leg.get_texts():
                tx.set_color("black")
    for leg in fig.legends:
        leg.get_frame().set_edgecolor("black")
        for tx in leg.get_texts():
            tx.set_color("black")
    for txt in fig.texts:
        txt.set_color("black")


def _save(fig, name):
    _finalize(fig)
    if all(ax.axison for ax in fig.axes) and not fig.legends:
        fig.tight_layout(w_pad=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def make_all(*args):
    _style()
    F.save = _save
    F.LIGHT = "#D9D9D9"
    F.make_all(*args)


if __name__ == "__main__":
    import pandas as pd
    import analysis as A
    R = A.load_results()
    T, ev = A.scenario_table(R)
    Fr = A.frontier_df(T)
    D = pd.read_csv(A.DATA_PROC / "psa_draws.csv")
    TOR = A.tornado()
    cons = A.consent_models()
    refc = A.refusal_cost(R)
    val = A.validation()
    npv, fin, hm = A.npv_tables()
    make_all(R, ev, T, Fr, D, TOR, cons, refc, val, npv)
    print("course figures written to", OUT)
