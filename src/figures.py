"""
figures.py
All manuscript figures. Typeface: Times New Roman for every glyph (text,
tick labels, math). Palette: basic professional colors; every filled series
also carries a distinct hatch so figures read in grayscale and for color-vision
deficiency, and every panel has a key.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram

from config import COST, FIN, HEALTH, INV, OPS, DATA_PROC, FIG_DIR
from welfare import iq_loss, cvd_excess_deaths, annuity_factor, unit_values

# ------------------------------------------------------------------ style
NAVY, RED, GREEN, GOLD, GRAY, BLACK = "#1F3A68", "#8B1A1A", "#2E6B30", "#B8860B", "#6E6E6E", "#000000"
LIGHT = "#D9D9D9"
SERIES = [NAVY, RED, GREEN, GOLD, GRAY, BLACK]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "++"]
LS = ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5)), (0, (1, 1))]
MK = ["o", "s", "^", "D", "v", "P"]
QFILL = ["#F2F2F2", "#C6D3E6", "#8FA8CC", "#4F6FA3", NAVY]           # SVI quintiles, light to dark
QHATCH = ["", "...", "///", "xxx", ""]
QLAB = ["Q1 (least vulnerable)", "Q2", "Q3", "Q4", "Q5 (most vulnerable)"]
SEQ5 = ["#F2F2F2", "#C6D3E6", "#8FA8CC", "#4F6FA3", NAVY]

plt.rcParams.update({
    "font.family": "Times New Roman", "font.serif": ["Times New Roman"], "font.size": 9.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold", "mathtext.cal": "Times New Roman:italic",
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.edgecolor": BLACK, "axes.linewidth": 0.7, "axes.grid": True, "grid.color": "#E3E3E3",
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "legend.frameon": True,
    "legend.framealpha": 0.95, "legend.edgecolor": "#BBBBBB", "legend.fancybox": False,
    "hatch.linewidth": 0.6, "hatch.color": BLACK, "lines.linewidth": 1.6, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220)
    plt.close(fig)


def _classed(ax, gdf, col, bins, labels, colors, title, hatches=None, missing="No data", legend_loc="lower left",
             key_title=None):
    v = gdf[col]
    cls = pd.cut(v, bins=bins, labels=False, include_lowest=True)
    for i, c in enumerate(colors):
        sub = gdf[cls == i]
        if len(sub):
            edge = "#9A9A9A" if c == "white" else "white"
            sub.plot(ax=ax, color=c, edgecolor=edge, linewidth=0.15 if c != "white" else 0.25,
                     hatch=(hatches[i] if hatches else None))
    na = gdf[cls.isna()]
    if len(na):
        na.plot(ax=ax, color="white", edgecolor=GRAY, linewidth=0.15, hatch="////")
    handles = [Patch(facecolor=c, edgecolor=BLACK, linewidth=0.4, hatch=(hatches[i] if hatches else None), label=l)
               for i, (c, l) in enumerate(zip(colors, labels))]
    if len(na):
        handles.append(Patch(facecolor="white", edgecolor=GRAY, hatch="////", label=missing))
    ax.legend(handles=handles, loc=legend_loc, fontsize=7, title=key_title, title_fontsize=7,
              handlelength=1.6, borderpad=0.4)
    ax.set_title(title, loc="left")
    ax.set_axis_off()


def _quantile_bins(v, k=5):
    q = np.nanquantile(v, np.linspace(0, 1, k + 1))
    q[0], q[-1] = -np.inf, np.inf
    return q


# ===================================================================== maps
def fig_inventory_maps(t: gpd.GeoDataFrame):
    t = t.copy()
    t["share_action"] = 100 * t["n_action"] / t["lines_total"]
    t.loc[t["lines_total"] < 25, "share_action"] = np.nan
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 8.6))
    _classed(axs[0, 0], t, "share_action", [-np.inf, 50, 70, 85, 95, np.inf],
             ["< 50%", "50-70%", "70-85%", "85-95%", "> 95%"], SEQ5,
             "(a) Lines requiring replacement", missing="< 25 lines", key_title="Lines requiring action (%)")
    b = _quantile_bins(t["svi"])
    _classed(axs[0, 1], t, "svi", b, [f"{lo:.2f}-{hi:.2f}" if np.isfinite(lo) and np.isfinite(hi) else
                                       (f"< {hi:.2f}" if not np.isfinite(lo) else f"> {lo:.2f}")
                                       for lo, hi in zip(b[:-1], b[1:])], SEQ5, "(b) CDC/ATSDR SVI 2022",
             key_title="SVI (percentile, 0-1)")
    b = _quantile_bins(t["kids_per_line"])
    _classed(axs[1, 0], t, "kids_per_line", b, [f"{lo:.2f}-{hi:.2f}" if np.isfinite(lo) and np.isfinite(hi) else
                                                 (f"< {hi:.2f}" if not np.isfinite(lo) else f"> {lo:.2f}")
                                                 for lo, hi in zip(b[:-1], b[1:])], SEQ5,
             "(c) Children under 6 per service line", key_title="Children < 6 per line")
    t["holc_dom"] = np.where(t["holc_graded"] < 0.5, 4,
                             t[["holc_A", "holc_B", "holc_C", "holc_D"]].values.argmax(axis=1))
    _classed(axs[1, 1], t, "holc_dom", [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
             ["A  Best", "B  Still desirable", "C  Declining", "D  Hazardous", "Ungraded (< 50%)"],
             [GREEN, NAVY, GOLD, RED, "white"], "(d) HOLC grade, 1930s (dominant)",
             hatches=["", "", "...", "///", ""], key_title="HOLC grade (1930s)")
    for ax in axs.flat:
        for c in ax.collections:
            pass
    fig.text(0.02, 0.03, "Sources: CDWM 2025 inventory via Inside Climate News/Grist; CDC/ATSDR SVI 2022; ACS 2019-2023; "
             "Mapping Inequality (Univ. of Richmond).", fontsize=6.5, color=GRAY)
    fig.subplots_adjust(wspace=0.0, hspace=0.06, left=0.0, right=1.0, top=0.97, bottom=0.05)
    save(fig, "fig01_inventory_maps")


def zone_polygons(z: pd.DataFrame, t: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    city = t.geometry.union_all() if hasattr(t.geometry, "union_all") else t.unary_union
    pts = MultiPoint([(x, y) for x, y in zip(z["x"], z["y"])])
    vor = voronoi_diagram(pts, envelope=city.envelope.buffer(2000))
    polys = list(vor.geoms)
    gz = gpd.GeoDataFrame(geometry=polys, crs=t.crs)
    pz = gpd.GeoDataFrame({"zone": z.index.values}, geometry=gpd.points_from_xy(z["x"].values, z["y"].values), crs=t.crs)
    gz = gpd.sjoin(gz, pz, predicate="contains").drop(columns="index_right")
    gz["geometry"] = gz.geometry.intersection(city)
    return gz.set_index("zone")


def fig_completion_maps(R, ev, z, t):
    gz = zone_polygons(z, t)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.9))
    bins = [0, 2032, 2036, 2040, 2044, 2060]
    labs = ["by 2031", "2032-35", "2036-39", "2040-43", "2044 or later"]
    for ax, k, ttl in zip(axs, ["SQ", "HW", "EW"], ["(a) Status quo", "(b) Health-welfare", "(c) Equity-weighted"]):
        W = pd.Series(ev[k]["W"], index=z.index) + INV.base_year - 1
        g = gz.join(W.rename("yr"))
        _classed(ax, g, "yr", bins, labs, SEQ5[::-1], ttl, hatches=["", "", "", "...", "///"])
        ax.get_legend().remove()
    handles = [Patch(facecolor=c, edgecolor=BLACK, linewidth=0.4, hatch=h, label=l)
               for c, h, l in zip(SEQ5[::-1], ["", "", "", "...", "///"], labs)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, title="Expected (line-weighted mean) replacement year in the zone",
               title_fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=0.93, bottom=0.14)
    save(fig, "fig07_completion_maps")


# ============================================================== validation
def fig_validation(ca: pd.DataFrame, val: dict):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    d = ca.dropna(subset=["bll_elev_2009_13"])
    ax = axs[0]
    ax.scatter(d["svi"], d["bll_elev_2009_13"], s=16, facecolor="white", edgecolor=NAVY, linewidth=0.9,
               label=f"Community area (n = {len(d)})", zorder=3)
    b = np.polyfit(d["svi"], d["bll_elev_2009_13"], 1)
    xs = np.linspace(d["svi"].min(), d["svi"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=RED, linestyle="--", label="OLS fit")
    r = val["rho_svi_2009_13"]
    ax.text(0.04, 0.70, f"Spearman $\\rho$ = {r[0]:.2f}\n95% CI [{r[1]:.2f}, {r[2]:.2f}]", transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, bbox=dict(facecolor="white", edgecolor="#BBBBBB", linewidth=0.5, pad=2))
    ax.set_xlabel("Mean CDC SVI (line-weighted)")
    ax.set_ylabel("Children screened with elevated BLL (%)")
    ax.set_title("(a) SVI vs. BLL, 2009-13", loc="left")
    ax.legend(loc="upper left", fontsize=7)

    ax = axs[1]
    h = val["holc"].reindex(["A", "B", "C", "D", "Ungraded"])
    cols = [GREEN, NAVY, GOLD, RED, "white"]
    for i, (g, rr) in enumerate(h.iterrows()):
        ax.bar(i, 100 * rr.action_share, color=cols[i], edgecolor=BLACK, linewidth=0.6, hatch=["", "", "...", "///", "xx"][i],
               label=f"{g} (n = {int(rr.tracts)})", width=0.7)
    ax.set_xticks(range(5), ["A", "B", "C", "D", "None"])
    ax.set_ylim(0, 168)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Dominant HOLC grade")
    ax.set_ylabel("Lines requiring replacement (%)")
    ax.set_title("(b) Inventory by HOLC grade", loc="left")
    ax.legend(fontsize=6.3, loc="upper center", ncol=3, title="Grade (number of tracts)", title_fontsize=6.3,
              handlelength=1.3, columnspacing=0.6, borderpad=0.3)

    ax = axs[2]
    cal = val["imputation"]["calibration"]
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=":", label="Perfect calibration")
    ax.plot(cal["p"], cal["y"], color=NAVY, marker="o", markersize=4, label="Model (deciles)")
    ax.text(0.04, 0.95, f"Spatial-CV AUC = {val['imputation']['auc']:.3f}\nBrier = {val['imputation']['brier']:.3f}"
            f" (null {val['imputation']['brier_null']:.3f})", transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlabel("Predicted P(lead)")
    ax.set_ylabel("Observed lead share")
    ax.set_title("(c) Unknown-material model", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig02_validation")


# ============================================================== functions
def fig_damage(z: pd.DataFrame):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axs[0]
    bll = np.linspace(0.5, 10, 300)
    for i, (bta, lab) in enumerate([(HEALTH.iq_beta, "$\\beta$ = 3.25 (corrected)"), (HEALTH.iq_beta_low, "$\\beta$ = 2.70 (original)")]):
        ax.plot(bll, -bta * np.log(bll), color=SERIES[i], linestyle=LS[i], label=lab)
    ax.axvspan(HEALTH.child_bll0, HEALTH.child_bll0 + HEALTH.child_dbll, color=GOLD, alpha=0.25, lw=0,
               label="Assumed LSL increment")
    ax.axvline(3.5, color=GRAY, linestyle=":", linewidth=1)
    ax.text(3.65, -7.2, "CDC BLRV 3.5", fontsize=7, color=GRAY)
    ax.set_xlabel("Child blood lead level (µg/dL)")
    ax.set_ylabel("IQ change relative to BLL = 1")
    ax.set_title("(a) Child IQ dose-response", loc="left")
    ax.set_ylim(-7.5, 2.5)
    ax.legend(fontsize=6.8, loc="upper right")

    ax = axs[1]
    db = np.linspace(0, 1.0, 100)
    for i, (bta, lab) in enumerate([(HEALTH.cvd_beta_low, "Aoki et al. (2016), central"),
                                    (HEALTH.cvd_beta_high, "Lanphear et al. (2018), high")]):
        ax.plot(db, 1e5 * cvd_excess_deaths(bta, HEALTH.adult_bll0, db), color=SERIES[i], linestyle=LS[i], label=lab)
    ax.axvline(HEALTH.adult_dbll, color=GRAY, linestyle=":", linewidth=1)
    ax.text(HEALTH.adult_dbll + 0.02, 1, "assumed\nincrement", fontsize=7, color=GRAY)
    ax.set_xlabel("Adult BLL increment from LSL (µg/dL)")
    ax.set_ylabel("Excess CVD deaths per 100,000 adults")
    ax.set_title("(b) Adult CVD mortality", loc="left")
    ax.legend(fontsize=6.8, loc="upper left")

    ax = axs[2]
    vc, va = unit_values(None, "central")
    q = z.groupby("svi_q").apply(lambda g: pd.Series({
        "child": np.average(g.lead_frac * g.kids_per_line * vc, weights=g.n_lsl),
        "adult": np.average(g.lead_frac * g.adults_per_line * va, weights=g.n_lsl)}))
    x = np.arange(1, 6)
    ax.bar(x, q["child"] / 1e3, color=NAVY, edgecolor=BLACK, linewidth=0.5, label="Child IQ, ADHD, special ed.", width=0.65)
    ax.bar(x, q["adult"] / 1e3, bottom=q["child"] / 1e3, color="white", edgecolor=BLACK, linewidth=0.5,
           hatch="///", label="Adult CVD mortality", width=0.65)
    ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("SVI quintile (Q5 = most vulnerable)")
    ax.set_ylabel("Harm per line-year (\\$ thousand)")
    ax.set_title("(c) Monetized harm $h_z$", loc="left")
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig03_damage_functions")


def fig_cost():
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    n = np.linspace(60, 2500, 300)
    ax.plot(n, (COST.c_block_marginal + COST.mobilization / n) / 1e3, color=NAVY, label="Block program, $c^B + F/n$")
    ax.axhline(COST.c_oneoff / 1e3, color=RED, linestyle="--", label="One-off replacement, $c^O$")
    ns = COST.mobilization / (COST.c_oneoff - COST.c_block_marginal)
    ax.axvline(ns, color=GRAY, linestyle=":", linewidth=1)
    ax.text(ns + 40, 52, f"break-even\nbatch $n^*$ = {ns:.0f}", fontsize=7.5, color=GRAY)
    ax.scatter([500], [34], marker="D", s=30, color=GOLD, edgecolor=BLACK, zorder=4, label="Reported block average (\\$34k)")
    ax.set_ylim(28, 60)
    ax.set_xlabel("Lines per zone-year mobilization $n$")
    ax.set_ylabel("Average cost (\\$ thousand per line)")
    ax.set_title("Mobilization economies of scale", loc="left")
    ax.legend(fontsize=7.2, loc="upper right")
    save(fig, "fig04_cost_function")


# ============================================================ status quo
def fig_status_quo(z: pd.DataFrame, cons: dict, refc: dict):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.05, 1.2, 0.9]})
    q = z.groupby("svi_q").apply(lambda g: pd.Series({
        "repl": 1000 * g.replaced_2021_25.sum() / g.n_lsl.sum(), "ref": 1000 * g.refused.sum() / g.n_lsl.sum()}))
    x = np.arange(1, 6)
    ax = axs[0]
    ax.bar(x - 0.19, q["repl"], 0.36, color=NAVY, edgecolor=BLACK, linewidth=0.5, label="Replaced, 2021-25")
    ax.bar(x + 0.19, q["ref"], 0.36, color="white", edgecolor=RED, linewidth=0.8, hatch="///", label="Access refused")
    ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("SVI quintile")
    ax.set_ylabel("Per 1,000 lines requiring action")
    ax.set_title("(a) Observed activity", loc="left")
    ax.set_ylim(0, 42)
    ax.legend(fontsize=7, loc="upper left")
    ax = axs[1]
    ct = cons["table"]
    ct = ct[(ct.model == "rate") & (ct.index != "const")]
    yy = np.arange(len(ct))[::-1]
    irr, lo, hi = np.exp(ct["coef"]), np.exp(ct["lo"]), np.exp(ct["hi"])
    sig = ct["p"] < 0.05
    ax.errorbar(irr, yy, xerr=[irr - lo, hi - irr], fmt="none", ecolor=BLACK, elinewidth=0.8, capsize=2)
    ax.scatter(irr[sig], yy[sig.values], marker="s", s=26, color=RED, zorder=3, label="p < 0.05")
    ax.scatter(irr[~sig], yy[~sig.values], marker="o", s=24, facecolor="white", edgecolor=NAVY, zorder=3, label="p $\\geq$ 0.05")
    ax.axvline(1, color=GRAY, linestyle=":")
    ax.set_yticks(yy, [s.replace(" (per 10 pp)", "") for s in ct.index], fontsize=7.5)
    ax.set_xlabel("Incidence-rate ratio of refusals (95% CI)")
    ax.set_title("(b) Refusal rate model", loc="left")
    ax.legend(fontsize=7, loc="lower left")
    ax = axs[2]
    bq = pd.Series(refc["by_q"]).sort_index()
    ax.bar(bq.index, bq.values, color=[QFILL[i - 1] for i in bq.index], edgecolor=BLACK, linewidth=0.5,
           hatch=None, width=0.65)
    for i, (k, v) in enumerate(bq.items()):
        ax.bar(k, v, color=QFILL[k - 1], edgecolor=BLACK, linewidth=0.5, hatch=QHATCH[k - 1], width=0.65)
    ax.set_xticks(bq.index, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("SVI quintile")
    ax.set_ylabel("PV harm of refusals (\\$ billion)")
    ax.set_title("(c) Cost of refusals", loc="left")
    ax.set_ylim(0, 2.3)
    ax.legend(handles=[Patch(facecolor=QFILL[i], edgecolor=BLACK, hatch=QHATCH[i], label=["Q1", "Q2", "Q3", "Q4", "Q5"][i])
                       for i in range(5)], fontsize=6.5, loc="upper left", ncol=2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig05_status_quo")


# ============================================================== schedules
def fig_schedules(R, z):
    keys = [("SQ", "(a) Status quo"), ("CM", "(b) Cost-minimizing"), ("HW", "(c) Health-welfare"),
            ("EW", "(d) Equity-weighted")]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    for ax, (k, ttl) in zip(axs.flat, keys):
        X = R[k]["X"].reindex(z.index).fillna(0)
        L = X.mul(z["n_lsl"], axis=0)
        yrs = INV.base_year - 1 + np.array(L.columns)
        bottom = np.zeros(len(yrs))
        for qq in range(1, 6):
            v = L[z["svi_q"] == qq].sum(axis=0).values / 1e3
            ax.bar(yrs, v, bottom=bottom, color=QFILL[qq - 1], edgecolor=BLACK, linewidth=0.3,
                   hatch=QHATCH[qq - 1], width=0.85)
            bottom += v
        ax.axvline(INV.base_year + INV.horizon - 0.5, color=RED, linestyle="--", linewidth=1)
        ax.set_title(ttl, loc="left")
        ax.set_xlim(INV.base_year - 1, INV.base_year + 22)
        ax.set_ylim(0, 26)
    for ax in axs[:, 0]:
        ax.set_ylabel("Lines replaced (thousand)")
    for ax in axs[1, :]:
        ax.set_xlabel("Year")
    handles = [Patch(facecolor=QFILL[i], edgecolor=BLACK, linewidth=0.4, hatch=QHATCH[i], label=QLAB[i]) for i in range(5)][::-1]
    handles.append(Line2D([0], [0], color=RED, linestyle="--", label="LCRI deferred deadline"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5, title="SVI quintile",
               title_fontsize=8)
    fig.tight_layout()
    save(fig, "fig06_schedules")


def fig_completion_paths(R, z):
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    N = z["n_lsl"].values
    ej = z["ej"].values
    T = INV.horizon + 3
    yrs = INV.base_year - 1 + np.arange(1, T + 1)
    pace = np.maximum(0, np.arange(1, T + 1) / INV.horizon - OPS.pace_slack).clip(max=1)
    ax.plot(yrs, 100 * pace, color=GRAY, linestyle=":", label="LCRI pace floor")
    for i, (k, lab) in enumerate([("SQ", "Status quo"), ("HW", "Health-welfare")]):
        X = R[k]["X"].reindex(z.index).fillna(0).reindex(columns=range(1, T + 1), fill_value=0)
        C = X.mul(N, axis=0).cumsum(axis=1)
        ax.plot(yrs, 100 * C[ej].sum().values / N[ej].sum(), color=SERIES[i], linestyle="-", marker=MK[i],
                markersize=3, label=f"{lab}: EJ zones")
        ax.plot(yrs, 100 * C[~ej].sum().values / N[~ej].sum(), color=SERIES[i], linestyle="--", marker=MK[i],
                markersize=3, markerfacecolor="white", label=f"{lab}: other zones")
    ax.set_xticks(np.arange(INV.base_year, INV.base_year + T + 1, 4))
    ax.set_ylabel("Cumulative share replaced (%)")
    ax.set_xlabel("Year")
    ax.set_title("Completion paths, environmental-justice vs. other zones", loc="left")
    ax.legend(fontsize=7.2, loc="lower right")
    save(fig, "fig08_completion_paths")


def fig_frontier(T: pd.DataFrame, F: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    eff = F[F.branch == "efficient"].sort_values("q5_gap")
    dom = F[F.branch == "dominated"].sort_values("q5_gap")
    pts = [("SQ", "X", RED, "Status quo"), ("PROP", "P", GOLD, "Proportional pace"),
           ("PRIO", "*", GRAY, "Harm priority, one-off"), ("CM", "^", GREEN, "Cost-minimizing"),
           ("HW", "D", BLACK, "Health-welfare"), ("EW", "v", NAVY, "Equity-weighted, linear"),
           ("EWc", "<", GOLD, "Equity-weighted, convex"), ("DW", ">", RED, "Distributional weights")]
    for j, ax in enumerate(axs):
        ax.plot(eff.q5_gap, eff.pv_social_bn, color=NAVY, marker="o", markersize=4,
                label="Efficient branch ($G \\leq \\varepsilon$)")
        ax.plot(dom.q5_gap, dom.pv_social_bn, color=GRAY, marker="s", markersize=3.5, linestyle="--",
                markerfacecolor="white", label="Dominated branch ($G \\geq \\varepsilon$)")
        for k, mk, c, lab in pts:
            if k in T.index:
                ax.scatter(T.loc[k, "q5_gap"], T.loc[k, "pv_social_bn"], marker=mk, s=52 if mk != "*" else 90,
                           color=c, edgecolor=BLACK, linewidth=0.5, zorder=5, label=lab)
        ax.axvline(0, color=GRAY, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Q5 wait gap $G$ (years)")
    axs[0].set_ylabel("PV social cost $H + C$ (\\$ billion)")
    axs[0].set_title("(a) All schedules", loc="left")
    lo = min(eff.pv_social_bn.min(), T.loc["HW", "pv_social_bn"]) - 0.15
    axs[1].set_ylim(lo, T.loc[["EW", "EWc", "DW"], "pv_social_bn"].max() + 0.25)
    axs[1].set_xlim(-8.2, 2.5)
    axs[1].set_title("(b) Optimized schedules (zoom)", loc="left")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.13), columnspacing=0.8)
    axs[0].text(0.03, 0.62, "negative $G$ = most vulnerable\nquintile served earlier", transform=axs[0].transAxes,
                fontsize=7, color=GRAY)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig09_frontier")


def fig_concentration(ev, z):
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    kids = z["kids_per_line"].values * z["n_lsl"].values
    o = np.argsort(-z["svi"].values)
    cw = np.r_[0, np.cumsum(kids[o]) / kids.sum()]
    for i, (k, lab) in enumerate([("SQ", "Status quo"), ("CM", "Cost-minimizing"), ("HW", "Health-welfare"),
                                  ("EW", "Equity-weighted")]):
        e = ev[k]["exp_pc"] * kids
        ce = np.r_[0, np.cumsum(e[o]) / e.sum()]
        ax.plot(cw, ce, color=SERIES[i], linestyle=LS[i], label=f"{lab} ($C$ = {ev[k]['conc_index_svi']:+.2f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=0.8, label="Equal burden")
    ax.set_xlabel("Cumulative share of children, most to least vulnerable")
    ax.set_ylabel("Cumulative share of exposure-years")
    ax.set_title("Concentration of residual exposure", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig10_concentration")


def fig_npv(npv: pd.DataFrame, TOR: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    cs = np.linspace(4_000, 45_000, 200)
    from analysis import zones
    zc = zones()
    from welfare import harm_per_line_year
    for i, (tier, lab) in enumerate([("child", "Child IQ only"), ("central", "Central"), ("high", "High CVD")]):
        h = np.average(harm_per_line_year(zc.kids_per_line, zc.adults_per_line, zc.lead_frac, 1.0, None, tier),
                       weights=zc.n_lsl)
        for j, r in enumerate((0.03,)):
            ax.plot(cs / 1e3, h * annuity_factor(r, HEALTH.benefit_life_yrs) / cs, color=SERIES[i], linestyle=LS[i],
                    label=f"{lab}")
    ax.axhline(1, color=BLACK, linewidth=0.9)
    for c, lab in [(4.7, "EPA\nnational"), (31, "Chicago\naverage"), (39, "Chicago\none-off")]:
        ax.axvline(c, color=GRAY, linestyle=":", linewidth=0.8)
        ax.text(c + 0.6, 0.56, lab, fontsize=6.5, color=GRAY, va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_ylim(0.5, 30)
    ax.set_xlabel("Cost per line (\\$ thousand)")
    ax.set_ylabel("Benefit-cost ratio ($r$ = 3%)")
    ax.set_title("(a) BCR by benefit tier", loc="left")
    ax.legend(fontsize=7, loc="upper right", title="Benefit tier", title_fontsize=7)
    ax.set_xlim(2, 46)
    ax = axs[1]
    T = TOR.sort_values("range")
    y = np.arange(len(T))
    base = T["base"].iloc[0]
    ax.barh(y, T["high"] - base, left=base, color=NAVY, edgecolor=BLACK, linewidth=0.5, height=0.6, label="Upper value")
    ax.barh(y, T["low"] - base, left=base, color="white", edgecolor=BLACK, linewidth=0.5, hatch="///", height=0.6,
            label="Lower value")
    ax.axvline(base, color=BLACK, linewidth=0.9)
    ax.axvline(1, color=RED, linestyle="--", linewidth=1, label="Break-even")
    ax.set_yticks(y, [s.split(" (")[0] for s in T["param"]], fontsize=7.5)
    ax.set_xlabel("Benefit-cost ratio at \\$31k, $r$ = 3%")
    ax.set_title("(b) One-way sensitivity", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig11_npv")


def fig_psa(D: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axs[0]
    sav = D["SQ_social"] - D["HW_social"]
    sp = D["CM_social"] - D["HW_social"]
    bins = np.linspace(min(sav.min(), sp.min()), max(sav.max(), sp.max()), 40)
    ax.hist(sav, bins=bins, color=NAVY, edgecolor=BLACK, linewidth=0.3, alpha=0.9, label="vs. status quo")
    ax.hist(sp, bins=bins, facecolor="none", edgecolor=RED, linewidth=0.6, hatch="///", label="vs. cost-minimizing")
    ax.axvline(0, color=BLACK, linewidth=0.9)
    ax.set_xlabel("PV saving of health-welfare schedule (\\$ billion)")
    ax.set_ylabel("Draws")
    ax.set_title(f"(a) Savings across {len(D):,} draws", loc="left")
    ax.legend(fontsize=7, loc="upper right")
    ax = axs[1]
    for i, r in enumerate((0.02, 0.03, 0.07)):
        v = np.sort(D.loc[D.rate == r, "bcr_31k"].values)
        ax.plot(v, np.linspace(0, 1, len(v)), color=SERIES[i], linestyle=LS[i], label=f"$r$ = {r:.0%}")
    ax.axvline(1, color=BLACK, linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_xlabel("Per-line benefit-cost ratio at \\$31k")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("(b) Distribution of the BCR", loc="left")
    ax.legend(fontsize=7, loc="lower right", title="Discount rate", title_fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig12_psa")


def make_all(R, ev, T, F, D, TOR, cons, refc, val, npv):
    from analysis import zones
    zc = zones()
    t = gpd.read_file(DATA_PROC / "tracts.gpkg")
    ca = pd.read_csv(DATA_PROC / "community_areas.csv")
    fig_inventory_maps(t)
    fig_validation(ca, val)
    fig_damage(zc)
    fig_cost()
    fig_status_quo(zc, cons, refc)
    fig_schedules(R, zc)
    fig_completion_maps(R, ev, zc, t)
    fig_completion_paths(R, zc)
    fig_frontier(T, F)
    fig_concentration(ev, zc)
    fig_npv(npv, TOR)
    fig_psa(D)
