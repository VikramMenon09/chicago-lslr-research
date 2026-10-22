"""
etl_pipeline.py
Builds the analysis tables from the raw public files fetched by fetch_data.py.

Outputs (data/processed/):
    lines.parquet      one row per service line (2025 CDWM inventory, geocoded)
    tracts.gpkg        tract features: inventory counts, ACS, CDC SVI, HOLC, CDPH
    zones.csv          optimization units (k-means corridors of lines requiring action)
    community_areas.csv  77 community areas with historical blood-lead screening
    imputation_cv.json cross-validated performance of the unknown-material model
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.neighbors import BallTree

from config import DATA_RAW, DATA_PROC, INV, SEED

CRS_LATLON, CRS_M = "EPSG:4326", "EPSG:26971"   # NAD83 / Illinois East (m)
ACTION = {"L", "GRR", "U"}                        # LCRI: replace, or verify then replace


# ----------------------------------------------------------- inventory
def load_inventory() -> pd.DataFrame:
    """CDWM 2025 inventory joined to the ICN/Grist geocodes by record index."""
    raw = pd.read_excel(DATA_RAW / "icn" / "2025_inventory.xlsx", header=None, dtype=str)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "FullAddress"][0]
    inv = raw.iloc[hdr + 1:].reset_index(drop=True)
    inv.columns = ["address", "lcr_site", "high_risk", "newly_identified", "yr_pws", "yr_cust",
                   "id_source", "gooseneck", "pws_material", "cust_material", "classification",
                   "galv_downstream", "date_notified", "prev_replaced", "date_replaced",
                   "replacement_material", "refused_access", "date_refused"]
    inv["row"] = np.arange(1, len(inv) + 1)
    geo = pd.concat([pd.read_csv(DATA_RAW / "icn" / f"service_lines{i}.csv", low_memory=False,
                                 usecols=["row", "lat", "long", "geoid",
                                          "classification_for_entire_service_line"])
                     for i in (1, 2)])
    df = inv.merge(geo, on="row", how="inner")
    assert (df["classification"] == df["classification_for_entire_service_line"]).mean() > 0.999
    df = df.drop(columns="classification_for_entire_service_line")
    df["geoid"] = df["geoid"].astype("Int64").astype(str)
    for c in ("high_risk", "prev_replaced", "refused_access"):
        df[c] = df[c].eq("Y")
    df["field_verified"] = df["id_source"].fillna("").str.contains("Field Verification")
    df["year_replaced"] = pd.to_datetime(df["date_replaced"], errors="coerce").dt.year
    df["action"] = df["classification"].isin(ACTION)
    df = df.dropna(subset=["lat", "long"])
    return df[["row", "lat", "long", "geoid", "classification", "action", "high_risk",
               "field_verified", "prev_replaced", "year_replaced", "refused_access"]]


# --------------------------------------------------------------- ACS
def _acs(table: str, cols: dict[str, str]) -> pd.DataFrame:
    a = pd.read_csv(DATA_RAW / "acs" / f"{table}.dat", sep="|", dtype=str)
    a["geoid"] = a["GEO_ID"].str[-11:]
    out = a[["geoid"]].copy()
    for src, name in cols.items():
        out[name] = pd.to_numeric(a[src], errors="coerce")
    return out


def load_acs() -> pd.DataFrame:
    pop = _acs("b01003", {"B01003_E001": "pop"})
    kids = _acs("b09001", {"B09001_E001": "u18", "B09001_E003": "u3", "B09001_E004": "a3_4",
                           "B09001_E005": "a5"})
    ten = _acs("b25003", {"B25003_E001": "occ_units", "B25003_E003": "renter_units"})
    uis = _acs("b25024", {"B25024_E001": "units", "B25024_E002": "u1det", "B25024_E003": "u1att",
                          "B25024_E004": "u2", "B25024_E005": "u3_4"})
    yb = _acs("b25034", {"B25034_E001": "units_yb", "B25034_E006": "yb1980s",
                         "B25034_E007": "yb1970s", "B25034_E008": "yb1960s", "B25034_E009": "yb1950s",
                         "B25034_E010": "yb1940s", "B25034_E011": "yb_pre1940"})
    a = pop.merge(kids, on="geoid").merge(ten, on="geoid").merge(uis, on="geoid").merge(yb, on="geoid")
    a["kids_u6"] = a["u3"] + a["a3_4"] + a["a5"]
    a["adults"] = (a["pop"] - a["u18"]).clip(lower=0)
    a["renter_share"] = a["renter_units"] / a["occ_units"].replace(0, np.nan)
    a["small_struct_share"] = (a["u1det"] + a["u1att"] + a["u2"] + a["u3_4"]) / a["units"].replace(0, np.nan)
    a["two_to_four_share"] = (a["u2"] + a["u3_4"]) / a["units"].replace(0, np.nan)
    a["pre1940_share"] = a["yb_pre1940"] / a["units_yb"].replace(0, np.nan)
    a["pre1986_share"] = (a[["yb1950s", "yb1960s", "yb1970s", "yb1940s", "yb_pre1940"]].sum(axis=1)
                          + 0.6 * a["yb1980s"]) / a["units_yb"].replace(0, np.nan)
    return a


# --------------------------------------------------------------- tracts
def load_tracts(lines: pd.DataFrame) -> gpd.GeoDataFrame:
    t = gpd.read_file(DATA_RAW / "icn" / "chicago_tracts_filled.geojson").to_crs(CRS_M)
    t["geoid"] = t["geoid"].astype(str)
    keep = ["geoid", "geometry", "pct_poverty", "median_household_income", "pct_hispanic",
            "pct_black_nonhispanic", "pct_white_nonhispanic", "pct_asian_nonhispanic",
            "pct_minority", "households", "pct_limited_or_no_english"]
    t = t[keep].merge(load_acs(), on="geoid", how="left")

    svi = pd.read_csv(DATA_RAW / "svi" / "SVI2022_Illinois.csv", dtype={"FIPS": str},
                      usecols=["FIPS", "RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"])
    svi = svi.rename(columns={"FIPS": "geoid", "RPL_THEMES": "svi"}).replace(-999, np.nan)
    t = t.merge(svi, on="geoid", how="left")

    holc = gpd.read_file(DATA_RAW / "holc" / "MIv3Areas_2020TractCrosswalk.gpkg",
                         columns=["city", "state", "grade", "GEOID", "pct_tract"], ignore_geometry=True)
    holc = holc[(holc["city"] == "Chicago") & (holc["state"] == "IL")]
    hs = holc.pivot_table(index="GEOID", columns="grade", values="pct_tract", aggfunc="sum").fillna(0)
    hs = hs.reindex(columns=list("ABCD"), fill_value=0).clip(upper=1)
    hs.columns = [f"holc_{c}" for c in hs.columns]
    hs["holc_graded"] = hs.sum(axis=1).clip(upper=1)
    t = t.merge(hs, left_on="geoid", right_index=True, how="left")
    t[[c for c in t.columns if c.startswith("holc_")]] = t.filter(like="holc_").fillna(0)

    ca = gpd.read_file(DATA_RAW / "icn" / "chicago_community_areas.geojson").to_crs(CRS_M)
    cent = t[["geoid"]].assign(geometry=t.geometry.representative_point())
    cent = gpd.GeoDataFrame(cent, crs=CRS_M)
    j = gpd.sjoin(cent, ca[["area_num_1", "community", "geometry"]], predicate="within")
    t = t.merge(j[["geoid", "area_num_1", "community"]], on="geoid", how="left")

    cnt = lines.groupby("geoid").agg(
        lines_total=("row", "size"), n_L=("classification", lambda s: (s == "L").sum()),
        n_GRR=("classification", lambda s: (s == "GRR").sum()),
        n_U=("classification", lambda s: (s == "U").sum()),
        n_NL=("classification", lambda s: (s == "NL").sum()),
        n_action=("action", "sum"), n_replaced_2021_25=("year_replaced", lambda s: s.between(2021, 2025).sum()),
        n_refused=("refused_access", "sum"), n_high_risk=("high_risk", "sum"),
        p_lead_U=("p_lead", lambda s: s[s.notna()].mean() if s.notna().any() else np.nan))
    t = t.merge(cnt, left_on="geoid", right_index=True, how="inner")
    t = t[t["lines_total"] >= 1].copy()

    # occupants served per line: small (1-4 unit) structures carry small-diameter services
    occ_lines = t["lines_total"].clip(lower=1)
    t["kids_per_line"] = (t["kids_u6"] * t["small_struct_share"].fillna(0) / occ_lines).clip(upper=1.5)
    t["adults_per_line"] = (t["adults"] * t["small_struct_share"].fillna(0) / occ_lines).clip(upper=8)
    return t


# --------------------------------------------- unknown-material imputation
def impute_unknowns(lines: pd.DataFrame, tract_feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """P(lead | unknown) from field-verified, never-replaced lines.

    Features: projected coordinates, share of lead among the 30 nearest verified
    neighbours (leave-one-out), and tract housing/demographic context. Spatially
    blocked 5-fold CV (folds = tracts) guards against neighbour leakage."""
    g = gpd.GeoDataFrame(lines, geometry=gpd.points_from_xy(lines["long"], lines["lat"]),
                         crs=CRS_LATLON).to_crs(CRS_M)
    lines = lines.assign(x=g.geometry.x.values, y=g.geometry.y.values)
    lines = lines.merge(tract_feats, on="geoid", how="left")
    train = lines[lines["field_verified"] & ~lines["prev_replaced"]
                  & lines["classification"].isin(["L", "GRR", "NL"])].copy()
    train["target"] = train["classification"].isin(["L", "GRR"]).astype(int)
    tree = BallTree(train[["x", "y"]].values)
    k = 30
    _, idx = tree.query(train[["x", "y"]].values, k=k + 1)
    tv = train["target"].values
    train["nn_lead"] = tv[idx[:, 1:]].mean(axis=1)                   # leave-one-out
    unk = lines["classification"].eq("U")
    _, idx_u = tree.query(lines.loc[unk, ["x", "y"]].values, k=k)
    lines.loc[unk, "nn_lead"] = tv[idx_u].mean(axis=1)

    feats = ["x", "y", "nn_lead", "pre1940_share", "pre1986_share", "small_struct_share",
             "median_household_income", "pct_black_nonhispanic", "pct_hispanic", "high_risk"]
    X, yv, grp = train[feats].astype(float), train["target"].values, train["geoid"].values
    oof = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(X, yv, grp):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                             random_state=SEED)
        clf.fit(X.iloc[tr], yv[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
    cv = {"n_train": int(len(train)), "base_rate": float(yv.mean()),
          "auc": float(roc_auc_score(yv, oof)), "brier": float(brier_score_loss(yv, oof)),
          "brier_null": float(brier_score_loss(yv, np.full_like(oof, yv.mean())))}
    # calibration by decile
    dec = pd.qcut(oof, 10, labels=False, duplicates="drop")
    cv["calibration"] = pd.DataFrame({"p": oof, "y": yv, "d": dec}).groupby("d").mean().round(4).to_dict("list")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         random_state=SEED).fit(X, yv)
    lines["p_lead"] = np.nan
    lines.loc[unk, "p_lead"] = clf.predict_proba(lines.loc[unk, feats].astype(float))[:, 1]
    lines.loc[lines["classification"].isin(["L", "GRR"]), "lead_prob"] = 1.0
    lines.loc[lines["classification"].eq("NL"), "lead_prob"] = 0.0
    lines.loc[unk, "lead_prob"] = lines.loc[unk, "p_lead"]
    return lines, cv


# ------------------------------------------------------------- zones
def build_zones(lines: pd.DataFrame, t: gpd.GeoDataFrame, k: int = INV.n_zones) -> tuple[pd.DataFrame, pd.DataFrame]:
    act = lines[lines["action"]].copy()
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10, batch_size=8192)
    act["zone"] = km.fit_predict(act[["x", "y"]].values)
    lines = lines.merge(act[["row", "zone"]], on="row", how="left")
    tf = t.set_index("geoid")
    cols = ["svi", "kids_per_line", "adults_per_line", "renter_share", "two_to_four_share",
            "median_household_income", "pct_hispanic", "pct_black_nonhispanic", "pct_poverty",
            "holc_D", "holc_graded", "pct_limited_or_no_english"]
    act = act.join(tf[cols], on="geoid", rsuffix="_t")
    z = act.groupby("zone").agg(
        n_lsl=("row", "size"), n_L=("classification", lambda s: (s == "L").sum()),
        n_GRR=("classification", lambda s: (s == "GRR").sum()),
        n_U=("classification", lambda s: (s == "U").sum()), lead_exp=("lead_prob", "sum"),
        x=("x", "mean"), y=("y", "mean"), lat=("lat", "mean"), lon=("long", "mean"),
        n_tracts=("geoid", "nunique"), high_risk=("high_risk", "sum"),
        **{c: (c, "mean") for c in cols})
    z["lead_frac"] = z["lead_exp"] / z["n_lsl"]
    # historical replacement and refusal counts, attributed to the zone of nearest action line
    tree = BallTree(act[["x", "y"]].values)
    _, nn = tree.query(lines[["x", "y"]].values, k=1)
    lines["zone_nn"] = act["zone"].values[nn[:, 0]]
    rep = lines[lines["year_replaced"].between(2021, 2025)].groupby("zone_nn").size()
    ref = lines[lines["refused_access"]].groupby("zone_nn").size()
    z["replaced_2021_25"] = rep.reindex(z.index).fillna(0).astype(int)
    z["refused"] = ref.reindex(z.index).fillna(0).astype(int)
    z["median_income"] = z.pop("median_household_income")
    z["owner_share"] = 1 - z["renter_share"]
    return lines, z


def build_community_areas(t: gpd.GeoDataFrame) -> pd.DataFrame:
    b = pd.read_csv(DATA_RAW / "cdph" / "bll_screening_ca_1999_2013.csv")
    yrs = range(2009, 2014)
    num = sum(b[f"elevated_blood_lead_level_in_{y}"].fillna(0) for y in yrs)
    den = sum(b[f"screened_for_lead_in_{y}"].fillna(0) for y in yrs)
    num99 = sum(b[f"elevated_blood_lead_level_in_{y}"].fillna(0) for y in range(1999, 2014))
    den99 = sum(b[f"screened_for_lead_in_{y}"].fillna(0) for y in range(1999, 2014))
    b = pd.DataFrame({"area_num_1": b["community_area_number"].astype(str),
                      "bll_elev_2009_13": 100 * num / den, "bll_elev_1999_13": 100 * num99 / den99,
                      "screened_2009_13": den})
    w = t["n_action"].clip(lower=1)
    agg = t.assign(w=w).groupby("area_num_1").apply(lambda g: pd.Series({
        "community": g["community"].iloc[0], "n_action": g["n_action"].sum(),
        "lines_total": g["lines_total"].sum(),
        "svi": np.average(g["svi"].fillna(g["svi"].mean()), weights=g["w"]),
        "kids_per_line": np.average(g["kids_per_line"], weights=g["w"]),
        "holc_D": np.average(g["holc_D"], weights=g["w"]),
        "median_income": np.average(g["median_household_income"].fillna(g["median_household_income"].median()),
                                    weights=g["w"])}))
    agg.index = agg.index.astype(str)
    return agg.merge(b, left_index=True, right_on="area_num_1", how="left")


def build_tables(n_zones: int = INV.n_zones, save: bool = True):
    lines = load_inventory()
    acs = load_acs()
    tf0 = gpd.read_file(DATA_RAW / "icn" / "chicago_tracts_filled.geojson")[
        ["geoid", "median_household_income", "pct_black_nonhispanic", "pct_hispanic"]]
    tf0["geoid"] = tf0["geoid"].astype(str)
    tf0 = tf0.merge(acs[["geoid", "pre1940_share", "pre1986_share", "small_struct_share"]], on="geoid")
    lines = lines[lines["geoid"].isin(set(tf0["geoid"]))]          # inside the city footprint
    lines, cv = impute_unknowns(lines, tf0)
    lines = lines[[c for c in lines.columns if c not in tf0.columns or c == "geoid"]]
    t = load_tracts(lines)
    lines, z = build_zones(lines, t, n_zones)
    ca = build_community_areas(t)
    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        lines.drop(columns=[c for c in ("nn_lead",) if c in lines]).to_parquet(DATA_PROC / "lines.parquet")
        t.to_file(DATA_PROC / "tracts.gpkg", driver="GPKG")
        z.to_csv(DATA_PROC / "zones.csv")
        ca.to_csv(DATA_PROC / "community_areas.csv", index=False)
        json.dump(cv, open(DATA_PROC / "imputation_cv.json", "w"), indent=2)
    return lines, t, z, ca, cv


if __name__ == "__main__":
    lines, t, z, ca, cv = build_tables()
    print(f"lines: {len(lines):,}  action: {lines.action.sum():,}  tracts: {len(t)}  zones: {len(z)}")
    print(f"expected lead among action lines: {z.lead_exp.sum():,.0f}")
    print("imputation CV:", {k: v for k, v in cv.items() if k != "calibration"})
