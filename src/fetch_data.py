"""
fetch_data.py
Downloads every raw input from its original public source into data/raw/.
Standard library only, so it runs before the virtual environment exists.

    python3 src/fetch_data.py          # ~250 MB download, a few minutes

Sources (pinned where versioned):
  1. 2025 Chicago service line inventory (CDWM submission to Illinois EPA, 491,705
     records) and its geocoded, tract-matched version, published by Inside Climate
     News / Grist / WBEZ (Aug. 2025), repository commit pinned below.
  2. CDC/ATSDR Social Vulnerability Index 2022, Illinois tracts.
  3. ACS 2019-2023 5-year table-based summary files (B01003, B09001, B25003,
     B25024, B25034), filtered to Cook County tracts while streaming.
  4. Mapping Inequality (HOLC) to 2020 tract crosswalk, Univ. of Richmond DSL.
  5. CDPH childhood blood-lead screening by community area, 1999-2013
     (City of Chicago Data Portal, dataset v2z5-jyrq).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

ICN = ("https://raw.githubusercontent.com/InsideClimateNews/2025-08-chicago-lead-service-lines/"
       "5b3a5d647e290d7203f6854418d94ceca8e54f0c")
HOLC = ("https://raw.githubusercontent.com/americanpanorama/mapping-inequality-census-crosswalk/"
        "44f372fba7bcececc955ae129e78e2dc52be3ec4")
ACS = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData"

FILES = {
    "icn/2025_inventory.xlsx": f"{ICN}/data/2025_inventory.xlsx",
    "icn/service_lines1.csv": f"{ICN}/processed_data/service_lines1.csv",
    "icn/service_lines2.csv": f"{ICN}/processed_data/service_lines2.csv",
    "icn/chicago_tracts_filled.geojson": f"{ICN}/processed_data/chicago_tracts_filled.geojson",
    "icn/chicago_community_areas.geojson": f"{ICN}/processed_data/chicago_community_areas.geojson",
    "svi/SVI2022_Illinois.csv": "https://svi.cdc.gov/Documents/Data/2022/csv/states/Illinois.csv",
    "holc/MIv3Areas_2020TractCrosswalk.gpkg": f"{HOLC}/MIv3Areas_2020TractCrosswalk.gpkg",
    "cdph/bll_screening_ca_1999_2013.csv":
        "https://data.cityofchicago.org/resource/v2z5-jyrq.csv?$limit=5000",
}
ACS_TABLES = ["b01003", "b09001", "b25003", "b25024", "b25034"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest.relative_to(RAW)}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "chicago-lslr-research/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print(f"  fetched {dest.relative_to(RAW)}  ({dest.stat().st_size/1e6:.1f} MB)")


def _acs_cook_tracts(table: str) -> None:
    """Stream a nationwide ACS summary file and keep only Cook County tracts."""
    dest = RAW / "acs" / f"{table}.dat"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  acs/{table}.dat")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ACS}/acsdt5y2023-{table}.dat"
    n = 0
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "w") as out:
        header = r.readline().decode()
        out.write(header)
        for raw in r:
            line = raw.decode()
            if line.startswith("1400000US17031"):
                out.write(line)
                n += 1
    print(f"  fetched acs/{table}.dat  ({n} Cook County tracts)")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> int:
    print(f"Downloading raw inputs into {RAW}")
    for rel, url in FILES.items():
        _download(url, RAW / rel)
    for t in ACS_TABLES:
        _acs_cook_tracts(t)
    with open(RAW / "MANIFEST.txt", "w") as m:
        for p in sorted(RAW.rglob("*")):
            if p.is_file() and p.name != "MANIFEST.txt":
                m.write(f"{sha256(p)}  {p.stat().st_size:>11}  {p.relative_to(RAW)}\n")
    print("Done. Checksums written to data/raw/MANIFEST.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
