"""
THAMAN — Expand Sales Dataset (2022–2024)
==========================================
Downloads NYC Annual Property Sales from the NYC Open Data Socrata API
(dataset w2pb-icbu, 760K rows, 2016–2024), geocodes/enriches via PLUTO,
and appends to data/raw/sales_geocoded.csv.

After running this script:
  1.  python training/feature_engineering.py   ← rebuilds features.csv
  2.  python training/train_stack_v2.py         ← retrains the model

Usage
  python scripts/download_more_sales.py             # 2022 + 2023 + 2024
  python scripts/download_more_sales.py --years 2023 2024
  python scripts/download_more_sales.py --dry-run   # count only
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import requests
from io import StringIO

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW       = os.path.join(ROOT, "data", "raw")
PLUTO_CSV = os.path.join(RAW, "nyc_pluto_25v4_csv", "pluto_25v4.csv")
SALES_OUT = os.path.join(RAW, "sales_geocoded.csv")

SOCRATA_URL    = "https://data.cityofnewyork.us/resource/w2pb-icbu.csv"
APP_TOKEN      = ""        # optional — raises rate limit from 1 → 10 req/s
DOWNLOAD_YEARS = [2022, 2023, 2024]
PAGE_SIZE      = 50_000
MIN_PRICE      = 10_000

# Rename Socrata columns → our internal schema
COL_RENAME = {
    "ease_ment":                    "easement",
    "building_class_as_of_final":   "building_class_at_present",
    "building_class_at_time_of":    "building_class_at_time_of_sale",
    "tax_class_as_of_final_roll":   "tax_class_at_present",
}

# PLUTO columns to join in (lat/lng already in Socrata; we only need these extras)
PLUTO_JOIN_COLS = [
    "bbl", "zonedist1", "bldgclass", "numfloors", "yearbuilt",
    "residfar", "commfar", "builtfar", "maxallwfar", "facilfar",
    "assessland", "assesstot",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--years", type=int, nargs="+", default=DOWNLOAD_YEARS)
    return p.parse_args()


def download_year(year: int) -> pd.DataFrame:
    """Download all rows for a calendar year via paginated Socrata API."""
    frames, offset = [], 0
    where  = (f"sale_date between '{year}-01-01T00:00:00.000' "
              f"and '{year}-12-31T23:59:59.999'")
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}

    while True:
        params = {
            "$where":  where,
            "$limit":  PAGE_SIZE,
            "$offset": offset,
            "$order":  "sale_date ASC",
        }
        try:
            resp = requests.get(SOCRATA_URL, params=params,
                                headers=headers, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"\n    ⚠  Request failed at offset {offset}: {exc}")
            break

        chunk = pd.read_csv(StringIO(resp.text))
        if chunk.empty:
            break

        frames.append(chunk)
        offset += len(chunk)
        print(f"    Fetched {offset:,} rows …", end="\r")

        if len(chunk) < PAGE_SIZE:
            break
        time.sleep(0.35)

    print()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean_numeric(series: pd.Series) -> pd.Series:
    """Strip dollar signs, commas, spaces then coerce to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,\s]", "", regex=True),
        errors="coerce",
    )


def load_pluto(path: str) -> pd.DataFrame:
    print(f"\n[PLUTO] Loading {os.path.basename(os.path.dirname(path))}/pluto …")
    pluto = pd.read_csv(path, usecols=lambda c: c in PLUTO_JOIN_COLS,
                        low_memory=False)
    pluto["bbl"] = pd.to_numeric(pluto["bbl"], errors="coerce")
    pluto = pluto.dropna(subset=["bbl"])
    pluto["bbl"] = pluto["bbl"].astype("int64")
    print(f"       {len(pluto):,} parcels loaded")
    return pluto


def main():
    args = parse_args()
    print("=" * 65)
    print("  THAMAN Data Expansion — NYC Annual Sales")
    print(f"  Years: {args.years}  |  Source: NYC Open Data w2pb-icbu")
    print("=" * 65)

    # ── 1. PLUTO enrichment table ──────────────────────────────────
    pluto = load_pluto(PLUTO_CSV)

    # ── 2. Download each year ──────────────────────────────────────
    all_new = []
    for year in args.years:
        print(f"\n[DL] {year} …")
        raw = download_year(year)
        if raw.empty:
            print(f"     ⚠  No data for {year} — skipping")
            continue
        print(f"     Raw rows downloaded: {len(raw):,}")

        # Normalise column names
        raw.columns = raw.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        raw = raw.rename(columns=COL_RENAME)

        # Clean numerics (sale_price / sq ft often come with commas)
        raw["sale_price"]        = clean_numeric(raw.get("sale_price",        pd.Series()))
        raw["gross_square_feet"] = clean_numeric(raw.get("gross_square_feet", pd.Series()))
        raw["land_square_feet"]  = clean_numeric(raw.get("land_square_feet",  pd.Series()))
        raw["sale_date"]         = pd.to_datetime(raw.get("sale_date"),        errors="coerce")
        raw["bbl"]               = pd.to_numeric(raw.get("bbl"),               errors="coerce")

        # Filter: real sale price, valid date, geocoded
        before = len(raw)
        raw = raw[
            raw["sale_price"].gt(MIN_PRICE) &
            raw["sale_date"].notna() &
            raw["latitude"].notna() &
            raw["longitude"].notna() &
            raw["bbl"].notna()
        ].copy()
        print(f"     After price/date/geo filter: {len(raw):,}  (dropped {before-len(raw):,})")

        # PLUTO join for zoning + building attributes (lat/lng already present)
        raw["bbl"] = raw["bbl"].astype("int64")
        merged = raw.merge(
            pluto.rename(columns={"bbl": "_bbl"}),
            left_on="bbl", right_on="_bbl",
            how="left",
        ).drop(columns=["_bbl"], errors="ignore")
        print(f"     After PLUTO enrichment: {len(merged):,} rows")
        all_new.append(merged)

    if not all_new:
        print("\n❌  Nothing downloaded. Check API or try again later.")
        sys.exit(1)

    new_df = pd.concat(all_new, ignore_index=True)
    print(f"\n  Total new rows: {len(new_df):,}")

    if args.dry_run:
        print("\n[DRY RUN] Nothing written.")
        return

    # ── 3. Load existing + merge ───────────────────────────────────
    print(f"\n[MERGE] Loading existing sales_geocoded.csv …")
    existing = pd.read_csv(SALES_OUT)
    existing["sale_date"] = pd.to_datetime(existing["sale_date"], errors="coerce")
    existing["bbl"]       = pd.to_numeric(existing["bbl"], errors="coerce")
    print(f"        Existing: {len(existing):,}  "
          f"({existing['sale_date'].min().date()} → {existing['sale_date'].max().date()})")

    # Add any new columns (e.g. assesstot/assessland) to existing with NaN
    for col in new_df.columns:
        if col not in existing.columns:
            existing[col] = np.nan

    # Keep union of columns; fill gaps with NaN
    all_cols = list(dict.fromkeys(existing.columns.tolist() + new_df.columns.tolist()))
    for col in all_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan
        if col not in existing.columns:
            existing[col] = np.nan

    combined = pd.concat([existing[all_cols], new_df[all_cols]], ignore_index=True)

    # Deduplicate on BBL + sale_date (keep earlier/existing record)
    combined["_key"] = combined["bbl"].astype(str) + "_" + combined["sale_date"].astype(str)
    combined = combined.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"])
    combined = combined.sort_values("sale_date").reset_index(drop=True)

    added = len(combined) - len(existing)
    print(f"\n  Existing rows:   {len(existing):,}")
    print(f"  New rows added:  {added:,}")
    print(f"  Total rows:      {len(combined):,}")
    print(f"  Date range:      {combined['sale_date'].min().date()} → {combined['sale_date'].max().date()}")
    print(f"\n  Borough breakdown:")
    print(combined["borough"].value_counts().sort_index().rename({
        "1":"Manhattan","2":"Bronx","3":"Brooklyn","4":"Queens","5":"Staten Island"}).to_string())

    combined.to_csv(SALES_OUT, index=False)
    print(f"\n  ✅ Saved → {SALES_OUT}")
    print(f"""
{'=' * 65}
  Next steps:
    1. python training/feature_engineering.py
    2. python training/train_stack_v2.py
{'=' * 65}""")


if __name__ == "__main__":
    main()
