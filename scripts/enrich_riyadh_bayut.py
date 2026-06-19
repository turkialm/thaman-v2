"""
enrich_riyadh_bayut.py
======================
Merge Bayut listing aggregates (per district_ar) into features_riyadh.csv
→ outputs features_riyadh_v2.csv with 6 new market-signal columns.

Bayut data = current asking prices (May 2026).
Used as static district-level signal — same pattern as geographic features.
NOT used as training target (those are MOJ actual transactions).

New features:
  bayut_listing_count   — # active listings (liquidity proxy)
  bayut_median_psqm     — median asking SAR/m² per district
  bayut_p25_psqm        — 25th pct asking price
  bayut_p75_psqm        — 75th pct asking price
  bayut_iqr_psqm        — interquartile spread (uncertainty signal)
  bayut_asking_premium  — bayut_median_psqm / district_median_price_sqm
                          (asking vs. actual transaction ratio — market heat)

Run: python scripts/enrich_riyadh_bayut.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

BAYUT_PATH    = PROC / "bayut_listings_riyadh.json"
IN_CSV        = PROC / "features_riyadh.csv"
OUT_CSV       = PROC / "features_riyadh_v2.csv"

PSQM_LO, PSQM_HI = 500, 50_000   # sanity filter

print("[1/5] Loading Bayut listings …")
raw = pd.DataFrame(json.load(open(BAYUT_PATH)))
print(f"  raw rows: {len(raw):,}  districts: {raw['district_ar'].nunique()}")

print("[2/5] Cleaning & filtering …")
clean = raw[(raw["psqm"] > PSQM_LO) & (raw["psqm"] < PSQM_HI) & raw["area"].notna()].copy()
print(f"  clean rows: {len(clean):,}  ({len(raw)-len(clean):,} dropped)")

print("[3/5] Aggregating per district_ar …")
agg = (
    clean.groupby("district_ar")
    .agg(
        bayut_listing_count=("psqm", "count"),
        bayut_median_psqm  =("psqm", "median"),
        bayut_p25_psqm     =("psqm", lambda x: x.quantile(0.25)),
        bayut_p75_psqm     =("psqm", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)
agg["bayut_iqr_psqm"] = agg["bayut_p75_psqm"] - agg["bayut_p25_psqm"]
print(f"  aggregated districts: {len(agg)}")

print("[4/5] Merging into training data …")
train = pd.read_csv(IN_CSV, encoding="utf-8-sig")
print(f"  training rows: {len(train):,}  training districts: {train['district_ar'].nunique()}")

merged = train.merge(agg, on="district_ar", how="left")

# Impute missing districts with global medians
glob_med   = agg["bayut_median_psqm"].median()
glob_p25   = agg["bayut_p25_psqm"].median()
glob_p75   = agg["bayut_p75_psqm"].median()
glob_iqr   = glob_p75 - glob_p25

n_missing  = merged["bayut_listing_count"].isna().sum()
print(f"  rows imputed (no Bayut district match): {n_missing:,}")

merged["bayut_listing_count"] = merged["bayut_listing_count"].fillna(0).astype(int)
merged["bayut_median_psqm"]   = merged["bayut_median_psqm"].fillna(glob_med)
merged["bayut_p25_psqm"]      = merged["bayut_p25_psqm"].fillna(glob_p25)
merged["bayut_p75_psqm"]      = merged["bayut_p75_psqm"].fillna(glob_p75)
merged["bayut_iqr_psqm"]      = merged["bayut_iqr_psqm"].fillna(glob_iqr)

# Asking premium = bayut median / MOJ district median (market heat ratio)
merged["bayut_asking_premium"] = (
    merged["bayut_median_psqm"] / merged["district_median_price_sqm"].replace(0, np.nan)
).fillna(1.0)

print(f"  global imputation values: median={glob_med:.0f}  p25={glob_p25:.0f}  p75={glob_p75:.0f}")
print(f"  bayut_asking_premium: mean={merged['bayut_asking_premium'].mean():.2f}  "
      f"min={merged['bayut_asking_premium'].min():.2f}  max={merged['bayut_asking_premium'].max():.2f}")

print(f"[5/5] Saving → {OUT_CSV.name} …")
merged.to_csv(OUT_CSV, index=False)
print(f"  done — {len(merged):,} rows × {len(merged.columns)} cols")
print()
print("New features added:")
for col in ["bayut_listing_count","bayut_median_psqm","bayut_p25_psqm",
            "bayut_p75_psqm","bayut_iqr_psqm","bayut_asking_premium"]:
    print(f"  {col}: {merged[col].describe()[['mean','min','max']].to_dict()}")
