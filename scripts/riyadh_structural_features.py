"""
Riyadh Structural Feature Enrichment
======================================
Adds per-district structural proxy features to features_riyadh.csv from:
  1. SA_Aqar rental listings  → aqar_* (size, age, bedrooms, rent/sqm)
  2. Bayut sales listings     → per-type area and rooms
  3. Haraj sales listings     → per-type area and age

Run:  python scripts/riyadh_structural_features.py
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
RAW  = _ROOT / "data" / "raw"
PROC = _ROOT / "data" / "processed"


def _normalize_ar(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"^حي\s+", "", s)          # strip leading "حي "
    s = re.sub(r"\s+", " ", s)
    return s


# ── Load master features ──────────────────────────────────────────────────────

print("Loading features_riyadh.csv...")
feat = pd.read_csv(PROC / "features_riyadh.csv", encoding="utf-8-sig")
feat["district_ar"] = feat["district_ar"].apply(_normalize_ar)
feat_districts = set(feat["district_ar"].unique())
print(f"  {len(feat)} rows | {len(feat_districts)} districts")

# ── 1. SA_Aqar rental listings ────────────────────────────────────────────────

print("\nProcessing SA_Aqar...")
df_aqar = pd.read_csv(RAW / "SA_Aqar.csv")
ryd = df_aqar[df_aqar["city"].str.strip() == "الرياض"].copy()
ryd["district"] = ryd["district"].apply(_normalize_ar)
ryd = ryd[ryd["size"] > 0].copy()
# rent per sqm (annual SAR / sqm)
ryd["rent_per_sqm"] = ryd["price"] / ryd["size"]
# Remove outliers: size > 3000 or < 50, rent > 2000 SAR/sqm/yr
ryd = ryd[(ryd["size"] >= 50) & (ryd["size"] <= 3000)]
ryd = ryd[(ryd["rent_per_sqm"] > 0) & (ryd["rent_per_sqm"] < 2000)]

aqar_agg = ryd.groupby("district").agg(
    aqar_median_size_sqm    = ("size",         "median"),
    aqar_median_bedrooms    = ("bedrooms",     "median"),
    aqar_median_property_age= ("property_age", "median"),
    aqar_rent_per_sqm       = ("rent_per_sqm", "median"),
    _aqar_n                 = ("size",         "count"),
).reset_index()

overlap = len(set(aqar_agg["district"]) & feat_districts)
print(f"  SA_Aqar: {len(ryd)} rows | {len(aqar_agg)} districts | {overlap} matching")

# Global fallbacks
aqar_global = {
    "aqar_median_size_sqm":     float(ryd["size"].median()),
    "aqar_median_bedrooms":     float(ryd["bedrooms"].median()),
    "aqar_median_property_age": float(ryd["property_age"].median()),
    "aqar_rent_per_sqm":        float(ryd["rent_per_sqm"].median()),
}
print(f"  Globals: {aqar_global}")

# ── 2. Bayut sales listings ───────────────────────────────────────────────────

print("\nProcessing Bayut listings...")
with open(PROC / "bayut_listings_riyadh.json") as f:
    bl = json.load(f)
df_b = pd.DataFrame(bl)
df_b["district_ar"] = df_b["district_ar"].apply(_normalize_ar)
df_b["area"]        = pd.to_numeric(df_b["area"],  errors="coerce")
df_b["rooms"]       = pd.to_numeric(df_b["rooms"], errors="coerce")
df_b["psqm"]        = pd.to_numeric(df_b["psqm"],  errors="coerce")
# Filter reasonable sizes and prices
df_b = df_b[(df_b["area"] >= 30) & (df_b["area"] <= 2000)]
df_b = df_b[(df_b["psqm"] > 500) & (df_b["psqm"] < 50000)]

# Per-type aggregations
type_map = {
    "Villas":             "villa",
    "Apartments":         "apt",
    "Floors":             "apt",        # floor units ≈ apartments
    "Residential Lands":  "plot",
}
df_b["type_group"] = df_b["property_type"].map(type_map)
df_b_res = df_b[df_b["type_group"].notna()].copy()

bayut_agg = df_b_res.groupby(["district_ar", "type_group"]).agg(
    _median_area  = ("area",  "median"),
    _median_rooms = ("rooms", "median"),
    _median_psqm  = ("psqm",  "median"),
    _n            = ("area",  "count"),
).reset_index()

# Pivot to wide format (one row per district)
bayut_wide = bayut_agg.pivot(index="district_ar", columns="type_group",
                              values=["_median_area", "_median_rooms", "_median_psqm"]).reset_index()
bayut_wide.columns = [
    "district_ar" if c[1] == "" else f"bayut_{c[1]}_median_{c[0][1:]}"
    for c in bayut_wide.columns
]

# Clean up column names
rename = {}
for col in bayut_wide.columns:
    if col == "district_ar": continue
    # e.g. "bayut_villa_median__median_area" → "bayut_villa_area_sqm"
    col_clean = col.replace("_median_area", "_area_sqm").replace("_median_rooms", "_rooms").replace("_median_psqm", "_psqm")
    rename[col] = col_clean
bayut_wide = bayut_wide.rename(columns=rename)

overlap_b = len(set(bayut_wide["district_ar"]) & feat_districts)
print(f"  Bayut: {len(df_b_res)} residential rows | {len(bayut_wide)} districts | {overlap_b} matching")
print(f"  Bayut columns: {[c for c in bayut_wide.columns if c != 'district_ar']}")

# ── 3. Haraj listings ─────────────────────────────────────────────────────────

print("\nProcessing Haraj listings...")
df_h = pd.read_csv(RAW / "saudi_listings_haraj_20260518.csv")
df_h["district"] = df_h["district"].apply(_normalize_ar)
df_h["area_sqm"]   = pd.to_numeric(df_h["area_sqm"],   errors="coerce")
df_h["age_years"]  = pd.to_numeric(df_h["age_years"],  errors="coerce")
df_h["bedrooms"]   = pd.to_numeric(df_h["bedrooms"],   errors="coerce")

# Normalize property types
def _haraj_type(s):
    if not isinstance(s, str): return None
    s = s.strip()
    if s in ("شقة", "Apartment", "شقة سكنية (غرفتين وصالة).", "شقة.", "شقة بالدور الثاني"): return "apt"
    if s in ("فيلا", "Villa", "فيلا.", "فيلا ", "فيلا درج داخلي + شقتين", "فيلا سكنية بقيمة أرض", "قطعة أرض عليها بناء (فيلا)"): return "villa"
    if s in ("ارض", "Land", "قطعة أرض سكنية", "أرض سكنية", "أرض"): return "plot"
    return None

df_h["type_group"] = df_h["property_type_ar"].apply(_haraj_type)
df_h_res = df_h[df_h["type_group"].notna() & df_h["area_sqm"].notna()].copy()
df_h_res = df_h_res[(df_h_res["area_sqm"] >= 30) & (df_h_res["area_sqm"] <= 3000)]

haraj_agg = df_h_res.groupby(["district", "type_group"]).agg(
    _median_area = ("area_sqm",  "median"),
    _median_age  = ("age_years", lambda x: x.dropna().median() if x.notna().sum() >= 2 else np.nan),
    _n           = ("area_sqm",  "count"),
).reset_index()

haraj_wide = haraj_agg.pivot(index="district", columns="type_group",
                              values=["_median_area", "_median_age"]).reset_index()
haraj_wide.columns = [
    "district_ar" if c[1] == "" else f"haraj_{c[1]}_{c[0][1:]}"
    for c in haraj_wide.columns
]

haraj_rename = {}
for col in haraj_wide.columns:
    if col == "district_ar": continue
    haraj_rename[col] = col.replace("_median_area", "_area_sqm").replace("_median_age", "_age_yr")
haraj_wide = haraj_wide.rename(columns=haraj_rename)

overlap_h = len(set(haraj_wide["district_ar"]) & feat_districts)
print(f"  Haraj: {len(df_h_res)} rows | {len(haraj_wide)} districts | {overlap_h} matching")
print(f"  Haraj columns: {[c for c in haraj_wide.columns if c != 'district_ar']}")

# ── 4. Merge all into features_riyadh.csv ────────────────────────────────────

print("\nMerging into features_riyadh.csv...")
# Get per-district unique rows (structural features are district-level, not per-row)
new_feat = feat.copy()

# SA_Aqar
new_feat = new_feat.merge(
    aqar_agg.drop(columns=["_aqar_n"]),
    left_on="district_ar", right_on="district", how="left"
).drop(columns=["district"], errors="ignore")

for col, val in aqar_global.items():
    new_feat[col] = new_feat[col].fillna(val)

# Bayut
new_feat = new_feat.merge(bayut_wide, on="district_ar", how="left")
# Fill Bayut NaN with global medians
bayut_globals = {
    col: float(bayut_wide[col].median())
    for col in bayut_wide.columns
    if col != "district_ar" and bayut_wide[col].notna().sum() > 5
}
for col, val in bayut_globals.items():
    if col in new_feat.columns:
        new_feat[col] = new_feat[col].fillna(val)

# Haraj
new_feat = new_feat.merge(haraj_wide, on="district_ar", how="left")
haraj_globals = {
    col: float(haraj_wide[col].median())
    for col in haraj_wide.columns
    if col != "district_ar" and haraj_wide[col].notna().sum() > 2
}
for col, val in haraj_globals.items():
    if col in new_feat.columns:
        new_feat[col] = new_feat[col].fillna(val)

# Report new columns
new_cols = [c for c in new_feat.columns if c not in feat.columns]
print(f"  Added {len(new_cols)} new columns: {new_cols}")
print(f"  Output rows: {len(new_feat)} (was {len(feat)})")
for col in new_cols:
    non_null = new_feat[col].notna().sum()
    print(f"    {col}: {non_null}/{len(new_feat)} non-null ({non_null/len(new_feat)*100:.0f}%)")

# ── Save enriched CSV ─────────────────────────────────────────────────────────

out_path = PROC / "features_riyadh.csv"
new_feat.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path}  ({len(new_feat)} rows x {len(new_feat.columns)} cols)")
