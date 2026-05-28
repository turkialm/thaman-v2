"""
THAMAN Riyadh Model — Stack Training v2
========================================
Extends v1 with 6 Bayut listing features (per-district asking price signals):
  bayut_listing_count, bayut_median_psqm, bayut_p25_psqm,
  bayut_p75_psqm, bayut_iqr_psqm, bayut_asking_premium

Run:  python scripts/enrich_riyadh_bayut.py   # generates features_riyadh_v2.csv
      python training/train_stack_riyadh_v2.py
"""

import json
import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.spatial import cKDTree as _cKDTree
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

BASE  = Path(__file__).parent.parent
PROC  = BASE / "data" / "processed"
MDIR  = BASE / "models"
MDIR.mkdir(exist_ok=True)

# ── Feature list ─────────────────────────────────────────────────────────────

FEATURES = [
    # Location
    "district_lat", "district_lon",
    # Property type
    "is_apartment", "is_villa", "is_residential_plot", "is_building",
    # Metro transit (old slot — dist_metro_m/log/bus_stops_500m handled by v10 block below)
    "metro_stations_1km",
    "nearest_metro_line_num", "nearest_metro_type_cd", "dist_metro_line1_m",
    # Bus (old slot — bus_stops_500m in v10 block; keep dist_bus_m if present)
    "dist_bus_m", "log_dist_bus_m", "brt_stops_500m",
    # Traffic
    "dist_major_intersection_m", "log_dist_intersection_m",
    "intersections_1km", "intersections_500m",
    # Commercial
    "commercial_count_1km", "commercial_density_score",
    "district_commercial_count", "district_commercial_mix",
    "hypermarket_count_1km", "supermarket_count_1km",
    "bank_count_1km", "restaurant_count_1km",
    "hotel_count_1km", "gas_station_count_1km",
    # Air quality
    "no2_nearest_mean", "so2_nearest_mean", "pm10_nearest_mean",
    "o3_nearest_mean", "dist_air_station_m", "air_quality_score",
    # Macro / price index
    "rei_residential_qtr_idx", "rei_apt_idx",
    "rei_yoy_change", "rei_qoq_change",
    "rei_type_idx",             # v8: type-matched REI (apt→Apt, villa→Villa, plot→Plot, bldg→Floor)
    # Geographic hub distances (v9) — explicit hub proximity (lat/lon alone too coarse for trees)
    "dist_kafd_m", "log_dist_kafd_m",           # KAFD (premium zone, corr=-0.517)
    "dist_old_city_m", "log_dist_old_city_m",   # historic core
    "dist_industrial_m", "log_dist_industrial_m", # industrial zone (negative premium)
    "dist_airport_m", "log_dist_airport_m",     # airport proximity
    # Riyadh Metro (v10) — 94 stations, 6 lines, opened Dec 2024 (Vision 2030)
    "dist_metro_m", "log_dist_metro_m",         # nearest metro station
    "metro_500m", "metro_1km",                  # walkable / near metro
    "nearest_metro_line",                        # line ordinal 1-6 (Blue=1 … Purple=6)
    "bus_stops_500m",                            # bus stop density (transit richness)
    "avg_saudi_salary_yr", "salary_yoy_change",
    # District aggregates
    "district_median_price_sqm", "district_transaction_volume",
    "district_price_vs_city_avg", "district_price_trend_slope",
    "district_median_price_apt_sqm",
    # Target-encoded
    "district_encoded", "district_type_encoded",
    "district_apt_encoded", "district_recent_encoded", "district_apt_recent_encoded",
    # Time
    "sale_year", "sale_quarter_sin", "sale_quarter_cos",
    "log_deed_count",
    # QoL POI proximity & density (mosques, malls, schools, hospitals, parks, entertainment)
    "dist_mosque_m", "log_dist_mosque_m", "mosque_count_500m",
    "dist_mall_m", "log_dist_mall_m", "mall_count_500m",
    "dist_school_m", "log_dist_school_m", "school_count_500m",
    "dist_hospital_m", "log_dist_hospital_m", "hospital_count_500m",
    "dist_park_m", "log_dist_park_m", "park_count_500m",
    "dist_entertain_m", "log_dist_entertain_m", "entertain_count_500m",
    # Connectivity
    "riyadh_connectivity_score",
    # Structural (SA_Aqar rental listings — district-level medians)
    "aqar_median_size_sqm",
    "aqar_median_bedrooms",
    "aqar_median_property_age",
    "aqar_rent_per_sqm",
    # Bayut listing signals (current asking prices — market demand/heat indicators)
    "bayut_listing_count",
    "bayut_median_psqm",
    "bayut_p25_psqm",
    "bayut_p75_psqm",
    "bayut_iqr_psqm",
    "bayut_asking_premium",
    # Bayut structural (apt/plot medians — villa has near-zero importance)
    "bayut_apt_median_psqm",
    "bayut_plot_median_psqm",
    "bayut_apt_median_area_sqm",
    "bayut_plot_median_area_sqm",
    "bayut_apt_median_rooms",
    "bayut_plot_median_rooms",
    # New QoL POIs (OSM, May 2026)
    "dist_pharmacy_m", "log_dist_pharmacy_m", "pharmacy_count_500m",
    "dist_gym_m", "log_dist_gym_m", "gym_count_500m",
    "dist_coffee_m", "log_dist_coffee_m", "coffee_count_500m",
    "dist_clinic_m", "log_dist_clinic_m", "clinic_count_500m",
    "dist_university_m", "log_dist_university_m", "university_count_500m",
    "dist_supermarket_m", "log_dist_supermarket_m", "supermarket_count_500m",
    "dist_cinema_m", "log_dist_cinema_m", "cinema_count_500m",
    "dist_sports_m", "log_dist_sports_m", "sports_count_500m",
    # Haraj listing signals (corr=0.735 with transaction prices)
    "haraj_listing_count",
    "haraj_median_psqm",
    "haraj_p25_psqm",
    "haraj_p75_psqm",
    "haraj_iqr_psqm",
    "haraj_asking_premium",
    # District temporal lag features (v4 — computed in training script)
    "district_lag1q_median_psqm",   # district median price last quarter
    "district_lag2q_median_psqm",   # district median price 2 quarters ago
    "district_lag_momentum",        # lag1 − lag2 (price trend direction)
    # v11: type-stratified lags + data-density signal
    "district_type_lag1q_psqm",     # lag-1q median per district × property_type
    "district_type_lag2q_psqm",     # lag-2q median per district × property_type
    "district_lag1q_std_psqm",      # lag-1q price std dev (volatility / sparse-market signal)
    "log_suhail_n_trans",           # log count of individual suhail txns in district-quarter
]

TARGET = "sale_price_sar_sqm"
GROUP_COL = "district_ar"

# ── Hyperparameters ───────────────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
    tree_method="hist", random_state=42,
)
LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
    min_split_gain=0.2, reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=1500, learning_rate=0.03, depth=5,
    l2_leaf_reg=2.0, random_strength=0.2, bagging_temperature=0.5,
    random_seed=42, verbose=0,
)


def medape(y_true, y_pred):
    return float(np.median(np.abs((np.expm1(y_pred) - np.expm1(y_true)) / np.expm1(y_true))) * 100)


def mae_sar(y_true, y_pred):
    return float(np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true))))


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading features_riyadh_v2.csv...")
df = pd.read_csv(PROC / "features_riyadh_v2.csv", encoding="utf-8-sig")
print(f"  Rows: {len(df)} | Cols: {len(df.columns)}")

# ── Fix 2024 quarter_id encoding bug ─────────────────────────────────────────
# 2024 rows have quarter stored as full quarter_id (20241/20243/20244) instead of
# quarter number (1/3/4), so quarter_id = year*10 + quarter = 2024*10 + 20241 = 40481.
# Correct: use year*10 + (quarter % 10) for those rows.
bug_mask = df["quarter_id"] >= 40000
if bug_mask.sum() > 0:
    df.loc[bug_mask, "quarter_id"] = (
        df.loc[bug_mask, "year"] * 10 + df.loc[bug_mask, "quarter"] % 10
    ).astype(int)
    print(f"  Fixed {bug_mask.sum()} rows with corrupt quarter_id (2024 data restored to training)")

# ── Extend with Suhail 2025Q4/2026Q1/Q2 ─────────────────────────────────────
# Individual Suhail transactions cover 2025Q2–2026Q2 but only 2025Q4+ are new.
# Aggregate by district×quarter×type → append as new training rows.
_suhail_path = BASE / "data" / "raw" / "suhail_riyadh_tx_raw.csv"
if _suhail_path.exists():
    import polars as _pl
    _type_map = {
        "شقة":        "شقة",
        "فيلا":       "فيلا",
        "مبنى سكني":  "عمارة",
        "قطعة أرض":   "قطعة أرض-سكنى",
        "أرض فضاء":   "قطعة أرض-سكنى",
    }
    _tx = (
        _pl.read_csv(_suhail_path, ignore_errors=True)
        .filter(_pl.col("province_name") == "الرياض")
        .with_columns([
            _pl.col("date").str.slice(0, 4).cast(_pl.Int64).alias("year"),
            _pl.col("date").str.slice(5, 2).cast(_pl.Int64).alias("month"),
        ])
        .with_columns(((_pl.col("month") - 1) // 3 + 1).alias("quarter"))
        .with_columns(_pl.col("property_type").replace(_type_map).alias("typecategoryar"))
        .filter(_pl.col("typecategoryar").is_in(list(set(_type_map.values()))))
        .filter((_pl.col("psqm") >= 500) & (_pl.col("psqm") <= 50_000))
        # Only quarters NOT already in training data (2025Q4, 2026Q1, 2026Q2)
        .filter(
            ((_pl.col("year") == 2025) & (_pl.col("quarter") == 4)) |
            (_pl.col("year") == 2026)
        )
    )
    _agg = (
        _tx.group_by(["district_ar", "year", "quarter", "typecategoryar"])
        .agg([
            _pl.col("psqm").median().alias(TARGET),
            _pl.len().alias("deed_counts"),
        ])
        .filter(_pl.col("deed_counts") >= 3)
    ).to_pandas()

    if len(_agg) > 0:
        # Spatial template: most recent row per district (forward-fill spatial features)
        _df_tmpl = (
            df.sort_values("quarter_id")
              .groupby("district_ar", as_index=False)
              .last()
        )
        # Columns to take from template (all spatial/static; override temporal below)
        _temporal_cols = [
            TARGET, "sale_year", "sale_quarter", "quarter", "year", "quarter_id",
            "deed_counts", "log_deed_count", "sale_quarter_sin", "sale_quarter_cos",
            "is_apartment", "is_villa", "is_residential_plot", "is_building",
            "typecategoryar", "district_lag1q_median_psqm",
            "district_lag2q_median_psqm", "district_lag_momentum",
            # v11 (computed post-merge, not from template)
            "district_type_lag1q_psqm", "district_type_lag2q_psqm",
            "district_lag1q_std_psqm", "log_suhail_n_trans", "_ptype",
        ]
        _spatial_cols = [c for c in _df_tmpl.columns if c not in _temporal_cols]
        _ext = _agg.merge(_df_tmpl[_spatial_cols], on="district_ar", how="inner")

        # Compute temporal features
        _q_arr = _ext["quarter"].values.astype(int)
        _ext["sale_year"]         = _ext["year"].astype(int)
        _ext["sale_quarter"]      = _q_arr
        _ext["quarter_id"]        = _ext["year"].astype(int) * 10 + _q_arr
        _ext["sale_quarter_sin"]  = np.sin(2 * np.pi * _q_arr / 4)
        _ext["sale_quarter_cos"]  = np.cos(2 * np.pi * _q_arr / 4)
        _ext["log_deed_count"]    = np.log1p(_ext["deed_counts"])
        _tc = _ext["typecategoryar"]
        _ext["is_apartment"]       = (_tc == "شقة").astype(int)
        _ext["is_villa"]           = (_tc == "فيلا").astype(int)
        _ext["is_residential_plot"]= (_tc == "قطعة أرض-سكنى").astype(int)
        _ext["is_building"]        = (_tc == "عمارة").astype(int)

        n_before = len(df)
        df = pd.concat([df, _ext], ignore_index=True)
        _new_qs = sorted(_ext[["year","quarter"]].drop_duplicates().apply(
            lambda r: f"{int(r.year)}Q{int(r.quarter)}", axis=1).tolist())
        print(f"  Suhail extension: +{len(df)-n_before} rows | quarters: {_new_qs} | total: {len(df)}")
    else:
        print("  Suhail extension: no new rows after filter")
else:
    print("  Suhail extension: file not found, skipping")

# ── Refresh haraj features from May 2026 snapshot ────────────────────────────
_haraj_new_path = BASE / "data" / "raw" / "saudi_listings_haraj_20260518.csv"
if _haraj_new_path.exists():
    import polars as _pl, re as _re
    _h = _pl.read_csv(_haraj_new_path, ignore_errors=True)
    _arabic_re = _re.compile(r'[؀-ۿ]')
    _h = (_h
        .filter(_h["district"].map_elements(lambda x: bool(_arabic_re.search(str(x))), return_dtype=_pl.Boolean))
        .filter((_pl.col("price_per_sqm") >= 500) & (_pl.col("price_per_sqm") <= 50_000))
        .filter(_pl.col("price_per_sqm").is_not_null())
        .rename({"district": "district_ar"})
    )
    # Overall aggregates (update existing haraj columns)
    _h_overall = (
        _h.group_by("district_ar")
        .agg([
            _pl.len().alias("haraj_listing_count"),
            _pl.col("price_per_sqm").median().alias("haraj_median_psqm"),
            _pl.col("price_per_sqm").quantile(0.25).alias("haraj_p25_psqm"),
            _pl.col("price_per_sqm").quantile(0.75).alias("haraj_p75_psqm"),
            (_pl.col("price_per_sqm").quantile(0.75) - _pl.col("price_per_sqm").quantile(0.25)).alias("haraj_iqr_psqm"),
        ])
    ).to_pandas()
    # Type-specific aggregates (new features)
    _type_map_h = {"apartment": "haraj_apt_median_psqm",
                   "villa":     "haraj_villa_median_psqm",
                   "plot":      "haraj_plot_median_psqm"}
    _h_type_dfs = []
    for _ptyp, _col in _type_map_h.items():
        _sub = (_h.filter(_pl.col("property_type_en") == _ptyp)
                .group_by("district_ar")
                .agg(_pl.col("price_per_sqm").median().alias(_col))
                .to_pandas())
        _h_type_dfs.append(_sub)

    # Update overall haraj columns in df via merge
    _haraj_update_cols = ["haraj_listing_count","haraj_median_psqm","haraj_p25_psqm","haraj_p75_psqm","haraj_iqr_psqm"]
    df = df.drop(columns=[c for c in _haraj_update_cols if c in df.columns])
    df = df.merge(_h_overall[["district_ar"] + _haraj_update_cols], on="district_ar", how="left")
    # Recompute haraj_asking_premium from updated median
    if "district_median_price_sqm" in df.columns:
        df["haraj_asking_premium"] = df["haraj_median_psqm"] / df["district_median_price_sqm"].replace(0, np.nan)

    # Add type-specific columns
    for _sub in _h_type_dfs:
        _col = [c for c in _sub.columns if c != "district_ar"][0]
        if _col in df.columns:
            df = df.drop(columns=[_col])
        df = df.merge(_sub, on="district_ar", how="left")

    _matched_pct = df["haraj_median_psqm"].notna().mean() * 100
    print(f"  Haraj refresh (May 2026): {len(_h_overall)} districts | match rate: {_matched_pct:.1f}%")
    # Save haraj district lookup for inference (so spatial.py uses same fresh values)
    _haraj_lu_src = _h_overall[["district_ar","haraj_listing_count","haraj_median_psqm","haraj_p25_psqm","haraj_p75_psqm","haraj_iqr_psqm"]].copy()
    _HARAJ_DISTRICT_LOOKUP = {}
    for _, _hrow in _haraj_lu_src.iterrows():
        _dname = _hrow["district_ar"]
        _dmedian = _hrow["haraj_median_psqm"]
        if pd.isna(_dmedian):
            continue
        _HARAJ_DISTRICT_LOOKUP[_dname] = {
            "haraj_listing_count": int(_hrow["haraj_listing_count"]) if not pd.isna(_hrow["haraj_listing_count"]) else None,
            "haraj_median_psqm":   round(float(_dmedian), 2),
            "haraj_p25_psqm":      round(float(_hrow["haraj_p25_psqm"]), 2) if not pd.isna(_hrow["haraj_p25_psqm"]) else None,
            "haraj_p75_psqm":      round(float(_hrow["haraj_p75_psqm"]), 2) if not pd.isna(_hrow["haraj_p75_psqm"]) else None,
            "haraj_iqr_psqm":      round(float(_hrow["haraj_iqr_psqm"]), 2) if not pd.isna(_hrow["haraj_iqr_psqm"]) else None,
        }
else:
    print("  Haraj refresh: file not found, skipping")

# ── Type-specific REI index (v8) ─────────────────────────────────────────────
# Existing rei_apt_idx applies apartment index to ALL property types.
# rei_type_idx: correct type-matched REI per row.
_REI_SECTOR_PATH = BASE / "data" / "raw" / "real-estate-price-index-by-sector-2023-100.csv"
if _REI_SECTOR_PATH.exists():
    _rei_df = pd.read_csv(_REI_SECTOR_PATH, sep=";")
    _rei_df = _rei_df[(_rei_df["Measure"] == "Index") & (_rei_df["Quarter"].str.startswith("Q"))].copy()
    _sector_map = {
        "Residential: Apartment":         "apartment",
        "Residential: Villa":             "villa",
        "Residential: Residential Plot":  "residential_plot",
        "Residential: Floor":             "building",
    }
    _rei_df = _rei_df[_rei_df["Sector"].isin(_sector_map)].copy()
    _rei_df["ptype"]      = _rei_df["Sector"].map(_sector_map)
    _rei_df["quarter_id"] = (_rei_df["Year"].astype(str) + _rei_df["Quarter"].str[1]).astype(int)
    # pivot: quarter_id × ptype → index value
    _rei_pivot = _rei_df.pivot_table(index="quarter_id", columns="ptype", values="value", aggfunc="first")
    # global fallbacks per type (mean of available values)
    _rei_glob = _rei_pivot.mean().to_dict()

    def _get_rei_type(row):
        """Return type-specific REI for this row's property type and quarter."""
        if row["is_apartment"] == 1:     pt = "apartment"
        elif row["is_villa"] == 1:       pt = "villa"
        elif row["is_residential_plot"] == 1: pt = "residential_plot"
        elif row["is_building"] == 1:    pt = "building"
        else:                            pt = "apartment"  # fallback
        qid = int(row["quarter_id"]) if not pd.isna(row["quarter_id"]) else 0
        if qid in _rei_pivot.index and pt in _rei_pivot.columns:
            val = _rei_pivot.loc[qid, pt]
            if not pd.isna(val):
                return float(val)
        return float(_rei_glob.get(pt, 100.0))

    df["rei_type_idx"] = df.apply(_get_rei_type, axis=1)
    print(f"  rei_type_idx: {df['rei_type_idx'].nunique()} unique values | "
          f"apt={df[df['is_apartment']==1]['rei_type_idx'].mean():.1f} "
          f"villa={df[df['is_villa']==1]['rei_type_idx'].mean():.1f} "
          f"plot={df[df['is_residential_plot']==1]['rei_type_idx'].mean():.1f}")
else:
    df["rei_type_idx"] = 100.0
    print("  rei_type_idx: sector REI file not found, defaulting to 100")

# ── Geographic hub distance features (v9) ────────────────────────────────────
# Explicit distance to key economic hubs — corr(dist_kafd, psqm)=-0.517
# Trees need many splits to rediscover this from raw lat/lon; explicit = cleaner signal.
_HUBS = {
    "kafd":       (24.771, 46.637),   # King Abdullah Financial District (premium zone)
    "old_city":   (24.690, 46.722),   # Historical Qasr Al-Hokm / Dira (traditional centre)
    "industrial": (24.620, 46.873),   # Industrial City (eastern industrial zone — negative premium)
    "airport":    (24.957, 46.699),   # King Khalid International Airport
}
_DEG2M_V9 = 111_000.0
for _hub, (_hlat, _hlon) in _HUBS.items():
    _col = f"dist_{_hub}_m"
    df[_col] = np.sqrt((df["district_lat"] - _hlat)**2 + (df["district_lon"] - _hlon)**2) * _DEG2M_V9
    df[f"log_{_col}"] = np.log1p(df[_col])
print(f"  Hub distances: {list(_HUBS.keys())} | kafd corr={df['dist_kafd_m'].corr(df[TARGET]):.3f}")

# ── Riyadh Metro + Bus transit features (v10) ────────────────────────────────
# 94 metro stations (6 lines, opened Dec 2024); 3,010 bus stops
_METRO_GJ  = BASE / "data/raw/metro-stations-in-riyadh-by-metro-line-and-station-type-2024.geojson"
_BUS_GJ    = BASE / "data/raw/bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024.geojson"
_DEG2M_V10 = 111_000.0

with open(_METRO_GJ) as _f:
    _metro_feats = json.load(_f)["features"]
_LINE_ORD = {"Line1": 1, "Line2": 2, "Line3": 3, "Line4": 4, "Line5": 5, "Line6": 6}
_metro_lats = np.array([_s["geometry"]["coordinates"][1] for _s in _metro_feats])
_metro_lons = np.array([_s["geometry"]["coordinates"][0] for _s in _metro_feats])
_metro_lines = np.array([_LINE_ORD.get(_s["properties"]["metro_line_cd"], 0) for _s in _metro_feats])
_metro_pts  = np.column_stack([_metro_lats, _metro_lons])
_metro_tree = _cKDTree(_metro_pts)

with open(_BUS_GJ) as _f:
    _bus_feats = json.load(_f)["features"]
_bus_pts = np.array([[_b["geometry"]["coordinates"][1], _b["geometry"]["coordinates"][0]]
                     for _b in _bus_feats])
_bus_tree = _cKDTree(_bus_pts)

_500m_deg = 500.0 / _DEG2M_V10

# Per unique district centroid → map back to df rows
_dist_metro = {}
_dist_keys  = df[["district_ar", "district_lat", "district_lon"]].drop_duplicates()
for _, _row in _dist_keys.iterrows():
    _pt = np.array([[_row["district_lat"], _row["district_lon"]]])
    _dd, _di = _metro_tree.query(_pt, k=1)
    _dm   = float(_dd[0]) * _DEG2M_V10
    _line = int(_metro_lines[_di[0]])
    _nb   = int(_bus_tree.query_ball_point(_pt[0], r=_500m_deg, return_length=True))
    _dist_metro[_row["district_ar"]] = (_dm, _line, _nb)

df["dist_metro_m"]       = df["district_ar"].map(lambda d: _dist_metro.get(d, (np.nan, 0, 0))[0])
df["log_dist_metro_m"]   = np.log1p(df["dist_metro_m"].fillna(df["dist_metro_m"].median()))
df["metro_500m"]         = (df["dist_metro_m"] < 500).astype(int)
df["metro_1km"]          = (df["dist_metro_m"] < 1000).astype(int)
df["nearest_metro_line"] = df["district_ar"].map(lambda d: _dist_metro.get(d, (np.nan, 0, 0))[1])
df["bus_stops_500m"]     = df["district_ar"].map(lambda d: _dist_metro.get(d, (np.nan, 0, 0))[2])

_metro_corr = df["dist_metro_m"].corr(df[TARGET])
_m500 = df["metro_500m"].mean() * 100
_m1km = df["metro_1km"].mean() * 100
print(f"  Metro: 94 stations | dist_metro corr={_metro_corr:.3f} | "
      f"metro_500m={_m500:.1f}% | metro_1km={_m1km:.1f}% | bus_stops_500m median={df['bus_stops_500m'].median():.0f}")

# ── District temporal lag features ───────────────────────────────────────────
# Compute district median price per quarter, then lag-1 and momentum.
# Computed pre-split so all quarters are available, but each row only sees past quarters.
_qseq = sorted(df["quarter_id"].unique())
_qprev = {q: _qseq[i - 1] for i, q in enumerate(_qseq) if i > 0}   # q → q-1 map

_dist_qmed = (
    df.groupby(["district_ar", "quarter_id"])[TARGET]
    .median()
    .rename("_dq_med")
    .reset_index()
)
_dist_qmed.columns = ["district_ar", "quarter_id", "_dq_med"]

def _lag_med(row):
    pq = _qprev.get(row["quarter_id"])
    if pq is None:
        return np.nan
    match = _dist_qmed[(
        _dist_qmed["district_ar"] == row["district_ar"]) &
        (_dist_qmed["quarter_id"] == pq)]["_dq_med"]
    return float(match.iloc[0]) if len(match) else np.nan

def _lag2_med(row):
    pq  = _qprev.get(row["quarter_id"])
    ppq = _qprev.get(pq) if pq else None
    if ppq is None:
        return np.nan
    match = _dist_qmed[(
        _dist_qmed["district_ar"] == row["district_ar"]) &
        (_dist_qmed["quarter_id"] == ppq)]["_dq_med"]
    return float(match.iloc[0]) if len(match) else np.nan

print("  Computing district lag features...")
# Vectorised merge is faster than row-apply
_lag1_df = _dist_qmed.copy()
_lag1_df["quarter_id"] = _lag1_df["quarter_id"].map(
    {v: k for k, v in _qprev.items()})  # shift: value becomes next quarter's lag
_lag1_df = _lag1_df.dropna(subset=["quarter_id"]).rename(columns={"_dq_med": "district_lag1q_median_psqm"})

_lag2_df = _dist_qmed.copy()
_qprev2 = {q: _qseq[i - 2] for i, q in enumerate(_qseq) if i > 1}
_lag2_df["quarter_id"] = _lag2_df["quarter_id"].map(
    {v: k for k, v in _qprev2.items()})
_lag2_df = _lag2_df.dropna(subset=["quarter_id"]).rename(columns={"_dq_med": "district_lag2q_median_psqm"})

df = df.merge(_lag1_df[["district_ar","quarter_id","district_lag1q_median_psqm"]],
              on=["district_ar","quarter_id"], how="left")
df = df.merge(_lag2_df[["district_ar","quarter_id","district_lag2q_median_psqm"]],
              on=["district_ar","quarter_id"], how="left")
df["district_lag_momentum"] = df["district_lag1q_median_psqm"] - df["district_lag2q_median_psqm"]

lag_fill_pct = df["district_lag1q_median_psqm"].notna().mean() * 100
print(f"  Lag-1 fill rate: {lag_fill_pct:.1f}%")

# ── v11: type-stratified lags + volatility + density ─────────────────────────
print("  Computing v11 features (type-stratified lag + std + suhail density)...")

# Infer property type per row (for grouping)
_type_col = pd.Series("other", index=df.index)
_type_col[df["is_apartment"] == 1]        = "apt"
_type_col[df["is_villa"] == 1]            = "villa"
_type_col[df["is_residential_plot"] == 1] = "plot"
_type_col[df["is_building"] == 1]         = "bldg"
df["_ptype"] = _type_col

# district × type × quarter → median psqm + std
_dtq_agg = (
    df.groupby(["district_ar", "_ptype", "quarter_id"])[TARGET]
    .agg(["median", "std"])
    .reset_index()
    .rename(columns={"median": "_dtq_med", "std": "_dtq_std"})
)
_dtq_dict   = {(r["district_ar"], r["_ptype"], r["quarter_id"]): r["_dtq_med"]
               for _, r in _dtq_agg.iterrows()}
_dtq_std_dict = {(r["district_ar"], r["_ptype"], r["quarter_id"]): r["_dtq_std"]
                 for _, r in _dtq_agg.iterrows()}
_city_med_v11 = float(df[TARGET].median())

# lag-1 type-stratified median
def _dtq_lag1(row):
    pq = _qprev.get(row["quarter_id"])
    if pq is None: return np.nan
    return _dtq_dict.get((row["district_ar"], row["_ptype"], pq), np.nan)

def _dtq_lag2(row):
    pq  = _qprev.get(row["quarter_id"])
    ppq = _qprev.get(pq) if pq else None
    if ppq is None: return np.nan
    return _dtq_dict.get((row["district_ar"], row["_ptype"], ppq), np.nan)

def _dq_lag1_std(row):
    pq = _qprev.get(row["quarter_id"])
    if pq is None: return np.nan
    return _dtq_std_dict.get((row["district_ar"], row["_ptype"], pq), np.nan)

df["district_type_lag1q_psqm"] = df.apply(_dtq_lag1, axis=1)
df["district_type_lag2q_psqm"] = df.apply(_dtq_lag2, axis=1)
df["district_lag1q_std_psqm"]  = df.apply(_dq_lag1_std, axis=1)

# Fill NaN with district-level lag or city median
df["district_type_lag1q_psqm"].fillna(df["district_lag1q_median_psqm"].fillna(_city_med_v11), inplace=True)
df["district_type_lag2q_psqm"].fillna(df["district_lag2q_median_psqm"].fillna(_city_med_v11), inplace=True)
df["district_lag1q_std_psqm"].fillna(df["district_lag1q_std_psqm"].median(), inplace=True)

v11_fill = df["district_type_lag1q_psqm"].notna().mean() * 100
print(f"  district_type_lag1q fill rate: {v11_fill:.1f}%")

# log_suhail_n_trans — individual suhail transaction count per district-quarter
_suhail_all_path = BASE / "data" / "raw" / "suhail_riyadh_tx_raw.csv"
if _suhail_all_path.exists():
    import polars as _pl2
    _stx = (
        _pl2.read_csv(_suhail_all_path, ignore_errors=True)
        .filter(_pl2.col("province_name") == "الرياض")
        .with_columns([
            _pl2.col("date").str.slice(0, 4).cast(_pl2.Int64).alias("_sy"),
            _pl2.col("date").str.slice(5, 2).cast(_pl2.Int64).alias("_sm"),
        ])
        .with_columns((_pl2.col("_sy") * 10 + (_pl2.col("_sm") - 1) // 3 + 1).alias("quarter_id"))
        .filter((_pl2.col("psqm") >= 500) & (_pl2.col("psqm") <= 50_000))
        .group_by(["district_ar", "quarter_id"])
        .agg(_pl2.len().alias("_s_cnt"))
    ).to_pandas()
    _s_dict = {(r["district_ar"], r["quarter_id"]): r["_s_cnt"]
               for _, r in _stx.iterrows()}

    def _suhail_lag1_cnt(row):
        pq = _qprev.get(row["quarter_id"])
        if pq is None: return 0
        return _s_dict.get((row["district_ar"], pq), 0)

    df["log_suhail_n_trans"] = np.log1p(df.apply(_suhail_lag1_cnt, axis=1))
    s_fill = (df["log_suhail_n_trans"] > 0).mean() * 100
    print(f"  log_suhail_n_trans coverage (>0): {s_fill:.1f}%")
else:
    df["log_suhail_n_trans"] = 0.0
    print("  log_suhail_n_trans: suhail file not found → 0")

df.drop(columns=["_ptype"], inplace=True, errors="ignore")

# Only keep columns that exist
feat_cols = [f for f in FEATURES if f in df.columns]
missing = set(FEATURES) - set(feat_cols)
if missing:
    print(f"  WARNING: {len(missing)} feature(s) not in CSV: {sorted(missing)}")
print(f"  Using {len(feat_cols)} features")

# Log-transform target
df["log_price"] = np.log1p(df[TARGET])

# ── Train / holdout split (time-based) ────────────────────────────────────────

all_qids = sorted(df["quarter_id"].unique())
# Fixed cutoff at 2025 Q1 — trains on 2018-2024, holds out 2025 Q1-Q3.
cutoff_qid = 20251

work_mask = df["quarter_id"] < cutoff_qid
work = df[work_mask].copy()
hold = df[~work_mask].copy()
print(f"  Work set: {len(work)} rows (quarter_id < {cutoff_qid})")
print(f"  Holdout:  {len(hold)} rows (quarter_id >= {cutoff_qid})")

X_work = work[feat_cols].fillna(0).values.astype(np.float32)
y_work = work["log_price"].values
groups = work[GROUP_COL].values

X_hold = hold[feat_cols].fillna(0).values.astype(np.float32)
y_hold = hold["log_price"].values

# ── 5-fold Spatial GroupKFold cross-validation ────────────────────────────────

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

oof_xgb = np.zeros(len(work))
oof_lgb = np.zeros(len(work))
oof_cat = np.zeros(len(work))

xgb_models, lgb_models, cat_models = [], [], []

print(f"\nRunning {N_FOLDS}-fold Spatial GroupKFold CV...")
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_work, y_work, groups)):
    X_tr, X_va = X_work[tr_idx], X_work[va_idx]
    y_tr, y_va = y_work[tr_idx], y_work[va_idx]
    print(f"  Fold {fold+1}: train={len(tr_idx)} val={len(va_idx)}", end=" ", flush=True)

    # XGBoost
    m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    m_xgb.fit(X_tr, y_tr,
               eval_set=[(X_va, y_va)],
               verbose=False)
    oof_xgb[va_idx] = m_xgb.predict(X_va)
    xgb_models.append(m_xgb)

    # LightGBM
    m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    m_lgb.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    oof_lgb[va_idx] = m_lgb.predict(X_va)
    lgb_models.append(m_lgb)

    # CatBoost
    m_cat = cb.CatBoostRegressor(**CAT_PARAMS)
    m_cat.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              early_stopping_rounds=50)
    oof_cat[va_idx] = m_cat.predict(X_va)
    cat_models.append(m_cat)

    fold_medape = medape(y_va, oof_xgb[va_idx])
    print(f"| XGB MedAPE={fold_medape:.1f}%")

# OOF evaluation
oof_stack = np.column_stack([oof_xgb, oof_lgb, oof_cat])
oof_r2 = r2_score(y_work, oof_stack.mean(axis=1))
oof_medape = medape(y_work, oof_stack.mean(axis=1))
print(f"\nOOF (ensemble mean): R²={oof_r2:.4f} | MedAPE={oof_medape:.2f}%")

# ── Ridge meta-learner ────────────────────────────────────────────────────────

print("Training Ridge meta-learner...")
meta = Ridge(alpha=1.0)
meta.fit(oof_stack, y_work)
oof_meta = meta.predict(oof_stack)
oof_meta_r2 = r2_score(y_work, oof_meta)
oof_meta_medape = medape(y_work, oof_meta)
print(f"OOF meta: R²={oof_meta_r2:.4f} | MedAPE={oof_meta_medape:.2f}%")

# ── Retrain final models on full work set ─────────────────────────────────────

print("\nRetraining final models on full work set...")
final_xgb = xgb.XGBRegressor(**XGB_PARAMS)
final_xgb.fit(X_work, y_work, verbose=False)

final_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
final_lgb.fit(X_work, y_work, callbacks=[lgb.log_evaluation(period=-1)])

final_cat = cb.CatBoostRegressor(**CAT_PARAMS)
final_cat.fit(X_work, y_work)

# ── Holdout evaluation ────────────────────────────────────────────────────────

print("\nEvaluating on holdout set...")
hold_preds = np.column_stack([
    final_xgb.predict(X_hold),
    final_lgb.predict(X_hold),
    final_cat.predict(X_hold),
])
hold_meta_preds = meta.predict(hold_preds)

hold_r2     = r2_score(y_hold, hold_meta_preds)
hold_medape = medape(y_hold, hold_meta_preds)
hold_mae    = mae_sar(y_hold, hold_meta_preds)

print(f"  Holdout R²:     {hold_r2:.4f}")
print(f"  Holdout MedAPE: {hold_medape:.2f}%")
print(f"  Holdout MAE:    {hold_mae:,.0f} SAR/sqm")

# Per-type metrics
_seg_by_type = {}
for ptype in ["apartment", "villa", "residential_plot", "building"]:
    mask_col = f"is_{ptype}"
    if mask_col in hold.columns:
        hmask = hold[mask_col].values.astype(bool)
        if hmask.sum() >= 10:
            t_r2     = r2_score(y_hold[hmask], hold_meta_preds[hmask])
            t_medape = medape(y_hold[hmask], hold_meta_preds[hmask])
            print(f"  [{ptype:>18}] R²={t_r2:.4f} | MedAPE={t_medape:.2f}% | n={hmask.sum()}")
            _seg_by_type[ptype] = {"r2": round(t_r2, 4), "medape": round(t_medape, 2), "n": int(hmask.sum())}

# ── Save models ───────────────────────────────────────────────────────────────

print("\nSaving models...")
# ── Build district lag lookup for prediction-time use ──────────────────────
# Use most recent available quarter as lag1, one before as lag2.
_recent_qids = sorted(df["quarter_id"].unique())[-4:]  # last 4 available quarters
_lkp = {}
for dist in df["district_ar"].unique():
    _ddf = _dist_qmed[_dist_qmed["district_ar"] == dist].sort_values("quarter_id")
    if len(_ddf) < 2:
        continue
    lag1_val = float(_ddf.iloc[-1]["_dq_med"])
    lag2_val = float(_ddf.iloc[-2]["_dq_med"]) if len(_ddf) >= 2 else lag1_val
    _lkp[dist] = {
        "lag1": lag1_val,
        "lag2": lag2_val,
        "momentum": lag1_val - lag2_val,
    }
city_lag1 = float(_dist_qmed.groupby("quarter_id")["_dq_med"].median().iloc[-1])
city_lag2 = float(_dist_qmed.groupby("quarter_id")["_dq_med"].median().iloc[-2])
print(f"  Lag lookup: {len(_lkp)} districts | city lag1={city_lag1:.0f} lag2={city_lag2:.0f}")

stack_path = MDIR / "riyadh_stack.pkl"
with open(stack_path, "wb") as f:
    pickle.dump({
        "xgb": final_xgb,
        "lgb": final_lgb,
        "cat": final_cat,
        "meta": meta,
        "district_lag_map":  _lkp,
        "city_lag1_median":  city_lag1,
        "city_lag2_median":  city_lag2,
    }, f, protocol=5)
print(f"  Saved: {stack_path}")

# Update riyadh_meta.json
meta_path = MDIR / "riyadh_meta.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta_dict = json.load(f)
else:
    meta_dict = {}

meta_dict.update({
    "feature_names": feat_cols,
    "n_features": len(feat_cols),
    "holdout_r2": round(hold_r2, 4),
    "holdout_medape_pct": round(hold_medape, 2),
    "holdout_mae_sar_sqm": round(hold_mae, 2),
    "oof_r2": round(oof_meta_r2, 4),
    "oof_medape_pct": round(oof_meta_medape, 2),
    "model_version": "riyadh_v11",
    "n_folds": N_FOLDS,
    "train_rows": len(work),
    "holdout_rows": len(hold),
    "holdout_cutoff_quarter_id": int(cutoff_qid),
    "meta_coefficients": meta.coef_.tolist(),
    "meta_intercept": float(meta.intercept_),
    "target": "log1p(sale_price_sar_sqm)",
    "y_unit": "SAR/sqm",
    "segment_by_type": _seg_by_type,
})

# Save haraj district lookup if refresh was run
try:
    if _HARAJ_DISTRICT_LOOKUP:
        meta_dict["haraj_district_lookup"] = _HARAJ_DISTRICT_LOOKUP
        print(f"  haraj_district_lookup: {len(_HARAJ_DISTRICT_LOOKUP)} districts saved to meta.json")
except NameError:
    pass

# Save rei_type_idx lookup for inference (most recent quarter per type)
try:
    _rei_latest_qid = sorted(_rei_pivot.index)[-1]
    meta_dict["rei_type_idx_latest"] = {
        pt: float(_rei_pivot.loc[_rei_latest_qid, pt])
        for pt in _rei_pivot.columns
        if not pd.isna(_rei_pivot.loc[_rei_latest_qid, pt])
    }
    meta_dict["rei_type_idx_lookup"] = {
        str(qid): {pt: float(val) for pt, val in row.items() if not pd.isna(val)}
        for qid, row in _rei_pivot.iterrows()
    }
    print(f"  rei_type_idx_lookup: {len(_rei_pivot)} quarters | latest={_rei_latest_qid}")
except NameError:
    pass

# v10: save metro district lookup for inference
try:
    meta_dict["metro_district_lookup"] = {
        d: {"dist_metro_m": round(v[0], 1), "nearest_metro_line": v[1], "bus_stops_500m": v[2]}
        for d, v in _dist_metro.items()
    }
    print(f"  metro_district_lookup: {len(_dist_metro)} districts saved to meta.json")
except NameError:
    pass

with open(meta_path, "w") as f:
    json.dump(meta_dict, f, indent=2)
print(f"  Updated: {meta_path}")

print("\n" + "=" * 60)
print(f"THAMAN Riyadh v11 — Training complete")
print(f"  OOF  R²={oof_meta_r2:.4f}  MedAPE={oof_meta_medape:.2f}%")
print(f"  Hold R²={hold_r2:.4f}  MedAPE={hold_medape:.2f}%  MAE={hold_mae:,.0f} SAR/sqm")
print(f"  vs v10: ΔR²={hold_r2-0.7634:+.4f}  ΔMedAPE={18.45-hold_medape:+.2f}pp")
print("=" * 60)
