"""
THAMAN Stack v12 — train_stack_v12.py
======================================
Key improvements over v11 (104 features → 109 features):

  Quarterly lagged NTA market features (zero leakage):
    • nta_lag1q_mean_logp   — NTA mean log-price from previous quarter
    • nta_lag1q_median_psf  — NTA median $/sqft from previous quarter
    • nta_lag1q_count       — NTA sale count from previous quarter (market heat)
    • nta_lag2q_mean_logp   — NTA mean log-price from 2 quarters ago
    • nta_logp_momentum     — Price trend: lag1 − lag2 (positive = appreciating NTA)

  All temporal stats computed from df_work only, joined on Q-1/Q-2 offset.
  Holdout rows see only training-window stats — no future leakage.

Data source: data/processed/features_v5.csv

Expected gain vs v11: MedAPE 20.24% → ~18.5–19.5%

Run:
  cd /Users/totam/Desktop/new_try
  python training/train_stack_v12.py
"""

import os, sys, json, warnings, joblib, datetime
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC      = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")

print("=" * 70)
print("  THAMAN Stack v12 — Quarterly NTA Temporal Lookback (109 features)")
print("=" * 70)

def safe_clip_min(arr, minval=1.0):
    return np.maximum(arr, minval)

def medape(y_true_usd, y_pred_usd):
    return float(np.median(np.abs(y_true_usd - y_pred_usd) / safe_clip_min(y_true_usd)) * 100)

# ── 1. Load data ───────────────────────────────────────────────────────
_FLOAT_OVERRIDES = {
    "dist_waterfront_m": pl.Float64, "dist_bike_lane_m": pl.Float64,
    "school_district": pl.Float64,   "district_avg_score": pl.Float64,
    "district_school_count": pl.Float64,
    "prior_sale_price": pl.Float64,  "years_since_prior_sale": pl.Float64,
    "price_appreciation": pl.Float64, "is_flip": pl.Float64,
}
print("\n[1/9] Loading features_v5.csv (v11 enriched) …")
_V5_PATH = os.path.join(PROC, "features_v5.csv")
_V4_PATH = os.path.join(PROC, "features_v4.csv")
_csv_path = _V5_PATH if os.path.exists(_V5_PATH) else _V4_PATH
if _csv_path == _V4_PATH:
    print("  ⚠ features_v5.csv not found — falling back to v4 (run prepare_v11_features.py first)")
df = (
    pl.read_csv(_csv_path, schema_overrides=_FLOAT_OVERRIDES)
    .with_columns(pl.col("sale_date").str.to_datetime(format=None, strict=False))
    .drop_nulls(subset=["sale_date", "sale_price", "latitude", "longitude"])
)
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── 1b. PLUTO join ─────────────────────────────────────────────────────
_PLUTO_PATH = os.path.join(BASE, "data", "raw", "nyc_pluto_25v4_csv", "pluto_25v4.csv")
if os.path.exists(_PLUTO_PATH):
    print("  Joining PLUTO …")
    _pluto = (
        pl.read_csv(_PLUTO_PATH, columns=["bbl", "assesstot", "assessland"])
        .with_columns(pl.col("bbl").cast(pl.Float64, strict=False))
        .drop_nulls(subset=["bbl"])
        .with_columns(pl.col("bbl").cast(pl.Int64).alias("_bbl_int"))
        .drop("bbl")
        .rename({"assesstot": "_at", "assessland": "_al"})
    )
    df = (df.with_columns(
            pl.col("bbl").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64).alias("_bbl_int"))
          .join(_pluto, on="_bbl_int", how="left").drop("_bbl_int"))
    for c, p in [("assesstot","_at"),("assessland","_al")]:
        if c not in df.columns: df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
        df = df.with_columns(pl.coalesce([pl.col(c), pl.col(p)]).alias(c)).drop(p)
    print(f"  assesstot coverage: {df['assesstot'].is_not_null().mean()*100:.1f}%")
else:
    for c in ["assesstot","assessland"]:
        if c not in df.columns: df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

# ── 2. Feature engineering ─────────────────────────────────────────────
print("\n[2/9] Engineering features …")

dist_cols = [c for c in df.columns if c.startswith("dist_")]
df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in dist_cols])
df = df.with_columns([pl.col(col).clip(lower_bound=0).log1p().alias(f"log_{col}") for col in dist_cols])

GRAVITY = {
    "midtown_manhattan":  (40.7549, -73.9840),
    "downtown_manhattan": (40.7074, -74.0113),
    "downtown_brooklyn":  (40.6928, -73.9903),
    "long_island_city":   (40.7447, -73.9485),
}
df = df.with_columns([
    (((pl.col("latitude") - clat) ** 2 + (pl.col("longitude") - clon) ** 2) ** 0.5 * 111_000)
    .alias(f"dist_{name}_m")
    for name, (clat, clon) in GRAVITY.items()
])
df = df.with_columns([
    (pl.col("borough") == 1).cast(pl.Int32).alias("is_manhattan"),
    (pl.col("crime_rate_nta") * (pl.col("borough") == 1).cast(pl.Int32)).alias("crime_x_manhattan"),
    (pl.col("crime_rate_nta") * (1 - (pl.col("borough") == 1).cast(pl.Int32))).alias("crime_x_non_manhattan"),
])

eps = 1e-6
walk_cols_order = ["transit", "bus", "amenities", "bike", "park"]
walk_comps_np = np.column_stack([
    1.0 / df["dist_subway_m"].clip(lower_bound=eps).to_numpy(),
    1.0 / df["dist_bus_m"].clip(lower_bound=eps).to_numpy(),
    df["poi_count_500m"].to_numpy(),
    1.0 / df["dist_bike_lane_m"].clip(lower_bound=eps).to_numpy(),
    1.0 / df["dist_park_m"].clip(lower_bound=eps).to_numpy(),
])
ws_scaler = MinMaxScaler()
walk_normed_np = ws_scaler.fit_transform(walk_comps_np)
walk_score_np = np.clip(walk_normed_np @ np.array([0.35, 0.15, 0.30, 0.10, 0.10]) * 100, 0, 100)
df = df.with_columns(pl.Series("walk_score_proxy", walk_score_np))
walk_score_scaler_params = {
    col: {"data_min": float(ws_scaler.data_min_[i]), "data_max": float(ws_scaler.data_max_[i]),
          "scale": float(ws_scaler.scale_[i])}
    for i, col in enumerate(walk_cols_order)
}
df = df.with_columns([
    (pl.col("sale_month") * (2 * np.pi / 12)).sin().alias("sale_month_sin"),
    (pl.col("sale_month") * (2 * np.pi / 12)).cos().alias("sale_month_cos"),
])

# ── v5 interaction features ────────────────────────────────────────────
df = df.with_columns([
    (pl.col("gross_square_feet") / pl.col("numfloors").clip(lower_bound=1)).alias("sqft_per_floor"),
    (pl.col("median_income_nta") / (pl.col("crime_rate_nta") + 1.0)).alias("income_over_crime"),
    (pl.col("residential_units") / pl.col("gross_square_feet").clip(lower_bound=1) * 1000.0).alias("density_index"),
    ((pl.col("gross_square_feet").clip(lower_bound=1).log1p()) *
     (pl.col("numfloors").clip(lower_bound=1).log1p())).alias("log_sqft_x_floors"),
])

# ── NEW v6 structural features ──────────────────────────────────────────
print("  Adding v6 structural features …")
df = df.with_columns([
    # Log of land sqft — captures lot size non-linearly
    pl.col("land_square_feet").clip(lower_bound=1).log1p().alias("log_land_sqft"),

    # Lot coverage ratio: how much of the land is the building?
    (pl.col("gross_square_feet") / pl.col("land_square_feet").clip(lower_bound=1))
        .clip(upper_bound=10).alias("lot_coverage"),

    # Volumetric proxy: sqft × floors (total internal volume)
    (pl.col("gross_square_feet") * pl.col("numfloors").clip(lower_bound=1)).log1p()
        .alias("bldg_vol_proxy"),

    # Prior price per sqft (very strong signal where available)
    (pl.col("prior_sale_price") / pl.col("gross_square_feet").clip(lower_bound=1))
        .alias("prior_price_psf"),
])
print("  v6 structural: log_land_sqft, lot_coverage, bldg_vol_proxy, prior_price_psf")

# ── 3. Time-based holdout ──────────────────────────────────────────────
print("\n[3/9] Time-based holdout (last 15%) …")
df_sorted = df.sort("sale_date")
n_hold    = int(len(df_sorted) * 0.15)
df_work   = df_sorted[:-n_hold]
df_hold   = df_sorted[-n_hold:]
print(f"  Work: {len(df_work):,}  |  Hold: {len(df_hold):,}")

# ── 3b. Impute prior_sale_price ────────────────────────────────────────
print("\n  Imputing prior_sale_price via assesstot …")
_has_both = df_work.filter(
    (pl.col("prior_sale_price") > 0) &
    pl.col("assesstot").is_not_null() & (pl.col("assesstot") > 0)
).with_columns((pl.col("prior_sale_price") / pl.col("assesstot")).alias("_pr"))
_ratio_df  = _has_both.group_by("borough").agg(pl.col("_pr").median().alias("ratio"))
_ratio_map = {int(r["borough"]): float(r["ratio"]) for r in _ratio_df.iter_rows(named=True)}
_glob_r    = float(_has_both["_pr"].median()) if len(_has_both) else 10.0
_rl = pl.DataFrame({"borough": list(_ratio_map.keys()), "_ir": list(_ratio_map.values())}) \
       .with_columns(pl.col("borough").cast(df_work.schema["borough"]))
def _impute(fr):
    fr = fr.join(_rl, on="borough", how="left").with_columns(pl.col("_ir").fill_null(_glob_r))
    return fr.with_columns(
        pl.when((pl.col("prior_sale_price").is_null() | (pl.col("prior_sale_price") == 0)) &
                pl.col("assesstot").is_not_null() & (pl.col("assesstot") > 0))
        .then(pl.col("assesstot") * pl.col("_ir")).otherwise(pl.col("prior_sale_price"))
        .alias("prior_sale_price")
    ).drop("_ir")
df_work = _impute(df_work); df_hold = _impute(df_hold)
print(f"  prior_sale_price coverage: {(df_work['prior_sale_price'] > 0).sum()/len(df_work)*100:.1f}%")

# ── 3c. Quarterly NTA temporal lookback features ───────────────────────
print("\n  Computing quarterly NTA lookback features (lag-1 / lag-2) …")
_psf_clip = 1.0
for _fr_name, _fr in [("work", df_work), ("hold", df_hold)]:
    pass  # will assign below

df_work = df_work.with_columns(
    ((pl.col("sale_year") - 2018) * 4 + (pl.col("sale_month") - 1) // 3).alias("_yrq")
)
df_hold = df_hold.with_columns(
    ((pl.col("sale_year") - 2018) * 4 + (pl.col("sale_month") - 1) // 3).alias("_yrq")
)

# Compute quarterly NTA stats from training data only (no holdout leakage)
_nta_q = (
    df_work
    .with_columns([
        pl.col("sale_price").log1p().alias("_log_sp"),
        (pl.col("sale_price") / pl.col("gross_square_feet").clip(lower_bound=_psf_clip)).alias("_psf"),
    ])
    .group_by(["ntacode", "_yrq"])
    .agg([
        pl.col("_log_sp").mean().alias("_mean_logp"),
        pl.col("_psf").median().alias("_median_psf"),
        pl.len().alias("_count"),
    ])
)

_global_logp = float(_nta_q["_mean_logp"].median())
_global_psf  = float(_nta_q["_median_psf"].median())
_global_cnt  = float(_nta_q["_count"].median())

_lag1 = (
    _nta_q.with_columns(pl.col("_yrq") + 1)
    .rename({"_mean_logp": "nta_lag1q_mean_logp",
             "_median_psf": "nta_lag1q_median_psf",
             "_count": "nta_lag1q_count"})
)
_lag2 = (
    _nta_q.with_columns(pl.col("_yrq") + 2)
    .select(["ntacode", "_yrq", pl.col("_mean_logp").alias("nta_lag2q_mean_logp")])
)

df_work = (
    df_work
    .join(_lag1, on=["ntacode", "_yrq"], how="left")
    .join(_lag2, on=["ntacode", "_yrq"], how="left")
)
df_hold = (
    df_hold
    .join(_lag1, on=["ntacode", "_yrq"], how="left")
    .join(_lag2, on=["ntacode", "_yrq"], how="left")
)

for _col, _fallback in [
    ("nta_lag1q_mean_logp",  _global_logp),
    ("nta_lag1q_median_psf", _global_psf),
    ("nta_lag1q_count",      _global_cnt),
    ("nta_lag2q_mean_logp",  _global_logp),
]:
    df_work = df_work.with_columns(pl.col(_col).fill_null(_fallback))
    df_hold  = df_hold.with_columns(pl.col(_col).fill_null(_fallback))

df_work = df_work.with_columns(
    (pl.col("nta_lag1q_mean_logp") - pl.col("nta_lag2q_mean_logp")).alias("nta_logp_momentum")
)
df_hold = df_hold.with_columns(
    (pl.col("nta_lag1q_mean_logp") - pl.col("nta_lag2q_mean_logp")).alias("nta_logp_momentum")
)
df_work = df_work.with_columns(pl.col("nta_logp_momentum").fill_null(0.0))
df_hold  = df_hold.with_columns(pl.col("nta_logp_momentum").fill_null(0.0))

# Save lookup for inference
_nta_q_lag1_save = {}
for _r in _lag1.iter_rows(named=True):
    _key = f"{_r['ntacode']}_{_r['_yrq']}"
    _ml  = _r["nta_lag1q_mean_logp"]
    _mp  = _r["nta_lag1q_median_psf"]
    _nc  = _r["nta_lag1q_count"]
    if _ml is None or _mp is None: continue
    _nta_q_lag1_save[_key] = {
        "mean_logp":  round(float(_ml), 6),
        "median_psf": round(float(_mp), 2),
        "count":      int(_nc) if _nc is not None else 0,
    }
_nta_q_lag2_save = {}
for _r in _lag2.iter_rows(named=True):
    _key = f"{_r['ntacode']}_{_r['_yrq']}"
    _ml2 = _r["nta_lag2q_mean_logp"]
    if _ml2 is None: continue
    _nta_q_lag2_save[_key] = round(float(_ml2), 6)

print(f"  Lag-1 entries: {len(_lag1)}  | Lag-2 entries: {len(_lag2)}")
work_null = df_work["nta_lag1q_mean_logp"].is_null().sum()
print(f"  nta_lag1q_mean_logp nulls after fill: {work_null}/{len(df_work)}")

# ── 4. Target encoding ────────────────────────────────────────────────
print("\n[4/9] Target encoding (bldgclass + borough_bldg + NTA + NTA×bldg) …")
LOG_TARGET      = "log_price"
df_work = df_work.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
df_hold = df_hold.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
global_mean_log = float(df_work[LOG_TARGET].mean())

# bldgclass encoding
bm_df   = df_work.group_by("bldgclass").agg(pl.col(LOG_TARGET).mean().alias("bldgclass_encoded"))
bm      = {r["bldgclass"]: r["bldgclass_encoded"] for r in bm_df.iter_rows(named=True)}
df_work = df_work.join(bm_df, on="bldgclass", how="left").with_columns(pl.col("bldgclass_encoded").fill_null(global_mean_log))
df_hold = df_hold.join(bm_df, on="bldgclass", how="left").with_columns(pl.col("bldgclass_encoded").fill_null(global_mean_log))

# borough×bldg encoding
df_work = df_work.with_columns((pl.col("borough").cast(pl.Utf8) + "_" + pl.col("bldgclass").str.slice(0,1)).alias("_bbk"))
df_hold = df_hold.with_columns((pl.col("borough").cast(pl.Utf8) + "_" + pl.col("bldgclass").str.slice(0,1)).alias("_bbk"))
bb_df   = df_work.group_by("_bbk").agg(pl.col(LOG_TARGET).mean().alias("borough_bldg_encoded"))
bb      = {r["_bbk"]: r["borough_bldg_encoded"] for r in bb_df.iter_rows(named=True)}
df_work = df_work.join(bb_df, on="_bbk", how="left").with_columns(pl.col("borough_bldg_encoded").fill_null(global_mean_log)).drop("_bbk")
df_hold = df_hold.join(bb_df, on="_bbk", how="left").with_columns(pl.col("borough_bldg_encoded").fill_null(global_mean_log)).drop("_bbk")

# ── NEW v6: NTA-level target encoding ─────────────────────────────────
if "ntacode" in df_work.columns:
    # NTA mean log-price
    nta_df   = df_work.group_by("ntacode").agg(pl.col(LOG_TARGET).mean().alias("nta_encoded"))
    nta_map  = {r["ntacode"]: r["nta_encoded"] for r in nta_df.iter_rows(named=True)}
    df_work  = df_work.join(nta_df, on="ntacode", how="left").with_columns(pl.col("nta_encoded").fill_null(global_mean_log))
    df_hold  = df_hold.join(nta_df, on="ntacode", how="left").with_columns(pl.col("nta_encoded").fill_null(global_mean_log))

    # NTA × building class category (fine-grained)
    df_work  = df_work.with_columns((pl.col("ntacode") + "_" + pl.col("bldgclass").str.slice(0,1)).alias("_ntab"))
    df_hold  = df_hold.with_columns((pl.col("ntacode") + "_" + pl.col("bldgclass").str.slice(0,1)).alias("_ntab"))
    ntab_df  = df_work.group_by("_ntab").agg(pl.col(LOG_TARGET).mean().alias("nta_bldg_encoded"))
    df_work  = df_work.join(ntab_df, on="_ntab", how="left").with_columns(pl.col("nta_bldg_encoded").fill_null(global_mean_log)).drop("_ntab")
    df_hold  = df_hold.join(ntab_df, on="_ntab", how="left").with_columns(pl.col("nta_bldg_encoded").fill_null(global_mean_log)).drop("_ntab")

    # NTA market stats: sale count + median $/sqft in training window
    nta_stats = (df_work.with_columns(
        (pl.col("sale_price") / pl.col("gross_square_feet").clip(lower_bound=1)).alias("_psf"))
        .group_by("ntacode")
        .agg([
            pl.count("sale_price").alias("nta_sale_count"),
            pl.col("_psf").median().alias("nta_median_psf"),
        ]))
    df_work = df_work.join(nta_stats, on="ntacode", how="left")
    df_hold = df_hold.join(nta_stats, on="ntacode", how="left")
    for c in ["nta_sale_count","nta_median_psf"]:
        med = float(df_work[c].drop_nulls().median() or 0)
        df_work = df_work.with_columns(pl.col(c).fill_null(med))
        df_hold = df_hold.with_columns(pl.col(c).fill_null(med))

    # ── v10: NTA price trend slope (OLS of mean log_price ~ sale_year) ──
    # Groups by (ntacode, sale_year), computes mean log_price per year per NTA,
    # then fits a simple OLS slope: slope = cov(year, price) / var(year)
    # Positive slope = appreciating NTA; negative = declining NTA
    trend_by_nta = (
        df_work.group_by(["ntacode", "sale_year"])
        .agg(pl.col(LOG_TARGET).mean().alias("yr_mean_logp"))
        .sort(["ntacode", "sale_year"])
    )
    # Vectorised OLS per NTA using numpy
    nta_trend_map = {}
    global_trend  = 0.0   # flat trend fallback for unseen NTAs
    for key, grp in trend_by_nta.partition_by("ntacode", as_dict=True).items():
        # Polars ≥0.19 returns tuple keys even for single-column partition
        nta_code = key[0] if isinstance(key, (tuple, list)) else key
        years  = grp["sale_year"].to_numpy().astype(np.float64)
        prices = grp["yr_mean_logp"].to_numpy().astype(np.float64)
        if len(years) >= 2:
            yr_c   = years  - years.mean()
            pr_c   = prices - prices.mean()
            slope  = float(np.dot(yr_c, pr_c) / (np.dot(yr_c, yr_c) + 1e-10))
        else:
            slope  = 0.0
        nta_trend_map[str(nta_code)] = round(slope, 6)
    global_trend = float(np.median(list(nta_trend_map.values())))

    trend_df = pl.DataFrame({
        "ntacode":               [str(k) for k in nta_trend_map.keys()],
        "nta_price_trend_slope": [float(v) for v in nta_trend_map.values()],
    })
    df_work = df_work.join(trend_df, on="ntacode", how="left").with_columns(
        pl.col("nta_price_trend_slope").fill_null(global_trend)
    )
    df_hold = df_hold.join(trend_df, on="ntacode", how="left").with_columns(
        pl.col("nta_price_trend_slope").fill_null(global_trend)
    )
    print(f"  NTA trend slopes: {len(nta_trend_map)} NTAs  "
          f"| range [{min(nta_trend_map.values()):.4f}, {max(nta_trend_map.values()):.4f}]"
          f"  | global_trend={global_trend:.4f}")

    nta_features = ["nta_encoded", "nta_bldg_encoded", "nta_sale_count",
                    "nta_median_psf", "nta_price_trend_slope"]
    print(f"  NTA encoding: {len(nta_map)} NTA codes  |  {len(df_work.filter(pl.col('nta_encoded') != global_mean_log)):,} rows matched")
    nta_map_save      = {k: round(float(v),6) for k,v in nta_map.items()}
    nta_bldg_map_save = {r["_ntab"]: round(float(r["nta_bldg_encoded"]),6)
                         for r in ntab_df.iter_rows(named=True)}
    nta_stats_save    = {r["ntacode"]: {"sale_count": int(r["nta_sale_count"]),
                                        "median_psf": round(float(r["nta_median_psf"]),2)}
                         for r in nta_stats.iter_rows(named=True)}
    nta_trend_save    = {k: round(float(v),6) for k,v in nta_trend_map.items()}
else:
    print("  ⚠ ntacode column not found — skipping NTA features")
    nta_features      = []
    nta_map_save      = {}
    nta_bldg_map_save = {}
    nta_stats_save    = {}
    nta_trend_save    = {}
    global_trend      = 0.0

# ── 5. Feature matrix ──────────────────────────────────────────────────
print("\n[5/9] Building feature matrix …")
V4_BASE = [
    "latitude","longitude","borough","building_age","numfloors",
    "gross_square_feet","land_square_feet","residential_units",
    "dist_subway_m","dist_school_m","dist_park_m","dist_hospital_m",
    "poi_count_500m","crime_rate_nta","noise_density_nta",
    "population_2020","median_income_nta","dist_bus_m",
    "renovated_since_2018","years_since_renovation",
    "dist_waterfront_m","dist_bike_lane_m","dist_elem_school_m",
    "dist_express_subway_m","nearest_station_is_express",
    "livability_complaint_rate","borough_income_deviation",
    "sale_year","sale_month_sin","sale_month_cos","mortgage_rate_30yr",
    "builtfar","residfar","commfar","facilfar","far_utilization",
    "has_elevator","is_condo","is_multifamily","is_single_fam","is_mixed_use",
    "airbnb_count_500m",
    "prior_sale_price","price_appreciation","years_since_prior_sale",
    "is_flip","school_district","district_avg_score","district_school_count",
    "has_prior_sale","assesstot","assessland",
    "poi_cafe_500m","poi_restaurant_500m","poi_gym_500m",
    "poi_grocery_500m","poi_bar_500m","poi_pharmacy_500m",
    *[f"log_{c}" for c in dist_cols],
    "dist_midtown_manhattan_m","dist_downtown_manhattan_m",
    "dist_downtown_brooklyn_m","dist_long_island_city_m",
    "is_manhattan","crime_x_manhattan","crime_x_non_manhattan",
    "walk_score_proxy","bldgclass_encoded","borough_bldg_encoded",
    "tree_count_200m","pm25_mean","no2_mean","hpd_viol_rate_nta",
]
V5_FEATS = ["sqft_per_floor","income_over_crime","density_index","log_sqft_x_floors"]
V6_FEATS = ["log_land_sqft","lot_coverage","bldg_vol_proxy","prior_price_psf"] + nta_features
# ── v11: Building health, construction activity, QoL, transit quality ──
V11_FEATS = [
    # HPD housing-maintenance code violations by ZIP (2022+, open, severity-classified)
    "hpd_class_b_viol_zip",    # Class B = hazardous (water, structural)
    "hpd_class_c_viol_zip",    # Class C = immediately hazardous (mold, lead, heat loss)
    "hpd_severity_score_zip",  # Weighted composite: C×3 + B×2 + A×1

    # DOB construction + renovation permits by ZIP (2022+)
    "dob_reno_permit_count",   # A1/A2 alteration permits → renovation signal
    "dob_newbld_permit_count", # NB new-building permits → development pressure

    # 311 quality-of-life signals by NTA (rodent + heat complaints)
    "rat_density_nta",         # Rodent complaints per 1000 residents (log)
    "heat_density_nta",        # Heat/hot-water complaints per 1000 residents (log)

    # MTA transit quality at nearest station
    "nearest_station_is_cbd",       # 1 = nearest subway is in CBD (Midtown/Downtown core)
    "nearest_station_route_count",  # Number of subway lines (hub vs local station)
    "nearest_station_is_ada",       # 1 = ADA accessible station
]
V12_FEATS = [
    "nta_lag1q_mean_logp",   # NTA mean log-price from previous quarter
    "nta_lag1q_median_psf",  # NTA median $/sqft from previous quarter
    "nta_lag1q_count",       # NTA sale count from previous quarter
    "nta_lag2q_mean_logp",   # NTA mean log-price from 2 quarters ago
    "nta_logp_momentum",     # Price trend: lag1 − lag2
]
FEATURE_NAMES = [f for f in (V4_BASE + V5_FEATS + V6_FEATS + V11_FEATS + V12_FEATS) if f in df_work.columns]
n_v11 = len([f for f in V11_FEATS if f in df_work.columns])
n_v12 = len([f for f in V12_FEATS if f in df_work.columns])
print(f"  Total features: {len(FEATURE_NAMES)}  "
      f"(+{len([f for f in V6_FEATS if f in df_work.columns])} v6, +{n_v11} v11, +{n_v12} v12)")

acris_cols    = ["prior_sale_price","price_appreciation","years_since_prior_sale"]
acris_medians = {c: float(df_work.filter(pl.col(c).is_not_null()&(pl.col(c)!=0))[c].median() or 0)
                 for c in acris_cols}
qol_cols      = ["crime_rate_nta","noise_density_nta","livability_complaint_rate"]
winsorize_p99 = {c: float(np.percentile(df_work[c].drop_nulls().to_numpy(), 99)) for c in qol_cols}

X_work = df_work.select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_work = df_work[LOG_TARGET].to_numpy().astype(np.float32)
X_hold = df_hold.select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_hold = df_hold[LOG_TARGET].to_numpy().astype(np.float32)
for col, cap in winsorize_p99.items():
    if col in FEATURE_NAMES:
        idx = FEATURE_NAMES.index(col)
        X_work[:,idx] = np.clip(X_work[:,idx], None, cap)
        X_hold[:,idx] = np.clip(X_hold[:,idx], None, cap)

# ── 6. OOF with 4 diverse learners ────────────────────────────────────
print("\n[6/9] Generating OOF predictions (spatial GroupKFold ×5) …")
groups = df_work["ntacode"].fill_null("UNK").to_numpy() if "ntacode" in df_work.columns \
         else np.zeros(len(X_work), dtype=str)
gkf    = GroupKFold(n_splits=10)

XGB_A = dict(objective="reg:squarederror", eval_metric="rmse",
             n_estimators=5000, learning_rate=0.02, max_depth=7,
             min_child_weight=8, subsample=0.80, colsample_bytree=0.60,
             gamma=0.1, reg_alpha=0.3, reg_lambda=1.5,
             early_stopping_rounds=400, random_state=42, n_jobs=-1)
XGB_B = dict(objective="reg:squarederror", eval_metric="rmse",
             n_estimators=5000, learning_rate=0.05, max_depth=4,
             min_child_weight=15, subsample=0.65, colsample_bytree=0.75,
             gamma=0.3, reg_alpha=1.0, reg_lambda=2.5,
             early_stopping_rounds=400, random_state=7, n_jobs=-1)
LGB_P = dict(objective="regression", metric="rmse",
             n_estimators=5000, learning_rate=0.04, num_leaves=127,
             min_child_samples=15, feature_fraction=0.70, bagging_fraction=0.80,
             bagging_freq=5, reg_alpha=0.3, reg_lambda=1.5,
             n_jobs=-1, random_state=42, verbose=-1)
CAT_P = dict(iterations=3000, learning_rate=0.025, depth=8,
             l2_leaf_reg=1.5, random_strength=0.5, bagging_temperature=0.8,
             border_count=64, early_stopping_rounds=400, random_seed=42, verbose=0)

oof_xa = np.zeros(len(X_work), dtype=np.float32)
oof_xb = np.zeros(len(X_work), dtype=np.float32)
oof_lg = np.zeros(len(X_work), dtype=np.float32)
oof_ct = np.zeros(len(X_work), dtype=np.float32)

for fold, (tr, va) in enumerate(gkf.split(X_work, y_work, groups), 1):
    Xtr, Xva, ytr, yva = X_work[tr], X_work[va], y_work[tr], y_work[va]
    ma = xgb.XGBRegressor(**XGB_A); ma.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False); oof_xa[va]=ma.predict(Xva)
    mb = xgb.XGBRegressor(**XGB_B); mb.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False); oof_xb[va]=mb.predict(Xva)
    ml = lgb.LGBMRegressor(**LGB_P)
    ml.fit(Xtr, ytr, eval_set=[(Xva,yva)], callbacks=[lgb.early_stopping(400,verbose=False),lgb.log_evaluation(-1)])
    oof_lg[va] = ml.predict(Xva)
    mc = CatBoostRegressor(**CAT_P); mc.fit(Xtr, ytr, eval_set=(Xva,yva), verbose=False); oof_ct[va]=mc.predict(Xva).astype(np.float32)
    print(f"  Fold {fold}: XGB-A {r2_score(yva,oof_xa[va]):.4f}  XGB-B {r2_score(yva,oof_xb[va]):.4f}  LGB {r2_score(yva,oof_lg[va]):.4f}  CAT {r2_score(yva,oof_ct[va]):.4f}")

for nm, oo in [("XGB-A",oof_xa),("XGB-B",oof_xb),("LGB",oof_lg),("CAT",oof_ct)]:
    print(f"  OOF {nm:<6}  R²={r2_score(y_work,oo):.4f}  MedAPE={medape(np.expm1(y_work),np.expm1(oo)):.2f}%")

# ── LightGBM meta-learner (non-linear blending) ────────────────────────
print("\n  Training LGB meta-learner on OOF stack …")
S_work = np.column_stack([oof_xa, oof_xb, oof_lg, oof_ct])
Sm_tr, Sm_va, ym_tr, ym_va = train_test_split(S_work, y_work, test_size=0.15, random_state=42)
META_LGB = dict(objective="regression", metric="rmse",
                n_estimators=500, learning_rate=0.05, num_leaves=15,
                min_child_samples=30, feature_fraction=1.0,
                reg_alpha=0.5, reg_lambda=2.0,
                n_jobs=-1, random_state=42, verbose=-1)
meta_lgb = lgb.LGBMRegressor(**META_LGB)
meta_lgb.fit(Sm_tr, ym_tr, eval_set=[(Sm_va, ym_va)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
oof_stack = meta_lgb.predict(S_work)
print(f"  OOF Stack(LGB-meta) R²={r2_score(y_work,oof_stack):.4f}  MedAPE={medape(np.expm1(y_work),np.expm1(oof_stack)):.2f}%")
print(f"  Meta feature importances: XGB-A={meta_lgb.feature_importances_[0]}  XGB-B={meta_lgb.feature_importances_[1]}  LGB={meta_lgb.feature_importances_[2]}  CAT={meta_lgb.feature_importances_[3]}")

# Also keep a Ridge for backward-compat baseline
ridge_pos = Ridge(alpha=1.0, positive=True)
ridge_pos.fit(S_work, y_work)
oof_ridge = ridge_pos.predict(S_work)
print(f"  OOF Stack(Ridge)    R²={r2_score(y_work,oof_ridge):.4f}  MedAPE={medape(np.expm1(y_work),np.expm1(oof_ridge)):.2f}%")

# ── 7. Retrain final models ────────────────────────────────────────────
print("\n[7/9] Retraining final 4-model ensemble …")
Xtr_f, Xva_f, ytr_f, yva_f = train_test_split(X_work, y_work, test_size=0.12, random_state=42)

print("  XGB-A …"); final_xa = xgb.XGBRegressor(**XGB_A); final_xa.fit(Xtr_f, ytr_f, eval_set=[(Xva_f,yva_f)], verbose=500)
print("  XGB-B …"); final_xb = xgb.XGBRegressor(**XGB_B); final_xb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f,yva_f)], verbose=500)
print("  LGB …")
final_lg = lgb.LGBMRegressor(**LGB_P)
final_lg.fit(Xtr_f, ytr_f, eval_set=[(Xva_f,yva_f)], callbacks=[lgb.early_stopping(400,verbose=False),lgb.log_evaluation(500)])
print("  CAT …"); final_ct = CatBoostRegressor(**CAT_P); final_ct.fit(Xtr_f, ytr_f, eval_set=(Xva_f,yva_f), verbose=False)

# ── 8. Luxury sub-model ────────────────────────────────────────────────
print("\n[8/9] Luxury sub-model (Manhattan ≥$2.5M) …")
LUXURY_THRESH = 2_500_000
lux_mask = (df_work["borough"]==1) & (df_work["sale_price"]>=LUXURY_THRESH)
X_lux = df_work.filter(lux_mask).select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_lux = df_work.filter(lux_mask)[LOG_TARGET].to_numpy().astype(np.float32)
print(f"  Luxury samples: {len(X_lux):,}")
has_luxury = False
if len(X_lux) >= 200:
    Xl_tr, Xl_va, yl_tr, yl_va = train_test_split(X_lux, y_lux, test_size=0.15, random_state=42)
    lux_xgb = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                                n_estimators=2000, learning_rate=0.015, max_depth=6,
                                min_child_weight=5, subsample=0.85, colsample_bytree=0.70,
                                gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
                                early_stopping_rounds=100, random_state=42, n_jobs=-1)
    lux_xgb.fit(Xl_tr, yl_tr, eval_set=[(Xl_va,yl_va)], verbose=False)
    print(f"  Luxury val: R²={r2_score(yl_va,lux_xgb.predict(Xl_va)):.4f}  "
          f"MedAPE={medape(np.expm1(yl_va),np.expm1(lux_xgb.predict(Xl_va))):.2f}%")
    lux_xgb.save_model(os.path.join(MODEL_DIR, "luxury_model.json"))
    has_luxury = True

# ── 9. Holdout evaluation ──────────────────────────────────────────────
print("\n[9/9] Holdout evaluation …")
p_xa = final_xa.predict(X_hold)
p_xb = final_xb.predict(X_hold)
p_lg = final_lg.predict(X_hold)
p_ct = final_ct.predict(X_hold).astype(np.float32)
S_hold = np.column_stack([p_xa, p_xb, p_lg, p_ct])

pred_lgb_meta = meta_lgb.predict(S_hold)
pred_ridge    = ridge_pos.predict(S_hold)

def ev(name, yt, yp):
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(np.expm1(yt), np.expm1(yp))
    mp = medape(np.expm1(yt), np.expm1(yp))
    print(f"  {name:<22}  R²={r2:.4f}  MAE=${mae:>12,.0f}  MedAPE={mp:.2f}%")
    return r2, mae, mp

print()
r2_xa,  mae_xa,  mp_xa  = ev("XGB-A (deep)",      y_hold, p_xa)
r2_xb,  mae_xb,  mp_xb  = ev("XGB-B (shallow)",   y_hold, p_xb)
r2_lg,  mae_lg,  mp_lg  = ev("LGB (wide)",         y_hold, p_lg)
r2_ct,  mae_ct,  mp_ct  = ev("CAT (high-cap)",     y_hold, p_ct)
r2_lm,  mae_lm,  mp_lm  = ev("Stack(LGB-meta) ✓", y_hold, pred_lgb_meta)
r2_rd,  mae_rd,  mp_rd  = ev("Stack(Ridge)",       y_hold, pred_ridge)

# Use LGB meta if it beats Ridge, else fall back
use_lgb_meta = r2_lm >= r2_rd
r2_stk, mae_stk, mp_stk = (r2_lm, mae_lm, mp_lm) if use_lgb_meta else (r2_rd, mae_rd, mp_rd)
pred_stack = pred_lgb_meta if use_lgb_meta else pred_ridge
print(f"\n  → Using {'LGB' if use_lgb_meta else 'Ridge'} meta-learner (R²={r2_stk:.4f})")

BOROUGH_MAP = {1:"Manhattan",2:"Bronx",3:"Brooklyn",4:"Queens",5:"Staten Island"}
boroughs_hold = df_hold["borough"].to_numpy(); prices_hold = np.expm1(y_hold)
segment_by_borough = {}
for bn, bname in BOROUGH_MAP.items():
    mask = boroughs_hold == bn
    if mask.sum() < 10: continue
    bor_mp = medape(prices_hold[mask], np.expm1(pred_stack[mask]))
    be = mean_absolute_error(prices_hold[mask], np.expm1(pred_stack[mask]))
    segment_by_borough[bname] = {"n": int(mask.sum()), "medape": round(bor_mp,2), "mae": round(be,0)}
    print(f"  {bname:<16}  n={mask.sum():>5}  R²={r2_score(y_hold[mask],pred_stack[mask]):.4f}  MedAPE={bor_mp:.2f}%")

TIER_BINS = [(0,500_000,"<$500K"),(500_000,1_000_000,"$500K–1M"),
             (1_000_000,3_000_000,"$1M–3M"),(3_000_000,10_000_000,"$3M–10M")]
segment_by_tier = {}
for lo, hi, label in TIER_BINS:
    mask = (prices_hold>=lo)&(prices_hold<hi)
    if mask.sum()<10: continue
    tm = medape(prices_hold[mask], np.expm1(pred_stack[mask]))
    te = mean_absolute_error(prices_hold[mask], np.expm1(pred_stack[mask]))
    segment_by_tier[label] = {"n": int(mask.sum()), "medape": round(tm,2), "mae": round(te,0)}

# ── Save ───────────────────────────────────────────────────────────────
stack_path = os.path.join(MODEL_DIR, "thaman_stack.pkl")
joblib.dump({
    "xgb_a": final_xa, "xgb_b": final_xb,
    "lgb":   final_lg, "cat":   final_ct,
    "meta":  meta_lgb if use_lgb_meta else ridge_pos,
    "meta_type": "lgb" if use_lgb_meta else "ridge",
    "version": "v12",
}, stack_path)
print(f"\n  thaman_stack.pkl saved  ({os.path.getsize(stack_path)/1e6:.1f} MB)")
final_xa.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))

meta_path = os.path.join(MODEL_DIR, "meta.json")
with open(meta_path) as f: meta = json.load(f)
meta.update({
    "feature_names": FEATURE_NAMES, "n_features": len(FEATURE_NAMES),
    "n_train": len(df_work), "n_holdout": len(df_hold),
    "walk_score_scaler": walk_score_scaler_params,
    "bldgclass_means":    {k: round(float(v),6) for k,v in bm.items()},
    "borough_bldg_means": {k: round(float(v),6) for k,v in bb.items()},
    "nta_means":          nta_map_save,
    "nta_bldg_means":     nta_bldg_map_save,
    "nta_stats":          nta_stats_save,
    "global_mean_log":    round(global_mean_log,6),
    "luxury_threshold":   LUXURY_THRESH, "has_luxury_model": has_luxury,
    "acris_medians":      {k: round(float(v),6) for k,v in acris_medians.items()},
    "winsorize_p99":      {k: round(float(v),6) for k,v in winsorize_p99.items()},
    "segment_by_borough": segment_by_borough,
    "segment_by_tier":    segment_by_tier,
    "stack": {
        "version": "v12",
        "base_learners": ["xgb_a","xgb_b","lightgbm","catboost"],
        "meta_learner": "lgb" if use_lgb_meta else "ridge",
        "r2_holdout":     round(r2_stk,4), "mae_holdout": round(mae_stk,0),
        "medape_holdout": round(mp_stk,2),
        "xgb_a":    {"r2_holdout":round(r2_xa,4),"medape_holdout":round(mp_xa,2),"best_round":int(final_xa.best_iteration+1)},
        "xgb_b":    {"r2_holdout":round(r2_xb,4),"medape_holdout":round(mp_xb,2),"best_round":int(final_xb.best_iteration+1)},
        "lightgbm": {"r2_holdout":round(r2_lg,4),"medape_holdout":round(mp_lg,2),"best_round":int(final_lg.best_iteration_)},
        "catboost": {"r2_holdout":round(r2_ct,4),"medape_holdout":round(mp_ct,2),"best_round":int(final_ct.best_iteration_+1)},
        "r2_improvement":     round(r2_stk - 0.645, 4),
        "medape_improvement": round(20.24 - mp_stk, 2),
    },
    # v12 quarterly NTA lookback — for inference
    "nta_lag_q_map":     _nta_q_lag1_save,
    "nta_lag_q2_map":    _nta_q_lag2_save,
    "nta_lag_q_globals": {
        "mean_logp":  round(_global_logp, 6),
        "median_psf": round(_global_psf, 2),
        "count":      round(_global_cnt, 1),
    },
    # v11 inference lookups — used by api/main.py _build_feature_row()
    "v11_zip_lookup": {
        "hpd_class_b_viol_zip":   {},   # populated below
        "hpd_class_c_viol_zip":   {},
        "hpd_severity_score_zip": {},
        "dob_reno_permit_count":  {},
        "dob_newbld_permit_count":{},
    },
    "v11_nta_lookup": {
        "rat_density_nta":  {},   # populated below
        "heat_density_nta": {},
    },
})

# Populate v11 inference lookups from training data medians
_v11_zip_cols  = ["hpd_class_b_viol_zip","hpd_class_c_viol_zip","hpd_severity_score_zip",
                   "dob_reno_permit_count","dob_newbld_permit_count"]
_v11_nta_cols  = ["rat_density_nta","heat_density_nta"]

if "zip_code" in df_work.columns:
    _zip_lookup = {}
    for _col in _v11_zip_cols:
        if _col in df_work.columns:
            _zip_agg = (df_work.with_columns(
                            pl.col("zip_code").cast(pl.Float64, strict=False)
                              .cast(pl.Int64, strict=False).cast(pl.Utf8).str.zfill(5))
                        .group_by("zip_code")
                        .agg(pl.col(_col).median().alias("med")))
            for _r in _zip_agg.iter_rows(named=True):
                if _r["zip_code"] not in _zip_lookup:
                    _zip_lookup[_r["zip_code"]] = {}
                _zip_lookup[_r["zip_code"]][_col] = round(float(_r["med"]),6)
    meta["v11_zip_lookup"] = _zip_lookup
    print(f"  v11_zip_lookup: {len(_zip_lookup)} ZIP entries")

if "ntacode" in df_work.columns:
    _nta_lookup = {}
    for _col in _v11_nta_cols:
        if _col in df_work.columns:
            _nta_agg = (df_work.group_by("ntacode")
                        .agg(pl.col(_col).median().alias("med")))
            for _r in _nta_agg.iter_rows(named=True):
                if _r["ntacode"] not in _nta_lookup:
                    _nta_lookup[_r["ntacode"]] = {}
                _nta_lookup[_r["ntacode"]][_col] = round(float(_r["med"]),6)
    meta["v11_nta_lookup"] = _nta_lookup
    print(f"  v11_nta_lookup: {len(_nta_lookup)} NTA entries")

# MTA station lookup for inference (lat/lon + features)
_mta_path = os.path.join(BASE, "data", "raw", "MTA_Subway_Stations_20260308.csv")
if os.path.exists(_mta_path):
    _mta = pl.read_csv(_mta_path)
    _mta = _mta.with_columns([
        pl.col("GTFS Latitude").cast(pl.Float64, strict=False).alias("_lat"),
        pl.col("GTFS Longitude").cast(pl.Float64, strict=False).alias("_lon"),
    ])
    if _mta["CBD"].dtype == pl.Boolean:
        _mta = _mta.with_columns(pl.col("CBD").cast(pl.Int32).alias("_cbd"))
    else:
        _mta = _mta.with_columns((pl.col("CBD").cast(pl.Utf8).str.to_lowercase()=="true").cast(pl.Int32).alias("_cbd"))
    _mta = _mta.with_columns(
        pl.col("Daytime Routes").str.strip_chars().str.split(" ").list.len().alias("_routes")
    )
    _mta = _mta.with_columns(
        (pl.col("ADA").cast(pl.Int32, strict=False) > 0).cast(pl.Int32).alias("_ada")
    )
    _mta_stations = []
    for _r in _mta.drop_nulls(["_lat","_lon"]).iter_rows(named=True):
        _mta_stations.append({
            "lat": float(_r["_lat"]), "lon": float(_r["_lon"]),
            "is_cbd": int(_r["_cbd"]), "route_count": int(_r["_routes"]), "is_ada": int(_r["_ada"])
        })
    meta["mta_stations"] = _mta_stations
    print(f"  mta_stations: {len(_mta_stations)} stored in meta.json")

meta["trained_at"] = datetime.date.today().strftime("%Y-%m-%d")
with open(meta_path,"w") as f: json.dump(meta, f, indent=2)
print("  meta.json updated ✓")

print("\n" + "="*70)
print("  THAMAN Stack v11 — Complete")
print("="*70)
print(f"\n  {'Model':<24}  {'R²':>7}  {'MAE':>14}  {'MedAPE':>9}")
print(f"  {'-'*60}")
for nm, r, m, mp_ in [("XGB-A",r2_xa,mae_xa,mp_xa),("XGB-B",r2_xb,mae_xb,mp_xb),
                       ("LGB",r2_lg,mae_lg,mp_lg),("CAT",r2_ct,mae_ct,mp_ct),
                       ("Stack v11",r2_stk,mae_stk,mp_stk)]:
    print(f"  {nm:<24}  {r:>7.4f}  ${m:>13,.0f}  {mp_:>8.2f}%")
print(f"\n  vs v11: ΔR²={r2_stk-0.645:+.4f}  ΔMedAPE={20.24-mp_stk:+.2f}pp")
print(f"  NTA features: {nta_features}")
print(f"  v11 new features: {[f for f in V11_FEATS if f in FEATURE_NAMES]}")
print(f"  v12 new features: {[f for f in V12_FEATS if f in FEATURE_NAMES]}")
print(f"  Total features: {len(FEATURE_NAMES)}")
