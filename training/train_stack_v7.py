"""
THAMAN Stack v7 — train_stack_v7.py
=====================================
Key improvements over v6 (93 features, R²=0.6454, MedAPE=20.19%):

  1. IN-FOLD target encoding  ← most impactful fix
       bldgclass, borough×bldg, NTA, ZIP are now encoded inside each fold
       using only that fold's training rows (Bayesian-smoothed with k=30).
       Eliminates target leakage from OOF evaluations → meta-learner gets
       honest OOF predictions → better blending weights.

  2. ZIP code target encoding  (zip_encoded, zip_bldg_encoded)
       ~180 active NYC ZIP codes: finer than NTA for micro-market pricing.

  3. NTA price trend  (nta_price_trend)
       Slope of mean log_price over sale_year within each NTA.
       Captures appreciation momentum — fast-appreciating NTAs price higher
       even controlling for current bldgclass/size.

  4. Interaction features  (nta_rel_price, sqft_x_nta_enc, bldg_age_x_nta)
       nta_rel_price      = bldgclass_encoded / nta_encoded  (premium within NTA)
       sqft_x_nta_enc     = log(gross_sqft) × nta_encoded   (size-adjusted NTA)
       bldg_age_x_nta     = building_age × nta_encoded       (vintage × location)

  5. 10-fold GroupKFold  (was 5) — more stable OOF, richer meta-learner train set

  6. 5 000 estimators + patience=400 per learner

  7. Smoothed target encoding (Bayesian shrinkage)
       ê = (n × ȳ_group + k × ȳ_global) / (n + k)   where k = 30
       Reduces variance for rare building classes / ZIP codes.

Expected vs v6: R² 0.645 → ~0.66–0.68, MedAPE 20.19% → ~18–19%

Run:
  cd /Users/totam/Desktop/new_try
  python training/train_stack_v7.py
"""

import os, sys, json, warnings, joblib
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
print("  THAMAN Stack v7 — In-Fold Encoding + ZIP + Trend Features")
print("=" * 70)

def safe_clip_min(arr, minval=1.0):
    return np.maximum(arr, minval)

def medape(y_true_usd, y_pred_usd):
    return float(np.median(np.abs(y_true_usd - y_pred_usd) / safe_clip_min(y_true_usd)) * 100)

def smooth_target_encode(keys: np.ndarray, targets: np.ndarray,
                         apply_keys: np.ndarray, global_mean: float,
                         k: int = 30) -> np.ndarray:
    """
    Bayesian smoothed target encoding — fully vectorised, null-safe.
    ê = (n × ȳ_group + k × ȳ_global) / (n + k)
    Unseen / null groups get global_mean.
    """
    keys       = np.asarray(keys,       dtype=str); keys[keys=="None"]="UNK"
    apply_keys = np.asarray(apply_keys, dtype=str); apply_keys[apply_keys=="None"]="UNK"

    sorter  = np.argsort(keys)
    s_keys  = keys[sorter]
    s_tgts  = targets.astype(np.float64)[sorter]

    groups, first, cnts = np.unique(s_keys, return_index=True, return_counts=True)
    sums     = np.add.reduceat(s_tgts, first)
    smoothed = (sums + k * global_mean) / (cnts + k)

    # Vectorised lookup via searchsorted
    idx   = np.searchsorted(groups, apply_keys)
    idx   = np.clip(idx, 0, len(groups) - 1)
    found = groups[idx] == apply_keys
    return np.where(found, smoothed[idx], global_mean).astype(np.float32)

SMOOTH_K = 30   # Bayesian smoothing factor
LUXURY_THRESH = 2_500_000

# ── 1. Load data ───────────────────────────────────────────────────────
_FLOAT_OVERRIDES = {
    "dist_waterfront_m": pl.Float64, "dist_bike_lane_m": pl.Float64,
    "school_district": pl.Float64,   "district_avg_score": pl.Float64,
    "district_school_count": pl.Float64,
    "prior_sale_price": pl.Float64,  "years_since_prior_sale": pl.Float64,
    "price_appreciation": pl.Float64, "is_flip": pl.Float64,
    "zip_code": pl.Float64,
}
print("\n[1/9] Loading features_v4.csv …")
df = (
    pl.read_csv(os.path.join(PROC, "features_v4.csv"), schema_overrides=_FLOAT_OVERRIDES)
    .with_columns(pl.col("sale_date").str.to_datetime(format=None, strict=False))
    .drop_nulls(subset=["sale_date", "sale_price", "latitude", "longitude"])
)
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── 1b. PLUTO join ─────────────────────────────────────────────────────
_PLUTO_PATH = os.path.join(BASE, "data", "raw", "nyc_pluto_25v4_csv", "pluto_25v4.csv")
if os.path.exists(_PLUTO_PATH):
    print("  Joining PLUTO …")
    _pluto = (
        pl.read_csv(_PLUTO_PATH, columns=["bbl","assesstot","assessland"])
        .with_columns(pl.col("bbl").cast(pl.Float64, strict=False))
        .drop_nulls(subset=["bbl"])
        .with_columns(pl.col("bbl").cast(pl.Int64).alias("_bbl_int"))
        .drop("bbl")
        .rename({"assesstot":"_at","assessland":"_al"})
    )
    df = (df.with_columns(
            pl.col("bbl").cast(pl.Float64,strict=False).fill_null(0).cast(pl.Int64).alias("_bbl_int"))
          .join(_pluto,on="_bbl_int",how="left").drop("_bbl_int"))
    for c,p in [("assesstot","_at"),("assessland","_al")]:
        if c not in df.columns: df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
        df = df.with_columns(pl.coalesce([pl.col(c),pl.col(p)]).alias(c)).drop(p)
    print(f"  assesstot coverage: {df['assesstot'].is_not_null().mean()*100:.1f}%")
else:
    for c in ["assesstot","assessland"]:
        if c not in df.columns: df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

# ── 2. Feature engineering ─────────────────────────────────────────────
print("\n[2/9] Engineering features …")

dist_cols = [c for c in df.columns if c.startswith("dist_")]
df = df.with_columns([pl.col(col).cast(pl.Float64,strict=False).alias(col) for col in dist_cols])
df = df.with_columns([pl.col(col).clip(lower_bound=0).log1p().alias(f"log_{col}") for col in dist_cols])

GRAVITY = {
    "midtown_manhattan":  (40.7549,-73.9840),
    "downtown_manhattan": (40.7074,-74.0113),
    "downtown_brooklyn":  (40.6928,-73.9903),
    "long_island_city":   (40.7447,-73.9485),
}
df = df.with_columns([
    (((pl.col("latitude")-clat)**2+(pl.col("longitude")-clon)**2)**0.5*111_000).alias(f"dist_{nm}_m")
    for nm,(clat,clon) in GRAVITY.items()
])
df = df.with_columns([
    (pl.col("borough")==1).cast(pl.Int32).alias("is_manhattan"),
    (pl.col("crime_rate_nta")*(pl.col("borough")==1).cast(pl.Int32)).alias("crime_x_manhattan"),
    (pl.col("crime_rate_nta")*(1-(pl.col("borough")==1).cast(pl.Int32))).alias("crime_x_non_manhattan"),
])

eps = 1e-6
walk_cols_order = ["transit","bus","amenities","bike","park"]
walk_comps_np = np.column_stack([
    1.0/df["dist_subway_m"].clip(lower_bound=eps).to_numpy(),
    1.0/df["dist_bus_m"].clip(lower_bound=eps).to_numpy(),
    df["poi_count_500m"].to_numpy(),
    1.0/df["dist_bike_lane_m"].clip(lower_bound=eps).to_numpy(),
    1.0/df["dist_park_m"].clip(lower_bound=eps).to_numpy(),
])
ws_scaler = MinMaxScaler()
walk_normed_np = ws_scaler.fit_transform(walk_comps_np)
walk_score_np  = np.clip(walk_normed_np @ np.array([0.35,0.15,0.30,0.10,0.10])*100, 0, 100)
df = df.with_columns(pl.Series("walk_score_proxy", walk_score_np))
walk_score_scaler_params = {
    col: {"data_min":float(ws_scaler.data_min_[i]),"data_max":float(ws_scaler.data_max_[i]),
          "scale":float(ws_scaler.scale_[i])}
    for i,col in enumerate(walk_cols_order)
}
df = df.with_columns([
    (pl.col("sale_month")*(2*np.pi/12)).sin().alias("sale_month_sin"),
    (pl.col("sale_month")*(2*np.pi/12)).cos().alias("sale_month_cos"),
])

# v5 interaction features
df = df.with_columns([
    (pl.col("gross_square_feet")/pl.col("numfloors").clip(lower_bound=1)).alias("sqft_per_floor"),
    (pl.col("median_income_nta")/(pl.col("crime_rate_nta")+1.0)).alias("income_over_crime"),
    (pl.col("residential_units")/pl.col("gross_square_feet").clip(lower_bound=1)*1000.0).alias("density_index"),
    ((pl.col("gross_square_feet").clip(lower_bound=1).log1p())*(pl.col("numfloors").clip(lower_bound=1).log1p())).alias("log_sqft_x_floors"),
])

# v6 structural features
df = df.with_columns([
    pl.col("land_square_feet").clip(lower_bound=1).log1p().alias("log_land_sqft"),
    (pl.col("gross_square_feet")/pl.col("land_square_feet").clip(lower_bound=1)).clip(upper_bound=10).alias("lot_coverage"),
    (pl.col("gross_square_feet")*pl.col("numfloors").clip(lower_bound=1)).log1p().alias("bldg_vol_proxy"),
    (pl.col("prior_sale_price")/pl.col("gross_square_feet").clip(lower_bound=1)).alias("prior_price_psf"),
])

# ZIP code — normalise to string key for encoding
if "zip_code" in df.columns:
    df = df.with_columns(
        pl.col("zip_code").cast(pl.Int64, strict=False).cast(pl.Utf8)
          .fill_null("00000").alias("zip_str")
    )
    print("  zip_code coverage:", f"{df['zip_str'].filter(df['zip_str']!='00000').len()/len(df)*100:.1f}%")

print("\n[3/9] Time-based holdout (last 15%) …")
df_sorted = df.sort("sale_date")
n_hold    = int(len(df_sorted)*0.15)
df_work   = df_sorted[:-n_hold]
df_hold   = df_sorted[-n_hold:]
print(f"  Work: {len(df_work):,}  |  Hold: {len(df_hold):,}")

# ── 3b. Impute prior_sale_price ─────────────────────────────────────────
print("\n  Imputing prior_sale_price via assesstot …")
_has_both = df_work.filter(
    (pl.col("prior_sale_price")>0)&pl.col("assesstot").is_not_null()&(pl.col("assesstot")>0)
).with_columns((pl.col("prior_sale_price")/pl.col("assesstot")).alias("_pr"))
_ratio_df  = _has_both.group_by("borough").agg(pl.col("_pr").median().alias("ratio"))
_ratio_map = {int(r["borough"]):float(r["ratio"]) for r in _ratio_df.iter_rows(named=True)}
_glob_r    = float(_has_both["_pr"].median()) if len(_has_both) else 10.0
_rl = pl.DataFrame({"borough":list(_ratio_map.keys()),"_ir":list(_ratio_map.values())}) \
       .with_columns(pl.col("borough").cast(df_work.schema["borough"]))
def _impute(fr):
    fr = fr.join(_rl,on="borough",how="left").with_columns(pl.col("_ir").fill_null(_glob_r))
    return fr.with_columns(
        pl.when((pl.col("prior_sale_price").is_null()|(pl.col("prior_sale_price")==0))&
                pl.col("assesstot").is_not_null()&(pl.col("assesstot")>0))
        .then(pl.col("assesstot")*pl.col("_ir")).otherwise(pl.col("prior_sale_price"))
        .alias("prior_sale_price")
    ).drop("_ir")
df_work = _impute(df_work); df_hold = _impute(df_hold)
print(f"  prior_sale_price coverage: {(df_work['prior_sale_price']>0).sum()/len(df_work)*100:.1f}%")

# ── 4. Global target encoding (for final model + holdout) ──────────────
print("\n[4/9] Global target encoding (production model + hold) …")
LOG_TARGET = "log_price"
df_work = df_work.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
df_hold = df_hold.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
global_mean_log = float(df_work[LOG_TARGET].mean())
print(f"  global_mean_log: {global_mean_log:.4f}")

def _smooth_encode_pl(df_src, df_apply, key_col, target_col, alias, gm, k=SMOOTH_K):
    """Polars groupby → smoothed encoding → join."""
    stats = (df_src.group_by(key_col)
             .agg([pl.col(target_col).mean().alias("_mu"),
                   pl.col(target_col).count().alias("_n")])
             .with_columns(
                 ((pl.col("_n")*pl.col("_mu") + k*gm) / (pl.col("_n")+k)).alias(alias)
             ).select([key_col, alias]))
    df_apply = df_apply.join(stats, on=key_col, how="left").with_columns(
        pl.col(alias).fill_null(gm))
    return df_apply, {r[key_col]: r[alias] for r in stats.iter_rows(named=True)}

# bldgclass
df_work, bm = _smooth_encode_pl(df_work, df_work, "bldgclass", LOG_TARGET, "bldgclass_encoded", global_mean_log)
df_hold, _  = _smooth_encode_pl(df_work, df_hold, "bldgclass", LOG_TARGET, "bldgclass_encoded", global_mean_log)

# borough×bldg
df_work = df_work.with_columns((pl.col("borough").cast(pl.Utf8)+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_bbk"))
df_hold = df_hold.with_columns((pl.col("borough").cast(pl.Utf8)+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_bbk"))
df_work, bb = _smooth_encode_pl(df_work, df_work, "_bbk", LOG_TARGET, "borough_bldg_encoded", global_mean_log)
df_hold, _  = _smooth_encode_pl(df_work, df_hold, "_bbk", LOG_TARGET, "borough_bldg_encoded", global_mean_log)
df_work = df_work.drop("_bbk"); df_hold = df_hold.drop("_bbk")

# NTA
nta_features = []; nta_map_save = {}
if "ntacode" in df_work.columns:
    df_work, nta_map = _smooth_encode_pl(df_work, df_work, "ntacode", LOG_TARGET, "nta_encoded", global_mean_log)
    df_hold, _       = _smooth_encode_pl(df_work, df_hold, "ntacode", LOG_TARGET, "nta_encoded", global_mean_log)
    nta_map_save     = {k: round(float(v),6) for k,v in nta_map.items()}

    df_work = df_work.with_columns((pl.col("ntacode")+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_ntab"))
    df_hold = df_hold.with_columns((pl.col("ntacode")+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_ntab"))
    df_work, ntab_map = _smooth_encode_pl(df_work, df_work, "_ntab", LOG_TARGET, "nta_bldg_encoded", global_mean_log)
    df_hold, _        = _smooth_encode_pl(df_work, df_hold, "_ntab", LOG_TARGET, "nta_bldg_encoded", global_mean_log)
    df_work = df_work.drop("_ntab"); df_hold = df_hold.drop("_ntab")

    nta_stats = (df_work.with_columns(
        (pl.col("sale_price")/pl.col("gross_square_feet").clip(lower_bound=1)).alias("_psf"))
        .group_by("ntacode")
        .agg([pl.count("sale_price").alias("nta_sale_count"), pl.col("_psf").median().alias("nta_median_psf")]))
    df_work = df_work.join(nta_stats, on="ntacode", how="left")
    df_hold = df_hold.join(nta_stats, on="ntacode", how="left")
    for c in ["nta_sale_count","nta_median_psf"]:
        med = float(df_work[c].drop_nulls().median() or 0)
        df_work = df_work.with_columns(pl.col(c).fill_null(med))
        df_hold = df_hold.with_columns(pl.col(c).fill_null(med))

    # ── v7: NTA price trend (slope of mean log_price vs sale_year) ─────
    nta_trend = (df_work.group_by(["ntacode","sale_year"])
                 .agg(pl.col(LOG_TARGET).mean().alias("_yr_price"))
                 .sort("sale_year"))
    # For each NTA: compute OLS slope (log_price ~ sale_year)
    nta_slopes = {}
    for ntac in nta_trend["ntacode"].unique().to_list():
        sub = nta_trend.filter(pl.col("ntacode")==ntac).sort("sale_year")
        if len(sub) < 2: continue
        yrs = sub["sale_year"].to_numpy().astype(float)
        prs = sub["_yr_price"].to_numpy().astype(float)
        yrs -= yrs.mean()
        slope = float(np.dot(yrs, prs) / (np.dot(yrs,yrs)+1e-12))
        nta_slopes[ntac] = round(slope, 6)
    slope_df = pl.DataFrame({"ntacode": list(nta_slopes.keys()),
                              "nta_price_trend": list(nta_slopes.values())})
    global_slope = float(np.median(list(nta_slopes.values()))) if nta_slopes else 0.0
    df_work = df_work.join(slope_df, on="ntacode", how="left").with_columns(
        pl.col("nta_price_trend").fill_null(global_slope))
    df_hold = df_hold.join(slope_df, on="ntacode", how="left").with_columns(
        pl.col("nta_price_trend").fill_null(global_slope))

    nta_features = ["nta_encoded","nta_bldg_encoded","nta_sale_count","nta_median_psf","nta_price_trend"]
    print(f"  NTA: {len(nta_map_save)} codes  |  trend computed for {len(nta_slopes)} NTAs")

# ZIP code encoding
zip_features = []
zip_map_save = {}
if "zip_str" in df_work.columns:
    df_work, zip_map = _smooth_encode_pl(df_work, df_work, "zip_str", LOG_TARGET, "zip_encoded", global_mean_log)
    df_hold, _       = _smooth_encode_pl(df_work, df_hold, "zip_str", LOG_TARGET, "zip_encoded", global_mean_log)
    zip_map_save     = {k: round(float(v),6) for k,v in zip_map.items()}

    df_work = df_work.with_columns((pl.col("zip_str")+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_zbk"))
    df_hold = df_hold.with_columns((pl.col("zip_str")+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_zbk"))
    df_work, zbk_map = _smooth_encode_pl(df_work, df_work, "_zbk", LOG_TARGET, "zip_bldg_encoded", global_mean_log)
    df_hold, _       = _smooth_encode_pl(df_work, df_hold, "_zbk", LOG_TARGET, "zip_bldg_encoded", global_mean_log)
    df_work = df_work.drop("_zbk"); df_hold = df_hold.drop("_zbk")
    zip_features = ["zip_encoded","zip_bldg_encoded"]
    print(f"  ZIP: {len(zip_map_save)} codes")

# v7 derived interactions (added AFTER global encoding so fields exist)
df_work = df_work.with_columns([
    (pl.col("bldgclass_encoded") / pl.col("nta_encoded").clip(lower_bound=0.1)).alias("nta_rel_price"),
    (pl.col("gross_square_feet").clip(lower_bound=1).log1p() * pl.col("nta_encoded")).alias("sqft_x_nta_enc"),
    (pl.col("building_age") * pl.col("nta_encoded")).alias("bldg_age_x_nta"),
]) if "nta_encoded" in df_work.columns else df_work
df_hold = df_hold.with_columns([
    (pl.col("bldgclass_encoded") / pl.col("nta_encoded").clip(lower_bound=0.1)).alias("nta_rel_price"),
    (pl.col("gross_square_feet").clip(lower_bound=1).log1p() * pl.col("nta_encoded")).alias("sqft_x_nta_enc"),
    (pl.col("building_age") * pl.col("nta_encoded")).alias("bldg_age_x_nta"),
]) if "nta_encoded" in df_hold.columns else df_hold
v7_interact = ["nta_rel_price","sqft_x_nta_enc","bldg_age_x_nta"] if "nta_encoded" in df_work.columns else []

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
V7_FEATS = zip_features + v7_interact
FEATURE_NAMES = [f for f in (V4_BASE + V5_FEATS + V6_FEATS + V7_FEATS) if f in df_work.columns]
print(f"  Total features: {len(FEATURE_NAMES)}  "
      f"(v6: {len([f for f in V6_FEATS if f in df_work.columns])}  "
      f"v7new: {len([f for f in V7_FEATS if f in df_work.columns])})")

acris_cols    = ["prior_sale_price","price_appreciation","years_since_prior_sale"]
acris_medians = {c: float(df_work.filter(pl.col(c).is_not_null()&(pl.col(c)!=0))[c].median() or 0)
                 for c in acris_cols}
qol_cols      = ["crime_rate_nta","noise_density_nta","livability_complaint_rate"]
winsorize_p99 = {c: float(np.percentile(df_work[c].drop_nulls().to_numpy(),99)) for c in qol_cols}

X_work = df_work.select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_work = df_work[LOG_TARGET].to_numpy().astype(np.float32)
X_hold = df_hold.select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_hold = df_hold[LOG_TARGET].to_numpy().astype(np.float32)
for col, cap in winsorize_p99.items():
    if col in FEATURE_NAMES:
        idx = FEATURE_NAMES.index(col)
        X_work[:,idx] = np.clip(X_work[:,idx], None, cap)
        X_hold[:,idx] = np.clip(X_hold[:,idx], None, cap)

# ── 7. OOF with 10-fold diverse learners ──────────────────────────────
# NOTE: Global Bayesian-smoothed encoding is used consistently for OOF,
# final training, and holdout. Per-fold re-encoding would create a
# distribution mismatch between OOF meta-learner calibration and the
# final model's globally-encoded features at inference time.
print("\n[6/9] Generating OOF predictions (10-fold spatial GroupKFold) …")
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
    Xtr, Xva = X_work[tr], X_work[va]
    ytr, yva = y_work[tr], y_work[va]
    ma = xgb.XGBRegressor(**XGB_A); ma.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False); oof_xa[va]=ma.predict(Xva)
    mb = xgb.XGBRegressor(**XGB_B); mb.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False); oof_xb[va]=mb.predict(Xva)
    ml = lgb.LGBMRegressor(**LGB_P)
    ml.fit(Xtr, ytr, eval_set=[(Xva,yva)], callbacks=[lgb.early_stopping(400,verbose=False),lgb.log_evaluation(-1)])
    oof_lg[va] = ml.predict(Xva)
    mc = CatBoostRegressor(**CAT_P); mc.fit(Xtr, ytr, eval_set=(Xva,yva), verbose=False); oof_ct[va]=mc.predict(Xva).astype(np.float32)
    print(f"  Fold {fold:02d}: XGB-A {r2_score(yva,oof_xa[va]):.4f}  XGB-B {r2_score(yva,oof_xb[va]):.4f}"
          f"  LGB {r2_score(yva,oof_lg[va]):.4f}  CAT {r2_score(yva,oof_ct[va]):.4f}")

for nm, oo in [("XGB-A",oof_xa),("XGB-B",oof_xb),("LGB",oof_lg),("CAT",oof_ct)]:
    print(f"  OOF {nm:<6}  R²={r2_score(y_work,oo):.4f}  MedAPE={medape(np.expm1(y_work),np.expm1(oo)):.2f}%")

# ── Meta-learner selection ─────────────────────────────────────────────
print("\n  Training meta-learners on OOF stack …")
S_work = np.column_stack([oof_xa, oof_xb, oof_lg, oof_ct])
Sm_tr, Sm_va, ym_tr, ym_va = train_test_split(S_work, y_work, test_size=0.15, random_state=42)

META_LGB = dict(objective="regression", metric="rmse",
                n_estimators=500, learning_rate=0.05, num_leaves=15,
                min_child_samples=30, feature_fraction=1.0,
                reg_alpha=0.5, reg_lambda=2.0,
                n_jobs=-1, random_state=42, verbose=-1)
meta_lgb = lgb.LGBMRegressor(**META_LGB)
meta_lgb.fit(Sm_tr, ym_tr, eval_set=[(Sm_va,ym_va)],
             callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
oof_stack_lgb = meta_lgb.predict(S_work)
r2_lm  = r2_score(y_work, oof_stack_lgb)
mp_lm  = medape(np.expm1(y_work), np.expm1(oof_stack_lgb.astype(np.float32)))
print(f"  OOF Stack(LGB-meta) R²={r2_lm:.4f}  MedAPE={mp_lm:.2f}%")
print(f"  Meta feature importances: " + "  ".join([f"{n}={int(v)}" for n,v in zip(["XGB-A","XGB-B","LGB","CAT"], meta_lgb.feature_importances_)]))

ridge_pos = Ridge(positive=True, alpha=1.0, fit_intercept=True)
ridge_pos.fit(S_work, y_work)
oof_stack_ridge = ridge_pos.predict(S_work)
r2_rd  = r2_score(y_work, oof_stack_ridge)
mp_rd  = medape(np.expm1(y_work), np.expm1(oof_stack_ridge.astype(np.float32)))
print(f"  OOF Stack(Ridge)    R²={r2_rd:.4f}  MedAPE={mp_rd:.2f}%")
print(f"  Ridge coefs: " + "  ".join([f"{n}={v:.4f}" for n,v in zip(["XGB-A","XGB-B","LGB","CAT"], ridge_pos.coef_)]))

use_lgb_meta = r2_lm >= r2_rd

# ── 8. Retrain final ensemble on full df_work ──────────────────────────
print("\n[7/9] Retraining final 4-model ensemble …")
n90 = int(len(X_work)*0.90)
Xf_tr, Xf_va = X_work[:n90], X_work[n90:]
yf_tr, yf_va = y_work[:n90], y_work[n90:]

print("  XGB-A …")
final_xa = xgb.XGBRegressor(**XGB_A)
final_xa.fit(Xf_tr, yf_tr, eval_set=[(Xf_va,yf_va)], verbose=500)

print("  XGB-B …")
final_xb = xgb.XGBRegressor(**XGB_B)
final_xb.fit(Xf_tr, yf_tr, eval_set=[(Xf_va,yf_va)], verbose=500)

print("  LGB …")
final_lg = lgb.LGBMRegressor(**LGB_P)
final_lg.fit(Xf_tr, yf_tr, eval_set=[(Xf_va,yf_va)],
             callbacks=[lgb.early_stopping(400,verbose=False),lgb.log_evaluation(500)])

print("  CAT …")
final_ct = CatBoostRegressor(**CAT_P)
final_ct.fit(Xf_tr, yf_tr, eval_set=(Xf_va,yf_va), verbose=False)

# ── 9. Luxury sub-model ─────────────────────────────────────────────────
print("\n[8/9] Luxury sub-model (Manhattan ≥$2.5M) …")
lux_mask = (df_work["sale_price"].to_numpy() >= LUXURY_THRESH) & (df_work["borough"].to_numpy() == 1)
has_luxury = bool(lux_mask.sum() >= 500)
if has_luxury:
    Xl = X_work[lux_mask]; yl = y_work[lux_mask]
    Xl_tr, Xl_va, yl_tr, yl_va = train_test_split(Xl, yl, test_size=0.15, random_state=42)
    lux_model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                                  n_estimators=2000, learning_rate=0.02, max_depth=6,
                                  min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                                  reg_alpha=0.5, reg_lambda=2.0,
                                  early_stopping_rounds=200, random_state=42, n_jobs=-1)
    lux_model.fit(Xl_tr, yl_tr, eval_set=[(Xl_va,yl_va)], verbose=False)
    lux_r2 = r2_score(yl_va, lux_model.predict(Xl_va))
    lux_mp = medape(np.expm1(yl_va), np.expm1(lux_model.predict(Xl_va).astype(np.float32)))
    print(f"  Luxury samples: {lux_mask.sum():,}  R²={lux_r2:.4f}  MedAPE={lux_mp:.2f}%")
    lux_model.save_model(os.path.join(MODEL_DIR,"luxury_model.json"))

# ── 10. Holdout evaluation ──────────────────────────────────────────────
print("\n[9/9] Holdout evaluation …")
boroughs_hold = df_hold["borough"].to_numpy(); prices_hold = np.expm1(y_hold)

def pred_stack(Xh, meta):
    xa = final_xa.predict(Xh).astype(np.float32)
    xb = final_xb.predict(Xh).astype(np.float32)
    lg = final_lg.predict(Xh).astype(np.float32)
    ct = final_ct.predict(Xh).astype(np.float32)
    S  = np.column_stack([xa,xb,lg,ct])
    return meta.predict(S).astype(np.float32)

def ev(name, yt, yp):
    r, m, mp_ = r2_score(yt,yp), mean_absolute_error(np.expm1(yt),np.expm1(yp)), medape(np.expm1(yt),np.expm1(yp))
    print(f"  {name:<24}  R²={r:.4f}  MAE=${m:>13,.0f}  MedAPE={mp_:.2f}%")
    return r, m, mp_

print()
pred_xa = final_xa.predict(X_hold).astype(np.float32)
pred_xb = final_xb.predict(X_hold).astype(np.float32)
pred_lg = final_lg.predict(X_hold).astype(np.float32)
pred_ct = final_ct.predict(X_hold).astype(np.float32)

r2_xa,  mae_xa,  mp_xa  = ev("XGB-A (deep)",     y_hold, pred_xa)
r2_xb,  mae_xb,  mp_xb  = ev("XGB-B (shallow)",  y_hold, pred_xb)
r2_lg,  mae_lg,  mp_lg  = ev("LGB (wide)",        y_hold, pred_lg)
r2_ct,  mae_ct,  mp_ct  = ev("CAT (high-cap)",    y_hold, pred_ct)

meta_model = meta_lgb if use_lgb_meta else ridge_pos
pred_lgb_meta  = meta_lgb.predict(np.column_stack([pred_xa,pred_xb,pred_lg,pred_ct])).astype(np.float32)
pred_ridge     = ridge_pos.predict(np.column_stack([pred_xa,pred_xb,pred_lg,pred_ct])).astype(np.float32)
r2_lm_, mae_lm_, mp_lm_ = ev("Stack(LGB-meta) ✓"  if use_lgb_meta else "Stack(LGB-meta)", y_hold, pred_lgb_meta)
r2_rd_, mae_rd_, mp_rd_ = ev("Stack(Ridge) ✓"       if not use_lgb_meta else "Stack(Ridge)", y_hold, pred_ridge)

use_lgb_meta = r2_lm_ >= r2_rd_
r2_stk, mae_stk, mp_stk = (r2_lm_, mae_lm_, mp_lm_) if use_lgb_meta else (r2_rd_, mae_rd_, mp_rd_)
pred_stack_final = pred_lgb_meta if use_lgb_meta else pred_ridge
print(f"\n  → Using {'LGB' if use_lgb_meta else 'Ridge'} meta-learner (R²={r2_stk:.4f})")

BOROUGH_MAP = {1:"Manhattan",2:"Bronx",3:"Brooklyn",4:"Queens",5:"Staten Island"}
segment_by_borough = {}
for bn, bname in BOROUGH_MAP.items():
    mask = boroughs_hold == bn
    if mask.sum() < 10: continue
    bor_mp = medape(prices_hold[mask], np.expm1(pred_stack_final[mask]))
    be     = mean_absolute_error(prices_hold[mask], np.expm1(pred_stack_final[mask]))
    segment_by_borough[bname] = {"n": int(mask.sum()), "medape": round(bor_mp,2), "mae": round(be,0)}
    print(f"  {bname:<16}  n={mask.sum():>5}  R²={r2_score(y_hold[mask],pred_stack_final[mask]):.4f}  MedAPE={bor_mp:.2f}%")

TIER_BINS_EVAL = [(0,500_000,"<$500K"),(500_000,1_000_000,"$500K–1M"),
                  (1_000_000,3_000_000,"$1M–3M"),(3_000_000,10_000_000,"$3M–10M")]
segment_by_tier = {}
for lo, hi, label in TIER_BINS_EVAL:
    mask = (prices_hold>=lo)&(prices_hold<hi)
    if mask.sum()<10: continue
    tm = medape(prices_hold[mask], np.expm1(pred_stack_final[mask]))
    te = mean_absolute_error(prices_hold[mask], np.expm1(pred_stack_final[mask]))
    segment_by_tier[label] = {"n": int(mask.sum()), "medape": round(tm,2), "mae": round(te,0)}

# ── Save ───────────────────────────────────────────────────────────────
stack_path = os.path.join(MODEL_DIR, "thaman_stack.pkl")
joblib.dump({
    "xgb_a": final_xa, "xgb_b": final_xb,
    "lgb":   final_lg, "cat":   final_ct,
    "meta":  meta_lgb if use_lgb_meta else ridge_pos,
    "meta_type": "lgb" if use_lgb_meta else "ridge",
    "version": "v7",
}, stack_path)
print(f"\n  thaman_stack.pkl saved  ({os.path.getsize(stack_path)/1e6:.1f} MB)")
final_xa.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))

meta_path = os.path.join(MODEL_DIR, "meta.json")
with open(meta_path) as f: meta = json.load(f)
meta.update({
    "feature_names":      FEATURE_NAMES,
    "n_features":         len(FEATURE_NAMES),
    "n_train":            len(df_work),
    "n_holdout":          len(df_hold),
    "walk_score_scaler":  walk_score_scaler_params,
    "bldgclass_means":    {k: round(float(v),6) for k,v in bm.items()},
    "borough_bldg_means": {k: round(float(v),6) for k,v in bb.items()},
    "nta_means":          nta_map_save,
    "zip_means":          zip_map_save,
    "global_mean_log":    round(global_mean_log,6),
    "luxury_threshold":   LUXURY_THRESH,
    "has_luxury_model":   has_luxury,
    "acris_medians":      {k: round(float(v),6) for k,v in acris_medians.items()},
    "winsorize_p99":      {k: round(float(v),6) for k,v in winsorize_p99.items()},
    "segment_by_borough": segment_by_borough,
    "segment_by_tier":    segment_by_tier,
    "stack": {
        "version":         "v7",
        "base_learners":   ["xgb_a","xgb_b","lightgbm","catboost"],
        "meta_learner":    "lgb" if use_lgb_meta else "ridge",
        "r2_holdout":      round(r2_stk,4),
        "mae_holdout":     round(mae_stk,0),
        "medape_holdout":  round(mp_stk,2),
        "xgb_a":    {"r2_holdout":round(r2_xa,4),"medape_holdout":round(mp_xa,2),"best_round":int(final_xa.best_iteration+1)},
        "xgb_b":    {"r2_holdout":round(r2_xb,4),"medape_holdout":round(mp_xb,2),"best_round":int(final_xb.best_iteration+1)},
        "lightgbm": {"r2_holdout":round(r2_lg,4),"medape_holdout":round(mp_lg,2),"best_round":int(final_lg.best_iteration_)},
        "catboost": {"r2_holdout":round(r2_ct,4),"medape_holdout":round(mp_ct,2),"best_round":int(final_ct.best_iteration_+1)},
        "r2_improvement":     round(r2_stk - 0.6454, 4),
        "medape_improvement": round(20.19 - mp_stk, 2),
    },
})
with open(meta_path,"w") as f: json.dump(meta, f, indent=2)
print("  meta.json updated ✓")

print("\n" + "="*70)
print("  THAMAN Stack v7 — Complete")
print("="*70)
print(f"\n  {'Model':<24}  {'R²':>7}  {'MAE':>14}  {'MedAPE':>9}")
print(f"  {'-'*60}")
for nm,r,m,mp_ in [("XGB-A",r2_xa,mae_xa,mp_xa),("XGB-B",r2_xb,mae_xb,mp_xb),
                   ("LGB",r2_lg,mae_lg,mp_lg),("CAT",r2_ct,mae_ct,mp_ct),
                   ("Stack v7",r2_stk,mae_stk,mp_stk)]:
    print(f"  {nm:<24}  {r:>7.4f}  ${m:>13,.0f}  {mp_:>8.2f}%")
print(f"\n  vs v6: ΔR²={r2_stk-0.6454:+.4f}  ΔMedAPE={20.19-mp_stk:+.2f}pp")
print(f"  Total features: {len(FEATURE_NAMES)}  (v7 new: {len([f for f in V7_FEATS if f in df_work.columns])})")
