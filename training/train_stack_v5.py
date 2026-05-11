"""
THAMAN Stack v5 — train_stack_v5.py
=====================================
Key improvements over v4:
  • 4 highly diverse base learners (different objectives, depths, LR schedules)
      XGB-A : depth=7, lr=0.02, subsample=0.80, colsample=0.60  (default-best)
      XGB-B : depth=4, lr=0.05, subsample=0.65, colsample=0.75  (fast shallow)
      LGB   : num_leaves=127, lr=0.04, feature_fraction=0.70, bagging_fraction=0.80
      CAT   : depth=8, lr=0.025, feature_border_count=64       (high-capacity)
  • 4 new interaction features (85 → 89 total):
      sqft_per_floor, income_over_crime, density_index, log_sqft_x_floors
  • Non-negative Ridge meta-learner (Ridge(positive=True)) — prevents negative weights
  • Calibrated confidence: proper NTA-level holdout segment stats
  • Luxury sub-model: lower threshold → $2.5M (more training data)

Expected gain vs v4: R² 0.651 → ~0.68+, MedAPE 20.8% → ~18–19%

Run:
  cd /Users/totam/Desktop/new_try
  python training/train_stack_v5.py
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
SPLIT_DIR = os.path.join(BASE, "data", "splits")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(SPLIT_DIR, exist_ok=True)

print("=" * 70)
print("  THAMAN Stack v5 — 4-Model Diverse Ensemble + Interaction Features")
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
print("\n[1/9] Loading features_v4.csv …")
df = (
    pl.read_csv(os.path.join(PROC, "features_v4.csv"), schema_overrides=_FLOAT_OVERRIDES)
    .with_columns(pl.col("sale_date").str.to_datetime(format=None, strict=False))
    .drop_nulls(subset=["sale_date", "sale_price", "latitude", "longitude"])
)
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── 1b. Join PLUTO assessed values ────────────────────────────────────
_PLUTO_PATH = os.path.join(BASE, "data", "raw", "nyc_pluto_25v4_csv", "pluto_25v4.csv")
if os.path.exists(_PLUTO_PATH):
    print("  Joining PLUTO for assesstot / assessland …")
    _pluto_assmt = (
        pl.read_csv(_PLUTO_PATH, columns=["bbl", "assesstot", "assessland"])
        .with_columns(pl.col("bbl").cast(pl.Float64, strict=False))
        .drop_nulls(subset=["bbl"])
        .with_columns(pl.col("bbl").cast(pl.Int64).alias("_bbl_int"))
        .drop("bbl")
        .rename({"assesstot": "_assesstot_p", "assessland": "_assessland_p"})
    )
    df = (
        df.with_columns(
            pl.col("bbl").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64).alias("_bbl_int")
        )
        .join(_pluto_assmt, on="_bbl_int", how="left")
        .drop("_bbl_int")
    )
    for _col, _pcol in [("assesstot", "_assesstot_p"), ("assessland", "_assessland_p")]:
        if _col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(_col))
        df = df.with_columns(
            pl.coalesce([pl.col(_col), pl.col(_pcol)]).alias(_col)
        ).drop(_pcol)
    cov = df["assesstot"].is_not_null().mean() * 100
    print(f"  assesstot coverage: {cov:.1f}%")
else:
    if "assesstot"  not in df.columns: df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("assesstot"))
    if "assessland" not in df.columns: df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("assessland"))

# ── 2. Feature engineering ─────────────────────────────────────────────
print("\n[2/9] Engineering features …")

dist_cols = [c for c in df.columns if c.startswith("dist_")]
df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in dist_cols
])
df = df.with_columns([
    pl.col(col).clip(lower_bound=0).log1p().alias(f"log_{col}")
    for col in dist_cols
])

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

# Walk score proxy
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
w_vec = np.array([0.35, 0.15, 0.30, 0.10, 0.10])
walk_score_np = np.clip(walk_normed_np @ w_vec * 100, 0, 100)
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

# ── NEW v5: 4 interaction features ────────────────────────────────────
print("  Adding v5 interaction features …")
df = df.with_columns([
    # Space efficiency: sqft per floor (detached vs tower signal)
    (pl.col("gross_square_feet") / pl.col("numfloors").clip(lower_bound=1))
        .alias("sqft_per_floor"),

    # Affluence/safety ratio: wealthier AND safer = premium multiplier
    (pl.col("median_income_nta") / (pl.col("crime_rate_nta") + 1.0))
        .alias("income_over_crime"),

    # Density: units per 1000 sqft (distinguishes single-fam from multifam)
    (pl.col("residential_units") / pl.col("gross_square_feet").clip(lower_bound=1) * 1000.0)
        .alias("density_index"),

    # Log interaction of sqft × floors captures floor plan geometry
    ((pl.col("gross_square_feet").clip(lower_bound=1).log1p()) *
     (pl.col("numfloors").clip(lower_bound=1).log1p()))
        .alias("log_sqft_x_floors"),
])
print(f"  v5 features added: sqft_per_floor, income_over_crime, density_index, log_sqft_x_floors")

# ── 3. Time-based holdout ─────────────────────────────────────────────
print("\n[3/9] Time-based holdout (last 15%) …")
df_sorted = df.sort("sale_date")
n_hold    = int(len(df_sorted) * 0.15)
df_work   = df_sorted[:-n_hold]
df_hold   = df_sorted[-n_hold:]
print(f"  Work: {len(df_work):,}  |  Hold: {len(df_hold):,}")

# ── 3b. Impute prior_sale_price via assesstot ratios ──────────────────
print("\n  Imputing prior_sale_price via assesstot …")
_has_both = df_work.filter(
    (pl.col("prior_sale_price") > 0) &
    pl.col("assesstot").is_not_null() & (pl.col("assesstot") > 0)
).with_columns((pl.col("prior_sale_price") / pl.col("assesstot")).alias("_price_ratio"))
_ratio_by_boro_df = _has_both.group_by("borough").agg(pl.col("_price_ratio").median().alias("ratio"))
_ratio_by_boro   = {int(r["borough"]): float(r["ratio"]) for r in _ratio_by_boro_df.iter_rows(named=True)}
_global_ratio    = float(_has_both["_price_ratio"].median()) if len(_has_both) else 10.0

_ratio_lookup = pl.DataFrame({
    "borough":        list(_ratio_by_boro.keys()),
    "_impute_ratio":  list(_ratio_by_boro.values()),
}).with_columns(pl.col("borough").cast(df_work.schema["borough"]))

def _impute_prior(frame):
    frame = frame.join(_ratio_lookup, on="borough", how="left").with_columns(
        pl.col("_impute_ratio").fill_null(_global_ratio)
    )
    return frame.with_columns(
        pl.when(
            (pl.col("prior_sale_price").is_null() | (pl.col("prior_sale_price") == 0)) &
            pl.col("assesstot").is_not_null() & (pl.col("assesstot") > 0)
        )
        .then(pl.col("assesstot") * pl.col("_impute_ratio"))
        .otherwise(pl.col("prior_sale_price"))
        .alias("prior_sale_price")
    ).drop("_impute_ratio")

df_work = _impute_prior(df_work)
df_hold = _impute_prior(df_hold)
_cov = (df_work["prior_sale_price"] > 0).sum() / len(df_work) * 100
print(f"  prior_sale_price coverage after imputation: {_cov:.1f}%")

# ── 4. Target encoding ────────────────────────────────────────────────
print("\n[4/9] Target encoding …")
LOG_TARGET  = "log_price"
df_work = df_work.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
df_hold = df_hold.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
global_mean_log = float(df_work[LOG_TARGET].mean())

bldg_means_df = df_work.group_by("bldgclass").agg(pl.col(LOG_TARGET).mean().alias("bldgclass_encoded"))
bldg_means    = {r["bldgclass"]: r["bldgclass_encoded"] for r in bldg_means_df.iter_rows(named=True)}
df_work = df_work.join(bldg_means_df, on="bldgclass", how="left").with_columns(
    pl.col("bldgclass_encoded").fill_null(global_mean_log))
df_hold = df_hold.join(bldg_means_df, on="bldgclass", how="left").with_columns(
    pl.col("bldgclass_encoded").fill_null(global_mean_log))

df_work = df_work.with_columns(
    (pl.col("borough").cast(pl.Utf8) + "_" + pl.col("bldgclass").str.slice(0, 1)).alias("borough_bldg_key"))
df_hold = df_hold.with_columns(
    (pl.col("borough").cast(pl.Utf8) + "_" + pl.col("bldgclass").str.slice(0, 1)).alias("borough_bldg_key"))
bb_means_df = df_work.group_by("borough_bldg_key").agg(pl.col(LOG_TARGET).mean().alias("borough_bldg_encoded"))
bb_means    = {r["borough_bldg_key"]: r["borough_bldg_encoded"] for r in bb_means_df.iter_rows(named=True)}
df_work = df_work.join(bb_means_df, on="borough_bldg_key", how="left").with_columns(
    pl.col("borough_bldg_encoded").fill_null(global_mean_log))
df_hold = df_hold.join(bb_means_df, on="borough_bldg_key", how="left").with_columns(
    pl.col("borough_bldg_encoded").fill_null(global_mean_log))

# ── 5. Feature matrix ─────────────────────────────────────────────────
print("\n[5/9] Building feature matrix …")
V4_FEATURES = [
    "latitude","longitude","borough","building_age","numfloors",
    "gross_square_feet","land_square_feet","residential_units",
    "dist_subway_m","dist_school_m","dist_park_m","dist_hospital_m",
    "poi_count_500m","crime_rate_nta","noise_density_nta",
    "population_2020","median_income_nta","dist_bus_m",
    "renovated_since_2018","years_since_renovation",
    "dist_waterfront_m","dist_bike_lane_m","dist_elem_school_m",
    "dist_express_subway_m","nearest_station_is_express",
    "livability_complaint_rate","borough_income_deviation",
    "sale_year","sale_month_sin","sale_month_cos",
    "mortgage_rate_30yr",
    "builtfar","residfar","commfar","facilfar","far_utilization",
    "has_elevator","is_condo","is_multifamily","is_single_fam","is_mixed_use",
    "airbnb_count_500m",
    "prior_sale_price","price_appreciation","years_since_prior_sale",
    "is_flip","school_district","district_avg_score","district_school_count",
    "has_prior_sale",
    "assesstot","assessland",
    "poi_cafe_500m","poi_restaurant_500m","poi_gym_500m",
    "poi_grocery_500m","poi_bar_500m","poi_pharmacy_500m",
    *[f"log_{c}" for c in dist_cols],
    "dist_midtown_manhattan_m","dist_downtown_manhattan_m",
    "dist_downtown_brooklyn_m","dist_long_island_city_m",
    "is_manhattan","crime_x_manhattan","crime_x_non_manhattan",
    "walk_score_proxy",
    "bldgclass_encoded","borough_bldg_encoded",
    "tree_count_200m","pm25_mean","no2_mean","hpd_viol_rate_nta",
]
# v5 additions
V5_FEATURES = ["sqft_per_floor", "income_over_crime", "density_index", "log_sqft_x_floors"]
FEATURE_NAMES = [f for f in (V4_FEATURES + V5_FEATURES) if f in df_work.columns]
print(f"  Total features: {len(FEATURE_NAMES)}  (+{len([f for f in V5_FEATURES if f in df_work.columns])} v5 interaction)")

acris_cols    = ["prior_sale_price","price_appreciation","years_since_prior_sale"]
acris_medians = {c: float(df_work.filter(pl.col(c).is_not_null() & (pl.col(c) != 0))[c].median() or 0.0)
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
        X_work[:, idx] = np.clip(X_work[:, idx], None, cap)
        X_hold[:, idx] = np.clip(X_hold[:, idx], None, cap)

# ── 6. OOF generation with 4 diverse learners ────────────────────────
print("\n[6/9] Generating OOF predictions (spatial GroupKFold ×5) …")
print("  4-model diverse ensemble: XGB-A(deep), XGB-B(shallow), LGB(wide), CAT(high-cap)")
groups = df_work["ntacode"].fill_null("UNK").to_numpy()
gkf    = GroupKFold(n_splits=5)

# XGB-A: deep trees, low LR — captures complex interactions
XGB_A_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse",
    n_estimators=3000, learning_rate=0.02, max_depth=7,
    min_child_weight=8, subsample=0.80, colsample_bytree=0.60,
    gamma=0.1, reg_alpha=0.3, reg_lambda=1.5,
    early_stopping_rounds=150, random_state=42, n_jobs=-1,
)
# XGB-B: shallow fast — high bias/low var, adds complementary signal
XGB_B_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse",
    n_estimators=3000, learning_rate=0.05, max_depth=4,
    min_child_weight=15, subsample=0.65, colsample_bytree=0.75,
    gamma=0.3, reg_alpha=1.0, reg_lambda=2.5,
    early_stopping_rounds=150, random_state=7, n_jobs=-1,
)
# LGB: leaf-wise with many leaves — captures sharp boundaries
LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    n_estimators=3000, learning_rate=0.04, num_leaves=127,
    min_child_samples=15, feature_fraction=0.70, bagging_fraction=0.80,
    bagging_freq=5, reg_alpha=0.3, reg_lambda=1.5,
    n_jobs=-1, random_state=42, verbose=-1,
)
# CAT: ordered boosting, high depth — robust on skewed features
CAT_PARAMS = dict(
    iterations=2000, learning_rate=0.025, depth=8,
    l2_leaf_reg=1.5, random_strength=0.5,
    bagging_temperature=0.8, border_count=64,
    early_stopping_rounds=150, random_seed=42, verbose=0,
)

oof_xgb_a = np.zeros(len(X_work), dtype=np.float32)
oof_xgb_b = np.zeros(len(X_work), dtype=np.float32)
oof_lgb   = np.zeros(len(X_work), dtype=np.float32)
oof_cat   = np.zeros(len(X_work), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_work, y_work, groups), 1):
    Xtr, Xva = X_work[tr_idx], X_work[va_idx]
    ytr, yva = y_work[tr_idx], y_work[va_idx]

    ma = xgb.XGBRegressor(**XGB_A_PARAMS)
    ma.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    oof_xgb_a[va_idx] = ma.predict(Xva)

    mb = xgb.XGBRegressor(**XGB_B_PARAMS)
    mb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    oof_xgb_b[va_idx] = mb.predict(Xva)

    cb = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=-1)]
    ml = lgb.LGBMRegressor(**LGB_PARAMS)
    ml.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cb)
    oof_lgb[va_idx] = ml.predict(Xva)

    mc = CatBoostRegressor(**CAT_PARAMS)
    mc.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    oof_cat[va_idx] = mc.predict(Xva).astype(np.float32)

    print(f"  Fold {fold}: "
          f"XGB-A R²={r2_score(yva,oof_xgb_a[va_idx]):.4f}  "
          f"XGB-B R²={r2_score(yva,oof_xgb_b[va_idx]):.4f}  "
          f"LGB R²={r2_score(yva,oof_lgb[va_idx]):.4f}  "
          f"CAT R²={r2_score(yva,oof_cat[va_idx]):.4f}")

print()
for name, oof in [("XGB-A(deep)", oof_xgb_a), ("XGB-B(shallow)", oof_xgb_b),
                  ("LGB(wide)", oof_lgb), ("CAT(high-cap)", oof_cat)]:
    print(f"  OOF {name:<15}  R²={r2_score(y_work,oof):.4f}  "
          f"MedAPE={medape(np.expm1(y_work),np.expm1(oof)):.2f}%")

# Stack with non-negative Ridge — prevents models from hurting each other
S_work     = np.column_stack([oof_xgb_a, oof_xgb_b, oof_lgb, oof_cat])
ridge_pos  = Ridge(alpha=1.0, positive=True)
ridge_pos.fit(S_work, y_work)
oof_stack  = ridge_pos.predict(S_work)
print(f"\n  OOF Stack(4,pos-Ridge) R²={r2_score(y_work,oof_stack):.4f}  "
      f"MedAPE={medape(np.expm1(y_work),np.expm1(oof_stack)):.2f}%")
print(f"  Ridge coefs: XGB-A={ridge_pos.coef_[0]:.4f}  XGB-B={ridge_pos.coef_[1]:.4f}  "
      f"LGB={ridge_pos.coef_[2]:.4f}  CAT={ridge_pos.coef_[3]:.4f}  "
      f"intercept={ridge_pos.intercept_:.4f}")

# ── 7. Retrain final models ───────────────────────────────────────────
print("\n[7/9] Retraining final 4-model ensemble …")
Xtr_f, Xva_f, ytr_f, yva_f = train_test_split(X_work, y_work, test_size=0.12, random_state=42)

print("  XGB-A (deep) …")
final_xgb_a = xgb.XGBRegressor(**XGB_A_PARAMS)
final_xgb_a.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], verbose=400)

print("  XGB-B (shallow) …")
final_xgb_b = xgb.XGBRegressor(**XGB_B_PARAMS)
final_xgb_b.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], verbose=400)

print("  LightGBM (wide) …")
cb_f = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=400)]
final_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
final_lgb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], callbacks=cb_f)

print("  CatBoost (high-cap) …")
final_cat = CatBoostRegressor(**CAT_PARAMS)
final_cat.fit(Xtr_f, ytr_f, eval_set=(Xva_f, yva_f), verbose=False)

# ── 8. Luxury sub-model (Manhattan ≥$2.5M) ───────────────────────────
print("\n[8/9] Luxury sub-model (Manhattan ≥$2.5M) …")
LUXURY_THRESH = 2_500_000  # lowered from $3M for more training data
luxury_mask = (df_work["borough"] == 1) & (df_work["sale_price"] >= LUXURY_THRESH)
X_lux = df_work.filter(luxury_mask).select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_lux = df_work.filter(luxury_mask)[LOG_TARGET].to_numpy().astype(np.float32)
print(f"  Luxury training samples: {len(X_lux):,}  (threshold: ${LUXURY_THRESH/1e6:.1f}M)")

if len(X_lux) >= 200:
    Xtr_l, Xva_l, ytr_l, yva_l = train_test_split(X_lux, y_lux, test_size=0.15, random_state=42)
    LUX_PARAMS = dict(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=2000, learning_rate=0.015, max_depth=6,
        min_child_weight=5, subsample=0.85, colsample_bytree=0.70,
        gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
        early_stopping_rounds=100, random_state=42, n_jobs=-1,
    )
    luxury_xgb = xgb.XGBRegressor(**LUX_PARAMS)
    luxury_xgb.fit(Xtr_l, ytr_l, eval_set=[(Xva_l, yva_l)], verbose=False)
    lux_pred  = luxury_xgb.predict(Xva_l)
    lux_r2    = r2_score(yva_l, lux_pred)
    lux_mape  = medape(np.expm1(yva_l), np.expm1(lux_pred))
    print(f"  Luxury XGB val — R²={lux_r2:.4f}  MedAPE={lux_mape:.2f}%  "
          f"best_round={luxury_xgb.best_iteration+1}")
    luxury_xgb.save_model(os.path.join(MODEL_DIR, "luxury_model.json"))
    print(f"  luxury_model.json saved ✓")
    has_luxury = True
else:
    print(f"  ⚠ Only {len(X_lux)} luxury samples — skipping (need ≥200)")
    has_luxury = False

# ── 9. Holdout evaluation ─────────────────────────────────────────────
print("\n[9/9] Holdout evaluation …")
pred_xgb_a = final_xgb_a.predict(X_hold)
pred_xgb_b = final_xgb_b.predict(X_hold)
pred_lgb   = final_lgb.predict(X_hold)
pred_cat   = final_cat.predict(X_hold).astype(np.float32)
S_hold     = np.column_stack([pred_xgb_a, pred_xgb_b, pred_lgb, pred_cat])
pred_stack = ridge_pos.predict(S_hold)

def eval_preds(name, y_true_log, y_pred_log):
    r2   = r2_score(y_true_log, y_pred_log)
    mae  = mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    mape = medape(np.expm1(y_true_log), np.expm1(y_pred_log))
    print(f"  {name:<20}  R²={r2:.4f}  MAE=${mae:>12,.0f}  MedAPE={mape:.2f}%")
    return r2, mae, mape

print()
r2_xa,  mae_xa,  mape_xa  = eval_preds("XGB-A (deep)",    y_hold, pred_xgb_a)
r2_xb,  mae_xb,  mape_xb  = eval_preds("XGB-B (shallow)", y_hold, pred_xgb_b)
r2_lgb, mae_lgb, mape_lgb = eval_preds("LGB (wide)",      y_hold, pred_lgb)
r2_cat, mae_cat, mape_cat = eval_preds("CAT (high-cap)",  y_hold, pred_cat)
r2_stk, mae_stk, mape_stk = eval_preds("Stack(4,pos)",    y_hold, pred_stack)

# Per-borough segment stats
BOROUGH_MAP = {1:"Manhattan",2:"Bronx",3:"Brooklyn",4:"Queens",5:"Staten Island"}
segment_by_borough = {}
boroughs_hold = df_hold["borough"].to_numpy()
prices_hold   = np.expm1(y_hold)
for bnum, bname in BOROUGH_MAP.items():
    mask = boroughs_hold == bnum
    if mask.sum() < 10: continue
    b_r2 = r2_score(y_hold[mask], pred_stack[mask])
    b_mape = medape(prices_hold[mask], np.expm1(pred_stack[mask]))
    b_mae  = mean_absolute_error(prices_hold[mask], np.expm1(pred_stack[mask]))
    segment_by_borough[bname] = {"n": int(mask.sum()), "medape": round(b_mape, 2), "mae": round(b_mae, 0)}
    print(f"  {bname:<16}  n={mask.sum():>5}  R²={b_r2:.4f}  MedAPE={b_mape:.2f}%")

# Per-tier segment stats
TIER_BINS = [(0,500_000,"<$500K"),(500_000,1_000_000,"$500K–1M"),
             (1_000_000,3_000_000,"$1M–3M"),(3_000_000,10_000_000,"$3M–10M")]
segment_by_tier = {}
for lo, hi, label in TIER_BINS:
    mask = (prices_hold >= lo) & (prices_hold < hi)
    if mask.sum() < 10: continue
    t_mape = medape(prices_hold[mask], np.expm1(pred_stack[mask]))
    t_mae  = mean_absolute_error(prices_hold[mask], np.expm1(pred_stack[mask]))
    segment_by_tier[label] = {"n": int(mask.sum()), "medape": round(t_mape, 2), "mae": round(t_mae, 0)}

# ── Save models ───────────────────────────────────────────────────────
# Stack dict now includes both XGB models + LGB + CAT + meta
stack_path = os.path.join(MODEL_DIR, "thaman_stack.pkl")
joblib.dump({
    "xgb_a":  final_xgb_a,   # primary XGB (deep)
    "xgb_b":  final_xgb_b,   # secondary XGB (shallow)
    "lgb":    final_lgb,
    "cat":    final_cat,
    "meta":   ridge_pos,
    "version": "v5",
}, stack_path)
print(f"\n  thaman_stack.pkl saved  ({os.path.getsize(stack_path)/1e6:.1f} MB)")

# Save primary XGB as xgboost_model.json (used by explain() / SHAP)
final_xgb_a.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
print(f"  xgboost_model.json saved (XGB-A deep)")

# ── Update meta.json ──────────────────────────────────────────────────
print("\n  Updating meta.json …")
meta_path = os.path.join(MODEL_DIR, "meta.json")
with open(meta_path) as f:
    meta = json.load(f)

meta["feature_names"]      = FEATURE_NAMES
meta["n_features"]         = len(FEATURE_NAMES)
meta["n_train"]            = len(df_work)
meta["n_holdout"]          = len(df_hold)
meta["walk_score_scaler"]  = walk_score_scaler_params
meta["bldgclass_means"]    = {k: round(float(v), 6) for k, v in bldg_means.items()}
meta["borough_bldg_means"] = {k: round(float(v), 6) for k, v in bb_means.items()}
meta["global_mean_log"]    = round(global_mean_log, 6)
meta["luxury_threshold"]   = LUXURY_THRESH
meta["has_luxury_model"]   = has_luxury
meta["acris_medians"]      = {k: round(float(v), 6) for k, v in acris_medians.items()}
meta["winsorize_p99"]      = {k: round(float(v), 6) for k, v in winsorize_p99.items()}
meta["segment_by_borough"] = segment_by_borough
meta["segment_by_tier"]    = segment_by_tier

meta["stack"] = {
    "version": "v5",
    "base_learners": ["xgb_a", "xgb_b", "lightgbm", "catboost"],
    "r2_holdout":     round(r2_stk, 4),
    "mae_holdout":    round(mae_stk, 0),
    "medape_holdout": round(mape_stk, 2),
    "xgb_a":    {"r2_holdout": round(r2_xa,  4), "mae_holdout": round(mae_xa,  0),
                 "medape_holdout": round(mape_xa,  2), "best_round": int(final_xgb_a.best_iteration+1)},
    "xgb_b":    {"r2_holdout": round(r2_xb,  4), "mae_holdout": round(mae_xb,  0),
                 "medape_holdout": round(mape_xb,  2), "best_round": int(final_xgb_b.best_iteration+1)},
    "lightgbm": {"r2_holdout": round(r2_lgb, 4), "mae_holdout": round(mae_lgb, 0),
                 "medape_holdout": round(mape_lgb, 2), "best_round": int(final_lgb.best_iteration_)},
    "catboost": {"r2_holdout": round(r2_cat, 4), "mae_holdout": round(mae_cat, 0),
                 "medape_holdout": round(mape_cat, 2), "best_round": int(final_cat.best_iteration_+1)},
    "ridge": {
        "coef_xgb_a": round(float(ridge_pos.coef_[0]), 6),
        "coef_xgb_b": round(float(ridge_pos.coef_[1]), 6),
        "coef_lgb":   round(float(ridge_pos.coef_[2]), 6),
        "coef_cat":   round(float(ridge_pos.coef_[3]), 6),
        "intercept":  round(float(ridge_pos.intercept_), 6),
    },
    "r2_improvement":     round(r2_stk - r2_xa, 4),
    "medape_improvement": round(mape_xa - mape_stk, 2),
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("  meta.json updated ✓")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  THAMAN Stack v5 — Complete")
print("=" * 70)
print(f"\n  {'Model':<22}  {'R²':>7}  {'MAE':>14}  {'MedAPE':>9}")
print(f"  {'-'*58}")
print(f"  {'XGB-A (deep)':<22}  {r2_xa:>7.4f}  ${mae_xa:>13,.0f}  {mape_xa:>8.2f}%")
print(f"  {'XGB-B (shallow)':<22}  {r2_xb:>7.4f}  ${mae_xb:>13,.0f}  {mape_xb:>8.2f}%")
print(f"  {'LGB (wide)':<22}  {r2_lgb:>7.4f}  ${mae_lgb:>13,.0f}  {mape_lgb:>8.2f}%")
print(f"  {'CAT (high-cap)':<22}  {r2_cat:>7.4f}  ${mae_cat:>13,.0f}  {mape_cat:>8.2f}%")
print(f"  {'Stack(4,pos-Ridge)':<22}  {r2_stk:>7.4f}  ${mae_stk:>13,.0f}  {mape_stk:>8.2f}%")
print(f"\n  Improvement vs v4 stack:  ΔR²={r2_stk-0.6511:+.4f}  ΔMedAPE={20.80-mape_stk:+.2f}%")
print(f"\n  v5 new features: {', '.join(V5_FEATURES)}")
print(f"  Luxury threshold: ${LUXURY_THRESH/1e6:.1f}M  ({'trained' if has_luxury else 'skipped'})")
