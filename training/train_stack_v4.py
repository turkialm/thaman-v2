"""
THAMAN Stack v4 — train_stack_v4.py
=====================================
Improvements over v3:
  • Real dist_waterfront_m (NYC coastline, 27K points) — was 100% NaN
  • Real dist_bike_lane_m  (28K bike lane segments)    — was 100% NaN
  • 6 POI category counts: cafe, restaurant, gym, grocery, bar, pharmacy
  • Luxury sub-model (XGBoost) for Manhattan $3M+ properties

Expected gain: +1.5–3% R², -2–4% MedAPE on luxury segment.

Run:
  cd /Users/totam/Desktop/new_try
  python training/train_stack_v4.py
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

print("=" * 65)
print("  THAMAN Stack v4 — XGB + LGB + CAT + Ridge + Luxury sub-model")
print("=" * 65)

def safe_clip_min(arr, minval=1.0):
    return np.maximum(arr, minval)

def medape(y_true_usd, y_pred_usd):
    return float(np.median(np.abs(y_true_usd - y_pred_usd) / safe_clip_min(y_true_usd)) * 100)

# ── 1. Load data ───────────────────────────────────────────────────
_FLOAT_OVERRIDES = {
    "dist_waterfront_m": pl.Float64, "dist_bike_lane_m": pl.Float64,
    "school_district": pl.Float64, "district_avg_score": pl.Float64,
    "district_school_count": pl.Float64,
    "prior_sale_price": pl.Float64, "years_since_prior_sale": pl.Float64,
    "price_appreciation": pl.Float64, "is_flip": pl.Float64,
}
print("\n[1/9] Loading features_v4.csv …")
df = (
    pl.read_csv(os.path.join(PROC, "features_v4.csv"), schema_overrides=_FLOAT_OVERRIDES)
    .with_columns(pl.col("sale_date").str.to_datetime(format=None, strict=False))
    .drop_nulls(subset=["sale_date", "sale_price", "latitude", "longitude"])
)
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── 1b. Join PLUTO assessed values ────────────────────────────────
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

# ── 2. Feature engineering ─────────────────────────────────────────
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

print(f"  Features engineered ✓  (log_dist ×{len(dist_cols)}, gravity ×4, walk, cyclic)")

# ── 3. Time-based holdout ─────────────────────────────────────────
print("\n[3/9] Time-based holdout (last 15%) …")
df_sorted = df.sort("sale_date")
n_hold    = int(len(df_sorted) * 0.15)
df_work   = df_sorted[:-n_hold]
df_hold   = df_sorted[-n_hold:]
print(f"  Work: {len(df_work):,}  |  Hold: {len(df_hold):,}")

# ── 3b. Impute prior_sale_price via assesstot ratios ──────────────
print("\n  Imputing prior_sale_price via assesstot …")
_has_both = df_work.filter(
    (pl.col("prior_sale_price") > 0) &
    pl.col("assesstot").is_not_null() & (pl.col("assesstot") > 0)
).with_columns((pl.col("prior_sale_price") / pl.col("assesstot")).alias("_price_ratio"))
_ratio_by_boro_df = _has_both.group_by("borough").agg(pl.col("_price_ratio").median().alias("ratio"))
_ratio_by_boro   = {int(r["borough"]): float(r["ratio"]) for r in _ratio_by_boro_df.iter_rows(named=True)}
_global_ratio    = float(_has_both["_price_ratio"].median()) if len(_has_both) else 10.0

_ratio_lookup = pl.DataFrame({
    "borough": list(_ratio_by_boro.keys()),
    "_impute_ratio": list(_ratio_by_boro.values()),
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
for b, r in sorted(_ratio_by_boro.items()):
    print(f"    Borough {b}: assesstot × {r:.1f}")

# ── 4. Target encoding ────────────────────────────────────────────
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

# ── 5. Feature matrix ─────────────────────────────────────────────
print("\n[5/9] Building feature matrix …")
KEEP_FROM_V1 = [
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
    "builtfar","residfar","commfar","facilfar","maxallwfar","far_utilization",
    "has_elevator","is_condo","is_multifamily","is_single_fam","is_mixed_use",
    "airbnb_count_500m",
    "prior_sale_price","price_appreciation","years_since_prior_sale",
    "is_flip","school_district","district_avg_score","district_school_count",
    "has_prior_sale",
    "assesstot","assessland",
    # ── v4 additions ────────────────────────────────────────────────
    "poi_cafe_500m","poi_restaurant_500m","poi_gym_500m",
    "poi_grocery_500m","poi_bar_500m","poi_pharmacy_500m",
]
NEW_FEATURES = [
    *[f"log_{c}" for c in dist_cols],
    "dist_midtown_manhattan_m","dist_downtown_manhattan_m",
    "dist_downtown_brooklyn_m","dist_long_island_city_m",
    "is_manhattan","crime_x_manhattan","crime_x_non_manhattan",
    "walk_score_proxy",
    "bldgclass_encoded","borough_bldg_encoded",
    "tree_count_200m","pm25_mean","no2_mean","hpd_viol_rate_nta",
]
FEATURE_NAMES = [f for f in (KEEP_FROM_V1 + NEW_FEATURES) if f in df_work.columns]
print(f"  Total features: {len(FEATURE_NAMES)}")

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

# ── 6. OOF generation ────────────────────────────────────────────
print("\n[6/9] Generating OOF predictions (spatial GroupKFold ×5) …")
groups = df_work["ntacode"].fill_null("UNK").to_numpy()
gkf    = GroupKFold(n_splits=5)

XGB_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse",
    n_estimators=2000, learning_rate=0.03, max_depth=5,
    min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
    early_stopping_rounds=100, random_state=42, n_jobs=-1,
)
LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    n_estimators=2000, learning_rate=0.03, max_depth=5,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
    min_split_gain=0.2, reg_alpha=0.5, reg_lambda=2.0,
    n_jobs=-1, random_state=42, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=2000, learning_rate=0.03, depth=5, l2_leaf_reg=2.0,
    random_strength=0.2, bagging_temperature=0.5,
    early_stopping_rounds=100, random_seed=42, verbose=0,
)

oof_xgb = np.zeros(len(X_work), dtype=np.float32)
oof_lgb = np.zeros(len(X_work), dtype=np.float32)
oof_cat = np.zeros(len(X_work), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_work, y_work, groups), 1):
    Xtr, Xva = X_work[tr_idx], X_work[va_idx]
    ytr, yva = y_work[tr_idx], y_work[va_idx]

    mx = xgb.XGBRegressor(**XGB_PARAMS)
    mx.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    oof_xgb[va_idx] = mx.predict(Xva)

    cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
    ml = lgb.LGBMRegressor(**LGB_PARAMS)
    ml.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cb)
    oof_lgb[va_idx] = ml.predict(Xva)

    mc = CatBoostRegressor(**CAT_PARAMS)
    mc.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    oof_cat[va_idx] = mc.predict(Xva).astype(np.float32)

    print(f"  Fold {fold}: XGB R²={r2_score(yva,oof_xgb[va_idx]):.4f}  "
          f"LGB R²={r2_score(yva,oof_lgb[va_idx]):.4f}  "
          f"CAT R²={r2_score(yva,oof_cat[va_idx]):.4f}")

print(f"\n  OOF XGB  R²={r2_score(y_work,oof_xgb):.4f}  "
      f"MedAPE={medape(np.expm1(y_work),np.expm1(oof_xgb)):.2f}%")
print(f"  OOF LGB  R²={r2_score(y_work,oof_lgb):.4f}  "
      f"MedAPE={medape(np.expm1(y_work),np.expm1(oof_lgb)):.2f}%")
print(f"  OOF CAT  R²={r2_score(y_work,oof_cat):.4f}  "
      f"MedAPE={medape(np.expm1(y_work),np.expm1(oof_cat)):.2f}%")

S_work  = np.column_stack([oof_xgb, oof_lgb, oof_cat])
ridge_3 = Ridge(alpha=1.0)
ridge_3.fit(S_work, y_work)
oof_stack = ridge_3.predict(S_work)
print(f"  OOF Stack(3) R²={r2_score(y_work,oof_stack):.4f}  "
      f"MedAPE={medape(np.expm1(y_work),np.expm1(oof_stack)):.2f}%")

# ── 7. Retrain final models ───────────────────────────────────────
print("\n[7/9] Retraining final XGB + LGB + CAT …")
Xtr_f, Xva_f, ytr_f, yva_f = train_test_split(X_work, y_work, test_size=0.12, random_state=42)

print("  XGBoost …")
final_xgb = xgb.XGBRegressor(**XGB_PARAMS)
final_xgb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], verbose=300)

print("  LightGBM …")
cb_f = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=300)]
final_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
final_lgb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], callbacks=cb_f)

print("  CatBoost …")
final_cat = CatBoostRegressor(**CAT_PARAMS)
final_cat.fit(Xtr_f, ytr_f, eval_set=(Xva_f, yva_f), verbose=False)

# ── 8. Luxury sub-model (Manhattan $3M+) ─────────────────────────
print("\n[8/9] Luxury sub-model (Manhattan $3M+) …")
LUXURY_THRESH = 3_000_000
luxury_mask = (df_work["borough"] == 1) & (df_work["sale_price"] >= LUXURY_THRESH)
X_lux = df_work.filter(luxury_mask).select(FEATURE_NAMES).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)
y_lux = df_work.filter(luxury_mask)[LOG_TARGET].to_numpy().astype(np.float32)
print(f"  Luxury training samples: {len(X_lux):,}")

if len(X_lux) >= 200:
    Xtr_l, Xva_l, ytr_l, yva_l = train_test_split(X_lux, y_lux, test_size=0.15, random_state=42)
    LUX_PARAMS = dict(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=1500, learning_rate=0.02, max_depth=4,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=1.0, reg_lambda=3.0,
        early_stopping_rounds=80, random_state=42, n_jobs=-1,
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

# ── 9. Holdout evaluation ─────────────────────────────────────────
print("\n[9/9] Holdout evaluation …")
pred_xgb   = final_xgb.predict(X_hold)
pred_lgb   = final_lgb.predict(X_hold)
pred_cat   = final_cat.predict(X_hold).astype(np.float32)
S_hold     = np.column_stack([pred_xgb, pred_lgb, pred_cat])
pred_stack = ridge_3.predict(S_hold)

def eval_preds(name, y_true_log, y_pred_log):
    r2   = r2_score(y_true_log, y_pred_log)
    mae  = mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    mape = medape(np.expm1(y_true_log), np.expm1(y_pred_log))
    print(f"  {name:<16}  R²={r2:.4f}  MAE=${mae:,.0f}  MedAPE={mape:.2f}%")
    return r2, mae, mape

print()
r2_xgb, mae_xgb, mape_xgb = eval_preds("XGBoost",  y_hold, pred_xgb)
r2_lgb, mae_lgb, mape_lgb = eval_preds("LightGBM", y_hold, pred_lgb)
r2_cat, mae_cat, mape_cat = eval_preds("CatBoost",  y_hold, pred_cat)
r2_stk, mae_stk, mape_stk = eval_preds("Stack(3)",  y_hold, pred_stack)

# Luxury holdout evaluation (Manhattan $3M+)
hold_lux_mask = (
    (df_hold["borough"] == 1) & (df_hold["sale_price"] >= LUXURY_THRESH)
).to_numpy()
if has_luxury and hold_lux_mask.sum() > 0:
    print(f"\n  --- Luxury segment ({hold_lux_mask.sum()} holdout samples) ---")
    _lux_stack = pred_stack[hold_lux_mask]
    _lux_pred  = luxury_xgb.predict(X_hold[hold_lux_mask])
    _lux_true  = y_hold[hold_lux_mask]
    eval_preds("Stack(3) lux",  _lux_true, _lux_stack)
    eval_preds("LuxuryXGB lux", _lux_true, _lux_pred)
    # Soft blend: ramp from alpha=0 at $2M to alpha=1 at $4M
    prices_hold = np.expm1(pred_stack[hold_lux_mask])
    alpha = np.clip((prices_hold - 2_000_000) / 2_000_000, 0.0, 1.0)
    blended = (1 - alpha) * _lux_stack + alpha * _lux_pred
    r2_b, mae_b, mape_b = eval_preds("Blended lux",  _lux_true, blended)

# ── Save models ───────────────────────────────────────────────────
stack_path = os.path.join(MODEL_DIR, "thaman_stack.pkl")
joblib.dump({"lgb": final_lgb, "cat": final_cat, "meta": ridge_3}, stack_path)
print(f"\n  thaman_stack.pkl saved  ({os.path.getsize(stack_path)/1e6:.1f} MB)")
final_xgb.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
print(f"  xgboost_model.json saved")

# ── Update meta.json ──────────────────────────────────────────────
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

meta["stack"] = {
    "version": "v4",
    "base_learners": ["xgboost", "lightgbm", "catboost"],
    "r2_holdout": round(r2_stk, 4),
    "mae_holdout": round(mae_stk, 0),
    "medape_holdout": round(mape_stk, 2),
    "xgboost":  {"r2_holdout": round(r2_xgb, 4), "mae_holdout": round(mae_xgb, 0),
                 "medape_holdout": round(mape_xgb, 2), "best_round": int(final_xgb.best_iteration+1)},
    "lightgbm": {"r2_holdout": round(r2_lgb, 4), "mae_holdout": round(mae_lgb, 0),
                 "medape_holdout": round(mape_lgb, 2), "best_round": int(final_lgb.best_iteration_)},
    "catboost": {"r2_holdout": round(r2_cat, 4), "mae_holdout": round(mae_cat, 0),
                 "medape_holdout": round(mape_cat, 2), "best_round": int(final_cat.best_iteration_+1)},
    "ridge": {"coef_xgb": round(float(ridge_3.coef_[0]),6), "coef_lgb": round(float(ridge_3.coef_[1]),6),
              "coef_cat": round(float(ridge_3.coef_[2]),6), "intercept": round(float(ridge_3.intercept_),6)},
    "r2_improvement":     round(r2_stk - r2_xgb, 4),
    "medape_improvement": round(mape_xgb - mape_stk, 2),
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("  meta.json updated ✓")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  THAMAN Stack v4 — Complete")
print("=" * 65)
print(f"\n  {'Model':<16}  {'R²':>7}  {'MAE':>12}  {'MedAPE':>9}")
print(f"  {'-'*52}")
print(f"  {'XGBoost':<16}  {r2_xgb:>7.4f}  ${mae_xgb:>11,.0f}  {mape_xgb:>8.2f}%")
print(f"  {'LightGBM':<16}  {r2_lgb:>7.4f}  ${mae_lgb:>11,.0f}  {mape_lgb:>8.2f}%")
print(f"  {'CatBoost':<16}  {r2_cat:>7.4f}  ${mae_cat:>11,.0f}  {mape_cat:>8.2f}%")
print(f"  {'Stack(3)':<16}  {r2_stk:>7.4f}  ${mae_stk:>11,.0f}  {mape_stk:>8.2f}%")
print(f"\n  New features added in v4:")
print(f"    dist_waterfront_m  (was 100% NaN → real 27K-point coastline)")
print(f"    dist_bike_lane_m   (was 100% NaN → 28K NYC bike lane segments)")
print(f"    poi_cafe/restaurant/gym/grocery/bar/pharmacy _500m (Overture POIs)")
print(f"    luxury_model.json  (Manhattan $3M+, {'trained' if has_luxury else 'skipped'})")
