"""
THAMAN Stacking Model v2 — train_stack_v2.py
=============================================
Adds CatBoost as 3rd base learner: XGB + LGB + CAT + Ridge meta-learner.
Expected: R² 0.593 → ~0.62–0.65

Steps:
  1. Load + engineer features (identical to v2)
  2. OOF predictions: XGBoost + LightGBM + CatBoost (5-fold spatial GroupKFold)
  3. Ridge meta-learner trained on 3-column OOF stack
  4. Retrain final XGB + LGB + CAT on full work set
  5. Evaluate: XGB / LGB / CAT / Stack(3) on holdout
  6. Save: thaman_stack.pkl  +  update meta.json
"""

import os, sys, json, warnings, joblib
import numpy as np
import pandas as pd
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
print("  THAMAN Stack v2 — XGBoost + LightGBM + CatBoost + Ridge")
print("=" * 65)

def safe_clip_min(arr, minval=1.0):
    return np.maximum(arr, minval)

def medape(y_true_usd, y_pred_usd):
    return float(np.median(np.abs(y_true_usd - y_pred_usd) / safe_clip_min(y_true_usd)) * 100)

# ── 1. Load data ───────────────────────────────────────────────────
print("\n[1/8] Loading features.csv …")
df = pd.read_csv(os.path.join(PROC, "features.csv"))
df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
df = df.dropna(subset=["sale_date", "sale_price", "latitude", "longitude"])
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── 2. Feature engineering (identical to v2) ──────────────────────
print("\n[2/8] Engineering features …")

# Log-transform distances
dist_cols = [c for c in df.columns if c.startswith("dist_")]
for col in dist_cols:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

# Urban gravity
GRAVITY = {
    "midtown_manhattan":  (40.7549, -73.9840),
    "downtown_manhattan": (40.7074, -74.0113),
    "downtown_brooklyn":  (40.6928, -73.9903),
    "long_island_city":   (40.7447, -73.9485),
}
for name, (clat, clon) in GRAVITY.items():
    df[f"dist_{name}_m"] = (
        np.sqrt((df["latitude"] - clat)**2 + (df["longitude"] - clon)**2) * 111_000
    )

# Manhattan flag + crime interactions
df["is_manhattan"]          = (df["borough"] == 1).astype(int)
df["crime_x_manhattan"]     = df["crime_rate_nta"] * df["is_manhattan"]
df["crime_x_non_manhattan"] = df["crime_rate_nta"] * (1 - df["is_manhattan"])

# Walk-score proxy — fit scaler on full df, save params
eps = 1e-6
walk_comps = pd.DataFrame({
    "transit":   1.0 / df["dist_subway_m"].clip(lower=eps),
    "bus":       1.0 / df["dist_bus_m"].clip(lower=eps),
    "amenities": df["poi_count_500m"],
    "bike":      1.0 / df["dist_bike_lane_m"].clip(lower=eps),
    "park":      1.0 / df["dist_park_m"].clip(lower=eps),
}, index=df.index)
ws_scaler = MinMaxScaler()
walk_normed = pd.DataFrame(
    ws_scaler.fit_transform(walk_comps),
    columns=walk_comps.columns, index=df.index
)
w = {"transit": 0.35, "bus": 0.15, "amenities": 0.30, "bike": 0.10, "park": 0.10}
df["walk_score_proxy"] = (
    sum(walk_normed[c] * wt for c, wt in w.items()) * 100
).clip(lower=0, upper=100)

# Save scaler params for inference
walk_score_scaler_params = {
    col: {
        "data_min":  float(ws_scaler.data_min_[i]),
        "data_max":  float(ws_scaler.data_max_[i]),
        "scale":     float(ws_scaler.scale_[i]),
    }
    for i, col in enumerate(walk_comps.columns)
}

# Cyclical month
df["sale_month_sin"] = np.sin(2 * np.pi * df["sale_month"] / 12)
df["sale_month_cos"] = np.cos(2 * np.pi * df["sale_month"] / 12)

print(f"  Features engineered ✓  (log_dist ×{len(dist_cols)}, gravity ×4, walk, cyclic)")

# ── 3. Time-based holdout (last 15%) ─────────────────────────────
print("\n[3/8] Time-based holdout (last 15%) …")
df_sorted = df.sort_values("sale_date").reset_index(drop=True)
n_hold    = int(len(df_sorted) * 0.15)
df_work   = df_sorted.iloc[:-n_hold].copy()
df_hold   = df_sorted.iloc[-n_hold:].copy()
print(f"  Work: {len(df_work):,}  |  Hold: {len(df_hold):,}")

# ── 4. Target encoding (no leakage — fit on work set only) ───────
print("\n[4/8] Target encoding …")
LOG_TARGET      = "log_price"
df_work[LOG_TARGET] = np.log1p(df_work["sale_price"])
df_hold[LOG_TARGET] = np.log1p(df_hold["sale_price"])
global_mean_log = float(df_work[LOG_TARGET].mean())

bldg_means = df_work.groupby("bldgclass")[LOG_TARGET].mean()
for frame in [df_work, df_hold]:
    frame["bldgclass_encoded"] = frame["bldgclass"].map(bldg_means).fillna(global_mean_log)

df_work["borough_bldg_key"] = (
    df_work["borough"].astype(str) + "_" + df_work["bldgclass"].str[:1]
)
df_hold["borough_bldg_key"] = (
    df_hold["borough"].astype(str) + "_" + df_hold["bldgclass"].str[:1]
)
bb_means = df_work.groupby("borough_bldg_key")[LOG_TARGET].mean()
for frame in [df_work, df_hold]:
    frame["borough_bldg_encoded"] = (
        frame["borough_bldg_key"].map(bb_means).fillna(global_mean_log)
    )
print(f"  bldgclass classes: {df_work['bldgclass'].nunique()}  "
      f"| borough_bldg combos: {df_work['borough_bldg_key'].nunique()}")

# ── 5. Feature matrix ─────────────────────────────────────────────
print("\n[5/8] Building feature matrix …")
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
]
NEW_FEATURES = [
    *[f"log_{c}" for c in dist_cols],
    "dist_midtown_manhattan_m","dist_downtown_manhattan_m",
    "dist_downtown_brooklyn_m","dist_long_island_city_m",
    "is_manhattan","crime_x_manhattan","crime_x_non_manhattan",
    "walk_score_proxy",
    "bldgclass_encoded","borough_bldg_encoded",
]
FEATURE_NAMES = [f for f in (KEEP_FROM_V1 + NEW_FEATURES) if f in df_work.columns]
print(f"  Total features: {len(FEATURE_NAMES)}")

# ACRIS medians + winsorize
acris_cols    = ["prior_sale_price","price_appreciation","years_since_prior_sale"]
acris_medians = {c: float(df_work[c].replace(0, np.nan).median()) for c in acris_cols}
qol_cols      = ["crime_rate_nta","noise_density_nta","livability_complaint_rate"]
winsorize_p99 = {c: float(np.percentile(df_work[c].dropna(), 99)) for c in qol_cols}

df_work[FEATURE_NAMES] = df_work[FEATURE_NAMES].fillna(0)
df_hold[FEATURE_NAMES] = df_hold[FEATURE_NAMES].fillna(0)

X_work = df_work[FEATURE_NAMES].values.astype(np.float32)
y_work = df_work[LOG_TARGET].values.astype(np.float32)
X_hold = df_hold[FEATURE_NAMES].values.astype(np.float32)
y_hold = df_hold[LOG_TARGET].values.astype(np.float32)

for col, cap in winsorize_p99.items():
    if col in FEATURE_NAMES:
        idx = FEATURE_NAMES.index(col)
        X_work[:, idx] = np.clip(X_work[:, idx], None, cap)
        X_hold[:, idx] = np.clip(X_hold[:, idx], None, cap)

# ── 6. OOF generation (5-fold spatial) ───────────────────────────
print("\n[6/8] Generating OOF predictions (spatial GroupKFold ×5) …")
print("       [XGBoost + LightGBM + CatBoost]")
groups = df_work["ntacode"].fillna("UNK").values
gkf    = GroupKFold(n_splits=5)

XGB_PARAMS = dict(
    objective         = "reg:squarederror",
    eval_metric       = "rmse",
    n_estimators      = 2000,
    learning_rate     = 0.03,
    max_depth         = 5,
    min_child_weight  = 10,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    gamma             = 0.2,
    reg_alpha         = 0.5,
    reg_lambda        = 2.0,
    early_stopping_rounds = 100,
    random_state      = 42,
    n_jobs            = -1,
)

LGB_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 2000,
    learning_rate     = 0.03,
    max_depth         = 5,
    min_child_samples = 20,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    min_split_gain    = 0.2,
    reg_alpha         = 0.5,
    reg_lambda        = 2.0,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)

CAT_PARAMS = dict(
    iterations          = 2000,
    learning_rate       = 0.03,
    depth               = 5,
    l2_leaf_reg         = 2.0,
    random_strength     = 0.2,
    bagging_temperature = 0.5,
    early_stopping_rounds = 100,
    random_seed         = 42,
    verbose             = 0,
)

oof_xgb = np.zeros(len(X_work), dtype=np.float32)
oof_lgb = np.zeros(len(X_work), dtype=np.float32)
oof_cat = np.zeros(len(X_work), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_work, y_work, groups), 1):
    Xtr, Xva = X_work[tr_idx], X_work[va_idx]
    ytr, yva = y_work[tr_idx], y_work[va_idx]

    # XGBoost
    mx = xgb.XGBRegressor(**XGB_PARAMS)
    mx.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    oof_xgb[va_idx] = mx.predict(Xva)

    # LightGBM
    cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
    ml = lgb.LGBMRegressor(**LGB_PARAMS)
    ml.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cb)
    oof_lgb[va_idx] = ml.predict(Xva)

    # CatBoost
    mc = CatBoostRegressor(**CAT_PARAMS)
    mc.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    oof_cat[va_idx] = mc.predict(Xva).astype(np.float32)

    r2x = r2_score(yva, oof_xgb[va_idx])
    r2l = r2_score(yva, oof_lgb[va_idx])
    r2c = r2_score(yva, oof_cat[va_idx])
    print(f"  Fold {fold}: XGB R²={r2x:.4f}  LGB R²={r2l:.4f}  CAT R²={r2c:.4f}")

print(f"\n  OOF XGB  R²={r2_score(y_work, oof_xgb):.4f}  "
      f"MedAPE={medape(np.expm1(y_work), np.expm1(oof_xgb)):.2f}%")
print(f"  OOF LGB  R²={r2_score(y_work, oof_lgb):.4f}  "
      f"MedAPE={medape(np.expm1(y_work), np.expm1(oof_lgb)):.2f}%")
print(f"  OOF CAT  R²={r2_score(y_work, oof_cat):.4f}  "
      f"MedAPE={medape(np.expm1(y_work), np.expm1(oof_cat)):.2f}%")

# ── Ridge meta-learner (3 base learners) ─────────────────────────
print("\n  Training Ridge meta-learner on 3-column OOF stack …")
S_work  = np.column_stack([oof_xgb, oof_lgb, oof_cat])
ridge_3 = Ridge(alpha=1.0)
ridge_3.fit(S_work, y_work)
oof_stack = ridge_3.predict(S_work)
print(f"  OOF Stack(3) R²={r2_score(y_work, oof_stack):.4f}  "
      f"MedAPE={medape(np.expm1(y_work), np.expm1(oof_stack)):.2f}%")
print(f"  Ridge weights: XGB={ridge_3.coef_[0]:.4f}  "
      f"LGB={ridge_3.coef_[1]:.4f}  CAT={ridge_3.coef_[2]:.4f}  "
      f"intercept={ridge_3.intercept_:.4f}")

# ── 7. Retrain final models on full work set ──────────────────────
print("\n[7/8] Retraining final XGB + LGB + CAT on full work set …")
Xtr_f, Xva_f, ytr_f, yva_f = train_test_split(
    X_work, y_work, test_size=0.12, random_state=42
)

print("  Training final XGBoost …")
final_xgb = xgb.XGBRegressor(**XGB_PARAMS)
final_xgb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], verbose=300)

print("  Training final LightGBM …")
cb_final = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=300)]
final_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
final_lgb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], callbacks=cb_final)

print("  Training final CatBoost …")
final_cat = CatBoostRegressor(**CAT_PARAMS)
final_cat.fit(Xtr_f, ytr_f, eval_set=(Xva_f, yva_f), verbose=False)

print(f"  XGB best round: {final_xgb.best_iteration + 1}")
print(f"  LGB best round: {final_lgb.best_iteration_}")
print(f"  CAT best round: {final_cat.best_iteration_ + 1}")

# ── 8. Holdout evaluation ─────────────────────────────────────────
print("\n[8/8] Evaluating on holdout …")
pred_xgb   = final_xgb.predict(X_hold)
pred_lgb   = final_lgb.predict(X_hold)
pred_cat   = final_cat.predict(X_hold).astype(np.float32)
S_hold     = np.column_stack([pred_xgb, pred_lgb, pred_cat])
pred_stack = ridge_3.predict(S_hold)

def eval_preds(name, y_true_log, y_pred_log):
    r2   = r2_score(y_true_log, y_pred_log)
    mae  = mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    mape = medape(np.expm1(y_true_log), np.expm1(y_pred_log))
    print(f"  {name:<14}  R²={r2:.4f}  MAE=${mae:,.0f}  MedAPE={mape:.2f}%")
    return r2, mae, mape

print()
r2_xgb, mae_xgb, mape_xgb = eval_preds("XGBoost",      y_hold, pred_xgb)
r2_lgb, mae_lgb, mape_lgb = eval_preds("LightGBM",     y_hold, pred_lgb)
r2_cat, mae_cat, mape_cat = eval_preds("CatBoost",      y_hold, pred_cat)
r2_stk, mae_stk, mape_stk = eval_preds("Stack(3)",      y_hold, pred_stack)

# ── Save stack pickle ─────────────────────────────────────────────
stack_path = os.path.join(MODEL_DIR, "thaman_stack.pkl")
joblib.dump({"lgb": final_lgb, "cat": final_cat, "meta": ridge_3}, stack_path)
print(f"\n  thaman_stack.pkl saved  ✓  ({os.path.getsize(stack_path)/1e6:.1f} MB)")

# Also save final XGB
final_xgb.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
print(f"  xgboost_model.json saved  ✓")

# ── Update meta.json ──────────────────────────────────────────────
print("\n  Updating meta.json …")
meta_path = os.path.join(MODEL_DIR, "meta.json")
with open(meta_path) as f:
    meta = json.load(f)

meta["walk_score_scaler"] = walk_score_scaler_params
meta["bldgclass_means"]   = {k: round(float(v), 6) for k, v in bldg_means.items()}
meta["borough_bldg_means"]= {k: round(float(v), 6) for k, v in bb_means.items()}
meta["global_mean_log"]   = round(global_mean_log, 6)

meta["stack"] = {
    "version": "v2",
    "base_learners": ["xgboost", "lightgbm", "catboost"],
    "xgboost": {
        "r2_holdout":     round(r2_xgb,  4),
        "mae_holdout":    round(mae_xgb,  0),
        "medape_holdout": round(mape_xgb, 2),
        "best_round":     int(final_xgb.best_iteration + 1),
    },
    "lightgbm": {
        "r2_holdout":     round(r2_lgb,  4),
        "mae_holdout":    round(mae_lgb,  0),
        "medape_holdout": round(mape_lgb, 2),
        "best_round":     int(final_lgb.best_iteration_),
    },
    "catboost": {
        "r2_holdout":     round(r2_cat,  4),
        "mae_holdout":    round(mae_cat,  0),
        "medape_holdout": round(mape_cat, 2),
        "best_round":     int(final_cat.best_iteration_ + 1),
    },
    "ridge": {
        "coef_xgb":   round(float(ridge_3.coef_[0]), 6),
        "coef_lgb":   round(float(ridge_3.coef_[1]), 6),
        "coef_cat":   round(float(ridge_3.coef_[2]), 6),
        "intercept":  round(float(ridge_3.intercept_), 6),
        "alpha":      1.0,
    },
    "r2_holdout":         round(r2_stk,  4),
    "mae_holdout":        round(mae_stk,  0),
    "medape_holdout":     round(mape_stk, 2),
    "r2_improvement":     round(r2_stk - r2_xgb, 4),
    "medape_improvement": round(mape_xgb - mape_stk, 2),
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  meta.json updated  ✓")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  THAMAN Stack v2 — Complete")
print("=" * 65)
print(f"\n  {'Model':<14}  {'R²':>7}  {'MAE':>12}  {'MedAPE':>9}")
print(f"  {'-'*50}")
print(f"  {'XGBoost':<14}  {r2_xgb:>7.4f}  ${mae_xgb:>11,.0f}  {mape_xgb:>8.2f}%")
print(f"  {'LightGBM':<14}  {r2_lgb:>7.4f}  ${mae_lgb:>11,.0f}  {mape_lgb:>8.2f}%")
print(f"  {'CatBoost':<14}  {r2_cat:>7.4f}  ${mae_cat:>11,.0f}  {mape_cat:>8.2f}%")
print(f"  {'Stack(3)':<14}  {r2_stk:>7.4f}  ${mae_stk:>11,.0f}  {mape_stk:>8.2f}%")
print(f"\n  R² improvement (XGB → Stack): +{r2_stk - r2_xgb:+.4f}")
print(f"  MedAPE improvement          : {mape_xgb:.2f}% → {mape_stk:.2f}%")
print(f"\n  Files saved:")
print(f"    models/thaman_stack.pkl   (LGB + CAT + Ridge meta, 3 coeffs)")
print(f"    models/xgboost_model.json (retrained XGB)")
print(f"    models/meta.json          (stack v2 metrics + catboost)")
