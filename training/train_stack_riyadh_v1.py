"""
THAMAN Riyadh Model — Stack Training v1
========================================
Trains XGBoost + LightGBM + CatBoost + Ridge meta-learner on features_riyadh.csv.
Mirrors the architecture of train_stack_v2.py but adapted for Riyadh district-level data.

Run:  python training/train_stack_riyadh_v1.py
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
    # Metro transit
    "dist_metro_m", "log_dist_metro_m", "metro_stations_1km",
    "nearest_metro_line_num", "nearest_metro_type_cd", "dist_metro_line1_m",
    # Bus
    "dist_bus_m", "log_dist_bus_m", "bus_stops_500m", "brt_stops_500m",
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
    "avg_saudi_salary_yr", "salary_yoy_change",
    # District aggregates
    "district_median_price_sqm", "district_transaction_volume",
    "district_price_vs_city_avg", "district_price_trend_slope",
    "district_median_price_apt_sqm",
    # Target-encoded
    "district_encoded", "district_type_encoded",
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

print("Loading features_riyadh.csv...")
df = pd.read_csv(PROC / "features_riyadh.csv", encoding="utf-8-sig")
print(f"  Rows: {len(df)} | Cols: {len(df.columns)}")

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
cutoff_idx = int(len(all_qids) * 0.80)
cutoff_qid = all_qids[cutoff_idx]

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
for ptype in ["apartment", "villa", "residential_plot", "building"]:
    mask_col = f"is_{ptype}"
    if mask_col in hold.columns:
        hmask = hold[mask_col].values.astype(bool)
        if hmask.sum() >= 10:
            t_r2     = r2_score(y_hold[hmask], hold_meta_preds[hmask])
            t_medape = medape(y_hold[hmask], hold_meta_preds[hmask])
            print(f"  [{ptype:>18}] R²={t_r2:.4f} | MedAPE={t_medape:.2f}% | n={hmask.sum()}")

# ── Save models ───────────────────────────────────────────────────────────────

print("\nSaving models...")
stack_path = MDIR / "riyadh_stack.pkl"
with open(stack_path, "wb") as f:
    pickle.dump({
        "xgb": final_xgb,
        "lgb": final_lgb,
        "cat": final_cat,
        "meta": meta,
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
    "model_version": "riyadh_v1",
    "n_folds": N_FOLDS,
    "train_rows": len(work),
    "holdout_rows": len(hold),
    "holdout_cutoff_quarter_id": int(cutoff_qid),
    "meta_coefficients": meta.coef_.tolist(),
    "meta_intercept": float(meta.intercept_),
    "target": "log1p(sale_price_sar_sqm)",
    "y_unit": "SAR/sqm",
})

with open(meta_path, "w") as f:
    json.dump(meta_dict, f, indent=2)
print(f"  Updated: {meta_path}")

print("\n" + "=" * 60)
print(f"THAMAN Riyadh v1 — Training complete")
print(f"  OOF  R²={oof_meta_r2:.4f}  MedAPE={oof_meta_medape:.2f}%")
print(f"  Hold R²={hold_r2:.4f}  MedAPE={hold_medape:.2f}%  MAE={hold_mae:,.0f} SAR/sqm")
print("=" * 60)
