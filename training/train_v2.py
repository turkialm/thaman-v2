"""
THAMAN Model v2 — Comprehensive Training Pipeline
==================================================
Implements all prioritised recommendations:

BLOCKERS:
  1. True time-based holdout — saved to disk, never used during tuning
  2. No price_per_sqft_nta leakage (confirmed absent)
  3. CV strictly on work set — holdout untouched

HIGH PRIORITY:
  4. Spatial CV (GroupKFold by NTA) + time-split evaluation
  5. Tighter regularisation (lower LR, higher reg)
  6. Luxury outlier experiment (cap at $10 M vs none)

FEATURES:
  7.  bldgclass target-encoded (replaces LabelEncoder)
  8.  Urban gravity distances (Midtown, Downtown Manhattan, BK, LIC)
  9.  Log-transform all 9 distance features
  10. Borough × bldgclass interaction (target-encoded)
  11. Walk-score proxy
  12. Month sin/cos cyclical encoding
  13. is_manhattan flag + crime×manhattan interaction

EVALUATION:
  17. Segment-level reporting + classification metrics (confusion matrix,
      accuracy, precision, recall, F1 by price tier)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC      = os.path.join(BASE, "data", "processed")
SPLIT_DIR = os.path.join(BASE, "data", "splits")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(SPLIT_DIR, exist_ok=True)

print("=" * 65)
print("  THAMAN v2 — Training Pipeline")
print("=" * 65)

# ── helper: safe clip on numpy arrays ───────────────────────────
def safe_clip_min(arr, minval=1.0):
    """Works on both numpy arrays and pandas Series."""
    return np.maximum(arr, minval)

# ── 1. Load data ─────────────────────────────────────────────────
print("\n[1/9] Loading features.csv …")
df = pd.read_csv(os.path.join(PROC, "features.csv"))
df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
df = df.dropna(subset=["sale_date", "sale_price", "latitude", "longitude"])
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── 2. Feature engineering ───────────────────────────────────────
print("\n[2/9] Engineering new features …")

# 9. Log-transform distances
dist_cols = [c for c in df.columns if c.startswith("dist_")]
for col in dist_cols:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
print(f"  + {len(dist_cols)} log-distance features")

# 8. Urban gravity distances
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
print(f"  + 4 urban-gravity distances")

# 13. Manhattan flag + crime interactions
df["is_manhattan"]          = (df["borough"] == 1).astype(int)
df["crime_x_manhattan"]     = df["crime_rate_nta"] * df["is_manhattan"]
df["crime_x_non_manhattan"] = df["crime_rate_nta"] * (1 - df["is_manhattan"])
print(f"  + is_manhattan + 2 crime interactions")

# 11. Walk-score proxy
def compute_walk_score(frame):
    eps = 1e-6
    comps = pd.DataFrame({
        "transit":   1.0 / (frame["dist_subway_m"].clip(lower=eps)),
        "bus":       1.0 / (frame["dist_bus_m"].clip(lower=eps)),
        "amenities": frame["poi_count_500m"],
        "bike":      1.0 / (frame["dist_bike_lane_m"].clip(lower=eps)),
        "park":      1.0 / (frame["dist_park_m"].clip(lower=eps)),
    }, index=frame.index)
    sc = MinMaxScaler()
    normed = pd.DataFrame(sc.fit_transform(comps), columns=comps.columns, index=frame.index)
    w = {"transit": 0.35, "bus": 0.15, "amenities": 0.30, "bike": 0.10, "park": 0.10}
    return (sum(normed[c] * wt for c, wt in w.items()) * 100).clip(lower=0, upper=100)

df["walk_score_proxy"] = compute_walk_score(df)
by_boro = {b: round(df.loc[df["borough"]==b,"walk_score_proxy"].mean(),1) for b in range(1,6)}
print(f"  + walk_score_proxy (by borough: {by_boro})")

# 12. Cyclical month encoding
df["sale_month_sin"] = np.sin(2 * np.pi * df["sale_month"] / 12)
df["sale_month_cos"] = np.cos(2 * np.pi * df["sale_month"] / 12)
print(f"  + sale_month_sin / sale_month_cos")

# ── 3. Blocker 1 — Time-based final holdout ─────────────────────
print("\n[3/9] Splitting — final holdout (last 15 % by date) …")
df_sorted = df.sort_values("sale_date").reset_index(drop=True)
n_hold    = int(len(df_sorted) * 0.15)
df_work   = df_sorted.iloc[:-n_hold].copy()
df_hold   = df_sorted.iloc[-n_hold:].copy()
df_hold.to_parquet(os.path.join(SPLIT_DIR, "final_holdout.parquet"), index=False)
print(f"  Work: {len(df_work):,}  ({df_work['sale_date'].min().date()} → {df_work['sale_date'].max().date()})")
print(f"  Hold: {len(df_hold):,}  ({df_hold['sale_date'].min().date()} → {df_hold['sale_date'].max().date()})")

# ── 4. Target encoding (blocker 2: no leakage) ──────────────────
print("\n[4/9] Target encoding bldgclass + borough×bldgclass …")
LOG_TARGET = "log_price"
df_work[LOG_TARGET] = np.log1p(df_work["sale_price"])
df_hold[LOG_TARGET] = np.log1p(df_hold["sale_price"])
global_mean_log = float(df_work[LOG_TARGET].mean())

bldg_means = df_work.groupby("bldgclass")[LOG_TARGET].mean()
for frame in [df_work, df_hold]:
    frame["bldgclass_encoded"] = frame["bldgclass"].map(bldg_means).fillna(global_mean_log)

df_work["borough_bldg_key"] = (df_work["borough"].astype(str) + "_"
                                + df_work["bldgclass"].str[:1])
df_hold["borough_bldg_key"] = (df_hold["borough"].astype(str) + "_"
                                + df_hold["bldgclass"].str[:1])
bb_means = df_work.groupby("borough_bldg_key")[LOG_TARGET].mean()
for frame in [df_work, df_hold]:
    frame["borough_bldg_encoded"] = (frame["borough_bldg_key"]
                                     .map(bb_means).fillna(global_mean_log))
print(f"  bldgclass classes: {df_work['bldgclass'].nunique()}  →  float")
print(f"  borough_bldg combos: {df_work['borough_bldg_key'].nunique()}")

# ── 5. Feature list ──────────────────────────────────────────────
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
]
NEW_FEATURES = [
    *[f"log_{c}" for c in dist_cols],                   # 9 log-distances
    "dist_midtown_manhattan_m","dist_downtown_manhattan_m",
    "dist_downtown_brooklyn_m","dist_long_island_city_m", # 4 gravity
    "is_manhattan","crime_x_manhattan","crime_x_non_manhattan",
    "walk_score_proxy",
    "bldgclass_encoded","borough_bldg_encoded",
]
FEATURE_NAMES = [f for f in (KEEP_FROM_V1 + NEW_FEATURES) if f in df_work.columns]
print(f"  Total features: {len(FEATURE_NAMES)}")

# Fill nulls
df_work[FEATURE_NAMES] = df_work[FEATURE_NAMES].fillna(0)
df_hold[FEATURE_NAMES] = df_hold[FEATURE_NAMES].fillna(0)

X_work = df_work[FEATURE_NAMES].values.astype(np.float32)
y_work = df_work[LOG_TARGET].values.astype(np.float32)
X_hold = df_hold[FEATURE_NAMES].values.astype(np.float32)
y_hold = df_hold[LOG_TARGET].values.astype(np.float32)

# ACRIS medians for imputation at inference
acris_cols    = ["prior_sale_price","price_appreciation","years_since_prior_sale"]
acris_medians = {c: float(df_work[c].replace(0, np.nan).median()) for c in acris_cols}
print(f"  ACRIS medians: { {k: round(v,3) for k,v in acris_medians.items()} }")

# Winsorise QoL at 99th pct
qol_cols       = ["crime_rate_nta","noise_density_nta","livability_complaint_rate"]
winsorize_p99  = {c: float(np.percentile(df_work[c].dropna(), 99)) for c in qol_cols}
for col, cap in winsorize_p99.items():
    if col in FEATURE_NAMES:
        idx = FEATURE_NAMES.index(col)
        X_work[:, idx] = np.clip(X_work[:, idx], None, cap)
        X_hold[:, idx] = np.clip(X_hold[:, idx], None, cap)
print(f"  Winsorise p99: { {k: round(v,1) for k,v in winsorize_p99.items()} }")

# ── 6. Spatial 5-fold CV ─────────────────────────────────────────
print("\n[6/9] Spatial CV (GroupKFold by NTA, 5 folds) …")
groups = df_work["ntacode"].fillna("UNK").values
gkf    = GroupKFold(n_splits=5)

PARAMS = dict(
    objective        = "reg:squarederror",
    eval_metric      = "rmse",
    n_estimators     = 2000,
    learning_rate    = 0.03,
    max_depth        = 5,
    min_child_weight = 10,
    subsample        = 0.7,
    colsample_bytree = 0.7,
    gamma            = 0.2,
    reg_alpha        = 0.5,
    reg_lambda       = 2.0,
    early_stopping_rounds = 100,
    random_state     = 42,
    n_jobs           = -1,
)

cv_r2, cv_mae, cv_mape = [], [], []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_work, y_work, groups), 1):
    Xtr, Xva = X_work[tr_idx], X_work[va_idx]
    ytr, yva = y_work[tr_idx], y_work[va_idx]
    m = xgb.XGBRegressor(**PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pv     = m.predict(Xva)
    pusd   = np.expm1(pv)
    tusd   = np.expm1(yva)
    r2     = r2_score(yva, pv)
    mae    = mean_absolute_error(tusd, pusd)
    medape = float(np.median(np.abs(tusd - pusd) / safe_clip_min(tusd)) * 100)
    cv_r2.append(r2); cv_mae.append(mae); cv_mape.append(medape)
    print(f"  Fold {fold}: R²={r2:.4f}  MAE=${mae:,.0f}  MedAPE={medape:.1f}%")

print(f"\n  Spatial CV R²    = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
print(f"  Spatial CV MAE   = ${np.mean(cv_mae):,.0f}")
print(f"  Spatial CV MedAPE= {np.mean(cv_mape):.2f}%")

# ── 7. Luxury-cap experiment ────────────────────────────────────
print("\n[7/9] Luxury outlier experiment …")
exp_results = {}
for cap, label in [(None, "no_cap"), (10_000_000, "cap_10M")]:
    if cap:
        m_w = df_work["sale_price"] <= cap
        Xe, ye, ge = X_work[m_w], y_work[m_w], groups[m_w]
    else:
        Xe, ye, ge = X_work, y_work, groups
    fr2, fm = [], []
    for ti, vi in gkf.split(Xe, ye, ge):
        m2 = xgb.XGBRegressor(**PARAMS)
        m2.fit(Xe[ti], ye[ti], eval_set=[(Xe[vi], ye[vi])], verbose=False)
        pv2 = m2.predict(Xe[vi])
        fr2.append(r2_score(ye[vi], pv2))
        fm.append(float(np.median(
            np.abs(np.expm1(ye[vi]) - np.expm1(pv2)) / safe_clip_min(np.expm1(ye[vi]))
        ) * 100))
    exp_results[label] = {"R²": round(float(np.mean(fr2)),4),
                          "MedAPE": round(float(np.mean(fm)),2), "n": int(len(ye))}
    print(f"  {label:10s}  R²={exp_results[label]['R²']:.4f}  "
          f"MedAPE={exp_results[label]['MedAPE']:.2f}%  n={exp_results[label]['n']:,}")

best_cap  = min(exp_results, key=lambda k: exp_results[k]["MedAPE"])
USE_CAP   = (best_cap == "cap_10M")
print(f"  → Best config: {best_cap}  (USE_CAP={USE_CAP})")

if USE_CAP:
    mw = df_work["sale_price"] <= 10_000_000
    mh = df_hold["sale_price"]  <= 10_000_000
    Xwf, ywf = X_work[mw], y_work[mw]
    Xhf, yhf = X_hold[mh], y_hold[mh]
    df_hold_eval = df_hold[mh].copy()
else:
    Xwf, ywf = X_work, y_work
    Xhf, yhf = X_hold, y_hold
    df_hold_eval = df_hold.copy()

# ── 8. Final model ───────────────────────────────────────────────
print("\n[8/9] Training final model on full work set …")
Xtr, Xva, ytr, yva = train_test_split(Xwf, ywf, test_size=0.15, random_state=42)
print(f"  Train: {len(Xtr):,}  |  Internal val: {len(Xva):,}")

final = xgb.XGBRegressor(**PARAMS)
final.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=200)
best_round = final.best_iteration + 1
print(f"\n  Best round: {best_round}")

pva   = final.predict(Xva);  r2_val  = r2_score(yva, pva)
ptr   = final.predict(Xtr);  r2_train= r2_score(ytr, ptr)
medape_val = float(np.median(
    np.abs(np.expm1(yva) - np.expm1(pva)) / safe_clip_min(np.expm1(yva))) * 100)
print(f"  Train R²={r2_train:.4f}  Val R²={r2_val:.4f}  "
      f"Overfitting gap={r2_train-r2_val:.4f}  Val MedAPE={medape_val:.2f}%")

# ── 9. Final holdout evaluation ──────────────────────────────────
print("\n[9/9] Final holdout evaluation …")
ph        = final.predict(Xhf)
ph_usd    = np.expm1(ph)
yh_usd    = np.expm1(yhf)
r2_hold   = r2_score(yhf, ph)
mae_hold  = mean_absolute_error(yh_usd, ph_usd)
rmse_hold = float(np.sqrt(np.mean((yhf - ph)**2)))
mape_hold = float(np.mean(np.abs(yh_usd - ph_usd) / safe_clip_min(yh_usd)) * 100)
medape_h  = float(np.median(np.abs(yh_usd - ph_usd) / safe_clip_min(yh_usd)) * 100)

print(f"\n  {'Metric':<22} {'v2':>10}  {'v1 (baseline)':>14}")
print(f"  {'-'*50}")
print(f"  {'R² (test):':<22} {r2_hold:>10.4f}  {'0.7354':>14}")
print(f"  {'R² (train):':<22} {r2_train:>10.4f}  {'0.8940':>14}")
print(f"  {'RMSE (log):':<22} {rmse_hold:>10.4f}  {'0.4966':>14}")
print(f"  {'MAE ($):':<22} ${mae_hold:>9,.0f}  {'$726,593':>14}")
print(f"  {'MAPE (%):':<22} {mape_hold:>9.2f}%  {'50.11%':>14}")
print(f"  {'MedAPE (%):':<22} {medape_h:>9.2f}%  {'17.78%':>14}")
print(f"  {'Features:':<22} {len(FEATURE_NAMES):>10}  {'50':>14}")
print(f"  {'Trees:':<22} {best_round:>10}  {'917':>14}")

# ── 17a. Segment reporting ───────────────────────────────────────
BORO = {1:"Manhattan",2:"Bronx",3:"Brooklyn",4:"Queens",5:"Staten Island"}
df_hold_eval = df_hold_eval.copy()
df_hold_eval["pred_usd"]   = ph_usd
df_hold_eval["true_usd"]   = yh_usd
df_hold_eval["pred_log"]   = ph
df_hold_eval["abs_error"]  = np.abs(yh_usd - ph_usd)
df_hold_eval["pct_error"]  = df_hold_eval["abs_error"] / df_hold_eval["true_usd"].clip(lower=1) * 100
df_hold_eval["boro_name"]  = df_hold_eval["borough"].map(BORO)
df_hold_eval["bldg_cat"]   = df_hold_eval["bldgclass"].str[:1]
df_hold_eval["price_tier"] = pd.cut(
    df_hold_eval["true_usd"],
    bins  =[0,500_000,1_000_000,3_000_000,10_000_000,np.inf],
    labels=["<$500K","$500K–1M","$1M–3M","$3M–10M","$10M+"]
)

print("\n── BY BOROUGH ───────────────────────────────────────────────")
print(df_hold_eval.groupby("boro_name").agg(
    N=("true_usd","count"),
    MAE=("abs_error", lambda x: f"${x.mean():,.0f}"),
    MedAPE=("pct_error","median"),
).round(1).to_string())

print("\n── BY PRICE TIER ────────────────────────────────────────────")
print(df_hold_eval.groupby("price_tier", observed=True).agg(
    N=("true_usd","count"),
    MAE=("abs_error", lambda x: f"${x.mean():,.0f}"),
    MedAPE=("pct_error","median"),
).round(1).to_string())

print("\n── BY BUILDING CLASS ────────────────────────────────────────")
print(df_hold_eval.groupby("bldg_cat").agg(
    N=("true_usd","count"),
    MedAPE=("pct_error","median"),
).sort_values("MedAPE").round(1).to_string())

# ── 17b. Classification metrics (price-tier classification) ─────
print("\n" + "=" * 65)
print("  CLASSIFICATION METRICS (Price Tier Prediction)")
print("=" * 65)

TIER_LABELS = ["<$500K","$500K–1M","$1M–3M","$3M–10M","$10M+"]
TIER_BINS   = [0, 500_000, 1_000_000, 3_000_000, 10_000_000, np.inf]

y_true_tier = pd.cut(yh_usd,   bins=TIER_BINS, labels=TIER_LABELS).astype(str)
y_pred_tier = pd.cut(ph_usd,   bins=TIER_BINS, labels=TIER_LABELS).astype(str)

# Overall accuracy
acc = accuracy_score(y_true_tier, y_pred_tier)
print(f"\n  Overall Tier Accuracy : {acc*100:.2f}%")

# ±1 tier tolerance accuracy
def within_n_tiers(true_tiers, pred_tiers, labels, n=1):
    correct = sum(
        abs(labels.index(t) - labels.index(p)) <= n
        for t, p in zip(true_tiers, pred_tiers)
    )
    return correct / len(true_tiers)

acc_1 = within_n_tiers(list(y_true_tier), list(y_pred_tier), TIER_LABELS, n=1)
print(f"  Within-1-tier Accuracy: {acc_1*100:.2f}%")

# Per-class precision, recall, F1
print(f"\n  Per-Tier Report:")
print(classification_report(y_true_tier, y_pred_tier,
                             labels=TIER_LABELS, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true_tier, y_pred_tier, labels=TIER_LABELS)
print("  Confusion Matrix (rows=actual, cols=predicted):")
cm_df = pd.DataFrame(cm, index=TIER_LABELS, columns=TIER_LABELS)
print(cm_df.to_string())

# Per-tier metrics table
prec  = precision_score(y_true_tier, y_pred_tier, labels=TIER_LABELS,
                         average=None, zero_division=0)
rec   = recall_score(y_true_tier, y_pred_tier, labels=TIER_LABELS,
                     average=None, zero_division=0)
f1    = f1_score(y_true_tier, y_pred_tier, labels=TIER_LABELS,
                 average=None, zero_division=0)
support = [np.sum(y_true_tier == t) for t in TIER_LABELS]
clf_per_tier = {
    t: {"precision": round(float(p),4), "recall": round(float(r),4),
        "f1": round(float(f),4), "support": int(s)}
    for t, p, r, f, s in zip(TIER_LABELS, prec, rec, f1, support)
}

macro_p = float(np.mean(prec)); macro_r = float(np.mean(rec)); macro_f = float(np.mean(f1))
print(f"\n  Macro averages:  Precision={macro_p:.4f}  Recall={macro_r:.4f}  F1={macro_f:.4f}")
print(f"  Overall accuracy: {acc:.4f}  |  Within-1-tier: {acc_1:.4f}")

# ── Save model + meta ────────────────────────────────────────────
print("\n── Saving model + meta.json ────────────────────────────────")
final.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))

# SHAP
try:
    import shap
    exp_shap  = shap.TreeExplainer(final)
    sv        = exp_shap.shap_values(Xhf[:500])
    ms        = np.abs(sv).mean(axis=0)
    top20_idx = np.argsort(ms)[::-1][:20]
    shap_top20= [{"feature": FEATURE_NAMES[i], "mean_shap": round(float(ms[i]),5)}
                 for i in top20_idx]
    print("  SHAP via shap library ✓")
except ImportError:
    imp = final.get_booster().get_fscore()
    tot = sum(imp.values()) or 1
    shap_top20 = sorted(
        [{"feature": k, "mean_shap": round(v/tot,5)} for k,v in imp.items()],
        key=lambda x: x["mean_shap"], reverse=True)[:20]
    print("  SHAP via get_fscore (install shap for SHAP values)")

# Segment dicts for meta
seg_borough = {}
for bn, grp in df_hold_eval.groupby("boro_name"):
    seg_borough[str(bn)] = {"n": int(len(grp)),
                             "medape": round(float(grp["pct_error"].median()),2),
                             "mae":    round(float(grp["abs_error"].mean()),0)}
seg_tier = {}
for t, grp in df_hold_eval.groupby("price_tier", observed=True):
    seg_tier[str(t)] = {"n": int(len(grp)),
                        "medape": round(float(grp["pct_error"].median()),2),
                        "mae":    round(float(grp["abs_error"].mean()),0)}

meta = {
    "feature_names":      FEATURE_NAMES,
    "n_features":         len(FEATURE_NAMES),
    "bldgclass_classes":  [],             # no longer used (target-encoded)
    "winsorize_p99":      winsorize_p99,
    "acris_medians":      acris_medians,
    "bldgclass_means":    {k: round(float(v),6) for k,v in bldg_means.items()},
    "borough_bldg_means": {k: round(float(v),6) for k,v in bb_means.items()},
    "global_mean_log":    round(global_mean_log, 6),
    "use_price_cap":      USE_CAP,
    "price_cap":          10_000_000 if USE_CAP else None,
    "n_train":            int(len(Xtr)),
    "n_val":              int(len(Xva)),
    "n_holdout":          int(len(Xhf)),
    "baseline":           {"r2": 0.2382, "mae": 1230842, "mape": 86.14},
    "xgboost": {
        "best_round":      best_round,
        "r2_test":         round(r2_hold,  4),
        "r2_train":        round(r2_train, 4),
        "r2_val":          round(r2_val,   4),
        "cv_r2_mean":      round(float(np.mean(cv_r2)),  4),
        "cv_r2_std":       round(float(np.std(cv_r2)),   4),
        "cv_medape_mean":  round(float(np.mean(cv_mape)),2),
        "rmse_test":       round(rmse_hold, 4),
        "mae_test":        round(mae_hold,  0),
        "mape_test":       round(mape_hold, 2),
        "medape_test":     round(medape_h,  2),
        "params":          {k: v for k, v in PARAMS.items()
                            if k != "early_stopping_rounds"},
    },
    "luxury_cap_experiment": exp_results,
    "classification": {
        "tier_labels":     TIER_LABELS,
        "tier_bins":       [0, 500_000, 1_000_000, 3_000_000, 10_000_000],
        "accuracy":        round(float(acc),  4),
        "within_1_tier":   round(float(acc_1),4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "macro_f1":        round(macro_f, 4),
        "per_tier":        clf_per_tier,
        "confusion_matrix": cm.tolist(),
    },
    "shap_top20":          shap_top20,
    "segment_by_borough":  seg_borough,
    "segment_by_tier":     seg_tier,
}

with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  meta.json saved  ✓")
print(f"  model saved      ✓")

print("\n" + "=" * 65)
print("  THAMAN v2 — Complete")
print("=" * 65)
print(f"\n  R² test   : {r2_hold:.4f}  (v1: 0.7354)")
print(f"  MedAPE    : {medape_h:.2f}%  (v1: 17.78%)")
print(f"  MAE       : ${mae_hold:,.0f}  (v1: $726,593)")
print(f"  Tier acc  : {acc*100:.1f}%  |  within-1: {acc_1*100:.1f}%")
print(f"  Macro F1  : {macro_f:.4f}")
print(f"  CV R²     : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}  (spatial)")
print(f"  Features  : {len(FEATURE_NAMES)}  (v1: 50)")
print(f"  Best round: {best_round}  (v1: 917)")
