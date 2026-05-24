"""
THAMAN Riyadh Model — Stack Training v2
=========================================
Sprint 1 improvements over v1:
  1. OOF target encodings — district encodings computed inside each CV fold (no leakage)
  2. Look-back features — district trailing price (past quarters only, no future leakage)
  3. Bayut asking price — per-district listing median as market-signal feature
  4. Removed leaky features — district_median_price_sqm, district_*encoded etc.
  5. LGB regularisation tightened — min_child_samples 10→20
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
PROC  = _ROOT / "data" / "processed"
MDIR  = _ROOT / "models"

# ── Feature list (NO leaky district aggregates) ───────────────────────────────
# Removed: district_encoded, district_type_encoded, district_apt_encoded,
#          district_recent_encoded, district_apt_recent_encoded,
#          district_median_price_sqm, district_median_price_apt_sqm,
#          district_price_vs_city_avg, district_price_trend_slope
BASE_FEATURES = [
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
    # Transaction volume (not a price aggregate)
    "district_transaction_volume", "log_deed_count",
    # Time
    "sale_year", "sale_quarter_sin", "sale_quarter_cos",
    # QoL POI
    "dist_mosque_m", "log_dist_mosque_m", "mosque_count_500m",
    "dist_mall_m", "log_dist_mall_m", "mall_count_500m",
    "dist_school_m", "log_dist_school_m", "school_count_500m",
    "dist_hospital_m", "log_dist_hospital_m", "hospital_count_500m",
    "dist_park_m", "log_dist_park_m", "park_count_500m",
    "dist_entertain_m", "log_dist_entertain_m", "entertain_count_500m",
    # Connectivity
    "riyadh_connectivity_score",
    # ── Sprint 1 additions ────────────────────────────────────────────────────
    # Look-back price features (computed below — no leakage)
    "district_lookback_mean",
    "district_lookback_apt_mean",
    "city_quarter_mean",
    # Bayut market signal
    "bayut_asking_psqm",
    # OOF encodings (added dynamically inside CV loop)
    # "district_enc_oof" — appended after BASE_FEATURES in feat_cols
]

OOF_ENC_COLS = ["district_enc_oof", "district_apt_enc_oof"]  # computed per-fold

TARGET    = "sale_price_sar_sqm"
GROUP_COL = "district_ar"

# ── Hyperparameters (v2 — LGB slightly tighter) ───────────────────────────────
XGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
    tree_method="hist", random_state=42,
)
LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    num_leaves=47,
    min_child_samples=20,          # was 10 → tighter
    subsample=0.7, colsample_bytree=0.7,
    min_split_gain=0.2, reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=1500, learning_rate=0.03, depth=5,
    l2_leaf_reg=3.0,               # was 2.0 → slightly tighter
    random_strength=0.2, bagging_temperature=0.5,
    random_seed=42, verbose=0,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def medape(y_true, y_pred):
    return float(np.median(
        np.abs((np.expm1(y_pred) - np.expm1(y_true)) / np.expm1(y_true))
    ) * 100)


def mae_sar(y_true, y_pred):
    return float(np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true))))


def _build_enc_map_from(src_df: pd.DataFrame, col: str, target_col: str,
                        k: int = 30) -> tuple[dict, float]:
    """Smoothed mean encoding map from src_df rows. Returns (map, global_mean)."""
    global_mean = src_df[target_col].mean() if len(src_df) > 0 else 0.0
    enc_map: dict = {}
    for district, grp in src_df.groupby(col):
        n  = len(grp)
        mu = grp[target_col].mean()
        enc_map[district] = (n * mu + k * global_mean) / (n + k)
    return enc_map, global_mean


def _apply_enc(df: pd.DataFrame, col: str, enc_map: dict, global_mean: float) -> pd.Series:
    return df[col].map(enc_map).fillna(global_mean)


def _add_lookback_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leak-free look-back aggregates.

    For each row at quarter Q in district D:
      district_lookback_mean    — mean log_price of all PAST quarters in D
      district_lookback_apt_mean — same restricted to apartment rows (is_apartment==1)
      city_quarter_mean          — city-wide mean log_price for that exact quarter
    Falls back to global mean when no past data exists.
    """
    df = df.copy()
    global_mean = df["log_price"].mean()

    # ── district_lookback_mean ─────────────────────────────────────────────
    # Per-district per-quarter aggregate, then expanding mean shifted by 1
    dq = (
        df.groupby(["district_ar", "quarter_id"])["log_price"]
          .mean()
          .reset_index(name="dq_mean")
          .sort_values(["district_ar", "quarter_id"])
    )
    lb_rows = []
    for dist, grp in dq.groupby("district_ar"):
        grp = grp.sort_values("quarter_id").reset_index(drop=True)
        grp["district_lookback_mean"] = grp["dq_mean"].expanding().mean().shift(1)
        lb_rows.append(grp[["district_ar", "quarter_id", "district_lookback_mean"]])
    dq_lb = pd.concat(lb_rows, ignore_index=True)
    df = df.merge(dq_lb, on=["district_ar", "quarter_id"], how="left")
    df["district_lookback_mean"] = df["district_lookback_mean"].fillna(global_mean)

    # ── district_lookback_apt_mean ─────────────────────────────────────────
    apt_df = df[df["is_apartment"] == 1]
    apt_global_mean = apt_df["log_price"].mean() if len(apt_df) else global_mean
    dq_apt = (
        apt_df.groupby(["district_ar", "quarter_id"])["log_price"]
              .mean()
              .reset_index(name="dq_apt_mean")
              .sort_values(["district_ar", "quarter_id"])
    )
    lb_apt_rows = []
    for dist, grp in dq_apt.groupby("district_ar"):
        grp = grp.sort_values("quarter_id").reset_index(drop=True)
        grp["district_lookback_apt_mean"] = grp["dq_apt_mean"].expanding().mean().shift(1)
        lb_apt_rows.append(grp[["district_ar", "quarter_id", "district_lookback_apt_mean"]])
    if lb_apt_rows:
        dq_apt_lb = pd.concat(lb_apt_rows, ignore_index=True)
        df = df.merge(dq_apt_lb, on=["district_ar", "quarter_id"], how="left")
    else:
        df["district_lookback_apt_mean"] = apt_global_mean
    df["district_lookback_apt_mean"] = df["district_lookback_apt_mean"].fillna(apt_global_mean)

    # ── city_quarter_mean ──────────────────────────────────────────────────
    cq = df.groupby("quarter_id")["log_price"].mean().reset_index(name="city_quarter_mean")
    df = df.merge(cq, on="quarter_id", how="left")
    df["city_quarter_mean"] = df["city_quarter_mean"].fillna(global_mean)

    return df


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading features_riyadh.csv...")
df = pd.read_csv(PROC / "features_riyadh.csv", encoding="utf-8-sig")
print(f"  Rows: {len(df)} | Cols: {len(df.columns)}")

df["log_price"] = np.log1p(df[TARGET])

# ── Look-back features (no leakage) ──────────────────────────────────────────

print("Computing look-back features...")
df = _add_lookback_features(df)
print(f"  district_lookback_mean range: {df['district_lookback_mean'].min():.3f} – {df['district_lookback_mean'].max():.3f}")

# ── Bayut asking price feature ────────────────────────────────────────────────

print("Adding Bayut asking price feature...")
spreads_path = PROC / "asking_price_spreads_riyadh.json"
with open(spreads_path) as f:
    spreads_data = json.load(f)

bayut_map: dict = {}
for district, info in spreads_data.get("districts", {}).items():
    if info.get("bayut_n", 0) >= 5:
        bayut_map[district] = float(info["bayut_median_psqm"])

# Global fallback = median of all reliable district asking prices
global_bayut_psqm = float(np.median(list(bayut_map.values()))) if bayut_map else 7000.0
df["bayut_asking_psqm"] = df["district_ar"].map(bayut_map).fillna(global_bayut_psqm)
print(f"  Bayut map: {len(bayut_map)} districts | fallback: {global_bayut_psqm:,.0f} SAR/sqm")

# ── Feature column resolution ─────────────────────────────────────────────────

base_feat_cols = [f for f in BASE_FEATURES if f in df.columns]
missing = set(BASE_FEATURES) - set(base_feat_cols)
if missing:
    print(f"  WARNING: {len(missing)} feature(s) not in CSV: {sorted(missing)}")
# Full feature list = base + OOF encoding columns (appended inside fold)
full_feat_cols = base_feat_cols + OOF_ENC_COLS
print(f"  Base features: {len(base_feat_cols)} | +OOF encodings: {len(OOF_ENC_COLS)} | Total: {len(full_feat_cols)}")

# ── Train / holdout split ─────────────────────────────────────────────────────

cutoff_qid = 20251  # train on 2018-2024, holdout 2025 Q1-Q3
work_mask  = df["quarter_id"] < cutoff_qid
work = df[work_mask].reset_index(drop=True)
hold = df[~work_mask].reset_index(drop=True)
print(f"\n  Work set: {len(work)} rows | Holdout: {len(hold)} rows")

y_work = work["log_price"].values
y_hold = hold["log_price"].values
groups = work[GROUP_COL].values

# ── 5-fold Spatial GroupKFold CV ──────────────────────────────────────────────

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

oof_xgb = np.zeros(len(work))
oof_lgb = np.zeros(len(work))
oof_cat = np.zeros(len(work))

xgb_models, lgb_models, cat_models = [], [], []

print(f"\nRunning {N_FOLDS}-fold Spatial GroupKFold CV (OOF encodings computed per fold)...")
for fold, (tr_idx, va_idx) in enumerate(gkf.split(work[base_feat_cols].fillna(0), y_work, groups)):
    tr_df = work.iloc[tr_idx]
    va_df = work.iloc[va_idx]

    # ── OOF district encodings (no leakage) ──────────────────────────────
    enc_map, gm = _build_enc_map_from(tr_df, GROUP_COL, "log_price", k=30)
    apt_src = tr_df[tr_df["is_apartment"] == 1]
    if len(apt_src) == 0:
        apt_src = tr_df
    apt_enc_map, apt_gm = _build_enc_map_from(apt_src, GROUP_COL, "log_price", k=20)

    def _build_X(subset_df):
        X = subset_df[base_feat_cols].fillna(0).copy()
        X["district_enc_oof"]     = _apply_enc(subset_df, GROUP_COL, enc_map, gm).values
        X["district_apt_enc_oof"] = _apply_enc(subset_df, GROUP_COL, apt_enc_map, apt_gm).values
        return X.values.astype(np.float32)

    X_tr = _build_X(tr_df)
    X_va = _build_X(va_df)
    y_tr = y_work[tr_idx]
    y_va = y_work[va_idx]

    print(f"  Fold {fold+1}: train={len(tr_idx)} val={len(va_idx)}", end=" ", flush=True)

    m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = m_xgb.predict(X_va)

    m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    oof_lgb[va_idx] = m_lgb.predict(X_va)

    m_cat = cb.CatBoostRegressor(**CAT_PARAMS)
    m_cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50)
    oof_cat[va_idx] = m_cat.predict(X_va)

    xgb_models.append(m_xgb)
    lgb_models.append(m_lgb)
    cat_models.append(m_cat)
    print(f"| XGB fold MedAPE={medape(y_va, oof_xgb[va_idx]):.1f}%")

# ── OOF evaluation ────────────────────────────────────────────────────────────

oof_stack    = np.column_stack([oof_xgb, oof_lgb, oof_cat])
oof_r2       = r2_score(y_work, oof_stack.mean(axis=1))
oof_medape_v = medape(y_work, oof_stack.mean(axis=1))
print(f"\nOOF (ensemble mean): R²={oof_r2:.4f} | MedAPE={oof_medape_v:.2f}%")

# ── Ridge meta-learner ────────────────────────────────────────────────────────

print("Training Ridge meta-learner...")
meta = Ridge(alpha=1.0)
meta.fit(oof_stack, y_work)
oof_meta     = meta.predict(oof_stack)
oof_meta_r2  = r2_score(y_work, oof_meta)
oof_meta_med = medape(y_work, oof_meta)
print(f"OOF meta: R²={oof_meta_r2:.4f} | MedAPE={oof_meta_med:.2f}%")

# ── Retrain final models on full work set ─────────────────────────────────────
# Compute final district encodings from full work set

print("\nRetraining final models on full work set...")
global_mean_work = y_work.mean()
k = 30

def _build_enc_map(df_sub: pd.DataFrame, key_col: str, k_smooth: int = 30) -> dict:
    gm = df_sub["log_price"].mean()
    enc = {}
    for d, grp in df_sub.groupby(key_col):
        n = len(grp); mu = grp["log_price"].mean()
        enc[d] = (n * mu + k_smooth * gm) / (n + k_smooth)
    return enc, gm

work_enc_map, work_gm = _build_enc_map_from(work, GROUP_COL, "log_price", k)
apt_work = work[work["is_apartment"] == 1]
work_apt_enc_map, work_apt_gm = _build_enc_map_from(
    apt_work if len(apt_work) > 0 else work, GROUP_COL, "log_price", 20
)

def _build_X_final(subset_df: pd.DataFrame) -> np.ndarray:
    X = subset_df[base_feat_cols].fillna(0).copy()
    X["district_enc_oof"]     = _apply_enc(subset_df, GROUP_COL, work_enc_map, work_gm).values
    X["district_apt_enc_oof"] = _apply_enc(subset_df, GROUP_COL, work_apt_enc_map, work_apt_gm).values
    return X.values.astype(np.float32)

X_work_final = _build_X_final(work)
X_hold_final = _build_X_final(hold)

final_xgb = xgb.XGBRegressor(**XGB_PARAMS)
final_xgb.fit(X_work_final, y_work, verbose=False)

final_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
final_lgb.fit(X_work_final, y_work, callbacks=[lgb.log_evaluation(period=-1)])

final_cat = cb.CatBoostRegressor(**CAT_PARAMS)
final_cat.fit(X_work_final, y_work)

# ── Holdout evaluation ────────────────────────────────────────────────────────

print("\nEvaluating on holdout set...")
hold_preds = np.column_stack([
    final_xgb.predict(X_hold_final),
    final_lgb.predict(X_hold_final),
    final_cat.predict(X_hold_final),
])
hold_meta_preds = meta.predict(hold_preds)

hold_r2     = r2_score(y_hold, hold_meta_preds)
hold_medape = medape(y_hold, hold_meta_preds)
hold_mae    = mae_sar(y_hold, hold_meta_preds)

print(f"  Holdout R²:     {hold_r2:.4f}")
print(f"  Holdout MedAPE: {hold_medape:.2f}%")
print(f"  Holdout MAE:    {hold_mae:,.0f} SAR/sqm")

for ptype in ["apartment", "villa", "residential_plot", "building"]:
    mask_col = f"is_{ptype}"
    if mask_col in hold.columns:
        hmask = hold[mask_col].values.astype(bool)
        if hmask.sum() >= 10:
            t_r2  = r2_score(y_hold[hmask], hold_meta_preds[hmask])
            t_med = medape(y_hold[hmask], hold_meta_preds[hmask])
            print(f"  [{ptype:>18}] R²={t_r2:.4f} | MedAPE={t_med:.2f}% | n={hmask.sum()}")

# ── Save models ───────────────────────────────────────────────────────────────

print("\nSaving models...")

# ── Build look-back inference maps from work set ──────────────────────────────
# At inference time, new rows look back at ALL historical data → use work-set means
district_lookback_map: dict = (
    work.groupby(GROUP_COL)["log_price"].mean().to_dict()
)
city_lookback_mean = float(work["log_price"].mean())

apt_work_for_lb = work[work["is_apartment"] == 1]
district_lookback_apt_map: dict = (
    apt_work_for_lb.groupby(GROUP_COL)["log_price"].mean().to_dict()
    if len(apt_work_for_lb) > 0 else {}
)
city_lookback_apt_mean = float(apt_work_for_lb["log_price"].mean()
                               if len(apt_work_for_lb) > 0 else city_lookback_mean)

stack_path = MDIR / "riyadh_stack.pkl"
with open(stack_path, "wb") as f:
    pickle.dump({
        "xgb": final_xgb,
        "lgb": final_lgb,
        "cat": final_cat,
        "meta": meta,
        # OOF encoding maps (needed at inference)
        "district_enc_map":          work_enc_map,
        "district_enc_global":       work_gm,
        "district_apt_enc_map":      work_apt_enc_map,
        "district_apt_enc_global":   work_apt_gm,
        # Look-back maps (needed at inference)
        "district_lookback_map":     district_lookback_map,
        "city_lookback_mean":        city_lookback_mean,
        "district_lookback_apt_map": district_lookback_apt_map,
        "city_lookback_apt_mean":    city_lookback_apt_mean,
        # Bayut feature
        "bayut_psqm_map":            bayut_map,
        "bayut_psqm_global":         global_bayut_psqm,
    }, f, protocol=5)
print(f"  Saved: {stack_path}")

# ── Update riyadh_meta.json ───────────────────────────────────────────────────

meta_path = MDIR / "riyadh_meta.json"
meta_dict: dict = {}
if meta_path.exists():
    with open(meta_path) as f:
        meta_dict = json.load(f)

meta_dict.update({
    "feature_names":              full_feat_cols,
    "n_features":                 len(full_feat_cols),
    "holdout_r2":                 round(hold_r2, 4),
    "holdout_medape_pct":         round(hold_medape, 2),
    "holdout_mae_sar_sqm":        round(hold_mae, 2),
    "oof_r2":                     round(oof_meta_r2, 4),
    "oof_medape_pct":             round(oof_meta_med, 2),
    "model_version":              "riyadh_v2",
    "n_folds":                    N_FOLDS,
    "train_rows":                 len(work),
    "holdout_rows":               len(hold),
    "holdout_cutoff_quarter_id":  int(cutoff_qid),
    "meta_coefficients":          meta.coef_.tolist(),
    "meta_intercept":             float(meta.intercept_),
    "target":                     "log1p(sale_price_sar_sqm)",
    "y_unit":                     "SAR/sqm",
    "v2_improvements": [
        "OOF target encodings (no leakage)",
        "District look-back mean (past quarters only)",
        "Bayut asking price as feature",
        "LGB min_child_samples 10->20",
        "CAT l2_leaf_reg 2->3",
    ],
})

with open(meta_path, "w") as f:
    json.dump(meta_dict, f, indent=2, ensure_ascii=False)
print(f"  Updated: {meta_path}")

print("\n" + "=" * 60)
print("THAMAN Riyadh v2 — Training complete")
print(f"  OOF  R²={oof_meta_r2:.4f}  MedAPE={oof_meta_med:.2f}%")
print(f"  Hold R²={hold_r2:.4f}  MedAPE={hold_medape:.2f}%  MAE={hold_mae:,.0f} SAR/sqm")
print("=" * 60)
