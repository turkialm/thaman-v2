"""
generate_scatter.py
===================
Generates predicted-vs-actual scatter plot data from the NYC v22 model.

Approach:
  1. Load features_v6.csv + apply same time-based 15% holdout split
  2. Apply minimal feature engineering (log transforms, encodings from meta.json)
  3. Run model inference on holdout rows
  4. Save data/processed/scatter_nyc.json  (actual, predicted, borough, bldgclass)

Run:
  cd /Users/totam/Desktop/new_try
  python scripts/generate_scatter.py
"""

import os, sys, json, warnings
import numpy as np
import polars as pl
import joblib

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC      = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")

print("=" * 60)
print("  THAMAN — Scatter Plot Data Generator (NYC v22)")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────────
_FLOAT_OVERRIDES = {
    "dist_waterfront_m": pl.Float64, "dist_bike_lane_m": pl.Float64,
    "prior_sale_price": pl.Float64, "years_since_prior_sale": pl.Float64,
    "price_appreciation": pl.Float64, "is_flip": pl.Float64,
}
_csv_path = os.path.join(PROC, "features_v6.csv")
if not os.path.exists(_csv_path):
    _csv_path = os.path.join(PROC, "features_v5.csv")
    print("  ⚠ features_v6.csv not found — using v5")

df = (
    pl.read_csv(_csv_path, schema_overrides=_FLOAT_OVERRIDES)
    .with_columns(pl.col("sale_date").str.to_datetime(format=None, strict=False))
    .drop_nulls(subset=["sale_date", "sale_price", "latitude", "longitude"])
)
print(f"  Loaded: {len(df):,} rows")

# ── 2. Time-based holdout (last 15%) ─────────────────────────────
df_sorted = df.sort("sale_date")
n_hold    = int(len(df_sorted) * 0.15)
df_hold   = df_sorted[-n_hold:]
print(f"  Holdout: {len(df_hold):,} rows (last 15%)")

# ── 3. Load model + meta ─────────────────────────────────────────
stack_path = os.path.join(MODEL_DIR, "thaman_stack.pkl")
meta_path  = os.path.join(MODEL_DIR, "meta.json")
print("  Loading model stack …")
stk  = joblib.load(stack_path)
with open(meta_path) as f:
    meta = json.load(f)

feat_names   = meta["feature_names"]
global_mean  = meta.get("global_mean_log", 13.5)
bldg_means   = meta.get("bldgclass_means", {})
bb_means     = meta.get("borough_bldg_means", {})
nta_means    = meta.get("nta_means", {})
ntab_means   = meta.get("nta_bldg_means", {})
nta_stats    = meta.get("nta_stats", {})
bbl_lookup   = meta.get("bbl_median_lookup", {})
bbl_global   = float(meta.get("bbl_hist_psf_global", 0.0))
nta_trend    = meta.get("nta_price_trend_slope_map", {})  # may not exist
lag1_map     = meta.get("nta_lag_q_map", {})
lag_glb      = meta.get("nta_lag_q_globals", {})
g_logp       = float(lag_glb.get("mean_logp",  13.0))
g_psf        = float(lag_glb.get("median_psf", 500.0))
g_cnt        = float(lag_glb.get("count",      50.0))
GRAVITY = {
    "midtown_manhattan":  (40.7549, -73.9840),
    "downtown_manhattan": (40.7074, -74.0113),
    "downtown_brooklyn":  (40.6928, -73.9903),
    "long_island_city":   (40.7447, -73.9485),
}
DIST_COLS = [
    "dist_subway_m","dist_school_m","dist_park_m","dist_hospital_m",
    "dist_bus_m","dist_bike_lane_m","dist_elem_school_m","dist_express_subway_m",
    "dist_waterfront_m","dist_commuter_rail_m","dist_large_park_m","dist_flagship_park_m",
]
v11_zip_lu = meta.get("v11_zip_lookup", {})
v11_nta_lu = meta.get("v11_nta_lookup", {})
v16_lu     = meta.get("v16_nta_lookup", {})
v17_lu     = meta.get("v17_nta_log_exempt", {})
v18_lu     = meta.get("v18_nta_log_income", {})
v16_gh     = float(meta.get("v16_global_hist_rate", 0.08))
v16_gf     = float(meta.get("v16_global_flood_rate", 0.08))
v17_glob   = float(meta.get("v17_global_log_exempt", 5.17))
v18_glob   = float(meta.get("v18_global_log_income", 11.37))
MTA_MEDIANS = {
    "nearest_station_is_cbd":      0,
    "nearest_station_route_count": 2,
    "nearest_station_is_ada":      0,
}

# ── 4. Build feature matrix for holdout ──────────────────────────
print("  Engineering features for holdout …")

# Add computed columns that training script creates
# Log-transformed distances
_dist_cols_exist = [c for c in DIST_COLS if c in df_hold.columns]
for col in _dist_cols_exist:
    df_hold = df_hold.with_columns(
        pl.col(col).cast(pl.Float64, strict=False).clip(lower_bound=0).log1p().alias(f"log_{col}")
    )

# Gravity distances
lats = df_hold["latitude"].to_numpy()
lons = df_hold["longitude"].to_numpy()
for name, (clat, clon) in GRAVITY.items():
    col_vals = np.sqrt((lats - clat)**2 + (lons - clon)**2) * 111_000.0
    df_hold = df_hold.with_columns(pl.Series(f"dist_{name}_m", col_vals, dtype=pl.Float64))

# Borough interaction
df_hold = df_hold.with_columns([
    (pl.col("borough") == 1).cast(pl.Int32).alias("is_manhattan"),
    (pl.col("crime_rate_nta") * (pl.col("borough") == 1).cast(pl.Int32)).alias("crime_x_manhattan"),
    (pl.col("crime_rate_nta") * (1 - (pl.col("borough") == 1).cast(pl.Int32))).alias("crime_x_non_manhattan"),
])

# Cyclical month
df_hold = df_hold.with_columns([
    (pl.col("sale_month") * (2 * np.pi / 12)).sin().alias("sale_month_sin"),
    (pl.col("sale_month") * (2 * np.pi / 12)).cos().alias("sale_month_cos"),
])

# v5 interactions
df_hold = df_hold.with_columns([
    (pl.col("gross_square_feet") / pl.col("numfloors").clip(lower_bound=1)).alias("sqft_per_floor"),
    (pl.col("median_income_nta") / (pl.col("crime_rate_nta") + 1.0)).alias("income_over_crime"),
    (pl.col("residential_units") / pl.col("gross_square_feet").clip(lower_bound=1) * 1000.0).alias("density_index"),
    ((pl.col("gross_square_feet").clip(lower_bound=1).log1p()) *
     (pl.col("numfloors").clip(lower_bound=1).log1p())).alias("log_sqft_x_floors"),
])

# v6 structural
df_hold = df_hold.with_columns([
    pl.col("land_square_feet").clip(lower_bound=1).log1p().alias("log_land_sqft"),
    (pl.col("gross_square_feet") / pl.col("land_square_feet").clip(lower_bound=1))
        .clip(upper_bound=10).alias("lot_coverage"),
    (pl.col("gross_square_feet") * pl.col("numfloors").clip(lower_bound=1)).log1p()
        .alias("bldg_vol_proxy"),
    (pl.col("prior_sale_price") / pl.col("gross_square_feet").clip(lower_bound=1))
        .alias("prior_price_psf"),
])

# Walk score proxy (simplified — use walk_score_proxy from CSV if present)
if "walk_score_proxy" not in df_hold.columns:
    df_hold = df_hold.with_columns(pl.lit(50.0).alias("walk_score_proxy"))

# Target encodings from meta.json
_bc_arr  = df_hold["bldgclass"].to_list()
_bor_arr = df_hold["borough"].to_list()
_nta_arr = df_hold["ntacode"].to_list() if "ntacode" in df_hold.columns else [""] * len(df_hold)

bldg_enc  = [float(bldg_means.get(bc or "", global_mean)) for bc in _bc_arr]
bb_enc    = [float(bb_means.get(f"{bo}_{(bc or '')[:1]}", global_mean)) for bo, bc in zip(_bor_arr, _bc_arr)]
nta_enc   = [float(nta_means.get(nt or "", global_mean)) for nt in _nta_arr]
ntab_enc  = [float(ntab_means.get(f"{(nt or '')}_{(bc or '')[:1]}",
             float(nta_means.get(nt or "", global_mean)))) for nt, bc in zip(_nta_arr, _bc_arr)]
nta_cnt   = [float(nta_stats.get(nt, {}).get("sale_count", 0)) for nt in _nta_arr]
nta_psf   = [float(nta_stats.get(nt, {}).get("median_psf", 0.0)) for nt in _nta_arr]

# NTA trend slope — stored in nta_stats per NTA, or use global fallback
_global_trend = float(meta.get("global_trend", 0.02))
nta_slope  = [float(nta_stats.get(nt or "", {}).get("trend_slope", _global_trend))
              for nt in _nta_arr]

df_hold = df_hold.with_columns([
    pl.Series("bldgclass_encoded",    bldg_enc, dtype=pl.Float64),
    pl.Series("borough_bldg_encoded", bb_enc,   dtype=pl.Float64),
    pl.Series("nta_encoded",          nta_enc,  dtype=pl.Float64),
    pl.Series("nta_bldg_encoded",     ntab_enc, dtype=pl.Float64),
    pl.Series("nta_sale_count",       nta_cnt,  dtype=pl.Float64),
    pl.Series("nta_median_psf",       nta_psf,  dtype=pl.Float64),
    pl.Series("nta_price_trend_slope",nta_slope,dtype=pl.Float64),
])

# v11 features: use NTA/ZIP lookup medians as fallbacks
for col in ["hpd_class_b_viol_zip","hpd_class_c_viol_zip","hpd_severity_score_zip",
            "dob_reno_permit_count","dob_newbld_permit_count"]:
    if col not in df_hold.columns:
        df_hold = df_hold.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
for col in ["rat_density_nta","heat_density_nta"]:
    if col not in df_hold.columns:
        df_hold = df_hold.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
for col, val in MTA_MEDIANS.items():
    if col not in df_hold.columns:
        df_hold = df_hold.with_columns(pl.lit(val).cast(pl.Float64).alias(col))

# v12: NTA temporal lag — use per-row sale quarter for correct lag lookup
import datetime as _dt
_yr_arr  = df_hold["sale_year"].to_list()  if "sale_year"  in df_hold.columns else [2023]*len(_nta_arr)
_mo_arr  = df_hold["sale_month"].to_list() if "sale_month" in df_hold.columns else [6]   *len(_nta_arr)

lag1_logp_arr = []; lag1_psf_arr = []; lag1_cnt_arr = []
lag2_logp_arr = []; momentum_arr  = []
for nt, yr, mo in zip(_nta_arr, _yr_arr, _mo_arr):
    # current quarter index and previous two
    yrq  = (int(yr) - 2018) * 4 + (int(mo) - 1) // 3
    yrq1 = max(0, yrq - 1)   # lag-1
    yrq2 = max(0, yrq - 2)   # lag-2
    r1 = lag1_map.get(f"{nt}_{yrq1}", lag1_map.get(f"{nt or ''}_{yrq1}", {}))
    r2 = lag1_map.get(f"{nt}_{yrq2}", lag1_map.get(f"{nt or ''}_{yrq2}", {}))
    ml1 = float(r1.get("mean_logp",  g_logp))
    mp1 = float(r1.get("median_psf", g_psf))
    mc1 = float(r1.get("count",      g_cnt))
    ml2 = float(r2.get("mean_logp",  g_logp))
    lag1_logp_arr.append(ml1); lag1_psf_arr.append(mp1); lag1_cnt_arr.append(mc1)
    lag2_logp_arr.append(ml2)
    momentum_arr.append(ml1 - ml2)

df_hold = df_hold.with_columns([
    pl.Series("nta_lag1q_mean_logp",  lag1_logp_arr, dtype=pl.Float64),
    pl.Series("nta_lag1q_median_psf", lag1_psf_arr,  dtype=pl.Float64),
    pl.Series("nta_lag1q_count",      lag1_cnt_arr,  dtype=pl.Float64),
    pl.Series("nta_lag2q_mean_logp",  lag2_logp_arr, dtype=pl.Float64),
    pl.Series("nta_logp_momentum",    momentum_arr,  dtype=pl.Float64),
])

# v16: historic/flood (NTA lookup)
v16_hist  = [float(v16_lu.get(nt, {}).get("is_historic_dist", v16_gh)) for nt in _nta_arr]
v16_flood = [float(v16_lu.get(nt, {}).get("in_flood_zone",    v16_gf)) for nt in _nta_arr]
v16_lm    = [float(v16_lu.get(nt, {}).get("is_landmark",      0.0))    for nt in _nta_arr]
v17_ex    = [float(v17_lu.get(nt, v17_glob)) for nt in _nta_arr]
v18_inc   = [float(v18_lu.get(nt, v18_glob)) for nt in _nta_arr]
df_hold = df_hold.with_columns([
    pl.Series("is_historic_dist",       v16_hist,  dtype=pl.Float64),
    pl.Series("in_flood_zone",          v16_flood, dtype=pl.Float64),
    pl.Series("is_landmark",            v16_lm,    dtype=pl.Float64),
    pl.Series("log_exempt_amount",      v17_ex,    dtype=pl.Float64),
    pl.Series("log_tract_median_income",v18_inc,   dtype=pl.Float64),
])

# v21: BBL hist psf
bbl_arr = df_hold["bbl"].cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64).to_list() \
          if "bbl" in df_hold.columns else [0] * len(df_hold)
bbl_hist = [float(bbl_lookup.get(b, nta_psf[i] or bbl_global))
            for i, b in enumerate(bbl_arr)]
df_hold = df_hold.with_columns(pl.Series("bbl_hist_psf", bbl_hist, dtype=pl.Float64))

# Also compute log_dist_citibike_m if dist_citibike_m exists
if "dist_citibike_m" in df_hold.columns and "log_dist_citibike_m" not in df_hold.columns:
    df_hold = df_hold.with_columns(
        pl.col("dist_citibike_m").cast(pl.Float64, strict=False).clip(lower_bound=0).log1p()
        .alias("log_dist_citibike_m")
    )

# ── 5. Build X_hold matrix ────────────────────────────────────────
print("  Building feature matrix …")
# Add any remaining missing feat_name columns as 0
for fn in feat_names:
    if fn not in df_hold.columns:
        df_hold = df_hold.with_columns(pl.lit(0.0).cast(pl.Float64).alias(fn))

missing_filled = [fn for fn in feat_names if fn not in df_hold.columns]
print(f"  Features: {len(feat_names)} total | missing still: {len(missing_filled)}")

X_hold = df_hold.select(feat_names).fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)

# ── 6. Run stack predictions ──────────────────────────────────────
print("  Running stack inference …")
final_xa = stk["xgb_a"]; final_xb = stk["xgb_b"]
final_lg = stk["lgb"];   final_ct = stk["cat"]
meta_model = stk["meta"]; meta_type  = stk.get("meta_type", "ridge")

p_xa = final_xa.predict(X_hold)
p_xb = final_xb.predict(X_hold)
p_lg = final_lg.predict(X_hold)
p_ct = final_ct.predict(X_hold).astype(np.float32)
S_hold = np.column_stack([p_xa, p_xb, p_lg, p_ct])
pred_log = meta_model.predict(S_hold)

y_actual   = np.expm1(df_hold["sale_price"].log1p().to_numpy())
y_pred     = np.expm1(pred_log)

# ── 7. Compute metrics ────────────────────────────────────────────
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
medape_val = float(np.median(np.abs(y_actual - y_pred) / np.maximum(y_actual, 1.0)) * 100)
r2_val     = r2_score(np.log1p(y_actual), pred_log)
print(f"  Scatter: R²={r2_val:.4f} | MedAPE={medape_val:.2f}%")

# ── 8. Sample for scatter plot (2000 pts, stratified by borough) ──
BOROUGH_MAP = {1:"Manhattan",2:"Bronx",3:"Brooklyn",4:"Queens",5:"Staten Island"}
boros       = df_hold["borough"].to_list()

np.random.seed(42)
out_rows = []
for bn, bname in BOROUGH_MAP.items():
    mask = [i for i, b in enumerate(boros) if b == bn]
    n_sample = min(400, len(mask))
    chosen = np.random.choice(mask, n_sample, replace=False)
    for idx in chosen:
        out_rows.append({
            "actual":    round(float(y_actual[idx])),
            "predicted": round(float(y_pred[idx])),
            "borough":   bname,
            "bldgclass": _bc_arr[idx],
            "error_pct": round(abs(y_actual[idx] - y_pred[idx]) / max(y_actual[idx], 1) * 100, 1),
        })

print(f"  Sample: {len(out_rows)} points ({sum(1 for r in out_rows if r['borough']=='Manhattan')} MAN, "
      f"{sum(1 for r in out_rows if r['borough']=='Brooklyn')} BKN, etc.)")

# ── 9. Save ───────────────────────────────────────────────────────
out = {
    "meta": {
        "model":   "NYC v22",
        "r2":      round(r2_val, 4),
        "medape":  round(medape_val, 2),
        "n_total": len(df_hold),
        "n_sample": len(out_rows),
        "generated": str(_dt.date.today()),
    },
    "points": out_rows,
}
out_path = os.path.join(PROC, "scatter_nyc.json")
with open(out_path, "w") as f:
    json.dump(out, f)
print(f"  Saved → {out_path}")

# ── Riyadh scatter ────────────────────────────────────────────────
print("\n  Generating Riyadh scatter …")
import pickle as _pkl
riy_csv = os.path.join(PROC, "features_riyadh_v2.csv")
rmeta_path = os.path.join(MODEL_DIR, "riyadh_meta.json")
rstack_path = os.path.join(MODEL_DIR, "riyadh_stack.pkl")

if os.path.exists(riy_csv) and os.path.exists(rstack_path):
    import pandas as _pd
    import warnings; warnings.filterwarnings("ignore")
    from math import radians, sin, cos, sqrt, atan2

    _rdf = _pd.read_csv(riy_csv, encoding="utf-8-sig")
    with open(rmeta_path) as f: rmeta = json.load(f)
    with open(rstack_path, "rb") as f: rstk = _pkl.load(f)

    # ── Fix corrupt 2024 quarter_id (same as training script) ────────────────
    _bug_mask = _rdf["quarter_id"] >= 40000
    if _bug_mask.sum() > 0:
        _rdf.loc[_bug_mask, "quarter_id"] = (
            _rdf.loc[_bug_mask, "year"].astype(int) * 10 +
            _rdf.loc[_bug_mask, "quarter_id"] % 10
        ).astype(int)
        print(f"  Fixed {_bug_mask.sum()} corrupt 2024 quarter_id rows")

    _TARGET = "sale_price_sar_sqm"
    _cutoff = 20251
    _train_r = _rdf[_rdf["quarter_id"] < _cutoff].copy()
    _hold_r  = _rdf[_rdf["quarter_id"] >= _cutoff].copy()
    print(f"  Riyadh holdout: {len(_hold_r)} rows (train: {len(_train_r)})")

    # ── Engineer missing features ─────────────────────────────────────────────

    # 1. Hub distances (haversine from district_lat/lon)
    _HUBS_R = {
        "kafd":       (24.771, 46.637),
        "old_city":   (24.690, 46.722),
        "industrial": (24.620, 46.873),
        "airport":    (24.957, 46.699),
    }
    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlam = radians(lon2 - lon1)
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    for _hub, (_hlat, _hlon) in _HUBS_R.items():
        col = f"dist_{_hub}_m"
        lcol = f"log_dist_{_hub}_m"
        _hold_r[col]  = _hold_r.apply(
            lambda r: _haversine_m(r["district_lat"], r["district_lon"], _hlat, _hlon)
            if _pd.notna(r.get("district_lat")) else 5000.0, axis=1).astype(np.float32)
        _hold_r[lcol] = np.log1p(_hold_r[col]).astype(np.float32)

    # 2. rei_type_idx — type-matched REI from meta lookup
    _rei_lookup  = rmeta.get("rei_type_idx_lookup", {})
    _rei_latest  = rmeta.get("rei_type_idx_latest", {})
    _TYPE_REI_MAP = {"is_apartment": "apartment", "is_villa": "villa",
                     "is_residential_plot": "residential_plot", "is_building": "building"}
    def _get_rei_type(row):
        qid = str(int(row["quarter_id"])) if _pd.notna(row.get("quarter_id")) else ""
        ptype = next((v for k, v in _TYPE_REI_MAP.items() if row.get(k, 0) == 1), "apartment")
        qdata = _rei_lookup.get(qid, _rei_latest)
        return float(qdata.get(ptype, _rei_latest.get(ptype, 100.0)))
    _hold_r["rei_type_idx"] = _hold_r.apply(_get_rei_type, axis=1).astype(np.float32)

    # 3. Metro binary features + nearest_metro_line from metro_district_lookup
    _metro_lkp = rmeta.get("metro_district_lookup", {})
    _METRO_500_THRESH = 500.0
    _METRO_1K_THRESH  = 1000.0
    def _metro_feats(row):
        dist = float(row.get("dist_metro_m", 9999))
        dist_m = dist if _pd.notna(dist) else 9999.0
        lkp = _metro_lkp.get(str(row.get("district_ar", "")), {})
        if lkp:
            dist_m = float(lkp.get("dist_metro_m", dist_m))
        return (
            int(dist_m <= _METRO_500_THRESH),
            int(dist_m <= _METRO_1K_THRESH),
            int(lkp.get("nearest_metro_line", 0)),
        )
    _metro_vals = _hold_r.apply(_metro_feats, axis=1, result_type="expand")
    _hold_r["metro_500m"]         = _metro_vals[0].astype(np.float32)
    _hold_r["metro_1km"]          = _metro_vals[1].astype(np.float32)
    _hold_r["nearest_metro_line"] = _metro_vals[2].astype(np.float32)

    # 4. District lag features from training portion
    _dq_med = (_train_r.groupby(["district_ar", "quarter_id"])[_TARGET]
               .median().reset_index().rename(columns={_TARGET: "_dq_med"}))
    _qseq = sorted(_train_r["quarter_id"].unique())
    _qprev = {q: _qseq[i-1] if i > 0 else None for i, q in enumerate(_qseq)}
    _qprev2 = {q: _qseq[i-2] if i > 1 else None for i, q in enumerate(_qseq)}
    _dq_dict = {(r["district_ar"], r["quarter_id"]): r["_dq_med"]
                for _, r in _dq_med.iterrows()}
    _city_med = float(_train_r[_TARGET].median())

    def _lag_feats(row):
        dist = str(row.get("district_ar", ""))
        q    = int(row.get("quarter_id", 0))
        # find prev quarter in train sequence (or nearest)
        pq  = _qprev.get(q)   or (_qseq[-1] if _qseq else None)
        pq2 = _qprev2.get(q)  or (_qseq[-2] if len(_qseq) > 1 else pq)
        l1 = _dq_dict.get((dist, pq),  _city_med) if pq  else _city_med
        l2 = _dq_dict.get((dist, pq2), _city_med) if pq2 else _city_med
        mom = (l1 - l2) / max(l2, 1.0)
        return float(l1), float(l2), float(mom)

    _lag_vals = _hold_r.apply(_lag_feats, axis=1, result_type="expand")
    _hold_r["district_lag1q_median_psqm"] = _lag_vals[0].astype(np.float32)
    _hold_r["district_lag2q_median_psqm"] = _lag_vals[1].astype(np.float32)
    _hold_r["district_lag_momentum"]      = _lag_vals[2].astype(np.float32)

    # 5. v11: type-stratified lags + price std + suhail density (from training portion)
    _TYPE_COL_MAP = {"is_apartment": "apt", "is_villa": "villa",
                     "is_residential_plot": "plot", "is_building": "bldg"}
    def _ptype(row):
        for col, label in _TYPE_COL_MAP.items():
            if row.get(col, 0) == 1: return label
        return "apt"

    # compute district×type×quarter agg from training portion
    _dtq_agg = (
        _train_r.copy()
        .assign(_ptype=[_ptype(r) for _, r in _train_r.iterrows()])
        .groupby(["district_ar", "_ptype", "quarter_id"])[_TARGET]
        .agg(["median", "std"])
        .reset_index()
        .rename(columns={"median": "_dtq_m", "std": "_dtq_s"})
    )
    _dtq_dict2 = {(r["district_ar"], r["_ptype"], r["quarter_id"]): r["_dtq_m"]
                  for _, r in _dtq_agg.iterrows()}
    _dtq_std2  = {(r["district_ar"], r["_ptype"], r["quarter_id"]): r["_dtq_s"]
                  for _, r in _dtq_agg.iterrows()}
    _global_std = float(_train_r[_TARGET].std())

    def _v11_feats(row):
        dist = str(row.get("district_ar", ""))
        q    = int(row.get("quarter_id", 0))
        pt   = _ptype(row)
        pq   = _qprev.get(q)  or (_qseq[-1] if _qseq else None)
        pq2  = _qprev2.get(q) or (_qseq[-2] if len(_qseq) > 1 else pq)
        l1t  = _dtq_dict2.get((dist, pt, pq),  _dq_dict.get((dist, pq),  _city_med)) if pq  else _city_med
        l2t  = _dtq_dict2.get((dist, pt, pq2), _dq_dict.get((dist, pq2), _city_med)) if pq2 else _city_med
        std1 = _dtq_std2.get((dist, pt, pq),   _global_std) if pq else _global_std
        return float(l1t), float(l2t), float(std1)

    _v11_vals = _hold_r.apply(_v11_feats, axis=1, result_type="expand")
    _hold_r["district_type_lag1q_psqm"] = _v11_vals[0].astype(np.float32)
    _hold_r["district_type_lag2q_psqm"] = _v11_vals[1].astype(np.float32)
    _hold_r["district_lag1q_std_psqm"]  = _v11_vals[2].astype(np.float32)

    # log_suhail_n_trans: lag-1q individual transaction count from suhail raw
    _suhail_all = os.path.join(BASE, "data", "raw", "suhail_riyadh_tx_raw.csv")
    if os.path.exists(_suhail_all):
        import polars as _pl3
        _stx = (
            _pl3.read_csv(str(_suhail_all), ignore_errors=True)
            .filter(_pl3.col("province_name") == "الرياض")
            .with_columns([
                _pl3.col("date").str.slice(0, 4).cast(_pl3.Int64).alias("_sy"),
                _pl3.col("date").str.slice(5, 2).cast(_pl3.Int64).alias("_sm"),
            ])
            .with_columns((_pl3.col("_sy") * 10 + (_pl3.col("_sm") - 1) // 3 + 1).alias("quarter_id"))
            .filter((_pl3.col("psqm") >= 500) & (_pl3.col("psqm") <= 50_000))
            .group_by(["district_ar", "quarter_id"])
            .agg(_pl3.len().alias("_s_cnt"))
        ).to_pandas()
        _s_dict2 = {(r["district_ar"], r["quarter_id"]): r["_s_cnt"] for _, r in _stx.iterrows()}

        def _suhail_cnt(row):
            pq = _qprev.get(int(row.get("quarter_id", 0)))
            return _s_dict2.get((str(row.get("district_ar", "")), pq), 0) if pq else 0

        _hold_r["log_suhail_n_trans"] = np.log1p(_hold_r.apply(_suhail_cnt, axis=1)).astype(np.float32)
    else:
        _hold_r["log_suhail_n_trans"] = np.float32(0.0)

    # ── Build feature matrix ──────────────────────────────────────────────────
    rfeat = rmeta.get("feature_names", [])
    rfeat_ok = [f for f in rfeat if f in _hold_r.columns]
    missing_r = [f for f in rfeat if f not in _hold_r.columns]
    if missing_r:
        print(f"  Still missing (→0): {missing_r}")
    X_r = np.zeros((len(_hold_r), len(rfeat)), dtype=np.float32)
    for i, fn in enumerate(rfeat):
        if fn in _hold_r.columns:
            X_r[:, i] = _hold_r[fn].fillna(0).values.astype(np.float32)

    rp = np.column_stack([
        rstk["xgb"].predict(X_r),
        rstk["lgb"].predict(X_r),
        rstk["cat"].predict(X_r),
    ])
    ry_pred_log = rstk["meta"].predict(rp)
    ry_actual   = _hold_r[_TARGET].values.astype(np.float32)  # already raw SAR/sqm
    ry_pred     = np.expm1(ry_pred_log)                       # model predicts log1p space

    TYPE_MAP = {"is_apartment":"Apartment","is_villa":"Villa",
                "is_residential_plot":"Plot","is_building":"Building"}
    r_types = []
    for _, row in _hold_r.iterrows():
        t = "Other"
        for col, label in TYPE_MAP.items():
            if col in row and row[col] == 1:
                t = label; break
        r_types.append(t)

    np.random.seed(42)
    chosen_r = np.random.choice(len(_hold_r), min(800, len(_hold_r)), replace=False)
    r_out = [{"actual": round(float(ry_actual[i])),
               "predicted": round(float(ry_pred[i])),
               "type": r_types[i],
               "error_pct": round(abs(ry_actual[i]-ry_pred[i])/max(ry_actual[i],1)*100,1)}
             for i in chosen_r]

    r_medape = float(np.median(np.abs(ry_actual-ry_pred)/np.maximum(ry_actual,1))*100)
    r_r2     = r2_score(np.log1p(np.maximum(ry_actual, 1)), ry_pred_log)

    rout = {
        "meta": {"model":"Riyadh v10","r2":round(r_r2,4),
                 "medape":round(r_medape,2),"n_total":len(_hold_r),"n_sample":len(r_out),
                 "generated":str(_dt.date.today())},
        "points": r_out,
    }
    rout_path = os.path.join(PROC, "scatter_riyadh.json")
    with open(rout_path, "w") as f:
        json.dump(rout, f)
    print(f"  Riyadh: R²={r_r2:.4f} MedAPE={r_medape:.2f}% | saved → {rout_path}")
else:
    print("  Riyadh data/model not found — skipping")

print("\n  Done ✓")
