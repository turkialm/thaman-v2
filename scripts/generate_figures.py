"""
THAMAN — Regenerate Model Figures (v2.1)
========================================
Produces updated SHAP importance, actual-vs-predicted, and error-by-borough
plots using the current v2.1 model (XGB + LGB + CAT + Ridge stack).

Run from project root:
    python scripts/generate_figures.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC      = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")

sys.path.insert(0, BASE)
from models.scorer import ThamanScorer

# ── Helpers ────────────────────────────────────────────────────────────
def medape(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)) * 100)

# ── 1. Load data ───────────────────────────────────────────────────────
print("[1/5] Loading features.csv …")
df = pd.read_csv(os.path.join(PROC, "features.csv"))
df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
df = df.dropna(subset=["sale_date", "sale_price", "latitude", "longitude"])
print(f"  Rows: {len(df):,}")

# ── 2. Replicate feature engineering from train_stack_v2.py ───────────
print("[2/5] Engineering features …")
from sklearn.preprocessing import MinMaxScaler

dist_cols = [c for c in df.columns if c.startswith("dist_")]
for col in dist_cols:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

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

df["is_manhattan"]          = (df["borough"] == 1).astype(int)
df["crime_x_manhattan"]     = df["crime_rate_nta"] * df["is_manhattan"]
df["crime_x_non_manhattan"] = df["crime_rate_nta"] * (1 - df["is_manhattan"])

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

df["sale_month_sin"] = np.sin(2 * np.pi * df["sale_month"] / 12)
df["sale_month_cos"] = np.cos(2 * np.pi * df["sale_month"] / 12)

# PLUTO assesstot coalesce
_PLUTO = os.path.join(BASE, "data", "raw", "nyc_pluto_25v4_csv", "pluto_25v4.csv")
if os.path.exists(_PLUTO):
    _pa = pd.read_csv(_PLUTO, usecols=["bbl","assesstot","assessland"], low_memory=False)
    _pa["bbl"] = pd.to_numeric(_pa["bbl"], errors="coerce")
    _pa = _pa.dropna(subset=["bbl"])
    _pa["bbl"] = _pa["bbl"].astype("int64")
    _bi = pd.to_numeric(df["bbl"], errors="coerce").fillna(0).astype("int64")
    df = df.assign(_bbl_int=_bi).merge(
        _pa.rename(columns={"bbl":"_bbl_int","assesstot":"_at","assessland":"_al"}),
        on="_bbl_int", how="left"
    ).drop(columns=["_bbl_int"])
    for _c, _p in [("assesstot","_at"),("assessland","_al")]:
        if _c not in df.columns: df[_c] = np.nan
        df[_c] = df[_c].combine_first(df[_p])
        df.drop(columns=[_p], inplace=True)
else:
    for _c in ["assesstot","assessland"]:
        if _c not in df.columns: df[_c] = np.nan

# Target encoding (on full df — only for visualisation, not holdout fidelity)
df["_log_price"] = np.log1p(df["sale_price"])
global_mean_log = float(df["_log_price"].mean())
bldg_means = df.groupby("bldgclass")["_log_price"].mean()
df["bldgclass_encoded"]   = df["bldgclass"].map(bldg_means).fillna(global_mean_log)
df["borough_bldg_key"]    = df["borough"].astype(str) + "_" + df["bldgclass"].str[:1]
bb_means = df.groupby("borough_bldg_key")["_log_price"].mean()
df["borough_bldg_encoded"] = df["borough_bldg_key"].map(bb_means).fillna(global_mean_log)

# Impute prior_sale_price via assesstot ratio
_has = df[(df["prior_sale_price"] > 0) & df["assesstot"].notna() & (df["assesstot"] > 0)]
_ratio_by_boro = {int(b): float((g["prior_sale_price"] / g["assesstot"]).median())
                  for b, g in _has.groupby("borough")}
_gr = float((_has["prior_sale_price"] / _has["assesstot"]).median()) if len(_has) else 10.0
_null_m = df["prior_sale_price"].isna() | (df["prior_sale_price"] == 0)
_ok_m   = df["assesstot"].notna() & (df["assesstot"] > 0)
df.loc[_null_m & _ok_m, "prior_sale_price"] = (
    df.loc[_null_m & _ok_m, "assesstot"] *
    df.loc[_null_m & _ok_m, "borough"].map(_ratio_by_boro).fillna(_gr)
)

# ── 3. Build feature matrix using scorer's feature list ───────────────
print("[3/5] Building X matrix …")
scorer = ThamanScorer()
feat_names = scorer.feature_names

missing = [f for f in feat_names if f not in df.columns]
if missing:
    print(f"  ⚠ Missing {len(missing)} features — filling with 0: {missing[:5]} …")
    for m in missing:
        df[m] = 0.0

X = df[feat_names].fillna(0).values.astype(np.float32)
y_true_usd = df["sale_price"].values

# Apply winsorization
for col, cap in scorer.winsorize.items():
    if col in feat_names:
        idx = feat_names.index(col)
        X[:, idx] = np.clip(X[:, idx], None, cap)

# Predictions using full stack
import joblib
from sklearn.linear_model import Ridge

dmat = xgb.DMatrix(X, feature_names=feat_names)
log_xgb = scorer.model.predict(dmat)
if scorer._stack is not None:
    log_lgb = scorer._stack["lgb"].predict(X).astype(np.float32)
    cols = [log_xgb, log_lgb]
    if "cat" in scorer._stack:
        log_cat = scorer._stack["cat"].predict(X).astype(np.float32)
        cols.append(log_cat)
    S = np.column_stack(cols)
    log_pred = scorer._stack["meta"].predict(S).astype(np.float32)
else:
    log_pred = log_xgb

y_pred_usd = np.expm1(log_pred)

# ── 4a. SHAP importance — random sample of 3000 ───────────────────────
print("[4/5] Computing SHAP values (sample=3000) …")
rng   = np.random.default_rng(42)
sidx  = rng.choice(len(X), size=min(3000, len(X)), replace=False)
X_shap = X[sidx]

explainer   = shap.TreeExplainer(scorer.model)
shap_vals   = explainer.shap_values(X_shap)
mean_abs    = np.abs(shap_vals).mean(axis=0)
top20_idx   = np.argsort(mean_abs)[::-1][:20]
top20_names = [feat_names[i] for i in top20_idx]
top20_vals  = mean_abs[top20_idx]

# Prettier feature names
_NAME_MAP = {
    "gross_square_feet":       "Building size (sq ft)",
    "bldgclass_encoded":       "Building class (encoded)",
    "land_square_feet":        "Land size (sq ft)",
    "longitude":               "Longitude (E-W location)",
    "latitude":                "Latitude (N-S location)",
    "prior_sale_price":        "Prior sale price",
    "assesstot":               "Assessed total value",
    "borough_bldg_encoded":    "Borough × bldg class",
    "median_income_nta":       "Median income (NTA)",
    "school_district":         "School district",
    "residential_units":       "Residential units",
    "building_age":            "Building age",
    "numfloors":               "Number of floors",
    "airbnb_count_500m":       "Airbnb density 500m",
    "dist_subway_m":           "Distance to subway",
    "log_dist_subway_m":       "Log dist. subway",
    "poi_count_500m":          "POI count 500m",
    "walk_score_proxy":        "Walk score (proxy)",
    "crime_rate_nta":          "Crime rate (NTA)",
    "dist_midtown_manhattan_m":"Dist. Midtown Manhattan",
    "dist_downtown_manhattan_m":"Dist. Downtown Manhattan",
    "price_appreciation":      "Price appreciation",
    "builtfar":                "Built FAR",
    "commfar":                 "Commercial FAR",
    "far_utilization":         "FAR utilization",
    "sale_year":               "Sale year",
    "dist_waterfront_m":       "Dist. waterfront",
    "mortgage_rate_30yr":      "Mortgage rate 30yr",
}
pretty_names = [_NAME_MAP.get(n, n.replace("_", " ").title()) for n in top20_names]

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#3b82f6" if i < 5 else "#93c5fd" for i in range(20)]
bars = ax.barh(range(20), top20_vals[::-1], color=colors[::-1], edgecolor="none", height=0.7)
ax.set_yticks(range(20))
ax.set_yticklabels(pretty_names[::-1], fontsize=9.5)
ax.set_xlabel("Mean |SHAP value| (log-price scale)", fontsize=10)
ax.set_title("THAMAN v2.1 — Top 20 Feature Importances (SHAP)", fontsize=13, fontweight="bold", pad=12)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

for i, (bar, val) in enumerate(zip(bars, top20_vals[::-1])):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, color="#374151")

plt.tight_layout()
out_shap = os.path.join(MODEL_DIR, "shap_importance.png")
plt.savefig(out_shap, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved {out_shap}")

# ── 4b. Actual vs Predicted ───────────────────────────────────────────
print("  Generating actual vs predicted scatter …")
mask     = (y_true_usd < 10_000_000) & (y_pred_usd < 10_000_000)
yt, yp   = y_true_usd[mask] / 1e6, y_pred_usd[mask] / 1e6
med_ape  = medape(y_true_usd, y_pred_usd)

# Density colouring
from scipy.stats import gaussian_kde
xy    = np.vstack([yt, yp])
dsamp = min(5000, len(yt))
didx  = rng.choice(len(yt), dsamp, replace=False)
kde   = gaussian_kde(xy[:, didx])
z     = kde(xy[:, didx])

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(yt[didx], yp[didx], c=z, s=6, cmap="YlOrRd", alpha=0.6, linewidths=0)
plt.colorbar(sc, ax=ax, label="Density")
lim = max(yt.max(), yp.max()) * 1.05
ax.plot([0, lim], [0, lim], "b--", lw=1.5, label="Perfect fit")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Actual Price ($M)", fontsize=11)
ax.set_ylabel("Predicted Price ($M)", fontsize=11)

from sklearn.metrics import r2_score
r2 = r2_score(yt, yp[didx[:len(yt)]] if len(yp) != len(yt) else yp)
# recalculate properly
r2 = r2_score(y_true_usd[mask], y_pred_usd[mask])

ax.set_title(f"THAMAN v2.1 — Actual vs Predicted\nR²={r2:.4f}  MedAPE={med_ape:.1f}%  (n={mask.sum():,}, cap $10M)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))

plt.tight_layout()
out_avp = os.path.join(MODEL_DIR, "actual_vs_predicted.png")
plt.savefig(out_avp, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved {out_avp}")

# ── 4c. Error by borough ──────────────────────────────────────────────
print("  Generating error-by-borough boxplot …")
pct_err  = np.abs(y_true_usd - y_pred_usd) / np.maximum(y_true_usd, 1.0) * 100
boroughs = df["borough"].values
bnames   = {1:"Manhattan", 2:"Bronx", 3:"Brooklyn", 4:"Queens", 5:"Staten\nIsland"}
colors_b = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]

data_by_b = []
labels_b  = []
for b in [1, 2, 3, 4, 5]:
    mask_b = (boroughs == b) & (pct_err < 200)  # cap outliers for visualisation
    data_by_b.append(pct_err[mask_b])
    labels_b.append(bnames[b])

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(data_by_b, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=2, alpha=0.3))

for patch, color in zip(bp["boxes"], colors_b):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for i, (d, lbl) in enumerate(zip(data_by_b, labels_b)):
    med = np.median(d)
    ax.text(i + 1, med + 1.5, f"{med:.0f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1f2937")

ax.set_xticks(range(1, 6))
ax.set_xticklabels(labels_b, fontsize=11)
ax.set_ylabel("Absolute % Error", fontsize=11)
ax.set_title("THAMAN v2.1 — Prediction Error by Borough\n(Median MedAPE shown above each box)",
             fontsize=12, fontweight="bold")
ax.axhline(med_ape, color="#6b7280", linestyle="--", lw=1.2, label=f"Overall MedAPE {med_ape:.1f}%")
ax.set_ylim(0, min(100, np.percentile(np.concatenate(data_by_b), 95) * 1.3))
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
out_err = os.path.join(MODEL_DIR, "error_by_borough.png")
plt.savefig(out_err, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved {out_err}")

# ── 5. Summary ─────────────────────────────────────────────────────────
print(f"\n[5/5] Done.")
print(f"  R²     : {r2:.4f}")
print(f"  MedAPE : {med_ape:.2f}%")
print(f"  Top feature: {pretty_names[0]}  ({top20_vals[0]:.3f})")
print(f"\n  Figures saved to models/")
print(f"  → shap_importance.png")
print(f"  → actual_vs_predicted.png")
print(f"  → error_by_borough.png")
