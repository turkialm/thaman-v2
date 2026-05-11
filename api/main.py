"""
THAMAN Property Valuation API  (v2 — 71 features)
==============================
FastAPI backend for the THAMAN AI-powered PropTech system.

Endpoints:
  GET  /             → redirect to map UI
  GET  /api          → API info
  GET  /health       → health check
  GET  /bldgclasses  → valid building class codes
  POST /predict      → property price prediction + SHAP drivers
  POST /batch        → batch prediction for multiple properties

Usage:
  cd new_try
  uvicorn api.main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""

import sys
import os
import json
import time
import asyncio
import datetime

import numpy as np
import polars as pl
import httpx as _httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from models.scorer import ThamanScorer
from api.spatial import SpatialLookup
from api.models import (
    PredictRequest, PredictResponse, FeatureDriver,
    BOROUGH_NAMES, BLDGCLASS_DESCRIPTIONS,
)


# ── Global state ──────────────────────────────────────────────────────
_scorer:      ThamanScorer  | None = None
_spatial:     SpatialLookup | None = None
_nearby_df                         = None   # pl.DataFrame — runtime sales lookup
_nearby_tree                       = None   # scipy cKDTree for nearby queries
_nta_geojson_cache: str | None     = None   # pre-built NTA choropleth GeoJSON

# ── Comps cache (zip_code → (result_dict, unix_timestamp)) ────────────
_comps_cache: dict[str, tuple[dict, float]] = {}
_COMPS_TTL   = 86_400   # 24 h

_NEARBY_COLS = [
    "latitude", "longitude", "sale_price", "address",
    "bldgclass", "gross_square_feet", "building_age", "sale_date",
]


def _build_nta_geojson() -> str:
    """Build NTA boundary GeoJSON enriched with per-NTA statistics from features CSV."""
    geojson_path = os.path.join(BASE, "data", "raw", "nta_boundaries.geojson")
    if not os.path.exists(geojson_path):
        return ""

    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    # Aggregate per-NTA stats from features files
    stats: dict[str, dict] = {}
    for csv_path, extra_cols in [
        (os.path.join(BASE, "data", "processed", "features.csv"),
         ["median_income_nta", "crime_rate_nta", "noise_density_nta",
          "livability_complaint_rate", "price_appreciation"]),
        (os.path.join(BASE, "data", "processed", "features_v3.csv"),
         ["tree_count_200m", "pm25_mean"]),
    ]:
        if not os.path.exists(csv_path):
            continue
        cols_needed = ["ntacode"] + [c for c in extra_cols]
        try:
            df = pl.read_csv(csv_path, columns=[c for c in cols_needed
                             if c in pl.read_csv(csv_path, n_rows=0).columns])
            for col in [c for c in extra_cols if c in df.columns]:
                agg = (df.group_by("ntacode")
                         .agg(pl.col(col).cast(pl.Float64, strict=False).median().alias(col)))
                for row in agg.iter_rows(named=True):
                    code = row["ntacode"]
                    if code not in stats:
                        stats[code] = {}
                    if row[col] is not None:
                        stats[code][col] = round(float(row[col]), 4)
        except Exception:
            pass

    # Merge stats into GeoJSON feature properties
    for feat in geojson.get("features", []):
        code = feat.get("properties", {}).get("ntacode", "")
        if code in stats:
            feat["properties"].update(stats[code])

    return json.dumps(geojson)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + spatial data once at startup."""
    global _scorer, _spatial, _nearby_df, _nearby_tree, _nta_geojson_cache
    print("=" * 60)
    print("THAMAN API — Starting up (v2)")
    print("=" * 60)
    _scorer  = ThamanScorer()
    _spatial = SpatialLookup()

    # Load lightweight nearby-sales index (subset of features.csv)
    try:
        from scipy.spatial import cKDTree as _KDTree
        _nearby_path = os.path.join(BASE, "data", "processed", "features.csv")
        available    = [c for c in _NEARBY_COLS
                        if c in pl.read_csv(_nearby_path, n_rows=0).columns]
        _nearby_df   = (
            pl.read_csv(_nearby_path, columns=available)
            .drop_nulls(subset=["latitude", "longitude", "sale_price"])
        )
        _nearby_tree = _KDTree(_nearby_df.select(["latitude", "longitude"]).to_numpy())
        print(f"  Nearby index: {len(_nearby_df):,} sales loaded")
    except Exception as e:
        print(f"  [nearby] Could not load nearby index: {e}")

    # Build NTA choropleth GeoJSON cache
    try:
        _nta_geojson_cache = _build_nta_geojson()
        if _nta_geojson_cache:
            print(f"  NTA layer: GeoJSON built ({len(_nta_geojson_cache)//1024} KB)")
        else:
            print("  NTA layer: nta_boundaries.geojson not found — /layers/nta unavailable")
    except Exception as e:
        print(f"  NTA layer: build failed — {e}")

    print("=" * 60)
    print("THAMAN API — Ready at http://localhost:8000")
    print("Docs:        http://localhost:8000/docs")
    print("=" * 60)
    yield
    print("THAMAN API — Shutting down")


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="THAMAN Property Valuation API",
    description=(
        "AI-powered NYC property price estimator. "
        "Combines structural attributes + Quality-of-Life indicators "
        "using GIS spatial lookups + XGBoost+LightGBM+CatBoost Stack (R²=0.651, MedAPE=20.80%, "
        "81 features, spatial CV validated, luxury sub-model for Manhattan $3M+)."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1_000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend at /ui  (index.html auto-served)
_FRONTEND_DIR = os.path.join(BASE, "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


# ── Urban gravity centres (same as training) ──────────────────────────
_GRAVITY = {
    "midtown_manhattan":  (40.7549, -73.9840),
    "downtown_manhattan": (40.7074, -74.0113),
    "downtown_brooklyn":  (40.6928, -73.9903),
    "long_island_city":   (40.7447, -73.9485),
}

# Distance columns that get log-transformed (must match train_v2.py order)
_DIST_COLS = [
    "dist_subway_m", "dist_school_m", "dist_park_m", "dist_hospital_m",
    "dist_bus_m", "dist_waterfront_m", "dist_bike_lane_m",
    "dist_elem_school_m", "dist_express_subway_m",
]


# ── Helper ────────────────────────────────────────────────────────────

def _build_feature_row(req: PredictRequest, spatial_feats: dict) -> dict:
    """
    Merge spatial auto-features with user-provided property attributes.
    Returns a flat dict matching feature_names from meta.json (v2 — 71 features).
    """
    feat = dict(spatial_feats)          # start with spatial features
    bc   = req.bldgclass.upper().strip()

    # ── Core property attributes ──────────────────────────────────────
    feat.update({
        "latitude":          req.latitude,
        "longitude":         req.longitude,
        "borough":           req.borough,
        "building_age":      req.building_age,
        "numfloors":         req.numfloors,
        "gross_square_feet": req.gross_square_feet,
        "land_square_feet":  (req.land_square_feet if req.land_square_feet is not None
                              else req.gross_square_feet * 0.9),
        "residential_units": req.residential_units,
    })

    # ── Building type flags (infer from bldgclass if not provided) ────
    elevator_classes = {
        "D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","DB",
        "R0","R1","R2","R3","R4","RR","RG","RH","RP","RS","RT","RW",
        "H1","H2","H3","H4","H7","H9","HB","HR",
    }
    elevator_prefixes = {"D", "H", "R"}
    feat["has_elevator"]   = (req.has_elevator   if req.has_elevator   is not None
                              else int(bc in elevator_classes or
                                       (len(bc) >= 1 and bc[0] in elevator_prefixes)))
    feat["is_condo"]       = (req.is_condo       if req.is_condo       is not None
                              else int(bc.startswith("R")))
    feat["is_multifamily"] = (req.is_multifamily if req.is_multifamily is not None
                              else int(bc.startswith("D")))
    feat["is_single_fam"]  = (req.is_single_fam  if req.is_single_fam  is not None
                              else int(bc.startswith("A")))
    feat["is_mixed_use"]   = (req.is_mixed_use   if req.is_mixed_use   is not None
                              else int(bc.startswith("S")))

    # ── FAR / Zoning ──────────────────────────────────────────────────
    if req.builtfar      is not None: feat["builtfar"]      = req.builtfar
    if req.residfar      is not None: feat["residfar"]      = req.residfar
    if req.commfar       is not None: feat["commfar"]       = req.commfar
    if req.facilfar      is not None: feat["facilfar"]      = req.facilfar
    if req.maxallwfar    is not None: feat["maxallwfar"]    = req.maxallwfar
    if req.far_utilization is not None:
        feat["far_utilization"] = req.far_utilization
    else:
        maf = float(feat.get("maxallwfar", 0) or 0)
        blt = float(feat.get("builtfar",   0) or 0)
        feat["far_utilization"] = min(blt / maf, 5.0) if maf > 0 else 0.0

    # ── ACRIS prior-sale data ─────────────────────────────────────────
    feat["prior_sale_price"]       = req.prior_sale_price
    feat["price_appreciation"]     = req.price_appreciation
    feat["years_since_prior_sale"] = req.years_since_prior_sale
    feat["has_prior_sale"]         = req.has_prior_sale or 0
    feat["is_flip"]                = req.is_flip or 0

    # ── Renovation ────────────────────────────────────────────────────
    feat["renovated_since_2018"]   = req.renovated_since_2018   or 0
    feat["years_since_renovation"] = req.years_since_renovation or 0.0

    # ── Time features — v2 uses cyclical month encoding ───────────────
    now   = datetime.datetime.now()
    year  = req.sale_year  or now.year
    month = req.sale_month or now.month
    feat["sale_year"]      = year
    feat["sale_month_sin"] = float(np.sin(2.0 * np.pi * month / 12.0))
    feat["sale_month_cos"] = float(np.cos(2.0 * np.pi * month / 12.0))
    # Note: raw sale_month is NOT in v2 feature_names — do not add it

    # ── v2 NEW FEATURES ───────────────────────────────────────────────

    # 1. Log-transformed distances (9 features)
    for dist_col in _DIST_COLS:
        val = float(feat.get(dist_col, 0) or 0)
        feat[f"log_{dist_col}"] = float(np.log1p(max(val, 0.0)))

    # 2. Urban gravity distances (4 features)
    lat, lon = req.latitude, req.longitude
    for name, (clat, clon) in _GRAVITY.items():
        feat[f"dist_{name}_m"] = float(
            np.sqrt((lat - clat) ** 2 + (lon - clon) ** 2) * 111_000.0
        )

    # 3. Manhattan flag + crime interactions
    is_manhattan = int(req.borough == 1)
    crime_rate   = float(feat.get("crime_rate_nta", 0.0) or 0.0)
    feat["is_manhattan"]          = is_manhattan
    feat["crime_x_manhattan"]     = crime_rate * is_manhattan
    feat["crime_x_non_manhattan"] = crime_rate * (1 - is_manhattan)

    # 4. Walk-score proxy (uses saved MinMaxScaler params from meta.json)
    eps = 1e-9
    raw_comps = {
        "transit":   1.0 / max(float(feat.get("dist_subway_m",    500) or 500), eps),
        "bus":       1.0 / max(float(feat.get("dist_bus_m",       200) or 200), eps),
        "amenities": float(feat.get("poi_count_500m", 50) or 0),
        "bike":      1.0 / max(float(feat.get("dist_bike_lane_m", 300) or 300), eps),
        "park":      1.0 / max(float(feat.get("dist_park_m",      200) or 200), eps),
    }
    ws_params = _scorer.meta.get("walk_score_scaler", {})
    walk_weights = {"transit": 0.35, "bus": 0.15, "amenities": 0.30, "bike": 0.10, "park": 0.10}
    walk_score   = 0.0
    for comp, wt in walk_weights.items():
        v = raw_comps[comp]
        if comp in ws_params:
            d_min  = ws_params[comp]["data_min"]
            d_max  = ws_params[comp]["data_max"]
            scale  = ws_params[comp]["scale"]
            normed = float(np.clip((v - d_min) * scale, 0.0, 1.0))
        else:
            # Fallback if scaler not yet saved (should not happen after train_stack)
            normed = float(np.clip(v / max(abs(v) * 2 + eps, eps), 0.0, 1.0))
        walk_score += wt * normed
    feat["walk_score_proxy"] = float(np.clip(walk_score * 100.0, 0.0, 100.0))

    # 5. Target-encoded bldgclass and borough×bldgclass
    bldg_means  = _scorer.meta["bldgclass_means"]
    bb_means    = _scorer.meta["borough_bldg_means"]
    global_mean = _scorer.meta["global_mean_log"]
    bb_key      = f"{req.borough}_{bc[0]}" if bc else f"{req.borough}_"

    feat["bldgclass_encoded"]    = float(bldg_means.get(bc, global_mean))
    feat["borough_bldg_encoded"] = float(bb_means.get(bb_key, global_mean))

    return feat


def _count_comparables(lat: float, lon: float, radius_m: int = 800) -> int:
    """
    Count training-set sales within radius_m metres using the already-loaded
    cKDTree.  Returns 0 if the index is not available. Target latency < 5ms.
    """
    if _nearby_df is None or _nearby_tree is None:
        return 0
    radius_deg = radius_m / 111_000.0
    idxs = _nearby_tree.query_ball_point([lat, lon], radius_deg)
    return len(idxs)


def _build_qc_flags(seg_medape: float, comps: int, price: float, borough: int) -> list:
    """Produce list of AVM QC flag strings for the given prediction context."""
    flags = []
    if comps < 5:                          flags.append("SPARSE_MARKET")
    if price > 3_000_000:                  flags.append("LUXURY_SEGMENT")
    if seg_medape > 30.0:                  flags.append("HIGH_UNCERTAINTY")
    if borough == 1 and price > 1_000_000: flags.append("METRO_CORE")
    return flags


def _get_shap_drivers(feat_dict: dict) -> list[FeatureDriver]:
    """Run SHAP explanation and return top 10 feature drivers."""
    row_dict  = {}
    acris_cols = set(_scorer.acris_medians.keys())

    for k in _scorer.feature_names:
        val = feat_dict.get(k, 0)
        if k in acris_cols and (val is None or (isinstance(val, float) and np.isnan(val))):
            row_dict[k] = None          # scorer will fill with training median
        elif val is None:
            row_dict[k] = None
        else:
            row_dict[k] = val

    df_row = pl.from_dicts([row_dict])

    try:
        shap_df   = _scorer.explain(df_row)
        row_vals  = {col: float(shap_df[col][0]) for col in shap_df.columns}
        top_feats = sorted(row_vals, key=lambda k: abs(row_vals[k]), reverse=True)[:10]
        drivers = []
        for fname in top_feats:
            impact  = row_vals[fname]
            raw_val = feat_dict.get(fname, 0)

            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                feat_val = 0.0
            else:
                try:
                    feat_val = float(raw_val)
                except (ValueError, TypeError):
                    feat_val = 0.0

            drivers.append(FeatureDriver(
                feature=fname,
                value=feat_val,
                impact=round(impact, 4),
                direction="positive" if impact > 0 else "negative",
                description=_spatial.get_feature_description(fname),
            ))
        return drivers

    except Exception as e:
        print(f"[SHAP warning] {e}")
        import traceback
        traceback.print_exc()
        return []


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"], include_in_schema=False)
@app.get("/ui", tags=["Info"], include_in_schema=False)
def root():
    """Serve the map UI directly (no redirect chain that breaks HF Spaces iframe)."""
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/api", tags=["Info"])
def api_info():
    """API info and available endpoints."""
    return {
        "name":        "THAMAN Property Valuation API",
        "version":     "2.1.0",
        "description": "AI-powered NYC property price estimator",
        "model":       "XGBoost + LightGBM + CatBoost Stack (71 features, spatial CV validated)",
        "performance": {
            "R2_holdout":   0.6509,
            "MedAPE_pct":   20.29,
            "MAE_usd":      1055713,
            "base_xgb_r2":  0.6537,
            "base_lgb_r2":  0.6511,
        },
        "endpoints": {
            "GET  /api":         "API info (this response)",
            "GET  /health":      "Health check",
            "GET  /bldgclasses": "List valid NYC building class codes",
            "POST /predict":     "Predict price for one property",
            "POST /batch":       "Predict prices for multiple properties",
            "GET  /ui":          "Interactive map (browser)",
        },
        "docs": "http://localhost:8000/docs",
        "ui":   "http://localhost:8000/ui",
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check — confirms model and spatial data are loaded."""
    return {
        "status":         "ok" if (_scorer and _spatial) else "loading",
        "model_loaded":   _scorer  is not None,
        "spatial_loaded": _spatial is not None,
        "timestamp":      datetime.datetime.now().isoformat(),
    }


@app.get("/bldgclasses", tags=["Reference"])
def get_bldgclasses():
    """Return all valid NYC building class codes that the model recognises."""
    if not _scorer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    known_classes = sorted(_scorer.bldgclass_means.keys())
    return {
        "total":       len(known_classes),
        "bldgclasses": known_classes,
        "common_examples": BLDGCLASS_DESCRIPTIONS,
        "note": (
            "Pass bldgclass to /predict. "
            "Unknown classes fall back to global mean. "
            "Note: D-class codes (D1–D4) represent entire elevator BUILDINGS, "
            "not individual units. For unit-level condos use R1."
        ),
    }


@app.post("/predict", response_model=None, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict the market value of a NYC property.

    **Required fields**: latitude, longitude, gross_square_feet,
    building_age, bldgclass, borough, numfloors, residential_units.

    All spatial features (subway distance, crime rate, school district, etc.)
    are automatically computed from the lat/lng coordinates.

    Returns predicted price with ±20.29% confidence interval and top SHAP drivers.
    """
    if not _scorer or not _spatial:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for startup.")

    # 1. Auto-compute spatial features from lat/lng
    try:
        spatial_feats = _spatial.lookup(req.latitude, req.longitude)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spatial lookup failed: {e}")

    # 2. Build full 70-feature row
    feat_dict = _build_feature_row(req, spatial_feats)

    # 3. Run XGBoost v2 prediction
    try:
        result = _scorer.predict_single(**feat_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # 3b. AVM QC: comparable count (hit rate) + quality flags
    comp_count  = _count_comparables(req.latitude, req.longitude)
    seg_medape  = result.get("segment_medape_pct", result["medape_test_pct"])
    qc_flags    = _build_qc_flags(seg_medape, comp_count,
                                   result["predicted_price"], req.borough)
    avm_qc_dict = {
        "confidence_score":    result.get("confidence_score", 0),
        "confidence_grade":    result.get("confidence_grade", "D"),
        "segment_medape_pct":  seg_medape,
        "comparables_found":   comp_count,
        "comparables_radius_m": 800,
        "sparse_market":       comp_count < 5,
        "qc_flags":            qc_flags,
    }

    # 4. SHAP explanations
    drivers = _get_shap_drivers(feat_dict)

    # 5. Build response
    bc_desc = BLDGCLASS_DESCRIPTIONS.get(
        req.bldgclass.upper().strip(),
        f"Building class {req.bldgclass.upper()}"
    )

    def _safe_int(v, default=0):
        try:
            return int(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        except (ValueError, TypeError):
            return default

    def _safe_round(v, n=0, default=0.0):
        try:
            return round(float(v), n) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        except (ValueError, TypeError):
            return default

    # Spatial summary (human-readable subset)
    spatial_summary = {
        "dist_subway_m":              _safe_round(spatial_feats.get("dist_subway_m")),
        "dist_bus_m":                 _safe_round(spatial_feats.get("dist_bus_m")),
        "dist_park_m":                _safe_round(spatial_feats.get("dist_park_m")),
        "dist_school_m":              _safe_round(spatial_feats.get("dist_school_m")),
        "dist_hospital_m":            _safe_round(spatial_feats.get("dist_hospital_m")),
        "nearest_station_is_express": _safe_int(spatial_feats.get("nearest_station_is_express")),
        "airbnb_count_500m":          _safe_int(spatial_feats.get("airbnb_count_500m")),
        "poi_count_500m":             _safe_round(spatial_feats.get("poi_count_500m")),
        "crime_rate_nta":             _safe_round(spatial_feats.get("crime_rate_nta"), 1),
        "noise_density_nta":          _safe_round(spatial_feats.get("noise_density_nta"), 1),
        "median_income_nta":          _safe_round(spatial_feats.get("median_income_nta")),
        "school_district":            _safe_int(spatial_feats.get("school_district")),
        "district_avg_score":         _safe_round(spatial_feats.get("district_avg_score"), 1),
        "mortgage_rate_30yr":         spatial_feats.get("mortgage_rate_30yr", 0.0),
    }

    return {
        "predicted_price":    result["predicted_price"],
        "confidence_low":     result["confidence_low"],
        "confidence_high":    result["confidence_high"],
        "confidence_note":    f"±{round(seg_medape, 1)}% segment MedAPE confidence interval",
        "model":              result["model"],
        "r2_test":            result["r2_test"],
        "medape_pct":         result["medape_test_pct"],
        "borough_name":       BOROUGH_NAMES.get(req.borough, str(req.borough)),
        "bldgclass_description": bc_desc,
        "spatial_features":   spatial_summary,
        "top_drivers":        [d.model_dump() for d in drivers],
        "avm_qc":             avm_qc_dict,
    }


@app.post("/batch", tags=["Prediction"])
def predict_batch(requests: list[PredictRequest]):
    """
    Predict prices for multiple properties at once (max 50).
    Returns a list of prediction results in the same order as input.
    """
    if not _scorer or not _spatial:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Batch size limit is 50 properties.")

    results = []
    for i, req in enumerate(requests):
        try:
            spatial_feats = _spatial.lookup(req.latitude, req.longitude)
            feat_dict     = _build_feature_row(req, spatial_feats)
            result        = _scorer.predict_single(**feat_dict)
            results.append({
                "index":           i,
                "predicted_price": result["predicted_price"],
                "confidence_low":  result["confidence_low"],
                "confidence_high": result["confidence_high"],
                "borough_name":    BOROUGH_NAMES.get(req.borough, str(req.borough)),
                "bldgclass":       req.bldgclass,
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return {"count": len(results), "results": results}


@app.get("/nearby", tags=["Reference"])
def nearby_sales(lat: float, lon: float, radius_m: int = 800, limit: int = 8):
    """
    Return up to `limit` recent property sales within `radius_m` metres of (lat, lon).
    Falls back to the nearest `limit` sales if none found within radius.
    """
    if _nearby_df is None or _nearby_tree is None:
        raise HTTPException(status_code=503, detail="Nearby index not loaded.")
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Invalid coordinates.")

    limit = min(max(1, limit), 20)

    # Search within radius (degree approximation: 1° ≈ 111 km)
    radius_deg = radius_m / 111_000.0
    idxs = _nearby_tree.query_ball_point([lat, lon], radius_deg)

    if not idxs:
        # Fallback: return nearest regardless of distance
        k = min(limit, len(_nearby_df))
        _, idxs = _nearby_tree.query([lat, lon], k=k)
        idxs = idxs.tolist() if hasattr(idxs, 'tolist') else list(idxs)

    subset = (
        _nearby_df[idxs]
        .with_columns(
            (((pl.col("latitude") - lat) ** 2 + (pl.col("longitude") - lon) ** 2) ** 0.5)
            .alias("_dist_deg")
        )
        .with_columns(
            (pl.col("_dist_deg") * 111_000).round(0).cast(pl.Int64).alias("distance_m")
        )
        .sort("_dist_deg")
        .head(limit)
    )

    nearby = []
    for row in subset.iter_rows(named=True):
        nearby.append({
            "address":          str(row.get("address") or "")[:80],
            "sale_price":       int(row["sale_price"]),
            "bldgclass":        str(row.get("bldgclass") or ""),
            "gross_square_feet":int(row.get("gross_square_feet") or 0),
            "building_age":     int(row.get("building_age") or 0),
            "sale_date":        str(row.get("sale_date") or "")[:10],
            "distance_m":       int(row["distance_m"]),
            "latitude":         float(row["latitude"]),
            "longitude":        float(row["longitude"]),
        })

    return {"count": len(nearby), "nearby": nearby}


@app.get("/market/comps", tags=["Reference"])
async def market_comps(lat: float, lon: float):
    """
    Fetch recent comparable sales from NYC DOF (official records) for the
    zip code nearest the given coordinates.
    - Nominatim + DOF calls run concurrently via asyncio.gather
    - Results cached 24 h per zip code (in-process dict)
    """
    import statistics

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Invalid coordinates.")

    # ── Step 1: reverse-geocode (async) ──────────────────────────────
    zip_code = ""
    try:
        async with _httpx.AsyncClient() as client:
            geo_r = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json"},
                headers={"User-Agent": "THAMAN-BSc-PropTech/1.0"},
                timeout=6,
            )
        zip_code = geo_r.json().get("address", {}).get("postcode", "")
    except Exception:
        pass

    if not zip_code:
        return {"available": False, "reason": "Could not determine zip code for this location."}

    # ── Step 2: cache hit? ────────────────────────────────────────────
    cached = _comps_cache.get(zip_code)
    if cached and (time.time() - cached[1]) < _COMPS_TTL:
        return cached[0]

    # ── Step 3: query NYC DOF — comps (18 mo) + trend (24 mo) in parallel ──
    cutoff_comps = (datetime.datetime.now() - datetime.timedelta(days=548)).strftime("%Y-%m-%d")
    cutoff_trend = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    rows: list = []
    trend_rows: list = []
    try:
        async with _httpx.AsyncClient() as client:
            comps_task = client.get(
                "https://data.cityofnewyork.us/resource/usep-8jbt.json",
                params={
                    "$where": (f"zip_code='{zip_code}' AND sale_date >= '{cutoff_comps}'"
                               " AND sale_price > '100000'"),
                    "$order": "sale_date DESC",
                    "$limit": 20,
                    "$select": ("address,zip_code,neighborhood,sale_price,sale_date,"
                                "building_class_at_present,gross_square_feet,"
                                "year_built,residential_units"),
                },
                headers={"User-Agent": "THAMAN-BSc-PropTech/1.0"},
                timeout=8,
            )
            trend_task = client.get(
                "https://data.cityofnewyork.us/resource/usep-8jbt.json",
                params={
                    "$where": (f"zip_code='{zip_code}' AND sale_date >= '{cutoff_trend}'"
                               " AND sale_price > '100000' AND gross_square_feet > '0'"),
                    "$order": "sale_date ASC",
                    "$limit": 300,
                    "$select": "sale_price,sale_date,gross_square_feet",
                },
                headers={"User-Agent": "THAMAN-BSc-PropTech/1.0"},
                timeout=8,
            )
            comps_r, trend_r = await asyncio.gather(comps_task, trend_task, return_exceptions=True)
        rows       = comps_r.json() if not isinstance(comps_r, Exception) and comps_r.is_success else []
        trend_rows = trend_r.json() if not isinstance(trend_r, Exception) and trend_r.is_success else []
    except Exception:
        pass

    if not rows:
        return {"available": False, "reason": f"No recent sales found in zip code {zip_code}."}

    # ── Step 4: build comps + summary ────────────────────────────────
    comps: list[dict] = []
    prices, psf_list = [], []
    for row in rows[:5]:
        price = int(row.get("sale_price") or 0)
        sqft  = int(row.get("gross_square_feet") or 0)
        psf   = round(price / sqft, 0) if sqft > 0 else None
        prices.append(price)
        if psf:
            psf_list.append(psf)
        comps.append({
            "address":      row.get("address", ""),
            "neighborhood": row.get("neighborhood", ""),
            "sale_price":   price,
            "sale_date":    str(row.get("sale_date", ""))[:10],
            "bldgclass":    row.get("building_class_at_present", ""),
            "sqft":         sqft or None,
            "psf":          psf,
            "year_built":   row.get("year_built"),
        })

    # ── Step 4b: build monthly price trend ───────────────────────────
    monthly: dict[str, list] = {}
    for row in trend_rows:
        d = str(row.get("sale_date", ""))[:7]   # "YYYY-MM"
        if not d or len(d) < 7:
            continue
        price = int(row.get("sale_price") or 0)
        if price > 0:
            monthly.setdefault(d, []).append(price)
    trend = [
        {"month": m, "count": len(ps), "median": int(statistics.median(ps))}
        for m, ps in sorted(monthly.items())
        if len(ps) >= 2
    ]

    result = {
        "available": True,
        "summary": {
            "zip_code":     zip_code,
            "comp_count":   len(comps),
            "median_price": int(statistics.median(prices)) if prices else None,
            "median_psf":   int(statistics.median(psf_list)) if psf_list else None,
            "source":       "NYC Dept of Finance — Official Property Sales Records",
            "source_url":   "https://data.cityofnewyork.us/d/usep-8jbt",
            "period":       f"Last 18 months in zip {zip_code}",
        },
        "comps": comps,
        "trend": trend,
    }

    # ── Step 5: cache and return ──────────────────────────────────────
    _comps_cache[zip_code] = (result, time.time())
    return result


@app.get("/layers/nta", tags=["Reference"])
def nta_layer():
    """Return NTA boundary GeoJSON enriched with per-NTA statistics for map choropleth layers."""
    if not _nta_geojson_cache:
        raise HTTPException(status_code=503, detail="NTA layer not available.")
    return Response(content=_nta_geojson_cache, media_type="application/json")
