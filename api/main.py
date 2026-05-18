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
from api.spatial import SpatialLookup, RiyadhSpatialLookup
from api.models import (
    PredictRequest, PredictResponse, FeatureDriver,
    BOROUGH_NAMES, BLDGCLASS_DESCRIPTIONS,
    RiyadhPredictRequest, RiyadhPredictResponse,
)


# ── Global state ──────────────────────────────────────────────────────
_scorer:      ThamanScorer  | None = None
_spatial:     SpatialLookup | None = None
_nearby_df                         = None   # pl.DataFrame — runtime sales lookup
_nearby_tree                       = None   # scipy cKDTree for nearby queries
_nta_geojson_cache: str | None     = None   # pre-built NTA choropleth GeoJSON
_riyadh_spatial: RiyadhSpatialLookup | None = None
_district_geojson_cache: str | None = None  # pre-built Riyadh district GeoJSON
_mta_tree                          = None   # KDTree for MTA station nearest-neighbor
_mta_feats: dict                   = {}     # arrays: is_cbd, route_count, is_ada

# ── Pre-baked sales tile cache (5km × 5km grid, built at startup) ─────
_TILE_DEG   = 0.05                          # ~5.5 km per tile
_NYC_MIN_LAT, _NYC_MIN_LON = 40.45, -74.30
_sales_tiles: dict = {}                     # (tx,ty) → list[dict]

# ── Comps cache (zip_code → (result_dict, unix_timestamp)) ────────────
_comps_cache: dict[str, tuple[dict, float]] = {}
_COMPS_TTL   = 86_400   # 24 h

_NEARBY_COLS = [
    "latitude", "longitude", "sale_price", "address",
    "bldgclass", "gross_square_feet", "building_age", "sale_date",
    "ntacode",   # needed for NTA lookup at inference
    "zip_code",  # needed for v11 HPD/DOB ZIP-level feature lookup
]


def _build_nta_geojson() -> str:
    """Build NTA boundary GeoJSON enriched with per-NTA statistics from features CSV.
    Prefers simplified geometry (436 KB) over raw (4.4 MB) for frontend performance."""
    # Prefer simplified version (10× smaller, visually identical at choropleth zoom)
    simplified_path = os.path.join(BASE, "data", "processed", "nta_simplified.geojson")
    raw_path        = os.path.join(BASE, "data", "raw",       "nta_boundaries.geojson")
    geojson_path    = simplified_path if os.path.exists(simplified_path) else raw_path
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
         ["tree_count_200m", "pm25_mean", "hpd_viol_rate_nta"]),
        (os.path.join(BASE, "data", "processed", "features_v5.csv"),
         ["rat_density_nta", "heat_density_nta", "hpd_viol_rate_nta",
          "livability_complaint_rate", "no2_mean",
          "dist_subway_m", "building_age", "airbnb_count_500m", "population_2020",
          "poi_restaurant_500m", "poi_cafe_500m", "poi_bar_500m",
          "poi_grocery_500m", "poi_gym_500m"]),
    ]:
        if not os.path.exists(csv_path):
            continue
        avail = pl.read_csv(csv_path, n_rows=0).columns
        cols_needed = ["ntacode"] + [c for c in extra_cols if c in avail]
        # also grab price/sqft source columns if available
        for _c in ["sale_price", "gross_square_feet", "poi_cafe_500m", "poi_bar_500m"]:
            if _c in avail and _c not in cols_needed:
                cols_needed.append(_c)
        try:
            df = pl.read_csv(csv_path, columns=cols_needed)
            # Derived columns
            if "sale_price" in df.columns and "gross_square_feet" in df.columns:
                df = df.with_columns(
                    (pl.col("sale_price").cast(pl.Float64, strict=False) /
                     pl.col("gross_square_feet").cast(pl.Float64, strict=False).clip(1))
                    .alias("price_psf")
                )
            if "poi_cafe_500m" in df.columns and "poi_bar_500m" in df.columns:
                df = df.with_columns(
                    (pl.col("poi_cafe_500m").cast(pl.Float64, strict=False) +
                     pl.col("poi_bar_500m").cast(pl.Float64, strict=False))
                    .alias("poi_nightlife_500m")
                )
            all_cols = [c for c in extra_cols + ["price_psf", "poi_nightlife_500m"] if c in df.columns]
            for col in all_cols:
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
    # NYC Open Data NTA 2020 uses "nta2020" field; older exports use "ntacode"
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        code = props.get("ntacode") or props.get("nta2020") or ""
        # Normalise: add "ntacode" key so frontend JS can reference it uniformly
        props["ntacode"] = code
        if code in stats:
            props.update(stats[code])

    return json.dumps(geojson)


def _build_sales_tiles():
    """Pre-bake sales into a 5.5-km tile grid for O(1) map queries."""
    global _sales_tiles
    if _nearby_df is None:
        return
    cols = [c for c in ["latitude","longitude","sale_price","address",
                         "bldgclass","gross_square_feet","sale_date"]
            if c in _nearby_df.columns]
    df = (
        _nearby_df
        .filter(
            (pl.col("latitude")  >= _NYC_MIN_LAT) & (pl.col("latitude")  <= 40.95) &
            (pl.col("longitude") >= _NYC_MIN_LON) & (pl.col("longitude") <= -73.70)
        )
        .select(cols)
        .sort("sale_date", descending=True)
        .with_columns([
            ((pl.col("latitude")  - _NYC_MIN_LAT) / _TILE_DEG).cast(pl.Int32).alias("_ty"),
            ((pl.col("longitude") - _NYC_MIN_LON) / _TILE_DEG).cast(pl.Int32).alias("_tx"),
        ])
    )
    tiles: dict = {}
    for row in df.iter_rows(named=True):
        key = (int(row["_tx"]), int(row["_ty"]))
        bucket = tiles.setdefault(key, [])
        if len(bucket) < 30:
            bucket.append({
                "latitude":          round(float(row["latitude"]),  6),
                "longitude":         round(float(row["longitude"]), 6),
                "sale_price":        int(row.get("sale_price") or 0),
                "address":           str(row.get("address") or "")[:60],
                "bldgclass":         str(row.get("bldgclass") or ""),
                "gross_square_feet": int(row.get("gross_square_feet") or 0),
                "sale_date":         str(row.get("sale_date") or "")[:10],
            })
    _sales_tiles = tiles
    print(f"  Sales tiles: {len(tiles)} tiles pre-baked")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + spatial data once at startup."""
    global _scorer, _spatial, _nearby_df, _nearby_tree, _nta_geojson_cache
    global _mta_tree, _mta_feats, _riyadh_spatial, _district_geojson_cache
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

    # Build MTA station KDTree for v11 transit-quality features
    try:
        from scipy.spatial import KDTree as _SciKDTree
        _stations = _scorer.meta.get("mta_stations", [])
        if _stations:
            _slats = np.array([s["lat"] for s in _stations], dtype=np.float64)
            _slons = np.array([s["lon"] for s in _stations], dtype=np.float64)
            _mta_tree  = _SciKDTree(np.column_stack([_slats, _slons]))
            _mta_feats = {
                "is_cbd":       np.array([s.get("is_cbd",       0) for s in _stations], dtype=np.int32),
                "route_count":  np.array([s.get("route_count",  1) for s in _stations], dtype=np.int32),
                "is_ada":       np.array([s.get("is_ada",       0) for s in _stations], dtype=np.int32),
            }
            print(f"  MTA station index: {len(_stations)} complexes loaded")
        else:
            print("  MTA station index: not in meta.json (train v11 first)")
    except Exception as e:
        print(f"  [MTA] KDTree build failed: {e}")

    # Build NTA choropleth GeoJSON cache
    try:
        _nta_geojson_cache = _build_nta_geojson()
        if _nta_geojson_cache:
            print(f"  NTA layer: GeoJSON built ({len(_nta_geojson_cache)//1024} KB)")
        else:
            print("  NTA layer: nta_boundaries.geojson not found — /layers/nta unavailable")
    except Exception as e:
        print(f"  NTA layer: build failed — {e}")

    # Pre-bake sales tile grid for instant map rendering
    try:
        _build_sales_tiles()
    except Exception as e:
        print(f"  Sales tiles: build failed — {e}")

    # Riyadh spatial lookup + district choropleth
    try:
        _riyadh_spatial = RiyadhSpatialLookup()
        _district_geojson_cache = _build_district_geojson(_riyadh_spatial)
        if _district_geojson_cache:
            print(f"  Riyadh district layer: {len(_district_geojson_cache)//1024} KB built")
    except Exception as e:
        print(f"  Riyadh spatial: init failed — {e}")

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
        "using GIS spatial lookups + XGBoost+LightGBM+CatBoost Stack (R²=0.646, MedAPE=20.16%, "
        "94 features, spatial CV validated, luxury sub-model for Manhattan $3M+)."
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


def _lookup_nta(lat: float, lon: float, bldgclass: str) -> dict:
    """
    Resolve lat/lon → ntacode via nearest-neighbor in the training sales index,
    then map ntacode to the 4 NTA model features from meta.json encoding maps.
    Returns a dict of feature overrides to merge into predict_single kwargs.
    Falls back to global_mean_log / global median when ntacode is unknown.
    """
    if _nearby_df is None or _nearby_tree is None or _scorer is None:
        return {}

    meta      = _scorer.meta
    gml       = meta.get("global_mean_log", 13.5)
    nta_means = meta.get("nta_means", {})
    ntab_means= meta.get("nta_bldg_means", {})
    nta_stats = meta.get("nta_stats", {})

    # Nearest-neighbor lookup → ntacode
    _, idx    = _nearby_tree.query([lat, lon], k=1)
    row       = _nearby_df.row(int(idx), named=True)
    ntacode   = row.get("ntacode") or ""

    # nta_encoded
    nta_enc   = nta_means.get(ntacode, gml)

    # nta_bldg_encoded  (ntacode + "_" + first letter of bldgclass)
    bldg_pfx  = (bldgclass or "")[:1].upper()
    ntab_key  = f"{ntacode}_{bldg_pfx}"
    ntab_enc  = ntab_means.get(ntab_key, nta_enc)   # fall back to nta_enc, then gml

    # nta_sale_count + nta_median_psf
    stats      = nta_stats.get(ntacode, {})
    sale_count = float(stats.get("sale_count", 0))
    med_psf    = float(stats.get("median_psf", 0.0))

    return {
        "nta_encoded":      nta_enc,
        "nta_bldg_encoded": ntab_enc,
        "nta_sale_count":   sale_count,
        "nta_median_psf":   med_psf,
        "_resolved_nta":    ntacode,   # for logging/response only
    }


def _lookup_v11_features(ntacode: str, lat: float, lon: float,
                          zip_code: str = "") -> dict:
    """
    Resolve v11 feature values for inference:
      1. ZIP-level HPD violation severity + DOB permit intensity → from v11_zip_lookup
      2. NTA-level rodent/heat complaint density             → from v11_nta_lookup
      3. MTA station quality (CBD, route count, ADA)         → KDTree on _mta_tree

    Falls back to global medians when lookup keys are missing.
    Called after _lookup_nta() so ntacode is already resolved.
    """
    if _scorer is None:
        return {}

    meta         = _scorer.meta
    zip_lookup   = meta.get("v11_zip_lookup",  {})
    nta_lookup   = meta.get("v11_nta_lookup",  {})
    out: dict    = {}

    # ── 1. ZIP-level HPD + DOB features ──────────────────────────────
    _ZIP_COLS = ["hpd_class_b_viol_zip", "hpd_class_c_viol_zip",
                 "hpd_severity_score_zip", "dob_reno_permit_count",
                 "dob_newbld_permit_count"]
    zip_data = zip_lookup.get(zip_code, {})
    if not zip_data and zip_lookup:
        # Fall back to median across all ZIPs
        zip_data = {}
    for col in _ZIP_COLS:
        if col in zip_data:
            out[col] = float(zip_data[col])
        elif zip_lookup:
            # Global median fallback: median of all zip values for this column
            vals = [float(v[col]) for v in zip_lookup.values() if col in v]
            out[col] = float(np.median(vals)) if vals else 0.0
        else:
            out[col] = 0.0

    # ── 2. NTA-level rodent + heat density ───────────────────────────
    _NTA_COLS = ["rat_density_nta", "heat_density_nta"]
    nta_data  = nta_lookup.get(ntacode, {})
    for col in _NTA_COLS:
        if col in nta_data:
            out[col] = float(nta_data[col])
        elif nta_lookup:
            vals = [float(v[col]) for v in nta_lookup.values() if col in v]
            out[col] = float(np.median(vals)) if vals else 0.0
        else:
            out[col] = 0.0

    # ── 3. MTA nearest-station quality ───────────────────────────────
    if _mta_tree is not None:
        try:
            _, idx = _mta_tree.query([lat, lon], k=1)
            out["nearest_station_is_cbd"]     = int(_mta_feats["is_cbd"][idx])
            out["nearest_station_route_count"] = int(_mta_feats["route_count"][idx])
            out["nearest_station_is_ada"]     = int(_mta_feats["is_ada"][idx])
        except Exception:
            out["nearest_station_is_cbd"]      = 0
            out["nearest_station_route_count"] = 1
            out["nearest_station_is_ada"]      = 0
    else:
        out["nearest_station_is_cbd"]      = 0
        out["nearest_station_route_count"] = 1
        out["nearest_station_is_ada"]      = 0

    return out


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
    last_trained = None
    model_version = None
    if _scorer and _scorer.meta:
        last_trained  = _scorer.meta.get("trained_at") or _scorer.meta.get("timestamp")
        model_version = _scorer.meta.get("stack", {}).get("version")
    return {
        "status":            "ok" if (_scorer and _spatial) else "loading",
        "model_loaded":      _scorer  is not None,
        "spatial_loaded":    _spatial is not None,
        "model_version":     model_version,
        "last_trained_date": last_trained,
        "timestamp":         datetime.datetime.now().isoformat(),
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

    # 2. Build full feature row
    feat_dict = _build_feature_row(req, spatial_feats)

    # 2b. NTA lookup: resolve lat/lon → ntacode → 4 NTA model features
    nta_override  = _lookup_nta(req.latitude, req.longitude, req.bldgclass)
    _resolved_nta = nta_override.pop("_resolved_nta", "")
    feat_dict.update(nta_override)   # overrides defaults (global_mean_log) with real NTA values

    # 2c. v11 features: HPD/DOB by ZIP + rat/heat by NTA + MTA station quality
    _zip_str = ""
    if _nearby_df is not None and _nearby_tree is not None and "zip_code" in _nearby_df.columns:
        try:
            from scipy.spatial import cKDTree as _KDT
            _, _nidx = _nearby_tree.query([req.latitude, req.longitude], k=1)
            _zip_raw = _nearby_df.row(int(_nidx), named=True).get("zip_code")
            _zip_str = str(int(_zip_raw)).zfill(5) if _zip_raw else ""
        except Exception:
            pass
    v11_feats = _lookup_v11_features(_resolved_nta, req.latitude, req.longitude, _zip_str)
    feat_dict.update(v11_feats)

    # 3. Run prediction
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

    # 4. SHAP explanations — prefer scorer's CatBoost SHAP (top_drivers), fall back to XGB explain()
    scorer_drivers = result.get("top_drivers", [])
    if scorer_drivers:
        drivers = [
            FeatureDriver(
                feature=d["feature"],
                value=d["value"],
                impact=d["impact"],
                direction=d["direction"],
                description=d["description"],
            )
            for d in scorer_drivers
        ]
    else:
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
        "nta_code":           _resolved_nta or None,
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
            # NTA lookup — same as /predict
            nta_ov = _lookup_nta(req.latitude, req.longitude, req.bldgclass)
            _b_nta = nta_ov.pop("_resolved_nta", "")
            feat_dict.update(nta_ov)
            # v11 features — HPD/DOB/rat/heat/MTA (same pattern as /predict)
            _b_zip = ""
            if _nearby_df is not None and _nearby_tree is not None and "zip_code" in _nearby_df.columns:
                try:
                    _, _bni = _nearby_tree.query([req.latitude, req.longitude], k=1)
                    _zr = _nearby_df.row(int(_bni), named=True).get("zip_code")
                    _b_zip = str(int(_zr)).zfill(5) if _zr else ""
                except Exception:
                    pass
            feat_dict.update(_lookup_v11_features(_b_nta, req.latitude, req.longitude, _b_zip))
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


# ── Riyadh analytics stats cache ─────────────────────────────────────
_riyadh_stats_cache: dict | None = None

def _build_riyadh_stats() -> dict:
    """Precompute Riyadh analytics stats from features_riyadh.csv."""
    import pandas as _pd, numpy as _np
    _csv = os.path.join(BASE, "data", "processed", "features_riyadh.csv")
    if not os.path.exists(_csv):
        return {}
    df = _pd.read_csv(_csv, encoding="utf-8-sig")

    # Overview
    overview = {
        "total_rows":     int(len(df)),
        "districts":      int(df["district_ar"].nunique()),
        "year_range":     f"{int(df['sale_year'].min())}–{int(df['sale_year'].max())}",
        "median_price_sqm": round(float(df["sale_price_sar_sqm"].median()), 0),
        "model_r2":       0.6747,
        "model_medape":   23.45,
        "oof_r2":         0.9427,
        "oof_medape":     8.28,
    }

    # Price by year
    py = df.groupby("sale_year")["sale_price_sar_sqm"].agg(
        median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
    ).reset_index()
    price_by_year = [
        {"year": int(r["sale_year"]), "median": round(float(r["median"]), 0),
         "q1": round(float(r["q1"]), 0), "q3": round(float(r["q3"]), 0)}
        for _, r in py.iterrows()
    ]

    # Price by property type
    type_map = {"is_apartment": "Apartment", "is_villa": "Villa",
                "is_residential_plot": "Residential Plot", "is_building": "Building"}
    price_by_type = []
    for col, label in type_map.items():
        if col in df.columns:
            sub = df[df[col] == 1]["sale_price_sar_sqm"]
            if len(sub) > 5:
                price_by_type.append({
                    "type": col, "label": label,
                    "median": round(float(sub.median()), 0),
                    "count":  int(len(sub))
                })

    # Top 25 districts by median price (min 10 transactions)
    dg = df.groupby("district_ar")["sale_price_sar_sqm"].agg(median="median", count="count")
    dg = dg[dg["count"] >= 10].sort_values("median", ascending=False).head(25).reset_index()
    top_districts = [
        {"district": str(r["district_ar"]), "median": round(float(r["median"]), 0), "count": int(r["count"])}
        for _, r in dg.iterrows()
    ]

    return {"overview": overview, "price_by_year": price_by_year,
            "price_by_type": price_by_type, "top_districts": top_districts}


@app.get("/riyadh/stats", tags=["Riyadh"])
def riyadh_stats():
    """Precomputed Riyadh analytics: overview KPIs, price by year, type breakdown, top districts."""
    global _riyadh_stats_cache
    if _riyadh_stats_cache is None:
        _riyadh_stats_cache = _build_riyadh_stats()
    return _riyadh_stats_cache


@app.post("/predict/riyadh", response_model=RiyadhPredictResponse, tags=["Prediction"])
def predict_riyadh(req: RiyadhPredictRequest):
    """
    Predict the market value of a **Riyadh** property (SAR/m² + total SAR).

    Uses the XGBoost+LightGBM+CatBoost+Ridge stack trained on Saudi open-data
    district-level quarterly transactions (2018–2025 Q3).

    **Required**: latitude, longitude, property_type, area_sqm.
    Spatial features (metro, bus, commercial, QoL POIs, air quality) are auto-computed
    from the coordinates. Year/quarter default to the current period.
    """
    if not _riyadh_spatial:
        raise HTTPException(status_code=503, detail="Riyadh spatial data not loaded.")
    if not _scorer:
        raise HTTPException(status_code=503, detail="Riyadh model not loaded.")
    if not hasattr(_scorer, "predict_riyadh"):
        raise HTTPException(status_code=503, detail="Riyadh scorer not available.")

    # Default year/quarter to current
    import datetime as _dt
    now = _dt.datetime.now()
    year    = req.year    or now.year
    quarter = req.quarter or ((now.month - 1) // 3 + 1)

    # Build feature row
    try:
        feat_dict = _riyadh_spatial.predict_features(
            lat=req.latitude, lon=req.longitude,
            property_type=req.property_type,
            year=year, quarter=quarter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature build failed: {e}")

    # Predict (SAR/sqm, log-space)
    try:
        result = _scorer.predict_riyadh(**feat_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Riyadh prediction failed: {e}")

    # Find nearest district name for display
    district_ar = None
    if (_riyadh_spatial._district_centroid_tree is not None
            and _riyadh_spatial._district_names):
        _, idx = _riyadh_spatial._district_centroid_tree.query([[req.latitude, req.longitude]], k=1)
        district_ar = _riyadh_spatial._district_names[int(idx[0])]

    psqm  = int(round(result["predicted_price_sqm"]))
    total = int(round(psqm * req.area_sqm))

    # District-adaptive confidence: look up per-district holdout MedAPE
    # Falls back to global model MedAPE (23.45%) if district not in table
    global_medape = result["medape_pct"]
    district_medape_tbl = {}
    if _scorer and hasattr(_scorer, '_riyadh_meta'):
        district_medape_tbl = _scorer._riyadh_meta.get("district_medape", {})
    raw_district_medape = district_medape_tbl.get(district_ar, global_medape) if district_ar else global_medape
    # Cap at 50%, floor at 10% to avoid absurd intervals
    conf_medape = float(max(10.0, min(50.0, raw_district_medape)))
    medape_frac = conf_medape / 100.0

    return RiyadhPredictResponse(
        predicted_price_sqm  = psqm,
        predicted_total_sar  = total,
        confidence_low_sqm   = int(psqm * (1 - medape_frac)),
        confidence_high_sqm  = int(psqm * (1 + medape_frac)),
        confidence_low_sar   = int(total * (1 - medape_frac)),
        confidence_high_sar  = int(total * (1 + medape_frac)),
        area_sqm             = req.area_sqm,
        property_type        = req.property_type,
        district_ar          = district_ar,
        model                = result.get("model", "riyadh_stack_v1"),
        r2_test              = result.get("r2_test", 0.675),
        medape_pct           = round(conf_medape, 2),
        spatial_features     = {k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in feat_dict.items()
                                 if k in ("dist_metro_m", "metro_stations_1km",
                                          "dist_bus_m", "bus_stops_500m",
                                          "commercial_count_1km", "dist_mosque_m",
                                          "dist_mall_m", "dist_school_m",
                                          "dist_hospital_m", "dist_park_m",
                                          "air_quality_score",
                                          "riyadh_connectivity_score")},
        top_drivers          = result.get("top_drivers", []),
    )


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

    limit = min(max(1, limit), 100)

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


@app.get("/sales/tile", tags=["Reference"])
def sales_tile(tx: int, ty: int):
    """
    Return pre-baked sales for a 5.5-km map tile (tx, ty).
    All tiles are computed at startup — response is O(1) dict lookup.
    """
    return {"sales": _sales_tiles.get((tx, ty), [])}


@app.get("/sales/bbox", tags=["Reference"])
def sales_bbox(
    min_lat: float, max_lat: float,
    min_lon: float, max_lon: float,
    limit: int = 200,
):
    """
    Return up to `limit` sales within a lat/lon bounding box, spatially sampled
    to give even coverage across the viewport. Used by the map bubble layer.
    """
    if _nearby_df is None:
        raise HTTPException(status_code=503, detail="Nearby index not loaded.")

    limit = min(max(1, limit), 500)

    subset = _nearby_df.filter(
        (pl.col("latitude")  >= min_lat) & (pl.col("latitude")  <= max_lat) &
        (pl.col("longitude") >= min_lon) & (pl.col("longitude") <= max_lon)
    )

    if len(subset) == 0:
        return {"count": 0, "sales": []}

    # Spatially sample: divide bbox into grid cells, pick one per cell
    if len(subset) > limit:
        grid = max(1, int(limit ** 0.5))          # e.g. limit=200 → 14×14 grid
        lat_step = (max_lat - min_lat) / grid
        lon_step = (max_lon - min_lon) / grid
        subset = (
            subset
            .with_columns([
                ((pl.col("latitude")  - min_lat) / lat_step).cast(pl.Int32).clip(0, grid - 1).alias("_gc"),
                ((pl.col("longitude") - min_lon) / lon_step).cast(pl.Int32).clip(0, grid - 1).alias("_gr"),
            ])
            .with_columns((pl.col("_gc") * grid + pl.col("_gr")).alias("_cell"))
            .sort("sale_date", descending=True)
            .unique(subset=["_cell"], keep="first")
            .drop(["_gc", "_gr", "_cell"])
        )

    _BBOX_COLS = [c for c in ["latitude","longitude","sale_price","address",
                               "bldgclass","gross_square_feet","sale_date"] if c in subset.columns]
    sales = []
    for row in subset.select(_BBOX_COLS).iter_rows(named=True):
        if not row.get("latitude") or not row.get("longitude"):
            continue
        sales.append({
            "latitude":         float(row["latitude"]),
            "longitude":        float(row["longitude"]),
            "sale_price":       int(row.get("sale_price") or 0),
            "address":          str(row.get("address") or ""),
            "bldgclass":        str(row.get("bldgclass") or ""),
            "gross_square_feet":int(row.get("gross_square_feet") or 0),
            "sale_date":        str(row.get("sale_date") or "")[:10],
        })

    return {"count": len(sales), "sales": sales}


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


def _build_district_geojson(riyadh_spatial) -> str:
    """
    Build Riyadh district choropleth GeoJSON.
    Priority: polygon file (data/processed/riyadh_district_polygons.geojson) → centroid points fallback.
    """
    import json as _json

    # ── Polygon path (preferred) ──────────────────────────────────────
    poly_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed",
        "riyadh_district_polygons.geojson"
    )
    if os.path.exists(poly_path):
        with open(poly_path, encoding="utf-8") as f:
            return f.read()

    # ── Fallback: centroid points from spatial lookup ─────────────────
    district_stats = riyadh_spatial.get_district_stats() if riyadh_spatial else {}
    if not district_stats:
        return ""

    METRIC_COLS = [
        "dist_metro_m", "metro_stations_1km",
        "commercial_count_1km", "hypermarket_count_1km",
        "bus_stops_500m", "no2_nearest_mean", "pm10_nearest_mean",
        "air_quality_score", "rei_residential_qtr_idx",
        "district_median_price_sqm", "district_price_trend_slope",
        "district_commercial_mix", "riyadh_connectivity_score",
    ]

    features = []
    for district_ar, stats in district_stats.items():
        lat = stats.get("district_lat")
        lon = stats.get("district_lon")
        if lat is None or lon is None:
            continue
        props = {"district_ar": district_ar}
        for col in METRIC_COLS:
            v = stats.get(col)
            if v is not None and not (isinstance(v, float) and v != v):
                props[col] = round(float(v), 4)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": props,
        })

    return _json.dumps({"type": "FeatureCollection", "features": features})


@app.get("/layers/district", tags=["Reference"])
def district_layer():
    """
    Return Riyadh district centroid GeoJSON enriched with per-district statistics
    (transit access, commercial density, air quality, price index) for map choropleth.
    """
    if not _district_geojson_cache:
        raise HTTPException(
            status_code=503,
            detail="District layer not available. Run scripts/riyadh_feature_engineering.py first.",
        )
    return Response(content=_district_geojson_cache, media_type="application/json")


@app.get("/layers/listings", tags=["Riyadh"])
def get_listings_layer():
    """Return scraped Haraj listings as GeoJSON for map display."""
    import glob, csv
    from pathlib import Path

    features = []
    pattern = str(Path(BASE) / "data" / "raw" / "saudi_listings_haraj_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return {"type": "FeatureCollection", "features": []}

    latest = files[-1]
    with open(latest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["lat"]); lon = float(row["lon"])
                psqm = float(row["price_per_sqm"])
                price = float(row["price_sar"])
                area = float(row["area_sqm"])
                if not (23.5 <= lat <= 26.0 and 45.5 <= lon <= 48.0):
                    continue
                if psqm <= 0:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "listing_id": row.get("listing_id", ""),
                        "district": row.get("district", ""),
                        "type_en": row.get("property_type_en", ""),
                        "type_ar": row.get("property_type_ar", ""),
                        "price_sar": round(price),
                        "area_sqm": round(area, 1),
                        "price_per_sqm": round(psqm),
                        "bedrooms": row.get("bedrooms", ""),
                        "url": row.get("url", ""),
                    }
                })
            except (ValueError, KeyError):
                continue

    return {"type": "FeatureCollection", "features": features}
