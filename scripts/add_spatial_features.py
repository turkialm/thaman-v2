"""
scripts/add_spatial_features.py
================================
Computes three missing feature groups and writes data/processed/features_v4.csv:

  1. dist_waterfront_m  — download NYC shoreline from Open Data, nearest-point distance
  2. dist_bike_lane_m   — download NYC bike routes from Open Data, nearest-point distance
  3. POI categories     — from existing overture_places.geojson:
       poi_cafe_500m, poi_restaurant_500m, poi_gym_500m,
       poi_grocery_500m, poi_bar_500m, poi_pharmacy_500m

Run:
  cd /Users/totam/Desktop/new_try
  python scripts/add_spatial_features.py
"""

import os, sys, json, time
import numpy as np
import polars as pl
import httpx
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW   = os.path.join(BASE, "data", "raw")
PROC  = os.path.join(BASE, "data", "processed")

WATERFRONT_PTS  = os.path.join(RAW, "nyc_coastline_pts.npy")
BIKE_PATH       = os.path.join(RAW, "nyc_bike_lanes.geojson")

_FLOAT_OVERRIDES = {
    "dist_waterfront_m": pl.Float64, "dist_bike_lane_m": pl.Float64,
    "school_district": pl.Float64, "district_avg_score": pl.Float64,
    "district_school_count": pl.Float64,
    "prior_sale_price": pl.Float64, "years_since_prior_sale": pl.Float64,
    "price_appreciation": pl.Float64, "is_flip": pl.Float64,
}

HEADERS = {"User-Agent": "THAMAN-BSc-PropTech/1.0"}


# ── helpers ───────────────────────────────────────────────────────────

def download_geojson(url: str, path: str, label: str) -> dict:
    if os.path.exists(path):
        print(f"  {label}: using cached {os.path.basename(path)}")
        with open(path) as f:
            return json.load(f)
    print(f"  {label}: downloading …", end="", flush=True)
    r = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True)
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    print(f" {len(data.get('features',[]))} features saved")
    return data


def geojson_to_points(geojson: dict) -> np.ndarray:
    """
    Extract a flat array of (lat, lon) points from any GeoJSON FeatureCollection.
    Handles Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon.
    Samples line/polygon vertices (every Nth point to keep memory reasonable).
    """
    pts = []
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Point":
            pts.append((coords[1], coords[0]))
        elif gtype == "MultiPoint":
            for c in coords:
                pts.append((c[1], c[0]))
        elif gtype == "LineString":
            for c in coords[::3]:        # sample every 3rd vertex
                pts.append((c[1], c[0]))
        elif gtype == "MultiLineString":
            for line in coords:
                for c in line[::3]:
                    pts.append((c[1], c[0]))
        elif gtype == "Polygon":
            for ring in coords:
                for c in ring[::5]:
                    pts.append((c[1], c[0]))
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for c in ring[::5]:
                        pts.append((c[1], c[0]))

    return np.array(pts, dtype=np.float64)


def kdtree_dist_m(tree: cKDTree, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    pts = np.column_stack([lats, lons])
    dists, _ = tree.query(pts, k=1, workers=-1)
    return dists * 111_000.0


def balltree_count_500m(coords_rad: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    bt = BallTree(coords_rad, metric="haversine")
    query_rad = np.radians(np.column_stack([lats, lons]))
    counts = bt.query_radius(query_rad, r=500 / 6_371_000, count_only=True)
    return counts.astype(np.int32)


# ── 1. Load features ─────────────────────────────────────────────────

print("\n[1/5] Loading features_v3.csv …")
in_path = os.path.join(PROC, "features_v3.csv")
if not os.path.exists(in_path):
    in_path = os.path.join(PROC, "features.csv")
    print(f"  features_v3.csv not found — using features.csv")

df = pl.read_csv(in_path, schema_overrides=_FLOAT_OVERRIDES)
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

lats = df["latitude"].to_numpy()
lons = df["longitude"].to_numpy()


# ── 2. Waterfront distances ───────────────────────────────────────────

print("\n[2/5] Waterfront distances …")
# Use NYC coastline extracted from NTA boundary union (27K points, accurate)
if not os.path.exists(WATERFRONT_PTS):
    print("  Building coastline from NTA boundaries …")
    from shapely.geometry import shape
    from shapely.ops import unary_union
    with open(os.path.join(RAW, "nta_boundaries.geojson")) as f:
        nta = json.load(f)
    polys = [shape(feat["geometry"]) for feat in nta["features"] if feat.get("geometry")]
    city  = unary_union(polys)
    rings = city.exterior if city.geom_type == "Polygon" else [p.exterior for p in city.geoms]
    if not isinstance(rings, list):
        rings = [rings]
    pts_list = []
    for ring in rings:
        for lon, lat in list(ring.coords)[::3]:
            pts_list.append([lat, lon])
    wf_pts = np.array(pts_list, dtype=np.float64)
    np.save(WATERFRONT_PTS, wf_pts)
    print(f"  Saved {len(wf_pts)} coastline points")
else:
    wf_pts = np.load(WATERFRONT_PTS)
    print(f"  Loaded {len(wf_pts)} coastline points from cache")

wf_tree = cKDTree(wf_pts)
dist_wf = kdtree_dist_m(wf_tree, lats, lons)
print(f"  dist_waterfront_m: min={dist_wf.min():.0f}m  median={np.median(dist_wf):.0f}m  max={dist_wf.max():.0f}m")


# ── 3. Bike lane distances ────────────────────────────────────────────

print("\n[3/5] Bike lane distances …")
bike_data = download_geojson(
    "https://data.cityofnewyork.us/resource/mzxg-pwib.geojson?$limit=50000",
    BIKE_PATH,
    "NYC bike routes",
)
bike_pts = geojson_to_points(bike_data)

if len(bike_pts) == 0:
    print("  ⚠ No bike lane features — distances will be set to 5000m fallback")
    dist_bike = np.full(len(df), 5000.0)
else:
    bike_tree = cKDTree(bike_pts)
    dist_bike = kdtree_dist_m(bike_tree, lats, lons)
    print(f"  dist_bike_lane_m: min={dist_bike.min():.0f}m  median={np.median(dist_bike):.0f}m  max={dist_bike.max():.0f}m")


# ── 4. POI category counts ────────────────────────────────────────────

print("\n[4/5] POI category counts …")
overture_path = os.path.join(RAW, "overture_places.geojson")
with open(overture_path) as f:
    op = json.load(f)

# Buckets: basic_category → column name
BUCKETS = {
    "cafe":        {"cafe", "coffee_shop"},
    "restaurant":  {"restaurant", "casual_eatery", "fast_food_restaurant", "pizzaria"},
    "gym":         {"gym", "fitness_center", "yoga_studio", "martial_arts_club"},
    "grocery":     {"grocery_store", "supermarket", "convenience_store"},
    "bar":         {"bar", "cocktail_bar", "night_club"},
    "pharmacy":    {"pharmacy", "drug_store"},
}

# Build per-bucket coordinate arrays
bucket_coords: dict[str, np.ndarray] = {}
for bname, cats in BUCKETS.items():
    bpts = []
    for feat in op["features"]:
        bc = feat.get("properties", {}).get("basic_category", "")
        if bc in cats:
            coords = feat.get("geometry", {}).get("coordinates", [])
            if coords and len(coords) >= 2:
                bpts.append([coords[1], coords[0]])  # lat, lon
    arr = np.array(bpts, dtype=np.float64) if bpts else np.zeros((0, 2))
    bucket_coords[bname] = arr
    print(f"  {bname}: {len(arr):,} POIs")

# Compute counts per bucket using BallTree haversine
poi_counts: dict[str, np.ndarray] = {}
for bname, arr in bucket_coords.items():
    if len(arr) == 0:
        poi_counts[bname] = np.zeros(len(df), dtype=np.int32)
    else:
        poi_counts[bname] = balltree_count_500m(
            np.radians(arr), lats, lons
        )
    print(f"  poi_{bname}_500m: median={np.median(poi_counts[bname]):.1f}  max={poi_counts[bname].max()}")


# ── 5. Merge into DataFrame and save ─────────────────────────────────

print("\n[5/5] Saving features_v4.csv …")

df = df.with_columns([
    pl.Series("dist_waterfront_m", dist_wf),
    pl.Series("dist_bike_lane_m",  dist_bike),
    *[pl.Series(f"poi_{k}_500m", v) for k, v in poi_counts.items()],
])

out_path = os.path.join(PROC, "features_v4.csv")
df.write_csv(out_path)
print(f"  Saved → {out_path}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Sanity check
for col in ["dist_waterfront_m", "dist_bike_lane_m"] + [f"poi_{k}_500m" for k in BUCKETS]:
    nulls = df[col].null_count()
    med   = df[col].median()
    print(f"  {col}: nulls={nulls}  median={med:.1f}")

print("\n✅  Done — run training/train_stack_v4.py next")
