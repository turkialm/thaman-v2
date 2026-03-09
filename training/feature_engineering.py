"""
THAMAN — Feature Engineering Pipeline
Builds the full feature matrix from raw data sources.
Output: data/processed/features.csv

Run from project root:  python training/feature_engineering.py
"""

import os, sys
# Ensure project root is on path when run from training/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)   # all relative paths resolve from project root

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import json
from datetime import datetime

CURRENT_YEAR = datetime.now().year
RADIUS_M = 500
EARTH_RADIUS_M = 6_371_000

print("=" * 60)
print("THAMAN Feature Engineering Pipeline")
print("=" * 60)

# ── Step 1: Load and filter property sales ─────────────────────
print("\n[1/7] Loading and filtering property sales...")

sales = pd.read_csv("data/raw/sales_geocoded.csv")
total = len(sales)

# Keep only geocoded rows with real sale prices
sales = sales[
    sales["latitude"].notna() &
    sales["longitude"].notna() &
    (sales["sale_price"] > 10_000)
].copy()

print(f"  Raw rows:       {total:,}")
print(f"  After filter:   {len(sales):,} (geocoded + price > $10k)")

# ── Step 2: Assign NTA to each property ────────────────────────
print("\n[2/7] Assigning NTA via spatial join...")

nta = gpd.read_file("data/raw/nta_boundaries.geojson")
sales_gdf = gpd.GeoDataFrame(
    sales,
    geometry=gpd.points_from_xy(sales.longitude, sales.latitude),
    crs="EPSG:4326"
)
sales_gdf = gpd.sjoin(
    sales_gdf,
    nta[["ntacode", "ntaname", "population_2020", "geometry"]],
    how="left",
    predicate="within"
)
sales_gdf = sales_gdf.drop(columns=["index_right", "geometry"])
matched_nta = sales_gdf["ntacode"].notna().sum()
print(f"  Properties with NTA: {matched_nta:,}/{len(sales_gdf):,}")

df = sales_gdf.copy()

# ── Step 3: Distance features via KD-tree ──────────────────────
print("\n[3/7] Computing distance features (subway, school, park)...")

def build_kdtree(coords_df, lat_col, lon_col):
    coords = coords_df[[lat_col, lon_col]].dropna().values
    return cKDTree(coords), coords

def nearest_dist_m(tree, prop_coords):
    distances, _ = tree.query(prop_coords, k=1)
    return distances * 111_000   # degrees -> meters (approx)

prop_coords = df[["latitude", "longitude"]].values

# Subway
subway = pd.read_csv("data/raw/MTA_Subway_Stations_20260308.csv")
subway_tree, _ = build_kdtree(subway, "GTFS Latitude", "GTFS Longitude")
df["dist_subway_m"] = nearest_dist_m(subway_tree, prop_coords)
print(f"  Subway: median {df['dist_subway_m'].median():.0f}m")

# Schools
schools = pd.read_csv("data/raw/schools.csv")
school_tree, _ = build_kdtree(schools, "latitude", "longitude")
df["dist_school_m"] = nearest_dist_m(school_tree, prop_coords)
print(f"  School: median {df['dist_school_m'].median():.0f}m")

# Parks
parks = pd.read_csv("data/raw/parks_with_coords.csv")
park_tree, _ = build_kdtree(parks, "latitude", "longitude")
df["dist_park_m"] = nearest_dist_m(park_tree, prop_coords)
print(f"  Park:   median {df['dist_park_m'].median():.0f}m")

# ── Step 4: Hospital distance from Overture POIs ───────────────
print("\n[4/7] Computing hospital/clinic distance from Overture POIs...")

with open("data/raw/overture_places.geojson") as f:
    overture = json.load(f)

rows = []
for feat in overture["features"]:
    cat = feat["properties"].get("basic_category", "") or ""
    geom = feat.get("geometry", {})
    if geom.get("type") == "Point":
        lon, lat = geom["coordinates"][:2]
        rows.append({"category": cat, "latitude": lat, "longitude": lon})

poi_df = pd.DataFrame(rows).dropna(subset=["latitude", "longitude"])
print(f"  Total Overture POIs with coordinates: {len(poi_df):,}")
print(f"  Categories (top 10): {poi_df['category'].value_counts().head(10).to_dict()}")

# Hospital/clinic subset
health_cats = ["health_and_medical", "hospital", "clinic", "medical_center",
               "doctor", "pharmacy", "urgent_care"]
health_df = poi_df[poi_df["category"].str.lower().str.contains(
    "health|hospital|clinic|medical|doctor|pharmacy|urgent", na=False
)]
print(f"  Health POIs: {len(health_df):,}")

if len(health_df) > 0:
    health_tree, _ = build_kdtree(health_df, "latitude", "longitude")
    df["dist_hospital_m"] = nearest_dist_m(health_tree, prop_coords)
    print(f"  Hospital: median {df['dist_hospital_m'].median():.0f}m")
else:
    df["dist_hospital_m"] = np.nan
    print("  Warning: no health POIs found — dist_hospital_m set to NaN")

# ── Step 5: POI count within 500m (BallTree) ───────────────────
print(f"\n[5/7] Counting POIs within {RADIUS_M}m radius (BallTree)...")

poi_rad = np.radians(poi_df[["latitude", "longitude"]].values)
ball_tree = BallTree(poi_rad, metric="haversine")

prop_rad = np.radians(df[["latitude", "longitude"]].values)
radius_rad = RADIUS_M / EARTH_RADIUS_M
counts = ball_tree.query_radius(prop_rad, r=radius_rad, count_only=True)
df["poi_count_500m"] = counts
print(f"  POI count: median {df['poi_count_500m'].median():.0f}, max {df['poi_count_500m'].max()}")

# ── Step 6: Crime rate and noise density per NTA ───────────────
print("\n[6/7] Computing crime_rate_nta and noise_density_nta...")

# Assign crimes to NTA
crimes = pd.read_parquet("data/raw/nypd_crimes.parquet")
crimes = crimes.dropna(subset=["latitude", "longitude"])
crimes["latitude"] = pd.to_numeric(crimes["latitude"], errors="coerce")
crimes["longitude"] = pd.to_numeric(crimes["longitude"], errors="coerce")
crimes = crimes.dropna(subset=["latitude", "longitude"])

crimes_gdf = gpd.GeoDataFrame(
    crimes,
    geometry=gpd.points_from_xy(crimes.longitude, crimes.latitude),
    crs="EPSG:4326"
)
crimes_nta = gpd.sjoin(
    crimes_gdf[["geometry"]],
    nta[["ntacode", "population_2020", "geometry"]],
    how="left",
    predicate="within"
)
crime_counts = crimes_nta.groupby("ntacode").size().reset_index(name="crime_count")
crime_counts = crime_counts.merge(
    nta[["ntacode", "population_2020"]], on="ntacode", how="left"
)
crime_counts["crime_rate_nta"] = (
    crime_counts["crime_count"] / crime_counts["population_2020"].clip(lower=1) * 1000
)
print(f"  Crime rate: median {crime_counts['crime_rate_nta'].median():.1f} per 1k residents")

# Assign noise complaints to NTA
noise = pd.read_parquet("data/raw/noise_complaints.parquet")
noise = noise.dropna(subset=["latitude", "longitude"])
noise["latitude"] = pd.to_numeric(noise["latitude"], errors="coerce")
noise["longitude"] = pd.to_numeric(noise["longitude"], errors="coerce")
noise = noise.dropna(subset=["latitude", "longitude"])

noise_gdf = gpd.GeoDataFrame(
    noise,
    geometry=gpd.points_from_xy(noise.longitude, noise.latitude),
    crs="EPSG:4326"
)
noise_nta = gpd.sjoin(
    noise_gdf[["geometry"]],
    nta[["ntacode", "population_2020", "geometry"]],
    how="left",
    predicate="within"
)
noise_counts = noise_nta.groupby("ntacode").size().reset_index(name="noise_count")
noise_counts = noise_counts.merge(
    nta[["ntacode", "population_2020"]], on="ntacode", how="left"
)
noise_counts["noise_density_nta"] = (
    noise_counts["noise_count"] / noise_counts["population_2020"].clip(lower=1) * 1000
)
print(f"  Noise density: median {noise_counts['noise_density_nta'].median():.1f} per 1k residents")

# Join NTA rates back to properties
df = df.merge(crime_counts[["ntacode", "crime_rate_nta"]], on="ntacode", how="left")
df = df.merge(noise_counts[["ntacode", "noise_density_nta"]], on="ntacode", how="left")

# ── Step 7: Building features ──────────────────────────────────
print("\n[7/7] Adding building features...")

df["building_age"] = CURRENT_YEAR - pd.to_numeric(df["yearbuilt"], errors="coerce")
df["building_age"] = df["building_age"].clip(lower=0)  # no negative ages
df["numfloors"] = pd.to_numeric(df["numfloors"], errors="coerce")

print(f"  building_age: median {df['building_age'].median():.0f} years")
print(f"  numfloors:    median {df['numfloors'].median():.0f}")

# ── Save final feature matrix ──────────────────────────────────
print("\n[SAVE] Building final feature matrix...")

FEATURE_COLS = [
    # Identifiers
    "address", "bbl", "latitude", "longitude", "ntacode", "ntaname",
    "borough", "neighborhood", "zip_code",
    # Target
    "sale_price", "sale_date",
    # Property attributes
    "building_age", "numfloors", "bldgclass",
    "gross_square_feet", "land_square_feet",
    "residential_units", "total_units",
    # Distance features
    "dist_subway_m", "dist_school_m", "dist_park_m", "dist_hospital_m",
    # Density features
    "poi_count_500m",
    # NTA-level features
    "crime_rate_nta", "noise_density_nta", "population_2020",
]

# Keep only columns that exist
available = [c for c in FEATURE_COLS if c in df.columns]
missing   = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  Missing cols (skipped): {missing}")

features = df[available].copy()
features.to_csv("data/processed/features.csv", index=False)

print(f"\n{'=' * 60}")
print(f"Feature matrix saved: data/processed/features.csv")
print(f"  Rows:     {len(features):,}")
print(f"  Columns:  {len(features.columns)}")
print(f"\nNull rates per feature column:")
feat_only = [c for c in available if c not in
             ["address","bbl","latitude","longitude","ntacode","ntaname",
              "borough","neighborhood","zip_code","sale_date","bldgclass"]]
for col in feat_only:
    null_pct = features[col].isna().mean() * 100
    print(f"  {col:<25} {null_pct:.1f}% null")
print("=" * 60)
