"""
THAMAN — Feature Engineering Pipeline (v2)
==========================================
Builds the full 50+ feature matrix from all raw data sources.
Output: data/processed/features.csv

Run from project root:  python training/feature_engineering.py
"""

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from datetime import datetime

CURRENT_YEAR   = datetime.now().year
RADIUS_M       = 500
EARTH_RADIUS_M = 6_371_000

print("=" * 65)
print("THAMAN Feature Engineering Pipeline v2")
print("=" * 65)

# ── Helper functions ────────────────────────────────────────────
def build_kdtree(df, lat_col, lon_col):
    coords = df[[lat_col, lon_col]].dropna().values
    return cKDTree(coords), coords

def nearest_dist_m(tree, coords):
    d, _ = tree.query(coords, k=1)
    return d * 111_000     # degrees → metres (approx)

def balltree_count(poi_rad, prop_rad, radius_m):
    bt = BallTree(poi_rad, metric="haversine")
    r  = radius_m / EARTH_RADIUS_M
    return bt.query_radius(prop_rad, r=r, count_only=True)

# ── Step 1: Load and filter property sales ──────────────────────
print("\n[1/13] Loading and filtering property sales …")
sales = pd.read_csv("data/raw/sales_geocoded.csv", low_memory=False)
total = len(sales)

sales["sale_price"] = pd.to_numeric(sales["sale_price"], errors="coerce")
sales["latitude"]   = pd.to_numeric(sales["latitude"],   errors="coerce")
sales["longitude"]  = pd.to_numeric(sales["longitude"],  errors="coerce")

sales = sales[
    sales["latitude"].notna()  &
    sales["longitude"].notna() &
    (sales["sale_price"] > 10_000)
].copy()
print(f"  Raw: {total:,}  →  After filter: {len(sales):,}")

# ── Step 2: Time features ────────────────────────────────────────
print("\n[2/13] Extracting time features …")
sales["sale_date"]  = pd.to_datetime(sales["sale_date"], errors="coerce")
sales["sale_year"]  = sales["sale_date"].dt.year
sales["sale_month"] = sales["sale_date"].dt.month

# ── Step 3: NTA spatial join + income ───────────────────────────
print("\n[3/13] NTA spatial join …")
nta = gpd.read_file("data/raw/nta_boundaries.geojson")
gdf = gpd.GeoDataFrame(
    sales,
    geometry=gpd.points_from_xy(sales.longitude, sales.latitude),
    crs="EPSG:4326",
)
gdf = gpd.sjoin(
    gdf,
    nta[["ntacode", "ntaname", "population_2020", "median_income_nta", "geometry"]],
    how="left", predicate="within",
).drop(columns=["index_right", "geometry"])
df = gdf.copy()
print(f"  NTA matched: {df['ntacode'].notna().sum():,}/{len(df):,}")

# Borough median income → income deviation feature
boro_median = df.groupby("borough")["median_income_nta"].transform("median")
df["borough_income_deviation"] = df["median_income_nta"] - boro_median

# ── Step 4: Subway distances ─────────────────────────────────────
print("\n[4/13] Subway distances (all + express) …")
subway   = pd.read_csv("data/raw/MTA_Subway_Stations_20260308.csv")
sub_tree, _ = build_kdtree(subway, "GTFS Latitude", "GTFS Longitude")
prop_coords = df[["latitude", "longitude"]].values
df["dist_subway_m"] = nearest_dist_m(sub_tree, prop_coords)

# Express stations: lines with ≥2 routes typically serve express
KNOWN_EXPRESS = {"A","C","E","B","D","F","M","N","Q","R","W","J","Z",
                 "2","3","4","5","6","7","L"}
def is_express_stop(routes_str):
    if pd.isna(routes_str): return False
    parts = str(routes_str).split()
    return any(r in KNOWN_EXPRESS for r in parts) and len(parts) >= 2

subway["is_express"] = subway["Daytime Routes"].apply(is_express_stop)
express_sub = subway[subway["is_express"]]
if len(express_sub) > 0:
    exp_tree, _ = build_kdtree(express_sub, "GTFS Latitude", "GTFS Longitude")
    df["dist_express_subway_m"]    = nearest_dist_m(exp_tree, prop_coords)
    df["nearest_station_is_express"] = (
        df["dist_express_subway_m"] <= df["dist_subway_m"] + 50
    ).astype(int)
else:
    df["dist_express_subway_m"]    = np.nan
    df["nearest_station_is_express"] = 0
print(f"  Express stations: {subway['is_express'].sum()}")

# ── Step 5: Bus, school, elementary school, park distances ───────
print("\n[5/13] Bus / school / park distances …")

bus_tree,  _ = build_kdtree(pd.read_csv("data/raw/mta_bus_stops.csv"), "latitude", "longitude")
df["dist_bus_m"] = nearest_dist_m(bus_tree, prop_coords)

sch_tree, _ = build_kdtree(pd.read_csv("data/raw/schools.csv"), "latitude", "longitude")
df["dist_school_m"] = nearest_dist_m(sch_tree, prop_coords)

elem = pd.read_csv("data/raw/elementary_schools.csv")
elem_tree, _ = build_kdtree(elem, "latitude", "longitude")
df["dist_elem_school_m"] = nearest_dist_m(elem_tree, prop_coords)

park_tree, _ = build_kdtree(pd.read_csv("data/raw/parks_with_coords.csv"), "latitude", "longitude")
df["dist_park_m"] = nearest_dist_m(park_tree, prop_coords)

# School district quality
schools_full = pd.read_csv("data/raw/schools.csv")
if "district" in schools_full.columns and "overall_score" in schools_full.columns:
    dist_stats = (schools_full.groupby("district")["overall_score"]
                  .agg(district_avg_score="mean", district_school_count="count")
                  .reset_index().rename(columns={"district": "school_district"}))
    # Assign district = borough×10 + block-based approximation via nearest school
    _, idx = sch_tree.query(prop_coords, k=1)
    df["school_district"] = schools_full.iloc[idx]["district"].values \
        if "district" in schools_full.columns else np.nan
    df = df.merge(dist_stats, on="school_district", how="left")
else:
    df["school_district"]       = np.nan
    df["district_avg_score"]    = np.nan
    df["district_school_count"] = np.nan

# ── Step 6: Hospital distance ────────────────────────────────────
print("\n[6/13] Hospital distance (Overture POIs) …")
import json
with open("data/raw/overture_places.geojson") as f:
    overture = json.load(f)

rows = [
    {"category": feat["properties"].get("basic_category",""),
     "latitude":  feat["geometry"]["coordinates"][1],
     "longitude": feat["geometry"]["coordinates"][0]}
    for feat in overture["features"]
    if feat.get("geometry",{}).get("type") == "Point"
]
poi_df  = pd.DataFrame(rows).dropna(subset=["latitude","longitude"])
health  = poi_df[poi_df["category"].str.contains(
    "health|hospital|clinic|medical|doctor|pharmacy|urgent", case=False, na=False)]
if len(health):
    h_tree, _ = build_kdtree(health, "latitude", "longitude")
    df["dist_hospital_m"] = nearest_dist_m(h_tree, prop_coords)
else:
    df["dist_hospital_m"] = np.nan

# ── Step 7: POI count + Airbnb count ────────────────────────────
print("\n[7/13] POI count + Airbnb density (500 m radius) …")
prop_rad  = np.radians(prop_coords)
poi_rad   = np.radians(poi_df[["latitude","longitude"]].values)
df["poi_count_500m"] = balltree_count(poi_rad, prop_rad, RADIUS_M)

airbnb    = pd.read_csv("data/raw/airbnb_listings.csv").dropna(subset=["latitude","longitude"])
ab_rad    = np.radians(airbnb[["latitude","longitude"]].values)
df["airbnb_count_500m"] = balltree_count(ab_rad, prop_rad, RADIUS_M)

# ── Step 8: NTA crime + noise + livability rates ─────────────────
print("\n[8/13] NTA crime / noise / livability rates …")

def nta_rate(parquet_path, nta_gdf, pop_col="population_2020", out_col="rate"):
    events = pd.read_parquet(parquet_path)
    events["latitude"]  = pd.to_numeric(events["latitude"],  errors="coerce")
    events["longitude"] = pd.to_numeric(events["longitude"], errors="coerce")
    events = events.dropna(subset=["latitude","longitude"])
    ev_gdf = gpd.GeoDataFrame(
        events, crs="EPSG:4326",
        geometry=gpd.points_from_xy(events.longitude, events.latitude),
    )
    ev_nta = gpd.sjoin(ev_gdf[["geometry"]], nta_gdf[["ntacode",pop_col,"geometry"]],
                       how="left", predicate="within")
    counts = ev_nta.groupby("ntacode").size().reset_index(name="count")
    counts = counts.merge(nta_gdf[["ntacode",pop_col]], on="ntacode", how="left")
    counts[out_col] = counts["count"] / counts[pop_col].clip(lower=1) * 1000
    return counts[["ntacode", out_col]]

crime_nta     = nta_rate("data/raw/nypd_crimes.parquet",          nta, out_col="crime_rate_nta")
noise_nta     = nta_rate("data/raw/noise_complaints.parquet",     nta, out_col="noise_density_nta")
livab_nta     = nta_rate("data/raw/livability_complaints.parquet",nta, out_col="livability_complaint_rate")

df = df.merge(crime_nta, on="ntacode", how="left")
df = df.merge(noise_nta, on="ntacode", how="left")
df = df.merge(livab_nta, on="ntacode", how="left")
print(f"  Crime rate  median: {df['crime_rate_nta'].median():.1f}/1k")
print(f"  Noise rate  median: {df['noise_density_nta'].median():.1f}/1k")
print(f"  Livability  median: {df['livability_complaint_rate'].median():.1f}/1k")

# ── Step 9: Building features + class flags ──────────────────────
print("\n[9/13] Building features + class flags …")

df["building_age"] = (CURRENT_YEAR
                      - pd.to_numeric(df.get("yearbuilt", df.get("year_built")),
                                      errors="coerce")).clip(lower=0)
df["numfloors"]    = pd.to_numeric(df.get("numfloors"), errors="coerce")

bldg = df["bldgclass"].fillna("").str.upper()
df["is_condo"]       = bldg.str.match(r"^R").astype(int)
df["is_multifamily"] = bldg.str.match(r"^[CD]").astype(int)
df["is_single_fam"]  = bldg.str.match(r"^A").astype(int)
df["is_mixed_use"]   = bldg.str.match(r"^S").astype(int)
df["has_elevator"]   = bldg.str.match(r"^(D|R[12])").astype(int)

# FAR utilisation (builtfar / facilfar as proxy where maxallwfar absent)
for c in ["builtfar","residfar","commfar","facilfar"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
denom = df["facilfar"].fillna(df["residfar"]).clip(lower=0.01)
df["far_utilization"] = (df["builtfar"] / denom).clip(0, 5)

# Assessed values (from PLUTO join in sales_geocoded — NaN for legacy rows is OK;
# train_stack_v2.py joins PLUTO itself as a fallback)
for c in ["assesstot","assessland"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")

# ── Step 10: Renovation features (DOB permits) ───────────────────
print("\n[10/13] Renovation features (DOB permits) …")
permits = pd.read_csv("data/raw/dob_permits.csv")
permits["bbl"] = pd.to_numeric(permits["bbl"], errors="coerce")
df["bbl_num"]  = pd.to_numeric(df["bbl"], errors="coerce")
df = df.merge(
    permits[["bbl","renovated_since_2018","years_since_renovation"]]
    .drop_duplicates("bbl"),
    left_on="bbl_num", right_on="bbl",
    how="left", suffixes=("","_p"),
).drop(columns=["bbl_p"], errors="ignore")
df["renovated_since_2018"]  = df["renovated_since_2018"].fillna(0).astype(int)
df["years_since_renovation"] = df["years_since_renovation"].fillna(
    df["building_age"]  # fallback: assume last renovation = original build
)
print(f"  Renovated since 2018: {df['renovated_since_2018'].sum():,} properties")

# ── Step 11: Mortgage rate (weekly → monthly join) ───────────────
print("\n[11/13] Mortgage rate …")
rates = pd.read_csv("data/raw/mortgage_rates.csv")
rates["date"]       = pd.to_datetime(rates["date"])
rates["year_month"] = rates["date"].dt.to_period("M")
monthly_rate = rates.groupby("year_month")["mortgage_rate_30yr"].mean().reset_index()
df["year_month"] = df["sale_date"].dt.to_period("M")
df = df.merge(monthly_rate, on="year_month", how="left").drop(columns=["year_month"])
print(f"  Mortgage rate range: {df['mortgage_rate_30yr'].min():.2f}% – "
      f"{df['mortgage_rate_30yr'].max():.2f}%")

# ── Step 12: Waterfront + bike lane (set NaN — requires OSM) ─────
print("\n[12/13] Waterfront / bike lane → NaN (OSM recompute skipped) …")
df["dist_waterfront_m"] = np.nan
df["dist_bike_lane_m"]  = np.nan

# ── Step 13: ACRIS prior-sale stubs (NaN — imputed in training) ──
print("\n[13/13] ACRIS stubs (NaN — imputed in train_stack_v2.py) …")
for c in ["prior_sale_price","prior_sale_date","years_since_prior_sale",
          "price_appreciation","is_flip","has_prior_sale"]:
    if c not in df.columns:
        df[c] = np.nan if c != "has_prior_sale" else 0

# ── Save ─────────────────────────────────────────────────────────
print("\n[SAVE] Building final feature matrix …")

FEATURE_COLS = [
    # Identifiers
    "address","bbl","latitude","longitude","ntacode","ntaname",
    "borough","neighborhood","zip_code",
    # Target + time
    "sale_price","sale_date","sale_year","sale_month",
    # Property attributes
    "building_age","numfloors","bldgclass",
    "gross_square_feet","land_square_feet",
    "residential_units","total_units",
    # Building class flags
    "is_condo","is_multifamily","is_single_fam","is_mixed_use","has_elevator",
    # Renovation
    "renovated_since_2018","years_since_renovation",
    # Distances
    "dist_subway_m","dist_express_subway_m","nearest_station_is_express",
    "dist_bus_m","dist_school_m","dist_elem_school_m",
    "dist_park_m","dist_hospital_m",
    "dist_waterfront_m","dist_bike_lane_m",
    # Density
    "poi_count_500m","airbnb_count_500m",
    # NTA-level
    "crime_rate_nta","noise_density_nta","livability_complaint_rate",
    "population_2020","median_income_nta","borough_income_deviation",
    # School
    "school_district","district_avg_score","district_school_count",
    # Zoning / FAR
    "builtfar","residfar","commfar","facilfar","far_utilization",
    # PLUTO assessed value
    "assesstot","assessland",
    # Macro
    "mortgage_rate_30yr",
    # ACRIS (mostly NaN — imputed in training)
    "prior_sale_price","prior_sale_date","years_since_prior_sale",
    "price_appreciation","is_flip","has_prior_sale",
]

available = [c for c in FEATURE_COLS if c in df.columns]
missing   = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  Columns not found (set to NaN): {missing}")
    for c in missing:
        df[c] = np.nan

df[FEATURE_COLS].to_csv("data/processed/features.csv", index=False)

print(f"\n{'=' * 65}")
print(f"  Saved → data/processed/features.csv")
print(f"  Rows: {len(df):,}  |  Columns: {len(FEATURE_COLS)}")
print(f"\n  Key null rates:")
check = ["prior_sale_price","dist_waterfront_m","dist_bike_lane_m",
         "gross_square_feet","mortgage_rate_30yr","assesstot"]
for c in check:
    pct = df[c].isna().mean() * 100 if c in df.columns else 100.0
    print(f"    {c:<28} {pct:.1f}% null")
print("=" * 65)
