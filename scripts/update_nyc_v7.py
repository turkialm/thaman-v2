"""
update_nyc_v7.py — Download new NYC sales (Feb–Jun 2026) and create features_v7.csv
=====================================================================================
Downloads records from NYC Open Data after 2026-01-31, geocodes via PLUTO BBL lookup,
computes spatial features from existing raw KD-trees, imputes NTA-level stats from
existing features_v6.csv NTA medians, then appends to create features_v7.csv.

Run: python scripts/update_nyc_v7.py
"""

import os, sys, json, requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.neighbors import BallTree

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

SOCRATA_URL = "https://data.cityofnewyork.us/resource/usep-8jbt.json"
CUTOFF_DATE = "2026-01-31T00:00:00"
PAGE_SIZE   = 10000
EARTH_R_M   = 6_371_000

def haversine_m(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))

def build_balltree(lat, lon):
    coords = np.radians(np.column_stack([lat, lon]))
    return BallTree(coords, metric='haversine')

def nearest_dist_m(tree, lat, lon):
    pts = np.radians(np.column_stack([lat, lon]))
    dist_rad, _ = tree.query(pts, k=1)
    return dist_rad.ravel() * EARTH_R_M

def count_within_m(tree, lat, lon, radius_m):
    pts = np.radians(np.column_stack([lat, lon]))
    counts = tree.query_radius(pts, r=radius_m/EARTH_R_M, count_only=True)
    return counts.astype(float)

print("=" * 65)
print("  THAMAN NYC Feature Update — v7")
print("=" * 65)

# ── 1. Download new sales from Open Data API ──────────────────────────
print(f"\n[1/8] Downloading new NYC sales after {CUTOFF_DATE[:10]}...")
records = []
offset = 0
while True:
    resp = requests.get(SOCRATA_URL, params={
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$where": f"sale_date>'{CUTOFF_DATE}'",
        "$order": "sale_date ASC",
    }, timeout=60)
    resp.raise_for_status()
    batch = resp.json()
    if not batch:
        break
    records.extend(batch)
    offset += len(batch)
    print(f"  Fetched {len(records):,} records...")
    if len(batch) < PAGE_SIZE:
        break

print(f"  Total new records: {len(records):,}")
new_raw = pd.DataFrame(records)

# Numeric coerce
for col in ['borough', 'block', 'lot', 'sale_price', 'gross_square_feet',
            'residential_units', 'commercial_units', 'total_units',
            'land_square_feet', 'year_built']:
    if col in new_raw.columns:
        new_raw[col] = pd.to_numeric(new_raw[col], errors='coerce')

new_raw['sale_date'] = pd.to_datetime(new_raw['sale_date'], errors='coerce')
new_raw['bbl'] = (new_raw['borough'] * 1_000_000_000 +
                  new_raw['block'] * 10_000 +
                  new_raw['lot'])

# Filter: price > 10K, valid date
new_raw = new_raw[
    new_raw['sale_price'] > 10_000
    & new_raw['sale_date'].notna()
    & new_raw['bbl'].notna()
].copy()
print(f"  After filter (price>10K, valid date): {len(new_raw):,}")

# ── 2. Load PLUTO — join for lat/lon + building attrs ─────────────────
print("\n[2/8] Loading PLUTO for geocoding...")
pluto = pd.read_csv(
    RAW / "nyc_pluto_25v4_csv" / "pluto_25v4.csv",
    usecols=['bbl', 'latitude', 'longitude', 'numfloors', 'yearbuilt',
             'bldgclass', 'assessland', 'assesstot', 'builtfar',
             'residfar', 'commfar', 'facilfar', 'zonedist1'],
    low_memory=False
)
pluto['bbl'] = pd.to_numeric(pluto['bbl'], errors='coerce')
pluto = pluto.dropna(subset=['bbl', 'latitude', 'longitude'])
print(f"  PLUTO: {len(pluto):,} lots with coordinates")

merged = new_raw.merge(pluto, on='bbl', how='left', suffixes=('', '_pluto'))

# building_class_at_time_of col may be named oddly
if 'bldgclass' not in merged.columns and 'building_class_at_time_of' in merged.columns:
    merged['bldgclass'] = merged['building_class_at_time_of']
elif 'bldgclass' not in merged.columns:
    merged['bldgclass'] = 'A1'

merged = merged.dropna(subset=['latitude', 'longitude'])
print(f"  After geocoding: {len(merged):,} rows")

# Derived
merged['sale_year']  = merged['sale_date'].dt.year
merged['sale_month'] = merged['sale_date'].dt.month
merged['building_age'] = 2026 - merged['yearbuilt'].fillna(merged.get('year_built', 2000))
merged['building_age'] = merged['building_age'].clip(0, 200)

# Building type flags
bc = merged['bldgclass'].fillna('').str[0]
merged['is_condo']       = (bc == 'R').astype(float)
merged['is_multifamily'] = bc.isin(['C','D','S']).astype(float)
merged['is_single_fam']  = bc.isin(['A','B']).astype(float)
merged['is_mixed_use']   = bc.isin(['K','O','S']).astype(float)
merged['has_elevator']   = bc.isin(['D','H','I','M']).astype(float)

# Renovation
merged['renovated_since_2018'] = (merged.get('yearbuilt', pd.Series([0]*len(merged))) >= 2018).astype(float)
merged['years_since_renovation'] = (2026 - merged.get('yearbuilt', pd.Series([2000]*len(merged)))).clip(0)

# FAR utilization
merged['far_utilization'] = (
    merged.get('builtfar', pd.Series([0.0]*len(merged))) /
    merged.get('residfar', pd.Series([1.0]*len(merged))).replace(0, np.nan)
).fillna(0).clip(0, 5)

# ── 3. NTA assignment via GeoDataFrame spatial join ───────────────────
print("\n[3/8] Assigning NTA codes via spatial join...")
nta_gdf = gpd.read_file(RAW / "nta_boundaries.geojson")
pts_gdf  = gpd.GeoDataFrame(
    merged,
    geometry=gpd.points_from_xy(merged['longitude'], merged['latitude']),
    crs=nta_gdf.crs if nta_gdf.crs else "EPSG:4326"
)
if pts_gdf.crs != nta_gdf.crs:
    pts_gdf = pts_gdf.to_crs(nta_gdf.crs)

joined = gpd.sjoin(pts_gdf, nta_gdf[['nta2020','ntaname','geometry']], how='left', predicate='within')
merged['ntacode'] = joined['nta2020'].values
merged['ntaname'] = joined['ntaname'].values
print(f"  NTA coverage: {merged['ntacode'].notna().sum():,}/{len(merged):,}")

# ── 4. Spatial features from KD-trees ────────────────────────────────
print("\n[4/8] Computing spatial distances...")
lat = merged['latitude'].values
lon = merged['longitude'].values

# Subway
subway = pd.read_csv(RAW / "MTA_Subway_Stations_20260308.csv")
subway_tree = build_balltree(subway['GTFS Latitude'].astype(float),
                              subway['GTFS Longitude'].astype(float))
merged['dist_subway_m'] = nearest_dist_m(subway_tree, lat, lon)

# Express subway (CBD col = true)
cbd_col = 'CBD' if 'CBD' in subway.columns else None
exp = subway[subway[cbd_col].astype(str).str.lower().str.strip() == 'true'] if cbd_col else subway.head(0)
if len(exp) > 0:
    exp_tree = build_balltree(exp['GTFS Latitude'].astype(float), exp['GTFS Longitude'].astype(float))
    merged['dist_express_subway_m'] = nearest_dist_m(exp_tree, lat, lon)
    merged['nearest_station_is_express'] = (merged['dist_express_subway_m'] < 400).astype(float)
else:
    merged['dist_express_subway_m'] = merged['dist_subway_m']
    merged['nearest_station_is_express'] = 0.0

# CBD station proximity
if cbd_col:
    cbd_sub = subway[subway[cbd_col].astype(str).str.lower().str.strip() == 'true']
    if len(cbd_sub) > 0:
        cbd_tree = build_balltree(cbd_sub['GTFS Latitude'].astype(float), cbd_sub['GTFS Longitude'].astype(float))
        merged['nearest_station_is_cbd'] = (nearest_dist_m(cbd_tree, lat, lon) < 500).astype(float)
    else:
        merged['nearest_station_is_cbd'] = 0.0
else:
    merged['nearest_station_is_cbd'] = 0.0

# Route count proxy (Daytime Routes column)
if 'Daytime Routes' in subway.columns:
    nearest_idx = subway_tree.query(np.radians(np.column_stack([lat, lon])), k=1)[1].ravel()
    routes = subway.iloc[nearest_idx]['Daytime Routes'].fillna('').astype(str).str.split().apply(len)
    merged['nearest_station_route_count'] = routes.values
    if 'ADA' in subway.columns:
        ada_flag = subway.iloc[nearest_idx]['ADA'].astype(str).str.strip()
        merged['nearest_station_is_ada'] = (ada_flag == '1').astype(float).values
    else:
        merged['nearest_station_is_ada'] = 0.0
else:
    merged['nearest_station_route_count'] = 2.0
    merged['nearest_station_is_ada'] = 0.0

# Bus
bus = pd.read_csv(RAW / "mta_bus_stops.csv")
bus_tree = build_balltree(bus['latitude'].astype(float), bus['longitude'].astype(float))
merged['dist_bus_m'] = nearest_dist_m(bus_tree, lat, lon)

# Schools
schools = pd.read_csv(RAW / "schools.csv")
sch_tree = build_balltree(schools['latitude'].astype(float), schools['longitude'].astype(float))
merged['dist_school_m'] = nearest_dist_m(sch_tree, lat, lon)

elem = pd.read_csv(RAW / "elementary_schools.csv") if (RAW / "elementary_schools.csv").exists() else schools.head(0)
if len(elem) > 0:
    elem_tree = build_balltree(elem['latitude'].astype(float), elem['longitude'].astype(float))
    merged['dist_elem_school_m'] = nearest_dist_m(elem_tree, lat, lon)
else:
    merged['dist_elem_school_m'] = merged['dist_school_m']

# Parks
parks = pd.read_csv(RAW / "parks_with_coords.csv")
parks = parks.dropna(subset=['latitude', 'longitude'])
parks_tree = build_balltree(parks['latitude'].astype(float), parks['longitude'].astype(float))
merged['dist_park_m'] = nearest_dist_m(parks_tree, lat, lon)

# Placeholder for features not easy to recompute
for col in ['dist_hospital_m', 'dist_waterfront_m', 'dist_bike_lane_m',
            'poi_count_500m', 'airbnb_count_500m', 'dist_citibike_m',
            'citibike_500m', 'dist_commuter_rail_m', 'log_dist_commuter_rail_m',
            'commuter_rail_1km']:
    merged[col] = np.nan

# ── 5. NTA-level features from existing v6 medians ───────────────────
print("\n[5/8] Imputing NTA-level features from v6 medians...")
v6 = pd.read_csv(PROC / "features_v6.csv", low_memory=False)

nta_agg_cols = [
    'crime_rate_nta', 'noise_density_nta', 'livability_complaint_rate',
    'population_2020', 'median_income_nta', 'borough_income_deviation',
    'school_district', 'district_avg_score', 'district_school_count',
    'hpd_viol_rate_nta', 'tree_count_200m', 'pm25_mean', 'no2_mean',
    'hpd_class_b_viol_zip', 'hpd_class_c_viol_zip', 'hpd_severity_score_zip',
    'dob_reno_permit_count', 'dob_newbld_permit_count',
    'rat_density_nta', 'heat_density_nta',
]
nta_medians = v6.groupby('ntacode')[nta_agg_cols].median()
poi_agg_cols = [c for c in v6.columns if c.startswith('poi_')]
nta_poi_medians = v6.groupby('ntacode')[poi_agg_cols].median()

merged = merged.merge(nta_medians, on='ntacode', how='left')
merged = merged.merge(nta_poi_medians, on='ntacode', how='left')

# Global medians for NTA misses
for col in nta_agg_cols + poi_agg_cols:
    if col in merged.columns:
        merged[col] = merged[col].fillna(v6[col].median() if col in v6.columns else 0)

# ── 6. Mortgage rate (static for 2026) ───────────────────────────────
try:
    mort = pd.read_csv(RAW / "mortgage_rates.csv")
    latest_rate = float(mort.sort_values(mort.columns[0]).iloc[-1].iloc[1])
except Exception:
    latest_rate = 6.8
merged['mortgage_rate_30yr'] = latest_rate

# ── 7. Prior sale (BBL lookup from v6) ───────────────────────────────
print("\n[6/8] Looking up prior sale from existing data...")
bbl_history = v6.groupby('bbl').agg(
    prior_sale_price=('sale_price', 'last'),
    prior_sale_date=('sale_date', 'last'),
).reset_index()
merged = merged.merge(bbl_history, on='bbl', how='left')
merged['years_since_prior_sale'] = (
    (merged['sale_date'] - pd.to_datetime(merged['prior_sale_date'], errors='coerce'))
    .dt.days / 365.25
).fillna(5.0).clip(0, 30)
merged['price_appreciation'] = (
    (merged['sale_price'] - merged['prior_sale_price']) /
    merged['prior_sale_price'].replace(0, np.nan)
).fillna(0).clip(-0.9, 10)
merged['is_flip'] = (merged['years_since_prior_sale'] < 2).astype(float)
merged['has_prior_sale'] = merged['prior_sale_price'].notna().astype(float)

# Spatial features not recomputed — fill from NTA neighbor median in v6
for col in ['dist_hospital_m', 'dist_waterfront_m', 'dist_bike_lane_m',
            'poi_count_500m', 'airbnb_count_500m', 'dist_citibike_m',
            'citibike_500m', 'dist_commuter_rail_m', 'log_dist_commuter_rail_m',
            'commuter_rail_1km']:
    if col in v6.columns:
        nta_fill = v6.groupby('ntacode')[col].median()
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = merged[col].fillna(merged['ntacode'].map(nta_fill))
        merged[col] = merged[col].fillna(v6[col].median())

# log_dist_commuter_rail_m
if 'dist_commuter_rail_m' in merged.columns:
    merged['log_dist_commuter_rail_m'] = np.log1p(merged['dist_commuter_rail_m'])

# ── 8. Align columns and save ─────────────────────────────────────────
print("\n[7/8] Aligning columns with features_v6.csv...")
v6_cols = list(v6.columns)
for col in v6_cols:
    if col not in merged.columns:
        merged[col] = v6[col].median() if v6[col].dtype != object else ''

# Rename if needed
if 'address' not in merged.columns and 'Address' in merged.columns:
    merged['address'] = merged['Address']
if 'address' not in merged.columns:
    merged['address'] = ''

if 'neighborhood' not in merged.columns and 'Neighborhood' in merged.columns:
    merged['neighborhood'] = merged['Neighborhood']

# Keep only v6 columns
out = merged[[c for c in v6_cols if c in merged.columns]].copy()
for col in v6_cols:
    if col not in out.columns:
        out[col] = np.nan

out = out[v6_cols]

print(f"  New rows aligned: {len(out):,} × {len(out.columns)} cols")

# Concatenate with v6
print("\n[8/8] Saving features_v7.csv...")
combined = pd.concat([v6, out], ignore_index=True)
print(f"  Combined: {len(combined):,} rows (was {len(v6):,})")
out_path = PROC / "features_v7.csv"
combined.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")
print("\nDone. Run: python training/train_stack_v12.py (update DATA_PATH to features_v7.csv)")
