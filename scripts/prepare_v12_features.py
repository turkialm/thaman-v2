"""
scripts/prepare_v12_features.py
================================
Enriches features_v5.csv with 7 new Overture POI categories → features_v6.csv

New feature columns (500m count):
  poi_atm_500m, poi_urgent_care_500m, poi_cinema_500m,
  poi_library_500m, poi_childcare_500m, poi_beauty_500m, poi_hotel_500m

Uses pre-extracted CSV files from process_nyc_overture_qol.py
(much faster than re-reading 425K overture_places.geojson).

Run:
    cd /Users/totam/Desktop/new_try
    python scripts/prepare_v12_features.py
"""

import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree

BASE   = Path(__file__).resolve().parent.parent
RAW    = BASE / "data" / "raw"
PROC   = BASE / "data" / "processed"
INPUT  = PROC / "features_v5.csv"
OUTPUT = PROC / "features_v6.csv"

RADIUS_M = 500
RADIUS_RAD = RADIUS_M / 6_371_000  # Earth radius in metres

# New POI categories: (column_name, csv_file)
NEW_POIS = [
    ("atm",          RAW / "nyc_overture_atms.csv"),
    ("urgent_care",  RAW / "nyc_overture_urgent_care.csv"),
    ("cinema",       RAW / "nyc_overture_cinemas.csv"),
    ("library",      RAW / "nyc_overture_libraries.csv"),
    ("childcare",    RAW / "nyc_overture_childcare.csv"),
    ("beauty",       RAW / "nyc_overture_beauty_salons.csv"),
    ("hotel",        RAW / "nyc_overture_hotels.csv"),
]


def count_within_radius(poi_lats: np.ndarray, poi_lons: np.ndarray,
                         query_lats: np.ndarray, query_lons: np.ndarray) -> np.ndarray:
    """Return count of POIs within RADIUS_M for each query point."""
    if len(poi_lats) == 0:
        return np.zeros(len(query_lats), dtype=np.int32)
    poi_rad  = np.radians(np.column_stack([poi_lats,  poi_lons]))
    qry_rad  = np.radians(np.column_stack([query_lats, query_lons]))
    tree     = BallTree(poi_rad, metric="haversine")
    counts   = tree.query_radius(qry_rad, r=RADIUS_RAD, count_only=True)
    return counts.astype(np.int32)


print(f"Loading {INPUT.name} …")
df = pl.read_csv(INPUT)
print(f"  {len(df):,} rows × {len(df.columns)} cols")

lats = df["latitude"].to_numpy()
lons = df["longitude"].to_numpy()

new_series = []

for col_name, csv_path in NEW_POIS:
    feat_col = f"poi_{col_name}_500m"

    # Skip if already present
    if feat_col in df.columns:
        print(f"  {feat_col}: already present — skip")
        continue

    if not csv_path.exists():
        print(f"  {feat_col}: {csv_path.name} not found — zeros")
        new_series.append(pl.Series(feat_col, np.zeros(len(df), dtype=np.int32)))
        continue

    poi_df = pd.read_csv(csv_path)
    poi_lats = poi_df["lat"].to_numpy(dtype=float)
    poi_lons = poi_df["lon"].to_numpy(dtype=float)

    # Drop NaN coordinates
    mask = ~(np.isnan(poi_lats) | np.isnan(poi_lons))
    poi_lats, poi_lons = poi_lats[mask], poi_lons[mask]

    counts = count_within_radius(poi_lats, poi_lons, lats, lons)
    new_series.append(pl.Series(feat_col, counts))
    print(f"  {feat_col}: {len(poi_lats):,} POIs | "
          f"median={np.median(counts):.0f}  max={counts.max()}")

df = df.with_columns(new_series)

print(f"\nSaving → {OUTPUT.name}")
df.write_csv(OUTPUT)
print(f"  Shape: {len(df):,} rows × {len(df.columns)} cols")

# Sanity check
print("\nNew column stats:")
for col_name, _ in NEW_POIS:
    feat_col = f"poi_{col_name}_500m"
    if feat_col in df.columns:
        arr = df[feat_col].to_numpy()
        print(f"  {feat_col:30s}  median={np.median(arr):.1f}  "
              f"p90={np.percentile(arr, 90):.0f}  max={arr.max()}")
