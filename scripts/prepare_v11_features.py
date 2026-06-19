"""
scripts/prepare_v11_features.py
================================
Enriches features_v4.csv with 4 new data groups → features_v5.csv

New feature groups (8 total new columns):
  A. HPD building-health by ZIP  — Class B/C open violation intensity (2022+)
  B. DOB construction activity    — Renovation + new-build permit density by ZIP
  C. Rodent / heat complaints     — 311-derived QoL signals by NTA (local parquet)
  D. MTA station quality          — CBD flag + route count at nearest station

Run:
    cd /Users/totam/Desktop/new_try
    python scripts/prepare_v11_features.py
"""

import os, sys, json, time
import urllib.request, urllib.parse
import numpy as np
import polars as pl
from scipy.spatial import KDTree

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW   = os.path.join(BASE, "data", "raw")
PROC  = os.path.join(BASE, "data", "processed")
INPUT  = os.path.join(PROC, "features_v4.csv")
OUTPUT = os.path.join(PROC, "features_v5.csv")

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def socrata_get(dataset_id, params, timeout=25, label=""):
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                "X-App-Token": ""})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        print(f"  ✓ {label or dataset_id}: {len(data)} rows ({time.time()-t0:.1f}s)")
        return data
    except Exception as e:
        print(f"  ✗ {label or dataset_id}: {e}")
        return []

# ── Load base data ─────────────────────────────────────────────────────
banner("Loading features_v4.csv")
df = pl.read_csv(INPUT, schema_overrides={
    "zip_code": pl.Float64, "latitude": pl.Float64, "longitude": pl.Float64,
    "population_2020": pl.Float64,
})
print(f"  Loaded: {len(df):,} rows × {df.shape[1]} cols")

# Normalise zip_code to 5-digit string
df = df.with_columns(
    pl.col("zip_code").cast(pl.Int64, strict=False).cast(pl.Utf8)
      .str.zfill(5).alias("_zip_str")
)

# ══════════════════════════════════════════════════════════════════════
# A. HPD Housing Maintenance Code Violations by ZIP (2022+)
#    Features: hpd_class_b_viol_zip, hpd_class_c_viol_zip
#    Signal: Class C = immediately hazardous (mold, heat loss, lead).
#            Class B = hazardous conditions. Depresses valuation.
# ══════════════════════════════════════════════════════════════════════
banner("A — HPD open violations by ZIP + class (2022+)")

hpd_raw = socrata_get("wvxf-dwi5", {
    "$select": "zip, class, count(*) as viol_count",
    "$where":  "violationstatus='Open' AND novissueddate >= '2022-01-01T00:00:00'",
    "$group":  "zip, class",
    "$limit":  "5000",
}, label="HPD violations agg")

if hpd_raw:
    # Filter rows with valid zip (Socrata omits the key when null)
    hpd_raw = [r for r in hpd_raw if r.get("zip") and r.get("class")]
    hpd_df = pl.DataFrame({
        "zip":        [r["zip"]              for r in hpd_raw],
        "viol_class": [r["class"]            for r in hpd_raw],
        "viol_count": [int(r["viol_count"])  for r in hpd_raw],
    })
    # Pivot to wide: one column per class (A/B/C)
    hpd_wide = (
        hpd_df.pivot(on="viol_class", index="zip", values="viol_count", aggregate_function="sum")
              .rename({c: f"hpd_class_{c.lower()}_raw" for c in ["A","B","C"]
                       if c in hpd_df["viol_class"].unique().to_list()})
    )
    for col in ["hpd_class_a_raw", "hpd_class_b_raw", "hpd_class_c_raw"]:
        if col not in hpd_wide.columns:
            hpd_wide = hpd_wide.with_columns(pl.lit(0).alias(col))
    hpd_wide = hpd_wide.with_columns([
        pl.col("hpd_class_b_raw").fill_null(0).alias("hpd_class_b_raw"),
        pl.col("hpd_class_c_raw").fill_null(0).alias("hpd_class_c_raw"),
        # Severity-weighted score: C counts 3×, B counts 2×, A counts 1×
        (pl.col("hpd_class_c_raw").fill_null(0) * 3.0 +
         pl.col("hpd_class_b_raw").fill_null(0) * 2.0 +
         pl.col("hpd_class_a_raw").fill_null(0) * 1.0
        ).alias("hpd_severity_score_zip"),
        pl.col("zip").str.zfill(5).alias("_zip_str"),
    ])
    # Keep only the two most informative (model sees B, C, and composite)
    hpd_wide = hpd_wide.select([
        "_zip_str",
        pl.col("hpd_class_b_raw").log1p().alias("hpd_class_b_viol_zip"),
        pl.col("hpd_class_c_raw").log1p().alias("hpd_class_c_viol_zip"),
        pl.col("hpd_severity_score_zip").log1p(),
    ])
    df = df.join(hpd_wide, on="_zip_str", how="left")
    for c in ["hpd_class_b_viol_zip", "hpd_class_c_viol_zip", "hpd_severity_score_zip"]:
        med = float(df[c].drop_nulls().median() or 0.0)
        df = df.with_columns(pl.col(c).fill_null(med))
    print(f"  Added: hpd_class_b_viol_zip, hpd_class_c_viol_zip, hpd_severity_score_zip")
    print(f"  Coverage: {(df['hpd_class_c_viol_zip'] > 0).sum()/len(df)*100:.1f}%")
else:
    print("  Skipped — API unavailable, filling zeros")
    for c in ["hpd_class_b_viol_zip","hpd_class_c_viol_zip","hpd_severity_score_zip"]:
        df = df.with_columns(pl.lit(0.0).alias(c))

# ══════════════════════════════════════════════════════════════════════
# B. DOB Construction + Renovation Permits by ZIP (2022+)
#    Features: dob_reno_permit_count, dob_newbld_permit_count
#    Signal: active renovation = building improvement → premium.
#            new building density = development pressure → appreciation.
# ══════════════════════════════════════════════════════════════════════
banner("B — DOB renovation + new-build permits by ZIP (2022+)")

dob_reno = socrata_get("ipu4-2q9a", {
    "$select": "zip_code, count(*) as permit_count",
    "$where":  "filing_date >= '01/01/2022' AND (job_type='A1' OR job_type='A2')",
    "$group":  "zip_code",
    "$limit":  "500",
}, label="DOB A1/A2 reno by ZIP")

dob_nb = socrata_get("ipu4-2q9a", {
    "$select": "zip_code, count(*) as nb_count",
    "$where":  "filing_date >= '01/01/2022' AND job_type='NB'",
    "$group":  "zip_code",
    "$limit":  "500",
}, label="DOB NB new-build by ZIP")

def build_dob_series(rows, count_key, col_name):
    if not rows:
        return None
    d = {r["zip_code"].zfill(5): int(r[count_key]) for r in rows if r.get("zip_code")}
    return pl.DataFrame({
        "_zip_str": list(d.keys()),
        col_name:   [float(v) for v in d.values()],
    })

reno_df = build_dob_series(dob_reno, "permit_count", "dob_reno_permit_count")
nb_df   = build_dob_series(dob_nb,   "nb_count",     "dob_newbld_permit_count")

for frame, cols in [(reno_df, ["dob_reno_permit_count"]),
                    (nb_df,   ["dob_newbld_permit_count"])]:
    if frame is not None:
        df = df.join(frame, on="_zip_str", how="left")
        for c in cols:
            med = float(df[c].drop_nulls().median() or 0.0)
            df = df.with_columns(
                pl.col(c).fill_null(med).log1p().alias(c)   # log-transform in place
            )
        print(f"  Added: {', '.join(cols)}")
    else:
        for c in cols:
            df = df.with_columns(pl.lit(0.0).alias(c))
            print(f"  Skipped {c} — API unavailable")

# ══════════════════════════════════════════════════════════════════════
# C. Rodent + Heat complaint density by NTA (from local parquet)
#    Features: rat_density_nta, heat_density_nta
#    Source: data/raw/livability_complaints.parquet
#    Method: shapely point-in-polygon → ntacode → count / population_2020
# ══════════════════════════════════════════════════════════════════════
banner("C — Rodent + heat complaint density by NTA (local parquet)")

LIVABILITY_PATH = os.path.join(RAW, "livability_complaints.parquet")
NTA_GJ_PATH     = os.path.join(RAW, "nta_boundaries.geojson")

try:
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree

    liv = pl.read_parquet(LIVABILITY_PATH)
    print(f"  Livability complaints: {len(liv):,} rows")
    print(f"  Types: {dict(zip(liv['complaint_type'].value_counts()['complaint_type'].to_list(), liv['complaint_type'].value_counts()['count'].to_list()))}")

    # Filter to rodent and heat/hot water
    rat_df  = liv.filter(pl.col("complaint_type") == "Rodent").drop_nulls(["latitude","longitude"])
    heat_df = liv.filter(pl.col("complaint_type").is_in(["Heat/Hot Water","Non-Residential Heat"])).drop_nulls(["latitude","longitude"])
    print(f"  Rodent: {len(rat_df):,}  |  Heat: {len(heat_df):,}")

    # Load NTA boundaries
    with open(NTA_GJ_PATH) as f:
        nta_gj = json.load(f)
    nta_geoms  = []
    nta_codes  = []
    for feat in nta_gj["features"]:
        props = feat.get("properties", {})
        code  = props.get("nta2020") or props.get("ntacode") or ""
        if code:
            try:
                geom = shape(feat["geometry"])
                nta_geoms.append(geom)
                nta_codes.append(code)
            except Exception:
                pass
    print(f"  NTA boundaries loaded: {len(nta_codes)} polygons")

    tree = STRtree(nta_geoms)

    def assign_nta_bulk(lat_arr, lon_arr):
        """Returns list of NTA codes (or None) for each point."""
        pts  = [Point(lon, lat) for lat, lon in zip(lat_arr, lon_arr)]
        results = []
        for pt in pts:
            idxs = tree.query(pt)
            matched = None
            for idx in idxs:
                if nta_geoms[idx].contains(pt):
                    matched = nta_codes[idx]
                    break
            results.append(matched)
        return results

    # Assign NTAs (batch — may take ~30s for 155K rodent + 7K heat)
    print("  Assigning NTAs to rodent complaints …")
    t0 = time.time()
    rat_nta  = assign_nta_bulk(rat_df["latitude"].to_numpy(),
                                rat_df["longitude"].to_numpy())
    print(f"    Done: {sum(x is not None for x in rat_nta):,} assigned ({time.time()-t0:.0f}s)")

    print("  Assigning NTAs to heat complaints …")
    t0 = time.time()
    heat_nta = assign_nta_bulk(heat_df["latitude"].to_numpy(),
                                heat_df["longitude"].to_numpy())
    print(f"    Done: {sum(x is not None for x in heat_nta):,} assigned ({time.time()-t0:.0f}s)")

    # Count per NTA
    rat_counts  = {}
    heat_counts = {}
    for code in rat_nta:
        if code:
            rat_counts[code] = rat_counts.get(code, 0) + 1
    for code in heat_nta:
        if code:
            heat_counts[code] = heat_counts.get(code, 0) + 1

    # Build per-NTA population lookup from training data
    nta_pop = (
        df.filter(pl.col("ntacode").is_not_null() & (pl.col("population_2020") > 0))
          .group_by("ntacode")
          .agg(pl.col("population_2020").median().alias("pop"))
    )
    pop_map = {r["ntacode"]: float(r["pop"]) for r in nta_pop.iter_rows(named=True)}
    global_pop = float(np.median(list(pop_map.values()))) if pop_map else 50000.0

    # Build NTA-level feature frame
    all_nta_codes = list(set(list(rat_counts) + list(heat_counts) + list(pop_map)))
    rat_feat  = []
    heat_feat = []
    for code in all_nta_codes:
        pop = pop_map.get(code, global_pop)
        # Per 1000 residents, log-scaled
        rat_feat.append(float(np.log1p(rat_counts.get(code, 0) / (pop / 1000.0 + 1e-6))))
        heat_feat.append(float(np.log1p(heat_counts.get(code, 0) / (pop / 1000.0 + 1e-6))))

    nta_feat_df = pl.DataFrame({
        "ntacode":         all_nta_codes,
        "rat_density_nta": rat_feat,
        "heat_density_nta": heat_feat,
    })

    df = df.join(nta_feat_df, on="ntacode", how="left")
    for c in ["rat_density_nta", "heat_density_nta"]:
        med = float(df[c].drop_nulls().median() or 0.0)
        df = df.with_columns(pl.col(c).fill_null(med))

    print(f"  Added: rat_density_nta, heat_density_nta")
    cov = (df["rat_density_nta"] > 0).sum() / len(df) * 100
    print(f"  Coverage: rat={cov:.1f}%  heat={(df['heat_density_nta'] > 0).sum()/len(df)*100:.1f}%")

except ImportError:
    print("  shapely not available — filling median zeros")
    for c in ["rat_density_nta", "heat_density_nta"]:
        df = df.with_columns(pl.lit(0.0).alias(c))
except Exception as e:
    print(f"  Error in NTA spatial join: {e}")
    for c in ["rat_density_nta", "heat_density_nta"]:
        df = df.with_columns(pl.lit(0.0).alias(c))

# ══════════════════════════════════════════════════════════════════════
# D. MTA Station Quality (from existing MTA_Subway_Stations CSV)
#    Features: nearest_station_is_cbd, nearest_station_route_count
#    Method: KDTree nearest-neighbor on property lat/lon
# ══════════════════════════════════════════════════════════════════════
banner("D — MTA station quality (CBD + route count)")

MTA_PATH = os.path.join(RAW, "MTA_Subway_Stations_20260308.csv")

try:
    mta = pl.read_csv(MTA_PATH)
    print(f"  MTA stations: {len(mta):,} rows | Columns: {mta.columns[:8]}")

    # Parse lat/lon
    mta = mta.with_columns([
        pl.col("GTFS Latitude").cast(pl.Float64, strict=False).alias("_slat"),
        pl.col("GTFS Longitude").cast(pl.Float64, strict=False).alias("_slon"),
    ]).drop_nulls(subset=["_slat", "_slon"])

    # CBD flag: can be bool or "true"/"false" string depending on Polars inference
    cbd_col = "CBD"
    if cbd_col in mta.columns:
        if mta[cbd_col].dtype == pl.Boolean:
            mta = mta.with_columns(pl.col(cbd_col).cast(pl.Int32).alias("_is_cbd"))
        else:
            mta = mta.with_columns(
                (pl.col(cbd_col).cast(pl.Utf8).str.to_lowercase() == "true")
                .cast(pl.Int32).alias("_is_cbd")
            )
    else:
        mta = mta.with_columns(pl.lit(0).alias("_is_cbd"))

    # Route count: parse "Daytime Routes" — e.g. "N W" → 2, "4 5 6" → 3
    routes_col = "Daytime Routes"
    if routes_col in mta.columns:
        mta = mta.with_columns(
            pl.col(routes_col).str.strip_chars()
              .str.split(" ")
              .list.len()
              .alias("_route_count")
        )
    else:
        mta = mta.with_columns(pl.lit(1).alias("_route_count"))

    # ADA accessibility
    ada_col = "ADA"
    if ada_col in mta.columns:
        mta = mta.with_columns(
            (pl.col(ada_col).cast(pl.Int32, strict=False) > 0).cast(pl.Int32).alias("_is_ada")
        )
    else:
        mta = mta.with_columns(pl.lit(0).alias("_is_ada"))

    # Complex-level dedup: one station per Complex ID, keep max route count, any CBD
    complex_col = "Complex ID"
    if complex_col in mta.columns:
        mta_cplx = (
            mta.group_by("Complex ID")
               .agg([
                   pl.col("_slat").first(),
                   pl.col("_slon").first(),
                   pl.col("_is_cbd").max(),
                   pl.col("_route_count").max(),
                   pl.col("_is_ada").max(),
               ])
        )
    else:
        mta_cplx = mta.select(["_slat","_slon","_is_cbd","_route_count","_is_ada"])

    station_lats = mta_cplx["_slat"].to_numpy()
    station_lons = mta_cplx["_slon"].to_numpy()
    station_cbd  = mta_cplx["_is_cbd"].to_numpy()
    station_rts  = mta_cplx["_route_count"].to_numpy()
    station_ada  = mta_cplx["_is_ada"].to_numpy()
    print(f"  Station complexes: {len(station_lats)}  |  CBD stations: {int(station_cbd.sum())}")

    # KDTree on station lat/lon (degree units — fine for nearest)
    ktree = KDTree(np.column_stack([station_lats, station_lons]))

    # Property lat/lon
    prop_ll = df.select(["latitude","longitude"]).fill_null(0).to_numpy()
    valid_mask = (prop_ll[:,0] != 0) & (prop_ll[:,1] != 0)

    is_cbd_col   = np.zeros(len(df), dtype=np.int32)
    route_ct_col = np.ones(len(df), dtype=np.int32)
    ada_col_arr  = np.zeros(len(df), dtype=np.int32)

    if valid_mask.sum() > 0:
        _, idxs = ktree.query(prop_ll[valid_mask], k=1)
        is_cbd_col[valid_mask]   = station_cbd[idxs]
        route_ct_col[valid_mask] = station_rts[idxs]
        ada_col_arr[valid_mask]  = station_ada[idxs]

    df = df.with_columns([
        pl.Series("nearest_station_is_cbd",    is_cbd_col.tolist()),
        pl.Series("nearest_station_route_count", route_ct_col.tolist()),
        pl.Series("nearest_station_is_ada",    ada_col_arr.tolist()),
    ])
    print(f"  Added: nearest_station_is_cbd, nearest_station_route_count, nearest_station_is_ada")
    print(f"  CBD coverage: {is_cbd_col.mean()*100:.1f}%  |  Mean routes: {route_ct_col.mean():.2f}")

except Exception as e:
    print(f"  Error in MTA join: {e}")
    for c in ["nearest_station_is_cbd", "nearest_station_route_count", "nearest_station_is_ada"]:
        df = df.with_columns(pl.lit(0).alias(c))

# ══════════════════════════════════════════════════════════════════════
# Final: drop helper column, save output
# ══════════════════════════════════════════════════════════════════════
banner("Saving features_v5.csv")

df = df.drop("_zip_str")

NEW_COLS = [
    "hpd_class_b_viol_zip", "hpd_class_c_viol_zip", "hpd_severity_score_zip",
    "dob_reno_permit_count", "dob_newbld_permit_count",
    "rat_density_nta", "heat_density_nta",
    "nearest_station_is_cbd", "nearest_station_route_count", "nearest_station_is_ada",
]
present = [c for c in NEW_COLS if c in df.columns]
print(f"\n  New feature columns ({len(present)}):")
for c in present:
    vals = df[c].drop_nulls()
    print(f"    {c:<38}  min={float(vals.min()):.3f}  med={float(vals.median()):.3f}  max={float(vals.max()):.3f}")

print(f"\n  Output: {OUTPUT}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
df.write_csv(OUTPUT)
print(f"  ✓ Saved successfully")
print(f"\n  Original features: {df.shape[1] - len(present)} → New total: {df.shape[1]}")
print(f"  These {len(present)} new columns feed directly into train_stack_v11.py")
