"""
Riyadh Feature Engineering Pipeline — THAMAN v2 (Polars)
Produces data/processed/features_riyadh.csv from raw Riyadh data files.

Sources:
  - data/raw/quarter_report SI.xlsx         (2018-2023, original)
  - data/raw/sales_riyadh_2024_Q*.csv       (2024 Q1/Q3/Q4, API)
  - data/raw/sales_riyadh_2025_Q*.csv       (2025 Q1/Q2/Q3, API; Q4 lacks district, skipped)

Steps:
  1  Load & normalise transactions from all sources (Polars)
  2  District centroid geocoding
  3  Metro transit features
  4  Bus stop features
  5  Traffic intersection features
  6  Commercial services features
  7  Air quality features
  8  Real estate price index
  9  Salary macro features
  10 District aggregate features + target encoding

Run:  python scripts/riyadh_feature_engineering.py
"""

import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.stats import linregress
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
RAW = _ROOT / "data" / "raw"
PROCESSED = _ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_ar(s: str) -> str:
    """Normalize Arabic string: NFKC, strip tatweel and harakat."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ـ", "")
    s = "".join(c for c in s if not (0x064B <= ord(c) <= 0x065F))
    return s.strip()


def _geojson_coords(feat):
    """Return (lat, lon) from GeoJSON feature geometry.coordinates [lon, lat]."""
    c = feat["geometry"]["coordinates"]
    return c[1], c[0]


def _geo_point_2d(feat):
    """Return (lat, lon) from geo_point_2d property; (None, None) if absent."""
    geo = feat["properties"].get("geo_point_2d")
    if not geo:
        return None, None
    return geo.get("lat"), geo.get("lon")


# ── Step 1: Load & normalise transactions ─────────────────────────────────────

print("Step 1 — Loading transactions...")

KEEP_TYPES = {"شقة", "فيلا", "قطعة أرض-سكنى", "عمارة"}

# ── 1a. XLSX (2018-2023) ──────────────────────────────────────────────────────

xlsx_raw = pl.read_excel(RAW / "quarter_report SI.xlsx")
print(f"  XLSX raw rows: {len(xlsx_raw)}")

xlsx = (
    xlsx_raw
    .filter(pl.col("city_ar") == "الرياض")
    .filter(pl.col("typecategoryar").is_in(list(KEEP_TYPES)))
    .filter(pl.col("deed_counts") >= 3)
    # Only keep rows where the IQR-winsorized per-sqm price is present.
    # The fallback RealEstatePrice_SUM / deed_counts gives average deal value,
    # not price per sqm, producing multi-million SAR/sqm garbage values.
    .filter(pl.col("Meter_Price_W_Avg_IQR").is_not_null())
    .with_columns([
        pl.col("Meter_Price_W_Avg_IQR").cast(pl.Float64, strict=False).alias("sale_price_sar_sqm"),
        pl.col("deed_counts").cast(pl.Int64).alias("deed_counts"),
        pl.col("yearnumber").cast(pl.Int64).alias("year"),
        pl.col("quarternumber").cast(pl.Int64).alias("quarter"),
        pl.col("district_ar").cast(pl.Utf8).alias("district_ar"),
        pl.col("region_ar").cast(pl.Utf8).alias("region_ar"),
        pl.col("city_ar").cast(pl.Utf8).alias("city_ar"),
    ])
    .filter(pl.col("sale_price_sar_sqm").is_not_null() & (pl.col("sale_price_sar_sqm") > 0))
    .filter(pl.col("district_ar").is_not_null())
    .filter(pl.col("district_ar") != "NULL")
    .select([
        "year", "quarter", "district_ar", "typecategoryar",
        "deed_counts", "sale_price_sar_sqm", "region_ar", "city_ar",
    ])
)
print(f"  XLSX after filter: {len(xlsx)} rows | years: {sorted(xlsx['year'].unique().to_list())}")


# ── 1b. 2024 API CSVs (format: رقم الربع  is the quarter number) ───────────────

def _load_2024_csv(path: Path) -> pl.DataFrame:
    # Read without renaming to avoid duplicate-column issues (two "الربع" variants)
    df = pl.read_csv(path, encoding="utf8-lossy", null_values=["NULL", "null", ""],
                     infer_schema_length=200)
    cols = df.columns
    # Quarter number: 'رقم الربع ' (has رقم)
    qcol = next((c for c in cols if "رقم" in c and "ربع" in c), None)
    if qcol is None:
        # Fallback: use الربع col that is numeric
        qcol = next((c for c in cols if "ربع" in c), cols[1])
    city_col     = next((c for c in cols if "المدينة" in c), cols[5])
    district_col = next((c for c in cols if "الحي" == c.strip()), next((c for c in cols if "الحي" in c), cols[6]))
    type_col     = next((c for c in cols if "نوع" in c and "عقار" in c), cols[7])
    deed_col     = next((c for c in cols if "صكوك" in c), cols[8])
    price_col    = next((c for c in cols if "متوسط" in c and "سعر" in c), cols[11])
    year_col     = next((c for c in cols if "السنة" in c), cols[0])
    region_col   = next((c for c in cols if "المنطقة" in c), cols[4])

    return (
        df
        .filter(pl.col(city_col).str.contains("الرياض"))
        .filter(pl.col(deed_col).cast(pl.Int64, strict=False) >= 3)
        .filter(pl.col(type_col).is_in(list(KEEP_TYPES)))
        .with_columns([
            pl.col(year_col).cast(pl.Int64, strict=False).alias("year"),
            pl.col(qcol).cast(pl.Int64, strict=False).alias("quarter"),
            pl.col(district_col).cast(pl.Utf8).alias("district_ar"),
            pl.col(type_col).cast(pl.Utf8).alias("typecategoryar"),
            pl.col(deed_col).cast(pl.Int64, strict=False).alias("deed_counts"),
            pl.col(price_col).cast(pl.Float64, strict=False).alias("sale_price_sar_sqm"),
            pl.col(region_col).cast(pl.Utf8).alias("region_ar"),
            pl.lit("الرياض").alias("city_ar"),
        ])
        .filter(pl.col("sale_price_sar_sqm").is_not_null() & (pl.col("sale_price_sar_sqm") > 0))
        .filter(pl.col("district_ar").is_not_null())
        .select(["year", "quarter", "district_ar", "typecategoryar",
                 "deed_counts", "sale_price_sar_sqm", "region_ar", "city_ar"])
    )


# ── 1c. 2025 API CSVs (format: الربع is the quarter number, تصنيف العقار present) ──

# 2025 property type → standardized mapping
# land type depends on classification column
def _load_2025_csv(path: Path) -> pl.DataFrame:
    # Read without global rename — 2025 CSVs have "الربع" (numeric) + "الربع " (text, trailing space)
    df = pl.read_csv(path, encoding="utf8-lossy", null_values=["NULL", "null", ""],
                     infer_schema_length=200)
    cols = df.columns

    city_col     = next((c for c in cols if "المدينة" in c), cols[4])
    district_col = next((c for c in cols if "الحي" == c.strip()), next((c for c in cols if "الحي" in c), cols[5]))
    type_col     = next((c for c in cols if "نوع" in c and "عقار" in c), cols[6])
    class_col    = next((c for c in cols if "تصنيف" in c), None)
    deed_col     = next((c for c in cols if "صكوك" in c), cols[8])
    price_col    = next((c for c in cols if "متوسط" in c and "سعر" in c), cols[11])
    year_col     = next((c for c in cols if "السنة" in c), cols[0])
    region_col   = next((c for c in cols if "المنطقة" in c), cols[3])

    # Quarter number: find the الربع column that is purely numeric
    q_candidates = [c for c in cols if "الربع" in c]
    qcol = q_candidates[0]
    for cand in q_candidates:
        try:
            n_null = df[cand].cast(pl.Int64, strict=False).null_count()
            if n_null < len(df) * 0.5:
                qcol = cand
                break
        except Exception:
            pass

    df = df.filter(pl.col(city_col).str.contains("الرياض"))
    df = df.filter(pl.col(deed_col).cast(pl.Int64, strict=False) >= 3)

    # Map 2025 property types → standardized
    if class_col and class_col in df.columns:
        type_expr = (
            pl.when(pl.col(type_col).is_in(["شقة", "دور"])).then(pl.lit("شقة"))
            .when(pl.col(type_col).is_in(["فيلا", "دوبلكس"])).then(pl.lit("فيلا"))
            .when((pl.col(type_col) == "أرض") & (pl.col(class_col) == "سكني")).then(pl.lit("قطعة أرض-سكنى"))
            .when(pl.col(type_col).is_in(["عمارة", "مبنى"])).then(pl.lit("عمارة"))
            .otherwise(pl.lit(None))
        )
    else:
        type_expr = (
            pl.when(pl.col(type_col).is_in(["شقة", "دور"])).then(pl.lit("شقة"))
            .when(pl.col(type_col).is_in(["فيلا", "دوبلكس"])).then(pl.lit("فيلا"))
            .when(pl.col(type_col).is_in(["عمارة", "مبنى"])).then(pl.lit("عمارة"))
            .otherwise(pl.lit(None))
        )

    return (
        df
        .with_columns([
            pl.col(year_col).cast(pl.Int64, strict=False).alias("year"),
            pl.col(qcol).cast(pl.Int64, strict=False).alias("quarter"),
            pl.col(district_col).cast(pl.Utf8).alias("district_ar"),
            type_expr.alias("typecategoryar"),
            pl.col(deed_col).cast(pl.Int64, strict=False).alias("deed_counts"),
            pl.col(price_col).cast(pl.Float64, strict=False).alias("sale_price_sar_sqm"),
            pl.col(region_col).cast(pl.Utf8).alias("region_ar"),
            pl.lit("الرياض").alias("city_ar"),
        ])
        .filter(pl.col("typecategoryar").is_not_null())
        .filter(pl.col("sale_price_sar_sqm").is_not_null() & (pl.col("sale_price_sar_sqm") > 0))
        .filter(pl.col("district_ar").is_not_null())
        .select(["year", "quarter", "district_ar", "typecategoryar",
                 "deed_counts", "sale_price_sar_sqm", "region_ar", "city_ar"])
    )


# Load all API CSVs
api_dfs = []
for fname, loader in [
    ("sales_riyadh_2024_Q1.csv", _load_2024_csv),
    ("sales_riyadh_2024_Q3.csv", _load_2024_csv),
    ("sales_riyadh_2024_Q4.csv", _load_2024_csv),
    ("sales_riyadh_2025_Q1.csv", _load_2025_csv),
    ("sales_riyadh_2025_Q2.csv", _load_2025_csv),
    ("sales_riyadh_2025_Q3.csv", _load_2025_csv),
]:
    p = RAW / fname
    if p.exists():
        part = loader(p)
        api_dfs.append(part)
        print(f"  {fname}: {len(part)} rows")
    else:
        print(f"  {fname}: NOT FOUND — skipping")

# Combine all sources
all_parts = [xlsx] + api_dfs
df = pl.concat(all_parts, how="diagonal_relaxed")

# Winsorise price P1-P99 across the full combined dataset
p01 = df["sale_price_sar_sqm"].quantile(0.01)
p99 = df["sale_price_sar_sqm"].quantile(0.99)
df = df.filter(
    (pl.col("sale_price_sar_sqm") >= p01) & (pl.col("sale_price_sar_sqm") <= p99)
)

# Derived columns
df = df.with_columns([
    (pl.col("year") * 10 + pl.col("quarter")).alias("quarter_id"),
    (pl.col("typecategoryar") == "شقة").cast(pl.Int8).alias("is_apartment"),
    (pl.col("typecategoryar") == "فيلا").cast(pl.Int8).alias("is_villa"),
    (pl.col("typecategoryar") == "قطعة أرض-سكنى").cast(pl.Int8).alias("is_residential_plot"),
    (pl.col("typecategoryar") == "عمارة").cast(pl.Int8).alias("is_building"),
    pl.col("deed_counts").cast(pl.Float64).log1p().alias("log_deed_count"),
    pl.col("year").alias("sale_year"),
    pl.col("quarter").alias("sale_quarter"),
    (2 * np.pi * pl.col("quarter") / 4).sin().alias("sale_quarter_sin"),
    (2 * np.pi * pl.col("quarter") / 4).cos().alias("sale_quarter_cos"),
    pl.col("district_ar").map_elements(_normalize_ar, return_dtype=pl.Utf8).alias("district_ar_norm"),
])

print(f"\n  Combined rows: {len(df)} | price range: {p01:.0f}–{p99:.0f} SAR/sqm")
print(f"  Districts: {df['district_ar'].n_unique()} | Years: {sorted(df['year'].unique().to_list())}")


# ── Step 2: District centroid geocoding ───────────────────────────────────────

print("\nStep 2 — Building district centroids...")

with open(RAW / "commercial-services-by-category-sub-municipality-and-district-2024.geojson") as f:
    comm_gj = json.load(f)

comm_rows = []
for feat in comm_gj["features"]:
    lat, lon = _geo_point_2d(feat)
    if lat is None or lon is None:
        continue
    dist = feat["properties"].get("districtar", "")
    if dist and dist not in ("NA", ""):
        comm_rows.append({"district_ar_norm": _normalize_ar(dist), "lat": lat, "lon": lon})

comm_pl = pl.DataFrame(comm_rows)
centroids = (
    comm_pl
    .group_by("district_ar_norm")
    .agg([pl.col("lat").mean().alias("district_lat"),
          pl.col("lon").mean().alias("district_lon")])
)

FALLBACK_CENTROIDS = {
    "الخزامى":  (24.8350, 46.6950),
    "الخير":    (24.8600, 46.7800),
    "الراية":   (24.7900, 46.6300),
    "الرسالة":  (24.7150, 46.5900),
    "الزاهر":   (24.6500, 46.7600),
    "الزهور":   (24.8100, 46.7200),
    "الشعلة":   (24.7400, 46.6100),
    "النخبة":   (24.8200, 46.6500),
    "عريض":     (24.8800, 46.8200),
    "مغرزات":   (24.7600, 46.5600),
}

fallback_pl = pl.DataFrame([
    {"district_ar_norm": _normalize_ar(d), "district_lat": lat, "district_lon": lon}
    for d, (lat, lon) in FALLBACK_CENTROIDS.items()
])

all_centroids = (
    pl.concat([centroids, fallback_pl], how="diagonal_relaxed")
    .unique("district_ar_norm", keep="first")
)

# Join district centroids → save reference CSV
norm_to_orig = dict(df.select(["district_ar_norm", "district_ar"]).unique().iter_rows())
all_centroids_save = all_centroids.with_columns(
    pl.col("district_ar_norm")
    .map_elements(lambda n: norm_to_orig.get(n, n), return_dtype=pl.Utf8)
    .alias("district_ar")
)
all_centroids_save.write_csv(PROCESSED / "district_centroids.csv")
print(f"  Centroids: {len(all_centroids)} districts")

df = df.join(
    all_centroids.select(["district_ar_norm", "district_lat", "district_lon"]),
    on="district_ar_norm", how="left"
)

# Fallback: Riyadh city center for unmatched districts
RIYADH_CENTER_LAT, RIYADH_CENTER_LON = 24.7136, 46.6753
missing_n = df["district_lat"].null_count()
if missing_n > 0:
    unmatched = df.filter(pl.col("district_lat").is_null())["district_ar"].unique().to_list()
    print(f"  {missing_n} rows ({len(unmatched)} districts) → Riyadh city center fallback")
    for d in unmatched:
        print(f"    '{d}'")
    df = df.with_columns([
        pl.col("district_lat").fill_null(RIYADH_CENTER_LAT),
        pl.col("district_lon").fill_null(RIYADH_CENTER_LON),
    ])

print(f"  After centroid join: {len(df)} rows")


# ── Numpy arrays for spatial queries ─────────────────────────────────────────

query_lats = df["district_lat"].to_numpy()
query_lons = df["district_lon"].to_numpy()
query_pts  = np.column_stack([query_lats, query_lons])   # (N, 2) in degrees

R_EARTH  = 6_371_000.0
DEG2M    = 111_000.0
r_1km    = 1000.0 / R_EARTH
r_500    = 500.0  / R_EARTH


# ── Step 3: Metro transit features ────────────────────────────────────────────

print("Step 3 — Metro transit features...")

with open(RAW / "metro-stations-in-riyadh-by-metro-line-and-station-type-2024.geojson") as f:
    metro_gj = json.load(f)

metro_coords, metro_lines, metro_types, line1_coords = [], [], [], []
for feat in metro_gj["features"]:
    lat, lon = _geojson_coords(feat)
    p = feat["properties"]
    metro_coords.append([lat, lon])
    metro_lines.append(p["metro_line_cd"])
    metro_types.append(int(p["metro_station_type_cd"]))
    if p["metro_line_cd"] == "Line1":
        line1_coords.append([lat, lon])

metro_arr = np.array(metro_coords, dtype=np.float64)
metro_tree = cKDTree(metro_arr)
metro_ball = BallTree(np.radians(metro_arr), metric="haversine")
line1_arr  = np.array(line1_coords, dtype=np.float64) if line1_coords else metro_arr
line1_tree = cKDTree(line1_arr)

LINE_ORDER = {"Line1": 1, "Line2": 2, "Line3": 3, "Line4": 4, "Line5": 5, "Line6": 6}

dists_deg, idxs = metro_tree.query(query_pts, k=1)
idxs_flat = idxs.ravel()
dist_metro_m = dists_deg.ravel() * DEG2M

d1_deg, _ = line1_tree.query(query_pts, k=1)
counts_1km = metro_ball.query_radius(np.radians(query_pts), r=r_1km, count_only=True)

df = df.with_columns([
    pl.Series("dist_metro_m",       dist_metro_m),
    pl.Series("log_dist_metro_m",   np.log1p(dist_metro_m)),
    pl.Series("nearest_metro_line_cd", [metro_lines[i] for i in idxs_flat]),
    pl.Series("nearest_metro_type_cd", [metro_types[i] for i in idxs_flat]),
    pl.Series("dist_metro_line1_m", d1_deg.ravel() * DEG2M),
    pl.Series("metro_stations_1km", counts_1km.astype(np.int32)),
    pl.Series("nearest_metro_line_num",
              [LINE_ORDER.get(metro_lines[i], 0) for i in idxs_flat]).cast(pl.Int32),
])
print(f"  Metro stations: {len(metro_coords)} | Line1: {len(line1_coords)}")


# ── Step 4: Bus stop features ─────────────────────────────────────────────────

print("Step 4 — Bus stop features...")

with open(RAW / "bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024.geojson") as f:
    bus_gj = json.load(f)

bus_coords, bus_shelters = [], []
for feat in bus_gj["features"]:
    lat, lon = _geo_point_2d(feat)
    p = feat["properties"]
    bus_coords.append([lat, lon])
    bus_shelters.append(1 if str(p.get("bsheltertypecode", "")).startswith("A") else 0)

bus_arr    = np.array(bus_coords, dtype=np.float64)
bus_tree   = cKDTree(bus_arr)
bus_ball   = BallTree(np.radians(bus_arr), metric="haversine")
shelter_arr = bus_arr[np.array(bus_shelters, dtype=bool)]
shelter_ball = BallTree(np.radians(shelter_arr), metric="haversine") if len(shelter_arr) > 0 else None

d_bus_deg, _ = bus_tree.query(query_pts, k=1)
bus_500 = bus_ball.query_radius(np.radians(query_pts), r=r_500, count_only=True)
brt_500 = shelter_ball.query_radius(np.radians(query_pts), r=r_500, count_only=True) if shelter_ball else np.zeros(len(df), dtype=np.int32)

df = df.with_columns([
    pl.Series("dist_bus_m",     d_bus_deg.ravel() * DEG2M),
    pl.Series("log_dist_bus_m", np.log1p(d_bus_deg.ravel() * DEG2M)),
    pl.Series("bus_stops_500m", bus_500.astype(np.int32)),
    pl.Series("brt_stops_500m", brt_500.astype(np.int32)),
])
print(f"  Bus stops: {len(bus_coords)} | BRT shelters: {len(shelter_arr)}")


# ── Step 5: Traffic intersection features ─────────────────────────────────────

print("Step 5 — Traffic intersection features...")

with open(RAW / "traffic-intersections-by-main-street-and-cross-street-2024.geojson") as f:
    int_gj = json.load(f)

int_coords = []
for feat in int_gj["features"]:
    lat, lon = _geo_point_2d(feat)
    if lat is None or lon is None:
        continue
    int_coords.append([lat, lon])

int_arr  = np.array(int_coords, dtype=np.float64)
int_tree = cKDTree(int_arr)
int_ball = BallTree(np.radians(int_arr), metric="haversine")

d_int_deg, _ = int_tree.query(query_pts, k=1)
int_1km = int_ball.query_radius(np.radians(query_pts), r=r_1km, count_only=True)
int_500 = int_ball.query_radius(np.radians(query_pts), r=r_500, count_only=True)

df = df.with_columns([
    pl.Series("dist_major_intersection_m", d_int_deg.ravel() * DEG2M),
    pl.Series("log_dist_intersection_m",   np.log1p(d_int_deg.ravel() * DEG2M)),
    pl.Series("intersections_1km",         int_1km.astype(np.int32)),
    pl.Series("intersections_500m",        int_500.astype(np.int32)),
])
print(f"  Traffic intersections: {len(int_coords)}")


# ── Step 6: Commercial services features ──────────────────────────────────────

print("Step 6 — Commercial services features...")

COMMERCIAL_BUCKETS = {
    "hypermarket":        {"HypMkt"},
    "supermarket":        {"SupMkt", "MktS", "GroS"},
    "bank":               {"Bank"},
    "restaurant":         {"Res"},
    "hotel":              {"Hot", "HotAp"},
    "gas_station":        {"GasStation", "PetStation"},
    "commercial_complex": {"ComC", "ComX"},
}

all_comm_coords, bucket_coords = [], {k: [] for k in COMMERCIAL_BUCKETS}
district_comm: dict[str, dict] = {}

for feat in comm_gj["features"]:
    p = feat["properties"]
    lat, lon = _geo_point_2d(feat)
    if lat is None or lon is None:
        continue
    cat  = p.get("comcatcode", "")
    dist = _normalize_ar(p.get("districtar", ""))
    all_comm_coords.append([lat, lon])
    if dist and dist not in (_normalize_ar("NA"), ""):
        entry = district_comm.setdefault(dist, {"count": 0, "cats": set()})
        entry["count"] += 1
        entry["cats"].add(cat)
    for bname, codes in COMMERCIAL_BUCKETS.items():
        if cat in codes:
            bucket_coords[bname].append([lat, lon])

comm_arr  = np.array(all_comm_coords, dtype=np.float64)
comm_ball = BallTree(np.radians(comm_arr), metric="haversine")

bucket_balls: dict = {}
for bname, pts in bucket_coords.items():
    if pts:
        bucket_balls[bname] = BallTree(np.radians(np.array(pts, dtype=np.float64)), metric="haversine")
        print(f"  Commercial {bname}: {len(pts)}")
    else:
        bucket_balls[bname] = None

comm_1km = comm_ball.query_radius(np.radians(query_pts), r=r_1km, count_only=True)
bucket_arrays: dict[str, np.ndarray] = {}
for bname, bt in bucket_balls.items():
    bucket_arrays[bname] = (
        bt.query_radius(np.radians(query_pts), r=r_1km, count_only=True)
        if bt else np.zeros(len(df), dtype=np.int32)
    )

df = df.with_columns(
    [pl.Series("commercial_count_1km", comm_1km.astype(np.int32))]
    + [pl.Series(f"{bname}_count_1km", arr.astype(np.int32))
       for bname, arr in bucket_arrays.items()]
)

# Density score
df = df.with_columns(
    (pl.col("hypermarket_count_1km") * 3
     + pl.col("supermarket_count_1km") * 2
     + pl.col("bank_count_1km")
     + pl.col("restaurant_count_1km")
     + pl.col("hotel_count_1km")).alias("commercial_density_score")
)

# District-level commercial stats
norms = df["district_ar_norm"].to_list()
df = df.with_columns([
    pl.Series("district_commercial_count",
              [district_comm.get(n, {}).get("count", 0) for n in norms]).cast(pl.Int32),
    pl.Series("district_commercial_mix",
              [len(district_comm.get(n, {}).get("cats", set())) for n in norms]).cast(pl.Int32),
])


# ── Step 6b: Mosque / Mall / School / Hospital / Park / Entertainment features ─

print("Step 6b — QoL POI features (mosques, malls, schools, hospitals, parks)...")


def _load_csv_coords(path: Path, lat_col: str = "latitude", lon_col: str = "longitude") -> np.ndarray:
    """Read a simple lat/lon CSV into an (N,2) float64 array, dropping nulls."""
    df_poi = pl.read_csv(path, encoding="utf8-lossy",
                         null_values=["NULL", "null", ""])
    # Rename BOM-prefixed column headers if present
    df_poi = df_poi.rename({c: c.lstrip("﻿").strip() for c in df_poi.columns})
    lat_col_act = next((c for c in df_poi.columns if lat_col in c.lower()), lat_col)
    lon_col_act = next((c for c in df_poi.columns if lon_col in c.lower()), lon_col)
    df_poi = df_poi.filter(
        pl.col(lat_col_act).is_not_null() & pl.col(lon_col_act).is_not_null()
    ).with_columns([
        pl.col(lat_col_act).cast(pl.Float64, strict=False),
        pl.col(lon_col_act).cast(pl.Float64, strict=False),
    ])
    lats = df_poi[lat_col_act].to_numpy()
    lons = df_poi[lon_col_act].to_numpy()
    # Filter Riyadh bbox
    mask = (lats > 23.5) & (lats < 26.0) & (lons > 45.5) & (lons < 48.0)
    return np.column_stack([lats[mask], lons[mask]])


QOL_POIS = {
    "mosque":      (RAW / "riyadh_mosques.csv",   "lat",  "lon"),
    "mall":        (RAW / "riyadh_malls.csv",      "lat",  "lon"),
    "school":      (RAW / "riyadh_schools.csv",    "lat",  "lon"),
    "hospital":    (RAW / "riyadh_hospitals.csv",  "lat",  "lon"),
    "park":        (RAW / "riyadh_parks.csv",      "lat",  "lon"),
    "entertain":   (RAW / "rcrc_entertainment.csv","lat",  "lon"),
}

qol_trees:  dict[str, cKDTree]  = {}
qol_balls:  dict[str, BallTree] = {}
qol_counts: dict[str, int]      = {}

for poi_name, (poi_path, lat_col, lon_col) in QOL_POIS.items():
    if not poi_path.exists():
        print(f"  {poi_name}: NOT FOUND — skipping")
        continue
    arr = _load_csv_coords(poi_path, lat_col=lat_col, lon_col=lon_col)
    if len(arr) < 2:
        print(f"  {poi_name}: too few points — skipping")
        continue
    qol_trees[poi_name]  = cKDTree(arr)
    qol_balls[poi_name]  = BallTree(np.radians(arr), metric="haversine")
    qol_counts[poi_name] = len(arr)
    print(f"  {poi_name}: {len(arr)} points")

new_cols = []
for poi_name, tree in qol_trees.items():
    d_deg, _ = tree.query(query_pts, k=1)
    dist_m   = d_deg.ravel() * DEG2M
    new_cols.append(pl.Series(f"dist_{poi_name}_m",     dist_m))
    new_cols.append(pl.Series(f"log_dist_{poi_name}_m", np.log1p(dist_m)))
    # 500m density count
    ball = qol_balls[poi_name]
    cnt  = ball.query_radius(np.radians(query_pts), r=r_500, count_only=True)
    new_cols.append(pl.Series(f"{poi_name}_count_500m", cnt.astype(np.int32)))

if new_cols:
    df = df.with_columns(new_cols)


# ── Step 7: Air quality features ──────────────────────────────────────────────

print("Step 7 — Air quality features...")

AQ_STATION_COORDS = {
    "At-Taawun":   (24.762272, 46.650878),
    "Al-Muruj":    (24.758315, 46.671171),
    "Al-Jazeera":  (24.700139, 46.678500),
    "Al-Uraija":   (24.685105, 46.703063),
    "Al-Khalidiya":(24.766047, 46.761886),
    "Ar-Rawabi":   (24.751314, 46.868278),
    "Ad-Dhubbat":  (24.723857, 46.756673),
    "Al-Ghurabi":  (24.648444, 46.721056),
    "Al-Khaleej":  (24.598469, 46.744378),
}

aq_raw = pl.read_csv(RAW / "air-quality.csv", separator=";", encoding="utf8-lossy",
                     null_values=["NULL", "null", ""])
aq_avg = (
    aq_raw
    .filter(pl.col("Indicator") == "Avg / Hourly")
    .filter(pl.col("Component").is_in(["NO2", "SO2", "PM10", "O3"]))
    .group_by(["Station", "Component"])
    .agg(pl.col("Value").cast(pl.Float64, strict=False).mean().alias("mean_val"))
    .pivot(on="Component", index="Station", values="mean_val")
)

# Build station coordinate arrays
valid_stations = [s for s in aq_avg["Station"].to_list() if s in AQ_STATION_COORDS]
aq_lats = np.array([AQ_STATION_COORDS[s][0] for s in valid_stations])
aq_lons = np.array([AQ_STATION_COORDS[s][1] for s in valid_stations])
aq_coords = np.column_stack([aq_lats, aq_lons])
aq_tree = cKDTree(aq_coords)

components = [c for c in ["NO2", "SO2", "PM10", "O3"] if c in aq_avg.columns]

# Station → component value dicts
aq_vals: dict[str, dict] = {}
for row in aq_avg.iter_rows(named=True):
    st = row["Station"]
    if st in AQ_STATION_COORDS:
        aq_vals[st] = {comp: (row.get(comp) or 0.0) for comp in components}

# IDW from 2 nearest stations
for comp in components:
    station_comp = np.array([aq_vals.get(s, {}).get(comp, 0.0) for s in valid_stations])
    comp_vals = np.zeros(len(df))
    for i, (lat, lon) in enumerate(zip(query_lats, query_lons)):
        d, idx = aq_tree.query([[lat, lon]], k=min(2, len(aq_coords)))
        dm = d.ravel() * DEG2M
        dm = np.where(dm < 1, 1, dm)
        w = 1.0 / dm
        comp_vals[i] = float(np.average(station_comp[idx.ravel()], weights=w))
    df = df.with_columns(pl.Series(f"{comp.lower()}_nearest_mean", comp_vals))

d_aq_deg, _ = aq_tree.query(query_pts, k=1)
df = df.with_columns(pl.Series("dist_air_station_m", d_aq_deg.ravel() * DEG2M))

# Composite air quality score (higher = cleaner)
poll_cols = [f"{c.lower()}_nearest_mean" for c in components]
poll_matrix = df.select(poll_cols).to_numpy()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(poll_matrix)
df = df.with_columns(pl.Series("air_quality_score", 100 - scaled.mean(axis=1) * 100))

print(f"  AQ stations: {len(valid_stations)} | components: {components}")


# ── Step 8: Real estate price index ───────────────────────────────────────────

print("Step 8 — Real estate price index...")

rei_raw = pl.read_csv(RAW / "real-estate-price-index-by-sector-2023-100.csv",
                      separator=";", encoding="utf8-lossy", null_values=["NULL","null",""])

rei_res = (
    rei_raw
    .filter(pl.col("Periodicity") == "Quarterly")
    .filter(pl.col("Sector") == "Residential: Total")
    .with_columns([
        pl.col("Year").cast(pl.Int64, strict=False).alias("Year"),
        pl.col("Quarter").str.extract(r"Q(\d)", 1).cast(pl.Int64, strict=False).alias("qnum"),
        pl.col("value").cast(pl.Float64, strict=False).alias("rei_residential_qtr_idx"),
    ])
    .with_columns((pl.col("Year") * 10 + pl.col("qnum")).alias("quarter_id"))
    .sort("quarter_id")
)

rei_apt = (
    rei_raw
    .filter(pl.col("Periodicity") == "Quarterly")
    .filter(pl.col("Sector") == "Residential: Apartment")
    .with_columns([
        pl.col("Year").cast(pl.Int64, strict=False).alias("Year"),
        pl.col("Quarter").str.extract(r"Q(\d)", 1).cast(pl.Int64, strict=False).alias("qnum"),
        pl.col("value").cast(pl.Float64, strict=False).alias("rei_apt_idx"),
    ])
    .with_columns((pl.col("Year") * 10 + pl.col("qnum")).alias("quarter_id"))
)

# Build lookup dicts
rei_idx_lookup:    dict[int, float] = {}
rei_yoy_lookup:    dict[int, float] = {}
rei_qoq_lookup:    dict[int, float] = {}

qids  = rei_res["quarter_id"].to_list()
idxs_ = rei_res["rei_residential_qtr_idx"].to_list()
for i, (qid, val) in enumerate(zip(qids, idxs_)):
    rei_idx_lookup[qid] = val
    yoy = (val / idxs_[i - 4] - 1) if i >= 4 else 0.0
    qoq = (val / idxs_[i - 1] - 1) if i >= 1 else 0.0
    rei_yoy_lookup[qid] = yoy
    rei_qoq_lookup[qid] = qoq

rei_apt_lookup: dict[int, float] = dict(
    zip(rei_apt["quarter_id"].to_list(), rei_apt["rei_apt_idx"].to_list())
)

available_qids = sorted(rei_idx_lookup.keys())

def _closest_qid(qid: int) -> int:
    return min(available_qids, key=lambda x: abs(x - qid)) if available_qids else qid

tx_qids = df["quarter_id"].to_list()
df = df.with_columns([
    pl.Series("rei_residential_qtr_idx",
              [rei_idx_lookup.get(q, rei_idx_lookup.get(_closest_qid(q), 100.0)) for q in tx_qids]),
    pl.Series("rei_yoy_change",
              [rei_yoy_lookup.get(q, rei_yoy_lookup.get(_closest_qid(q), 0.0)) or 0.0 for q in tx_qids]),
    pl.Series("rei_qoq_change",
              [rei_qoq_lookup.get(q, rei_qoq_lookup.get(_closest_qid(q), 0.0)) or 0.0 for q in tx_qids]),
    pl.Series("rei_apt_idx",
              [rei_apt_lookup.get(q, rei_idx_lookup.get(q, rei_idx_lookup.get(_closest_qid(q), 100.0)))
               for q in tx_qids]),
])
print(f"  REI quarters: {sorted(rei_idx_lookup.keys())}")


# ── Step 9: Salary macro features ─────────────────────────────────────────────

print("Step 9 — Salary macro features...")

sal_raw = pl.read_csv(
    RAW / "average-salaries-in-the-private-sector-by-main-profession-nationality-and-gende0.csv",
    separator=";", encoding="utf8-lossy", null_values=["NULL", "null", ""]
)
saudi_sal = (
    sal_raw
    .filter(pl.col("Nationality") == "Saudis")
    .group_by("Year")
    .agg(pl.col("Average Salary").cast(pl.Float64, strict=False).mean().alias("avg_saudi_salary_yr"))
    .sort("Year")
)
sal_years  = saudi_sal["Year"].to_list()
sal_values = saudi_sal["avg_saudi_salary_yr"].to_list()

sal_dict = dict(zip(sal_years, sal_values))
sal_yoy  = {y: (sal_dict[y] / sal_dict[sal_years[i - 1]] - 1) if i > 0 else 0.0
            for i, y in enumerate(sal_years)}

def _closest_sal_year(yr: int) -> int:
    return min(sal_years, key=lambda y: abs(y - yr))

tx_years = df["year"].to_list()
df = df.with_columns([
    pl.Series("avg_saudi_salary_yr",
              [sal_dict.get(y, sal_dict.get(_closest_sal_year(y), 0.0)) for y in tx_years]),
    pl.Series("salary_yoy_change",
              [sal_yoy.get(y, sal_yoy.get(_closest_sal_year(y), 0.0)) for y in tx_years]),
])
print(f"  Salary years: {sal_years}")


# ── Step 10: District aggregate features + target encoding ────────────────────

print("Step 10 — District aggregate features...")

# Time-based 80/20 work/hold split to avoid leakage
all_qids_sorted = sorted(df["quarter_id"].unique().to_list())
holdout_cutoff  = all_qids_sorted[int(len(all_qids_sorted) * 0.80)]
work_df = df.filter(pl.col("quarter_id") < holdout_cutoff)

# District median price & volume
district_stats = (
    work_df
    .group_by("district_ar")
    .agg([
        pl.col("sale_price_sar_sqm").median().alias("district_median_price_sqm"),
        pl.col("sale_price_sar_sqm").len().alias("district_transaction_volume"),
    ])
)

city_median = float(work_df["sale_price_sar_sqm"].median())
district_stats = district_stats.with_columns(
    (pl.col("district_median_price_sqm") / city_median - 1.0).alias("district_price_vs_city_avg")
)

# Price trend slope per district (OLS)
trend_rows = []
for dist, grp in work_df.group_by("district_ar"):
    grp_pl = grp.sort("quarter_id")
    if len(grp_pl) < 3:
        trend_rows.append({"district_ar": dist[0], "district_price_trend_slope": 0.0})
        continue
    try:
        slope, *_ = linregress(grp_pl["quarter_id"].to_numpy(),
                               grp_pl["sale_price_sar_sqm"].to_numpy())
        trend_rows.append({"district_ar": dist[0], "district_price_trend_slope": float(slope)})
    except Exception:
        trend_rows.append({"district_ar": dist[0], "district_price_trend_slope": 0.0})

trend_pl = pl.DataFrame(trend_rows)

# Apartment-specific district median
apt_median = (
    work_df
    .filter(pl.col("is_apartment") == 1)
    .group_by("district_ar")
    .agg(pl.col("sale_price_sar_sqm").median().alias("district_median_price_apt_sqm"))
)

# Target encoding: district mean log-price (work set only)
global_mean_log = float(np.log1p(work_df["sale_price_sar_sqm"].to_numpy()).mean())

work_log = work_df.with_columns(
    pl.col("sale_price_sar_sqm").log1p().alias("_log_price")
)
district_enc = (
    work_log
    .group_by("district_ar")
    .agg(pl.col("_log_price").mean().alias("district_encoded"))
)
district_type_enc = (
    work_log
    .group_by(["district_ar", "typecategoryar"])
    .agg(pl.col("_log_price").mean().alias("district_type_encoded"))
)

# Apartment-specific district encoding (filters to is_apartment==1)
district_apt_enc = (
    work_log
    .filter(pl.col("is_apartment") == 1)
    .group_by("district_ar")
    .agg(pl.col("_log_price").mean().alias("district_apt_encoded"))
)

# Recent-period district encoding (2023+)
district_recent_enc = (
    work_log
    .filter(pl.col("year") >= 2023)
    .group_by("district_ar")
    .agg(pl.col("_log_price").mean().alias("district_recent_encoded"))
)

# Apartment + recent (2023+) district encoding
district_apt_recent_enc = (
    work_log
    .filter((pl.col("is_apartment") == 1) & (pl.col("year") >= 2023))
    .group_by("district_ar")
    .agg(pl.col("_log_price").mean().alias("district_apt_recent_encoded"))
)

# Join all district stats into main df
df = (
    df
    .join(district_stats.select(["district_ar", "district_median_price_sqm",
                                  "district_transaction_volume", "district_price_vs_city_avg"]),
          on="district_ar", how="left")
    .join(trend_pl, on="district_ar", how="left")
    .join(apt_median, on="district_ar", how="left")
    .join(district_enc, on="district_ar", how="left")
    .join(district_type_enc, on=["district_ar", "typecategoryar"], how="left")
    .join(district_apt_enc, on="district_ar", how="left")
    .join(district_recent_enc, on="district_ar", how="left")
    .join(district_apt_recent_enc, on="district_ar", how="left")
)

# Fill nulls in aggregate columns with medians/global mean
agg_fill_cols = ["district_median_price_sqm", "district_transaction_volume",
                 "district_price_vs_city_avg", "district_price_trend_slope",
                 "district_median_price_apt_sqm"]
for col in agg_fill_cols:
    med = df[col].median()
    df = df.with_columns(pl.col(col).fill_null(med if med is not None else 0.0))

df = df.with_columns([
    pl.col("district_encoded").fill_null(global_mean_log),
    pl.col("district_type_encoded").fill_null(global_mean_log),
    pl.col("district_apt_encoded").fill_null(global_mean_log),
    pl.col("district_recent_encoded").fill_null(global_mean_log),
    pl.col("district_apt_recent_encoded").fill_null(global_mean_log),
])


# ── Riyadh Connectivity Score ─────────────────────────────────────────────────

print("Building Riyadh connectivity score...")

conn_raw = np.column_stack([
    df["metro_stations_1km"].to_numpy().astype(float),
    df["commercial_count_1km"].to_numpy().astype(float),
    df["bus_stops_500m"].to_numpy().astype(float),
    1.0 / np.maximum(df["dist_metro_m"].to_numpy(), 1.0),
    df["intersections_1km"].to_numpy().astype(float),
])
conn_scaler = MinMaxScaler()
conn_scaled = conn_scaler.fit_transform(np.nan_to_num(conn_raw))
WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
df = df.with_columns(
    pl.Series("riyadh_connectivity_score", (conn_scaled * WEIGHTS).sum(axis=1) * 100)
)

# Save scaler params
conn_feat_names = ["metro_stations_1km", "commercial_count_1km", "bus_stops_500m",
                   "inv_dist_metro", "intersections_1km"]
conn_scaler_params = {
    "feature_names": conn_feat_names,
    "weights": WEIGHTS.tolist(),
    "data_min_": conn_scaler.data_min_.tolist(),
    "data_max_": conn_scaler.data_max_.tolist(),
}
meta_path = Path("models/riyadh_meta.json")
meta_path.parent.mkdir(exist_ok=True)
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
meta.update({
    "riyadh_connectivity_scaler": conn_scaler_params,
    "global_mean_log": global_mean_log,
    "city_median_price_sqm": city_median,
    "holdout_cutoff_quarter_id": int(holdout_cutoff),
})
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"  Connectivity scaler saved to {meta_path}")


# ── Final output ──────────────────────────────────────────────────────────────

print("Writing output...")

ID_COLS  = ["district_ar", "district_lat", "district_lon",
            "year", "quarter", "quarter_id",
            "typecategoryar", "deed_counts", "region_ar", "city_ar"]
TARGET   = ["sale_price_sar_sqm"]
TYPE_COLS= ["is_apartment", "is_villa", "is_residential_plot", "is_building"]
METRO_COLS = ["dist_metro_m", "log_dist_metro_m", "metro_stations_1km",
              "nearest_metro_line_num", "nearest_metro_type_cd", "dist_metro_line1_m"]
BUS_COLS   = ["dist_bus_m", "log_dist_bus_m", "bus_stops_500m", "brt_stops_500m"]
INT_COLS   = ["dist_major_intersection_m", "log_dist_intersection_m",
              "intersections_1km", "intersections_500m"]
COMM_COLS  = (["commercial_count_1km", "commercial_density_score",
               "district_commercial_count", "district_commercial_mix"]
              + [f"{b}_count_1km" for b in COMMERCIAL_BUCKETS])
QOL_COLS   = [f"dist_{n}_m" for n in qol_trees] + [f"log_dist_{n}_m" for n in qol_trees] + [f"{n}_count_500m" for n in qol_trees]
AIR_COLS   = [f"{c.lower()}_nearest_mean" for c in components] + ["dist_air_station_m", "air_quality_score"]
MACRO_COLS = ["rei_residential_qtr_idx", "rei_apt_idx", "rei_yoy_change", "rei_qoq_change",
              "avg_saudi_salary_yr", "salary_yoy_change"]
DIST_COLS  = ["district_median_price_sqm", "district_transaction_volume",
              "district_price_vs_city_avg", "district_price_trend_slope",
              "district_median_price_apt_sqm", "district_encoded", "district_type_encoded",
              "district_apt_encoded", "district_recent_encoded", "district_apt_recent_encoded"]
TIME_COLS  = ["sale_year", "sale_quarter", "sale_quarter_sin", "sale_quarter_cos", "log_deed_count"]
SCORE_COLS = ["riyadh_connectivity_score"]

all_cols  = (ID_COLS + TARGET + TYPE_COLS + METRO_COLS + BUS_COLS
             + INT_COLS + COMM_COLS + QOL_COLS + AIR_COLS + MACRO_COLS + DIST_COLS + TIME_COLS + SCORE_COLS)
final_cols = [c for c in all_cols if c in df.columns]

out = df.select(final_cols)
out_path = PROCESSED / "features_riyadh.csv"
out.write_csv(out_path)

print(f"\n{'='*60}")
print(f"Output: {out_path}")
print(f"  Rows:      {len(out)}")
print(f"  Cols:      {len(out.columns)}")
print(f"  Districts: {out['district_ar'].n_unique()}")
print(f"  Years:     {sorted(out['year'].unique().to_list())}")
price_ser = out["sale_price_sar_sqm"]
print(f"  Price range: {price_ser.min():.0f} – {price_ser.max():.0f} SAR/sqm")
print(f"  Median price: {price_ser.median():.0f} SAR/sqm")

# NaN audit
total = len(out)
bad = [(c, out[c].null_count() / total) for c in out.columns if out[c].null_count() / total > 0.20]
if bad:
    print("\nWARNING — columns with >20% null:")
    for col, pct in bad:
        print(f"  {col}: {pct:.1%}")
else:
    print("\nAll columns < 20% null — OK")
print("=" * 60)
