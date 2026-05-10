"""
THAMAN API — Spatial Lookup Module
====================================
Loads all spatial reference data at startup and computes distance /
density / NTA-level features for any lat/lng coordinate.

Data loaded (all from local raw/ files):
  - MTA Subway stations   → dist_subway_m, dist_express_subway_m, nearest_station_is_express
  - MTA Bus stops         → dist_bus_m
  - High schools          → dist_school_m
  - Elementary schools    → dist_elem_school_m
  - Parks                 → dist_park_m
  - Airbnb listings       → airbnb_count_500m (BallTree 500m radius)
  - NTA boundaries        → point-in-polygon → NTA code
  - NTA stats (from features.csv) → crime_rate, income, school_district, etc.
  - Mortgage rates CSV    → mortgage_rate_30yr (latest)
"""

import os
import json
import math
import numpy as np
import polars as pl
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")

# Express subway routes in NYC (trunk lines that run express segments)
EXPRESS_ROUTES = {"A", "C", "E", "2", "3", "4", "5", "D", "F", "B", "N", "Q", "J", "Z"}

# Feature descriptions for human-readable explanations
FEATURE_DESCRIPTIONS = {
    "gross_square_feet":        "Building size (sq ft)",
    "bldgclass":                "Building class / type",
    "land_square_feet":         "Land area (sq ft)",
    "longitude":                "East-West location (Manhattan premium)",
    "latitude":                 "North-South location",
    "school_district":          "School district number",
    "median_income_nta":        "Neighborhood median income",
    "residential_units":        "Number of residential units",
    "building_age":             "Age of building (years)",
    "airbnb_count_500m":        "Airbnb density within 500m",
    "poi_count_500m":           "Points of interest within 500m",
    "price_appreciation":       "Prior price appreciation rate",
    "crime_rate_nta":           "Neighborhood crime rate (per 1k residents)",
    "borough_income_deviation": "Income deviation from borough average",
    "numfloors":                "Number of floors",
    "residfar":                 "Residential floor area ratio allowance",
    "prior_sale_price":         "Prior recorded sale price",
    "builtfar":                 "Actual built floor area ratio",
    "borough":                  "Borough (1=Manhattan … 5=Staten Island)",
    "dist_waterfront_m":        "Distance to waterfront (m)",
    "dist_subway_m":            "Distance to nearest subway station (m)",
    "dist_park_m":              "Distance to nearest park (m)",
    "dist_school_m":            "Distance to nearest high school (m)",
    "dist_elem_school_m":       "Distance to nearest elementary school (m)",
    "dist_bus_m":               "Distance to nearest bus stop (m)",
    "dist_hospital_m":          "Distance to nearest hospital/clinic (m)",
    "dist_bike_lane_m":         "Distance to nearest bike lane (m)",
    "dist_express_subway_m":    "Distance to nearest express subway (m)",
    "noise_density_nta":        "Neighborhood noise complaint density",
    "livability_complaint_rate":"Neighborhood livability complaint rate",
    "population_2020":          "NTA population (2020 Census)",
    "mortgage_rate_30yr":       "30-year fixed mortgage rate (%)",
    "sale_year":                "Year of sale",
    "has_elevator":             "Building has elevator",
    "is_condo":                 "Condo unit (R-class)",
    "is_multifamily":           "Multi-family elevator building (D-class)",
    "is_single_fam":            "Single family home (A-class)",
    "is_mixed_use":             "Mixed-use building (S-class)",
    "renovated_since_2018":     "Major permit/renovation since 2018",
    "years_since_renovation":   "Years since last major renovation",
    "far_utilization":          "FAR utilization (built / max allowed)",
    "commfar":                  "Commercial floor area ratio allowance",
    "facilfar":                 "Community facility FAR allowance",
    "maxallwfar":               "Maximum allowable FAR",
    "district_avg_score":       "Average school quality score in district",
    "district_school_count":    "Number of schools in district",
    "has_prior_sale":           "Property has a prior sale record",
    "is_flip":                  "Resold within 2 years (flip)",
    "years_since_prior_sale":   "Years since prior sale",
    "nearest_station_is_express":"Nearest subway station runs express trains",
    # v2 new features
    "log_dist_subway_m":        "Log-distance to nearest subway station",
    "log_dist_school_m":        "Log-distance to nearest high school",
    "log_dist_park_m":          "Log-distance to nearest park",
    "log_dist_hospital_m":      "Log-distance to nearest hospital",
    "log_dist_bus_m":           "Log-distance to nearest bus stop",
    "log_dist_waterfront_m":    "Log-distance to waterfront",
    "log_dist_bike_lane_m":     "Log-distance to nearest bike lane",
    "log_dist_elem_school_m":   "Log-distance to nearest elementary school",
    "log_dist_express_subway_m":"Log-distance to nearest express subway",
    "dist_midtown_manhattan_m": "Distance to Midtown Manhattan (gravity centre)",
    "dist_downtown_manhattan_m":"Distance to Downtown Manhattan (gravity centre)",
    "dist_downtown_brooklyn_m": "Distance to Downtown Brooklyn (gravity centre)",
    "dist_long_island_city_m":  "Distance to Long Island City (gravity centre)",
    "is_manhattan":             "Property is in Manhattan (borough 1)",
    "crime_x_manhattan":        "Crime rate × Manhattan flag interaction",
    "crime_x_non_manhattan":    "Crime rate × non-Manhattan flag interaction",
    "walk_score_proxy":         "Walkability/transit composite score (0–100)",
    "bldgclass_encoded":        "Building class target-encoded mean log-price",
    "borough_bldg_encoded":     "Borough × building-class target-encoded mean log-price",
    "sale_month_sin":           "Cyclical month encoding — sine component",
    "sale_month_cos":           "Cyclical month encoding — cosine component",
}


OVERTURE_BUCKETS = {
    "cafe":       {"cafe", "coffee_shop"},
    "restaurant": {"restaurant", "casual_eatery", "fast_food_restaurant", "pizzaria"},
    "gym":        {"gym", "fitness_center", "yoga_studio", "martial_arts_club"},
    "grocery":    {"grocery_store", "supermarket", "convenience_store"},
    "bar":        {"bar", "cocktail_bar", "night_club"},
    "pharmacy":   {"pharmacy", "drug_store"},
}


class SpatialLookup:
    """
    Load all spatial reference data once at startup; serve fast lookups at request time.
    """

    def __init__(self):
        print("[SpatialLookup] Loading spatial reference data...")
        self._load_nta_stats()
        self._load_subway()
        self._load_bus()
        self._load_schools()
        self._load_parks()
        self._load_airbnb()
        self._load_waterfront()
        self._load_bike_lanes()
        self._load_poi_buckets()
        self._load_mortgage_rate()
        print("[SpatialLookup] All data loaded. Ready.")

    # ── Loaders ────────────────────────────────────────────────────────

    def _load_nta_stats(self):
        """Precompute per-NTA medians from features_v4.csv + load NTA boundaries."""
        feat_path = os.path.join(PROC, "features_v4.csv")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(PROC, "features.csv")
        features = pl.read_csv(feat_path)

        NTA_STAT_COLS = [
            "crime_rate_nta", "noise_density_nta", "livability_complaint_rate",
            "population_2020", "median_income_nta", "borough_income_deviation",
            "school_district", "district_avg_score", "district_school_count",
            "poi_count_500m", "dist_hospital_m", "dist_waterfront_m", "dist_bike_lane_m",
            "poi_cafe_500m", "poi_restaurant_500m", "poi_gym_500m",
            "poi_grocery_500m", "poi_bar_500m", "poi_pharmacy_500m",
            "builtfar", "residfar", "commfar", "facilfar", "maxallwfar", "far_utilization",
        ]
        available = [c for c in NTA_STAT_COLS if c in features.columns]

        nta_stats = features.group_by("ntacode").agg(
            [pl.col(c).median() for c in available]
        )
        self._nta_stats = {
            row["ntacode"]: {k: v for k, v in row.items() if k != "ntacode"}
            for row in nta_stats.iter_rows(named=True)
        }
        self._global_medians = {c: float(features[c].median() or 0.0) for c in available}

        # Borough-level income for borough_income_deviation fallback
        self._borough_median_income = {
            row["borough"]: row["median_income_nta"]
            for row in features.group_by("borough")
            .agg(pl.col("median_income_nta").median())
            .iter_rows(named=True)
        }

        # NTA GeoDataFrame for point-in-polygon
        self._nta_gdf = gpd.read_file(os.path.join(RAW, "nta_boundaries.geojson"))
        print(f"  NTA stats: {len(self._nta_stats)} NTAs | global medians computed")

    def _load_subway(self):
        subway = (
            pl.read_csv(os.path.join(RAW, "MTA_Subway_Stations_20260308.csv"))
            .with_columns([
                pl.col("GTFS Latitude").cast(pl.Float64, strict=False),
                pl.col("GTFS Longitude").cast(pl.Float64, strict=False),
            ])
            .drop_nulls(subset=["GTFS Latitude", "GTFS Longitude"])
        )

        all_coords = subway.select(["GTFS Latitude", "GTFS Longitude"]).to_numpy()
        self._subway_tree = cKDTree(all_coords)

        # Express: any station whose Daytime Routes overlap with EXPRESS_ROUTES
        def is_express(routes_str):
            if routes_str is None:
                return False
            return bool(set(str(routes_str).split()) & EXPRESS_ROUTES)

        routes       = subway["Daytime Routes"].to_list()
        express_mask = pl.Series([is_express(r) for r in routes])
        express_coords = (
            subway.filter(express_mask)
            .select(["GTFS Latitude", "GTFS Longitude"])
            .to_numpy()
        )
        self._express_tree = cKDTree(express_coords) if len(express_coords) > 0 else self._subway_tree
        print(f"  Subway: {len(all_coords)} stations | express: {len(express_coords)}")

    def _load_bus(self):
        bus = (
            pl.read_csv(os.path.join(RAW, "mta_bus_stops.csv"))
            .with_columns([
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
            ])
            .drop_nulls(subset=["latitude", "longitude"])
        )
        self._bus_tree = cKDTree(bus.select(["latitude", "longitude"]).to_numpy())
        print(f"  Bus stops: {len(bus)}")

    def _load_schools(self):
        hs   = pl.read_csv(os.path.join(RAW, "schools.csv")).drop_nulls(subset=["latitude", "longitude"])
        elem = pl.read_csv(os.path.join(RAW, "elementary_schools.csv")).drop_nulls(subset=["latitude", "longitude"])
        self._hs_tree   = cKDTree(hs.select(["latitude", "longitude"]).to_numpy())
        self._elem_tree = cKDTree(elem.select(["latitude", "longitude"]).to_numpy())
        print(f"  Schools: {len(hs)} HS | {len(elem)} elementary")

    def _load_parks(self):
        parks = pl.read_csv(os.path.join(RAW, "parks_with_coords.csv"), ignore_errors=True).drop_nulls(subset=["latitude", "longitude"])
        self._park_tree = cKDTree(parks.select(["latitude", "longitude"]).to_numpy())
        print(f"  Parks: {len(parks)}")

    def _load_airbnb(self):
        airbnb = (
            pl.read_csv(os.path.join(RAW, "airbnb_listings.csv"))
            .with_columns([
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
            ])
            .drop_nulls(subset=["latitude", "longitude"])
        )
        coords_rad = np.radians(airbnb.select(["latitude", "longitude"]).to_numpy())
        self._airbnb_balltree = BallTree(coords_rad, metric="haversine")
        print(f"  Airbnb: {len(airbnb)} listings")

    def _load_waterfront(self):
        wf_path = os.path.join(RAW, "nyc_coastline_pts.npy")
        if os.path.exists(wf_path):
            pts = np.load(wf_path)
            self._waterfront_tree = cKDTree(pts)
            print(f"  Waterfront: {len(pts)} coastline points")
        else:
            self._waterfront_tree = None
            print("  Waterfront: coastline file missing — will use NTA median fallback")

    def _load_bike_lanes(self):
        import json as _json
        bike_path = os.path.join(RAW, "nyc_bike_lanes.geojson")
        if os.path.exists(bike_path):
            with open(bike_path) as f:
                gj = _json.load(f)
            pts = []
            for feat in gj.get("features", []):
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates", [])
                gtype  = geom.get("type", "")
                if gtype == "LineString":
                    for c in coords[::3]:
                        pts.append((c[1], c[0]))
                elif gtype == "MultiLineString":
                    for line in coords:
                        for c in line[::3]:
                            pts.append((c[1], c[0]))
            if pts:
                self._bike_tree = cKDTree(np.array(pts, dtype=np.float64))
                print(f"  Bike lanes: {len(pts)} sampled vertices")
            else:
                self._bike_tree = None
                print("  Bike lanes: no features parsed — will use NTA median fallback")
        else:
            self._bike_tree = None
            print("  Bike lanes: file missing — will use NTA median fallback")

    def _load_poi_buckets(self):
        import json as _json
        op_path = os.path.join(RAW, "overture_places.geojson")
        self._poi_balltrees: dict[str, BallTree | None] = {}
        if not os.path.exists(op_path):
            print("  POI buckets: overture_places.geojson missing — counts will be 0")
            for bname in OVERTURE_BUCKETS:
                self._poi_balltrees[bname] = None
            return
        with open(op_path) as f:
            op = _json.load(f)
        for bname, cats in OVERTURE_BUCKETS.items():
            bpts = []
            for feat in op["features"]:
                bc = feat.get("properties", {}).get("basic_category", "")
                if bc in cats:
                    c = feat.get("geometry", {}).get("coordinates", [])
                    if c and len(c) >= 2:
                        bpts.append([c[1], c[0]])
            if bpts:
                arr = np.array(bpts, dtype=np.float64)
                self._poi_balltrees[bname] = BallTree(np.radians(arr), metric="haversine")
            else:
                self._poi_balltrees[bname] = None
            print(f"  POI {bname}: {len(bpts):,}")

    def _load_mortgage_rate(self):
        mort = pl.read_csv(os.path.join(RAW, "mortgage_rates.csv"))
        self._mortgage_rate = float(mort["mortgage_rate_30yr"].drop_nulls()[-1])
        print(f"  Mortgage rate (latest): {self._mortgage_rate}%")

    # ── KD-tree helper ─────────────────────────────────────────────────

    @staticmethod
    def _kdtree_dist_m(tree: cKDTree, lat: float, lon: float) -> float:
        """Nearest neighbor distance in meters using degree approximation."""
        d, _ = tree.query([[lat, lon]], k=1)
        return float(d[0]) * 111_000  # 1 degree ≈ 111 km

    # ── NTA lookup ─────────────────────────────────────────────────────

    def _nta_for_point(self, lat: float, lon: float) -> str | None:
        """Find the NTA code for a given lat/lon via point-in-polygon."""
        try:
            pt = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy([lon], [lat]),
                crs="EPSG:4326"
            )
            joined = gpd.sjoin(pt, self._nta_gdf[["ntacode", "geometry"]], how="left", predicate="within")
            ntacode = joined["ntacode"].iloc[0]
            return ntacode if (ntacode is not None and not (isinstance(ntacode, float) and math.isnan(ntacode))) else None
        except Exception:
            return None

    # ── Main lookup ────────────────────────────────────────────────────

    def lookup(self, lat: float, lon: float) -> dict:
        """
        Compute all auto-derived spatial features for the given lat/lon.
        Returns a dict of feature_name → value (matching feature_names in meta.json).
        """
        feats = {}

        # ── Distance features ──────────────────────────────────────────
        feats["dist_subway_m"]          = self._kdtree_dist_m(self._subway_tree,  lat, lon)
        feats["dist_express_subway_m"]  = self._kdtree_dist_m(self._express_tree, lat, lon)
        # nearest_station_is_express: true when express station is as close as any station
        feats["nearest_station_is_express"] = int(
            abs(feats["dist_express_subway_m"] - feats["dist_subway_m"]) < 150
        )
        feats["dist_bus_m"]             = self._kdtree_dist_m(self._bus_tree,     lat, lon)
        feats["dist_school_m"]          = self._kdtree_dist_m(self._hs_tree,      lat, lon)
        feats["dist_elem_school_m"]     = self._kdtree_dist_m(self._elem_tree,    lat, lon)
        feats["dist_park_m"]            = self._kdtree_dist_m(self._park_tree,    lat, lon)

        # ── Airbnb density within 500m ─────────────────────────────────
        radius_rad = 500.0 / 6_371_000.0
        cnt = self._airbnb_balltree.query_radius(
            np.radians([[lat, lon]]), r=radius_rad, count_only=True
        )
        feats["airbnb_count_500m"] = int(cnt[0])

        # ── Waterfront distance (real per-point) ───────────────────────
        ntacode  = self._nta_for_point(lat, lon)
        nta_data = self._nta_stats.get(ntacode, {}) if ntacode else {}

        if self._waterfront_tree is not None:
            feats["dist_waterfront_m"] = self._kdtree_dist_m(self._waterfront_tree, lat, lon)
        else:
            feats["dist_waterfront_m"] = nta_data.get("dist_waterfront_m", self._global_medians.get("dist_waterfront_m", 1414.0))

        # ── Bike lane distance (real per-point) ────────────────────────
        if self._bike_tree is not None:
            feats["dist_bike_lane_m"] = self._kdtree_dist_m(self._bike_tree, lat, lon)
        else:
            feats["dist_bike_lane_m"] = nta_data.get("dist_bike_lane_m", self._global_medians.get("dist_bike_lane_m", 152.0))

        # ── POI category counts within 500m (real per-point) ──────────
        radius_rad_poi = 500.0 / 6_371_000.0
        for bname, bt in self._poi_balltrees.items():
            col = f"poi_{bname}_500m"
            if bt is not None:
                cnt = bt.query_radius(np.radians([[lat, lon]]), r=radius_rad_poi, count_only=True)
                feats[col] = int(cnt[0])
            else:
                feats[col] = int(nta_data.get(col, self._global_medians.get(col, 0)))

        # ── NTA-level features ─────────────────────────────────────────
        for col in [
            "crime_rate_nta", "noise_density_nta", "livability_complaint_rate",
            "population_2020", "median_income_nta", "borough_income_deviation",
            "school_district", "district_avg_score", "district_school_count",
            "poi_count_500m", "dist_hospital_m",
            "builtfar", "residfar", "commfar", "facilfar", "maxallwfar", "far_utilization",
        ]:
            feats[col] = nta_data.get(col, self._global_medians.get(col, 0.0))

        # ── Macro features ─────────────────────────────────────────────
        feats["mortgage_rate_30yr"] = self._mortgage_rate

        now = datetime.now()
        feats["sale_year"] = now.year
        # Note: sale_month removed in v2 — main.py computes sale_month_sin/cos instead

        return feats

    def get_feature_description(self, feature_name: str) -> str:
        return FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace("_", " ").title())
