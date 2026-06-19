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
    "cafe":         {"cafe", "coffee_shop"},
    "restaurant":   {"restaurant", "casual_eatery", "fast_food_restaurant", "pizzaria"},
    "gym":          {"gym", "fitness_center", "yoga_studio", "martial_arts_club"},
    "grocery":      {"grocery_store", "supermarket", "convenience_store"},
    "bar":          {"bar", "cocktail_bar", "night_club"},
    "pharmacy":     {"pharmacy", "drug_store"},
    # New QoL categories (May 2026)
    "atm":          {"atm"},
    "urgent_care":  {"urgent_care", "medical_clinic", "healthcare_location"},
    "cinema":       {"movie_theater", "cinema"},
    "library":      {"library", "public_library"},
    "childcare":    {"childcare", "child_care_facility", "preschool"},
    "beauty":       {"beauty_salon", "hair_salon", "nail_salon"},
    "hotel":        {"hotel", "hostel", "motel"},
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
        self._load_citibike()
        self._load_commuter_rail()
        self._load_waterfront()
        self._load_bike_lanes()
        self._load_poi_buckets()
        self._load_mortgage_rate()
        print("[SpatialLookup] All data loaded. Ready.")

    # ── Loaders ────────────────────────────────────────────────────────

    def _load_nta_stats(self):
        """Precompute per-NTA medians from features_v6.csv (latest) + load NTA boundaries."""
        for _p in ["features_v6.csv", "features_v5.csv", "features_v4.csv", "features.csv"]:
            feat_path = os.path.join(PROC, _p)
            if os.path.exists(feat_path):
                break
        features = pl.read_csv(feat_path)

        NTA_STAT_COLS = [
            "crime_rate_nta", "noise_density_nta", "livability_complaint_rate",
            "population_2020", "median_income_nta", "borough_income_deviation",
            "school_district", "district_avg_score", "district_school_count",
            "poi_count_500m", "dist_hospital_m", "dist_waterfront_m", "dist_bike_lane_m",
            "poi_cafe_500m", "poi_restaurant_500m", "poi_gym_500m",
            "poi_grocery_500m", "poi_bar_500m", "poi_pharmacy_500m",
            # New Overture POI categories (v6)
            "poi_atm_500m", "poi_urgent_care_500m", "poi_cinema_500m",
            "poi_library_500m", "poi_childcare_500m", "poi_beauty_500m", "poi_hotel_500m",
            # Citi Bike access (v14)
            "citibike_500m", "dist_citibike_m",
            # LPC historic district + FEMA flood zone (v16)
            "is_historic_dist", "in_flood_zone", "is_landmark",
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

        # Merge v16 NTA lookup (historic district + flood zone rates from meta.json)
        _meta_path = os.path.join(BASE, "models", "meta.json")
        if os.path.exists(_meta_path):
            with open(_meta_path) as _f:
                _meta_j = json.load(_f)
            _v16_lu = _meta_j.get("v16_nta_lookup", {})
            _v16_glob_hist  = _meta_j.get("v16_global_hist_rate",  0.0838)
            _v16_glob_flood = _meta_j.get("v16_global_flood_rate", 0.0834)
            for _nta, _vals in _v16_lu.items():
                if _nta not in self._nta_stats:
                    self._nta_stats[_nta] = {}
                self._nta_stats[_nta].update(_vals)
            # Fill global medians for fallback
            self._global_medians.setdefault("is_historic_dist", _v16_glob_hist)
            self._global_medians.setdefault("in_flood_zone",    _v16_glob_flood)
            self._global_medians.setdefault("is_landmark",      0.0)

            # v17: tax exemption NTA lookup
            _v17_lu = _meta_j.get("v17_nta_log_exempt", {})
            _v17_glob = _meta_j.get("v17_global_log_exempt", 0.0)
            for _nta, _val in _v17_lu.items():
                if _nta not in self._nta_stats:
                    self._nta_stats[_nta] = {}
                self._nta_stats[_nta]["log_exempt_amount"] = _val
            self._global_medians.setdefault("log_exempt_amount", _v17_glob)

            # v18: census tract median household income NTA lookup
            _v18_lu = _meta_j.get("v18_nta_log_income", {})
            _v18_glob = _meta_j.get("v18_global_log_income", 10.8)
            for _nta, _val in _v18_lu.items():
                if _nta not in self._nta_stats:
                    self._nta_stats[_nta] = {}
                self._nta_stats[_nta]["log_tract_median_income"] = _val
            self._global_medians.setdefault("log_tract_median_income", _v18_glob)

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
        # v19: size-stratified park trees
        _parks_v19 = parks.with_columns(
            pl.col("ACRES").cast(pl.Float64, strict=False).fill_null(0.0)
        )
        _lrg  = _parks_v19.filter(pl.col("ACRES") >= 10.0).select(["latitude","longitude"]).to_numpy()
        _flag = _parks_v19.filter(pl.col("ACRES") >= 100.0).select(["latitude","longitude"]).to_numpy()
        self._park_large_tree    = cKDTree(_lrg)  if len(_lrg)  > 0 else None
        self._park_flagship_tree = cKDTree(_flag) if len(_flag) > 0 else None
        print(f"  Parks: {len(parks)} total | large≥10ac: {len(_lrg)} | flagship≥100ac: {len(_flag)}")

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

    def _load_citibike(self):
        cb_path = os.path.join(RAW, "nyc_citibike_stations.csv")
        self._citibike_tree = None
        if not os.path.exists(cb_path):
            return
        cb = (
            pl.read_csv(cb_path)
            .with_columns([
                pl.col("lat").cast(pl.Float64, strict=False),
                pl.col("lon").cast(pl.Float64, strict=False),
            ])
            .drop_nulls(subset=["lat", "lon"])
        )
        coords_rad = np.radians(cb.select(["lat", "lon"]).to_numpy())
        self._citibike_tree = BallTree(coords_rad, metric="haversine")
        print(f"  Citi Bike: {len(cb)} stations")

    def _load_commuter_rail(self):
        cr_path = os.path.join(RAW, "nyc_commuter_rail_stations.csv")
        self._commuter_rail_tree = None
        if not os.path.exists(cr_path):
            print("  Commuter rail: file missing — dist_commuter_rail_m will use NTA fallback")
            return
        cr = (
            pl.read_csv(cr_path)
            .with_columns([
                pl.col("lat").cast(pl.Float64, strict=False),
                pl.col("lon").cast(pl.Float64, strict=False),
            ])
            .drop_nulls(subset=["lat", "lon"])
        )
        coords_rad = np.radians(cr.select(["lat", "lon"]).to_numpy())
        self._commuter_rail_tree = BallTree(coords_rad, metric="haversine")
        print(f"  Commuter rail: {len(cr)} stations (LIRR/Metro-North/SIR)")

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
        # v19: size-stratified park distances
        _d_lp = self._kdtree_dist_m(self._park_large_tree,    lat, lon) if self._park_large_tree    else float("nan")
        _d_fp = self._kdtree_dist_m(self._park_flagship_tree, lat, lon) if self._park_flagship_tree else float("nan")
        feats["dist_large_park_m"]        = _d_lp
        feats["dist_flagship_park_m"]     = _d_fp
        feats["log_dist_large_park_m"]    = float(np.log1p(_d_lp))   if not np.isnan(_d_lp) else float("nan")
        feats["log_dist_flagship_park_m"] = float(np.log1p(_d_fp))   if not np.isnan(_d_fp) else float("nan")

        # ── Airbnb density within 500m ─────────────────────────────────
        radius_rad = 500.0 / 6_371_000.0
        cnt = self._airbnb_balltree.query_radius(
            np.radians([[lat, lon]]), r=radius_rad, count_only=True
        )
        feats["airbnb_count_500m"] = int(cnt[0])

        # ── Citi Bike stations within 500m + nearest distance ──────────
        if self._citibike_tree is not None:
            pt_rad = np.radians([[lat, lon]])
            cb_cnt = self._citibike_tree.query_radius(pt_rad, r=radius_rad, count_only=True)
            feats["citibike_500m"] = int(cb_cnt[0])
            cb_dist, _ = self._citibike_tree.query(pt_rad, k=1)
            feats["dist_citibike_m"] = round(float(cb_dist[0][0]) * 6_371_000, 1)
        else:
            feats["citibike_500m"] = int(nta_data.get("citibike_500m", 0))
            feats["dist_citibike_m"] = float(nta_data.get("dist_citibike_m", 9999.0))

        # ── Commuter rail distance + 1km flag ─────────────────────────
        if self._commuter_rail_tree is not None:
            pt_rad = np.radians([[lat, lon]])
            cr_dist, _ = self._commuter_rail_tree.query(pt_rad, k=1)
            cr_dist_m = round(float(cr_dist[0][0]) * 6_371_000, 1)
            feats["dist_commuter_rail_m"] = cr_dist_m
            feats["commuter_rail_1km"]    = int(cr_dist_m <= 1000)
        else:
            feats["dist_commuter_rail_m"] = 9999.0
            feats["commuter_rail_1km"]    = 0

        # ── Waterfront distance (real per-point) ───────────────────────
        ntacode  = self._nta_for_point(lat, lon)
        nta_data = self._nta_stats.get(ntacode, {}) if ntacode else {}

        if self._waterfront_tree is not None:
            feats["dist_waterfront_m"] = self._kdtree_dist_m(self._waterfront_tree, lat, lon)
        else:
            feats["dist_waterfront_m"] = nta_data.get("dist_waterfront_m", self._global_medians.get("dist_waterfront_m", 1414.0))
        feats["log_dist_waterfront_m"] = float(np.log1p(feats["dist_waterfront_m"]))
        feats["waterfront_200m"]       = int(feats["dist_waterfront_m"] < 200)

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
            # v16: LPC historic district rate + FEMA flood zone rate (NTA-level averages)
            "is_historic_dist", "in_flood_zone", "is_landmark",
            # v17: tax exemption NTA-level mean
            "log_exempt_amount",
            # v18: census tract median household income NTA-level mean
            "log_tract_median_income",
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


# ══════════════════════════════════════════════════════════════════════════════
#  Riyadh Spatial Lookup
# ══════════════════════════════════════════════════════════════════════════════

import unicodedata as _ucd
import pandas as _pd

RIYADH_COMMERCIAL_BUCKETS = {
    "hypermarket":        {"HypMkt"},
    "supermarket":        {"SupMkt", "MktS", "GroS"},
    "bank":               {"Bank"},
    "restaurant":         {"Res"},
    "hotel":              {"Hot", "HotAp"},
    "gas_station":        {"GasStation", "PetStation"},
    "commercial_complex": {"ComC", "ComX"},
}

# English station name → (lat, lon) mapping for air quality CSV
_AQ_STATION_COORDS = {
    "At-Taawun":    (24.762272, 46.650878),
    "Al-Muruj":     (24.758315, 46.671171),
    "Al-Jazeera":   (24.700139, 46.678500),
    "Al-Uraija":    (24.685105, 46.703063),
    "Al-Khalidiya": (24.766047, 46.761886),
    "Ar-Rawabi":    (24.751314, 46.868278),
    "Ad-Dhubbat":   (24.723857, 46.756673),
    "Al-Ghurabi":   (24.648444, 46.721056),
    "Al-Khaleej":   (24.598469, 46.744378),
}

_LINE_ORDER = {"Line1": 1, "Line2": 2, "Line3": 3, "Line4": 4, "Line5": 5, "Line6": 6}


def _normalize_ar(s: str) -> str:
    """Normalize Arabic string: NFKC, strip tatweel and harakat."""
    if not isinstance(s, str):
        return ""
    s = _ucd.normalize("NFKC", s)
    s = s.replace("ـ", "")  # tatweel
    s = "".join(c for c in s if not (0x064B <= ord(c) <= 0x065F))
    return s.strip()


class RiyadhSpatialLookup:
    """
    Parallel spatial index for Riyadh-specific data sources.
    Loaded once at API startup; lookup() returns ~25 features for any lat/lon.
    """

    def __init__(self):
        print("[RiyadhSpatialLookup] Loading Riyadh spatial data...")
        self._load_metro()
        self._load_bus()
        self._load_intersections()
        self._load_commercial()
        self._load_poi_csvs()
        self._load_air_quality()
        self._load_district_stats()
        print("[RiyadhSpatialLookup] Ready.")

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_metro(self):
        path = os.path.join(RAW, "metro-stations-in-riyadh-by-metro-line-and-station-type-2024.geojson")
        with open(path) as f:
            gj = json.load(f)
        coords, lines, types = [], [], []
        line1 = []
        for feat in gj["features"]:
            c = feat["geometry"]["coordinates"]  # [lon, lat]
            lat, lon = c[1], c[0]
            p = feat["properties"]
            coords.append([lat, lon])
            lines.append(p["metro_line_cd"])
            types.append(int(p["metro_station_type_cd"]))
            if p["metro_line_cd"] == "Line1":
                line1.append([lat, lon])
        self._metro_arr   = np.array(coords, dtype=np.float64)
        self._metro_tree  = cKDTree(self._metro_arr)
        self._metro_ball  = BallTree(np.radians(self._metro_arr), metric="haversine")
        self._metro_lines = lines
        self._metro_types = types
        self._line1_tree  = cKDTree(np.array(line1, dtype=np.float64)) if line1 else self._metro_tree
        print(f"  Metro: {len(coords)} stations | Line1: {len(line1)}")

    def _load_bus(self):
        path = os.path.join(RAW, "bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024.geojson")
        with open(path) as f:
            gj = json.load(f)
        coords, shelters = [], []
        for feat in gj["features"]:
            geo = feat["properties"].get("geo_point_2d") or {}
            lat, lon = geo.get("lat"), geo.get("lon")
            if lat is None:
                continue
            coords.append([lat, lon])
            btype = str(feat["properties"].get("bsheltertypecode", ""))
            shelters.append(btype.startswith("A"))
        arr = np.array(coords, dtype=np.float64)
        self._bus_tree = cKDTree(arr)
        self._bus_ball = BallTree(np.radians(arr), metric="haversine")
        shelter_arr = arr[np.array(shelters)]
        self._shelter_ball = BallTree(np.radians(shelter_arr), metric="haversine") if len(shelter_arr) else None
        print(f"  Bus: {len(coords)} stops | BRT: {shelter_arr.shape[0]}")

    def _load_intersections(self):
        path = os.path.join(RAW, "traffic-intersections-by-main-street-and-cross-street-2024.geojson")
        with open(path) as f:
            gj = json.load(f)
        coords = []
        for feat in gj["features"]:
            geo = feat["properties"].get("geo_point_2d") or {}
            lat, lon = geo.get("lat"), geo.get("lon")
            if lat is None:
                continue
            coords.append([lat, lon])
        arr = np.array(coords, dtype=np.float64)
        self._int_tree = cKDTree(arr)
        self._int_ball = BallTree(np.radians(arr), metric="haversine")
        print(f"  Intersections: {len(coords)}")

    def _load_commercial(self):
        path = os.path.join(RAW, "commercial-services-by-category-sub-municipality-and-district-2024.geojson")
        with open(path) as f:
            gj = json.load(f)
        all_coords = []
        bucket_coords = {k: [] for k in RIYADH_COMMERCIAL_BUCKETS}
        for feat in gj["features"]:
            geo = feat["properties"].get("geo_point_2d") or {}
            lat, lon = geo.get("lat"), geo.get("lon")
            if lat is None:
                continue
            all_coords.append([lat, lon])
            cat = feat["properties"].get("comcatcode", "")
            for bname, codes in RIYADH_COMMERCIAL_BUCKETS.items():
                if cat in codes:
                    bucket_coords[bname].append([lat, lon])
        all_arr = np.array(all_coords, dtype=np.float64)
        self._comm_ball_all = BallTree(np.radians(all_arr), metric="haversine")
        self._comm_balls = {}
        for bname, pts in bucket_coords.items():
            if pts:
                self._comm_balls[bname] = BallTree(
                    np.radians(np.array(pts, dtype=np.float64)), metric="haversine"
                )
            else:
                self._comm_balls[bname] = None
            print(f"  Commercial {bname}: {len(pts)}")

    def _load_poi_csvs(self):
        """Load mosque/mall/school/hospital/park/entertainment POI files from saudi_thaman."""
        _POI_FILES = {
            "mosque":     os.path.join(RAW, "riyadh_mosques.csv"),
            "mall":       os.path.join(RAW, "riyadh_malls.csv"),
            "school":     os.path.join(RAW, "riyadh_schools.csv"),
            "hospital":   os.path.join(RAW, "riyadh_hospitals.csv"),
            "park":       os.path.join(RAW, "riyadh_parks.csv"),
            "entertain":  os.path.join(RAW, "rcrc_entertainment.csv"),
            # New QoL POIs (OSM, May 2026) — used in v3 model features
            "pharmacy":   os.path.join(RAW, "riyadh_qol_pharmacies.csv"),
            "gym":        os.path.join(RAW, "riyadh_qol_gyms.csv"),
            "coffee":     os.path.join(RAW, "riyadh_qol_coffee_shops.csv"),
            "clinic":     os.path.join(RAW, "riyadh_qol_clinics.csv"),
            "university": os.path.join(RAW, "riyadh_qol_universities.csv"),
            "supermarket":os.path.join(RAW, "riyadh_qol_supermarkets.csv"),
            "cinema":        os.path.join(RAW, "riyadh_qol_cinemas.csv"),
            "sports":        os.path.join(RAW, "riyadh_qol_sports_centres.csv"),
            # Batch-2 QoL POIs — display only (not in v3 model training)
            "restaurant":    os.path.join(RAW, "riyadh_qol_restaurants.csv"),
            "library":       os.path.join(RAW, "riyadh_qol_libraries.csv"),
            "atm":           os.path.join(RAW, "riyadh_qol_atms.csv"),
            "kindergarten":  os.path.join(RAW, "riyadh_qol_kindergartens.csv"),
            "swimming_pool": os.path.join(RAW, "riyadh_qol_swimming_pools.csv"),
        }
        self._poi_trees: dict = {}
        self._poi_balls: dict = {}
        for poi_name, path in _POI_FILES.items():
            if not os.path.exists(path):
                continue
            df = _pd.read_csv(path, encoding="utf-8-sig")
            df.columns = [c.lstrip("﻿").strip() for c in df.columns]
            lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
            lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
            if lat_col is None or lon_col is None:
                continue
            df = df[[lat_col, lon_col]].dropna()
            df[lat_col] = _pd.to_numeric(df[lat_col], errors="coerce")
            df[lon_col] = _pd.to_numeric(df[lon_col], errors="coerce")
            df = df.dropna()
            # Filter Riyadh bbox
            df = df[(df[lat_col] > 23.5) & (df[lat_col] < 26.0) &
                    (df[lon_col] > 45.5) & (df[lon_col] < 48.0)]
            if len(df) < 2:
                continue
            arr = df[[lat_col, lon_col]].values.astype(np.float64)
            self._poi_trees[poi_name] = cKDTree(arr)
            self._poi_balls[poi_name] = BallTree(np.radians(arr), metric="haversine")
            print(f"  POI {poi_name}: {len(arr)}")

    def _load_air_quality(self):
        aq_csv = os.path.join(RAW, "air-quality.csv")
        aq_df = _pd.read_csv(aq_csv, sep=";")
        aq_avg = aq_df[aq_df["Indicator"] == "Avg / Hourly"].copy()
        aq_means = (
            aq_avg[aq_avg["Component"].isin(["NO2", "SO2", "PM10", "O3"])]
            .groupby(["Station", "Component"])["Value"]
            .mean()
            .unstack(fill_value=0)
            .reset_index()
        )
        aq_means["lat"] = aq_means["Station"].map(lambda s: _AQ_STATION_COORDS.get(s, (None, None))[0])
        aq_means["lon"] = aq_means["Station"].map(lambda s: _AQ_STATION_COORDS.get(s, (None, None))[1])
        aq_means = aq_means[aq_means["lat"].notna()].copy()
        self._aq_arr      = aq_means[["lat", "lon"]].values.astype(np.float64)
        self._aq_tree     = cKDTree(self._aq_arr)
        self._aq_means_df = aq_means
        self._aq_components = [c for c in ["NO2", "SO2", "PM10", "O3"] if c in aq_means.columns]
        print(f"  Air quality: {len(self._aq_arr)} stations | {self._aq_components}")

    def _load_district_stats(self):
        """Load per-district aggregated stats from features_riyadh.csv."""
        feat_path = os.path.join(PROC, "features_riyadh.csv")
        if not os.path.exists(feat_path):
            self._district_stats = {}
            self._district_feat_df = None
            self._district_centroid_tree = None
            self._district_names = []
            print("  District stats: features_riyadh.csv not found — run riyadh_feature_engineering.py")
            return

        df = _pd.read_csv(feat_path)
        # Choropleth stats (median per district across all columns)
        STAT_COLS = [
            "district_lat", "district_lon",
            "dist_metro_m", "metro_stations_1km",
            "commercial_count_1km", "hypermarket_count_1km",
            "bus_stops_500m", "no2_nearest_mean", "pm10_nearest_mean",
            "air_quality_score", "rei_residential_qtr_idx",
            "district_median_price_sqm", "district_price_trend_slope",
            "district_commercial_mix", "riyadh_connectivity_score",
        ]
        available = [c for c in STAT_COLS if c in df.columns]
        agg = df.groupby("district_ar")[available].median().reset_index()
        self._district_stats = agg.set_index("district_ar").to_dict("index")

        # Full feature medians per district (for prediction)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        feat_cols = [c for c in num_cols if c not in ("year", "quarter", "quarter_id",
                                                       "sale_year", "sale_quarter",
                                                       "sale_price_sar_sqm",
                                                       "is_apartment", "is_villa",
                                                       "is_residential_plot", "is_building",
                                                       "district_encoded", "district_type_encoded")]
        self._district_feat_df = (
            df.groupby("district_ar")[feat_cols]
            .median()
            .reset_index()
        )
        # ── Haraj override from riyadh_meta.json (fresher May 2026 snapshot) ──────
        _rmeta_path = os.path.join(os.path.dirname(PROC), "models", "riyadh_meta.json")
        _haraj_lu = {}
        if os.path.exists(_rmeta_path):
            try:
                with open(_rmeta_path) as _rf:
                    _rmeta = json.load(_rf)
                _haraj_lu = _rmeta.get("haraj_district_lookup", {})
            except Exception:
                pass
        if _haraj_lu:
            _haraj_cols = ["haraj_listing_count","haraj_median_psqm","haraj_p25_psqm","haraj_p75_psqm","haraj_iqr_psqm"]
            for _col in _haraj_cols:
                if _col not in self._district_feat_df.columns:
                    self._district_feat_df[_col] = float("nan")
            for _idx, _drow in self._district_feat_df.iterrows():
                _dname = _drow["district_ar"]
                if _dname in _haraj_lu:
                    _lu_row = _haraj_lu[_dname]
                    for _col in _haraj_cols:
                        _val = _lu_row.get(_col)
                        if _val is not None:
                            self._district_feat_df.at[_idx, _col] = float(_val)
            # Recompute haraj_asking_premium from updated median
            if "haraj_median_psqm" in self._district_feat_df.columns and "district_median_price_sqm" in self._district_feat_df.columns:
                _dmed = self._district_feat_df["district_median_price_sqm"].replace(0, float("nan"))
                self._district_feat_df["haraj_asking_premium"] = self._district_feat_df["haraj_median_psqm"] / _dmed
            print(f"  Haraj override: {len(_haraj_lu)} districts from riyadh_meta.json")

        # Separate lookup for target-encoded district features (excluded from feat_cols
        # to avoid leakage during training, but valid and important at inference time).
        _enc_cols = [c for c in [
            "district_encoded", "district_type_encoded",
            "district_apt_encoded", "district_recent_encoded", "district_apt_recent_encoded",
        ] if c in df.columns]
        _enc_agg = df.groupby("district_ar")[_enc_cols].median()
        self._district_encoded_map: dict = _enc_agg.to_dict("index")  # district_ar → {col: val}
        # KDTree on district centroids — use district_centroids.csv + a hardcoded
        # fallback table for 13 districts whose geocodes defaulted to city-centre
        # (24.7136, 46.6753) in features_riyadh.csv and are absent from the CSV.
        cent_csv = os.path.join(PROC, "district_centroids.csv")
        _cent_override: dict = {}
        if os.path.exists(cent_csv):
            import pandas as _pd2
            _c = _pd2.read_csv(cent_csv)
            _cent_override = {
                row["district_ar"]: (float(row["district_lat"]), float(row["district_lon"]))
                for _, row in _c.iterrows()
                if not (_pd2.isna(row["district_lat"]) or _pd2.isna(row["district_lon"]))
            }

        # Hardcoded real centroids for districts missing from district_centroids.csv
        # (verified against Google Maps / OSM district polygons)
        _HARDCODED: dict = {
            "ظهره العودة غرب":  (24.7389, 46.5165),  # Zahrat Al Awda West — Diriyah fringe
            "ظهرة العودة شرق":  (24.7530, 46.5454),  # Zahrat Al Awda East — Diriyah fringe
            "الشفاء":           (24.5608, 46.6930),  # Al Shifa — south Riyadh
            "الصفاء":           (24.6717, 46.7700),  # Al Safa — east-central Riyadh
            "المنصورة":         (24.6091, 46.7444),  # Al Mansourah — south Riyadh
            "الوسام":           (24.6275, 46.6800),  # Al Wisam — south Riyadh
            "وادي لبن":         (24.6200, 46.5650),  # Wadi Laban — west Riyadh
            "الرابية":          (24.8000, 46.6600),  # Al Rabi'a — north Riyadh
            "السحاب":           (24.7700, 46.7150),  # Al Sahab — north-east Riyadh
            "الملك سلمان":      (24.7630, 46.6440),  # King Salman — north Riyadh (KAFD area)
            "المرجان":          (24.6600, 46.7500),  # Al Murjan — east Riyadh
            "سدرة":             (24.7650, 46.7000),  # Sidra — north Riyadh
            "أخرى":             (99.0, 0.0),          # "Other" catch-all — sentinel far outside Riyadh so it never wins
        }
        _cent_override.update(_HARDCODED)

        _DEFAULT_LAT, _DEFAULT_LON = 24.7136, 46.6753
        cent_lats, cent_lons = [], []
        _fixed = 0
        for _, row in self._district_feat_df.iterrows():
            d_ar = row["district_ar"]
            feat_lat = float(row.get("district_lat", _DEFAULT_LAT) or _DEFAULT_LAT)
            feat_lon = float(row.get("district_lon", _DEFAULT_LON) or _DEFAULT_LON)
            is_default = (abs(feat_lat - _DEFAULT_LAT) < 0.001 and abs(feat_lon - _DEFAULT_LON) < 0.001)
            if is_default and d_ar in _cent_override:
                cent_lats.append(_cent_override[d_ar][0])
                cent_lons.append(_cent_override[d_ar][1])
                _fixed += 1
            else:
                cent_lats.append(feat_lat)
                cent_lons.append(feat_lon)

        cents = np.array(list(zip(cent_lats, cent_lons)), dtype=np.float64)
        self._district_centroid_tree = cKDTree(cents)
        self._district_names = self._district_feat_df["district_ar"].tolist()
        print(f"  District stats: {len(self._district_stats)} districts | predict features: {len(feat_cols)} | centroid-corrected: {_fixed}")

    # ── Lookup ────────────────────────────────────────────────────────────────

    def lookup(self, lat: float, lon: float) -> dict:
        """Return all Riyadh spatial features for a (lat, lon) coordinate."""
        feats = {}
        pt = np.array([[lat, lon]])
        pt_rad = np.radians(pt)
        r_500  = 500.0  / 6_371_000
        r_1km  = 1000.0 / 6_371_000

        # Metro
        d, idx = self._metro_tree.query(pt, k=1)
        feats["dist_metro_m"]          = float(d[0]) * 111_000
        feats["log_dist_metro_m"]      = float(np.log1p(feats["dist_metro_m"]))
        feats["nearest_metro_line_num"] = int(_LINE_ORDER.get(self._metro_lines[int(idx[0])], 0))
        feats["nearest_metro_type_cd"] = int(self._metro_types[int(idx[0])])
        feats["metro_stations_1km"]    = int(self._metro_ball.query_radius(pt_rad, r=r_1km, count_only=True)[0])
        d1, _ = self._line1_tree.query(pt, k=1)
        feats["dist_metro_line1_m"]    = float(d1[0]) * 111_000

        # Bus
        d_b, _ = self._bus_tree.query(pt, k=1)
        feats["dist_bus_m"]     = float(d_b[0]) * 111_000
        feats["log_dist_bus_m"] = float(np.log1p(feats["dist_bus_m"]))
        feats["bus_stops_500m"] = int(self._bus_ball.query_radius(pt_rad, r=r_500, count_only=True)[0])
        feats["brt_stops_500m"] = int(
            self._shelter_ball.query_radius(pt_rad, r=r_500, count_only=True)[0]
            if self._shelter_ball else 0
        )

        # Intersections
        d_i, _ = self._int_tree.query(pt, k=1)
        feats["dist_major_intersection_m"] = float(d_i[0]) * 111_000
        feats["log_dist_intersection_m"]   = float(np.log1p(feats["dist_major_intersection_m"]))
        feats["intersections_1km"]         = int(self._int_ball.query_radius(pt_rad, r=r_1km, count_only=True)[0])
        feats["intersections_500m"]        = int(self._int_ball.query_radius(pt_rad, r=r_500, count_only=True)[0])

        # Commercial
        feats["commercial_count_1km"] = int(self._comm_ball_all.query_radius(pt_rad, r=r_1km, count_only=True)[0])
        for bname, bt in self._comm_balls.items():
            feats[f"{bname}_count_1km"] = int(bt.query_radius(pt_rad, r=r_1km, count_only=True)[0]) if bt else 0
        feats["commercial_density_score"] = (
            feats.get("hypermarket_count_1km", 0) * 3
            + feats.get("supermarket_count_1km", 0) * 2
            + feats.get("bank_count_1km", 0)
            + feats.get("restaurant_count_1km", 0)
            + feats.get("hotel_count_1km", 0)
        )

        # QoL POIs (mosque, mall, school, hospital, park, entertainment)
        for poi_name, tree in self._poi_trees.items():
            d_poi, _ = tree.query(pt, k=1)
            dist_m = float(d_poi[0]) * 111_000
            feats[f"dist_{poi_name}_m"]     = dist_m
            feats[f"log_dist_{poi_name}_m"] = float(np.log1p(dist_m))
            feats[f"{poi_name}_count_500m"] = int(
                self._poi_balls[poi_name].query_radius(pt_rad, r=r_500, count_only=True)[0]
            )

        # Air quality (IDW from 2 nearest stations)
        d_aq, idx_aq = self._aq_tree.query(pt, k=min(2, len(self._aq_arr)))
        d_m = d_aq.ravel() * 111_000
        d_m = np.where(d_m < 1, 1, d_m)
        weights = 1.0 / d_m
        for comp in self._aq_components:
            if comp in self._aq_means_df.columns:
                vals = self._aq_means_df.iloc[idx_aq.ravel()][comp].values
                feats[f"{comp.lower()}_nearest_mean"] = float(np.average(vals, weights=weights))
        feats["dist_air_station_m"] = float(d_aq.ravel()[0]) * 111_000

        return feats

    def get_district_stats(self) -> dict:
        """Return dict of district_ar → per-district metric medians."""
        return self._district_stats

    def predict_features(self, lat: float, lon: float,
                         property_type: str, year: int, quarter: int) -> dict:
        """
        Build a complete 72-feature dict for Riyadh model prediction.
        Combines live spatial lookup + district medians + type flags + macro.
        """
        feats: dict = {}

        # 1. Type flags
        feats["is_apartment"]      = int(property_type == "شقة")
        feats["is_villa"]          = int(property_type == "فيلا")
        feats["is_residential_plot"] = int(property_type == "قطعة أرض-سكنى")
        feats["is_building"]       = int(property_type == "عمارة")

        # 2. Time features
        feats["sale_year"]         = float(year)
        feats["sale_quarter_sin"]  = float(np.sin(2 * np.pi * quarter / 4))
        feats["sale_quarter_cos"]  = float(np.cos(2 * np.pi * quarter / 4))
        quarter_id = year * 10 + quarter

        # 3. Nearest district baseline features
        _matched_district: str = ""
        if self._district_centroid_tree is not None and self._district_feat_df is not None:
            _, idx = self._district_centroid_tree.query([[lat, lon]], k=1)
            row = self._district_feat_df.iloc[int(idx[0])].to_dict()
            _matched_district = str(row.get("district_ar", ""))
            feats.update({
                k: v for k, v in row.items()
                if k != "district_ar"
                and (not isinstance(v, float) or not np.isnan(v))
            })

        # 4. Live spatial features from the exact lat/lon (override district centroid)
        live = self.lookup(lat, lon)
        feats.update(live)

        # 5. Log deed count (use district median as a proxy for a typical transaction)
        feats.setdefault("log_deed_count", float(np.log1p(5)))

        # 6. District target encoding — look up per-district medians.
        # These were excluded from feat_cols to avoid leakage during training but
        # are valid and important at inference time (they carry district price signal).
        _GLOBAL_ENC  = 7.8
        _GLOBAL_TYPE = 7.8
        if _matched_district and hasattr(self, "_district_encoded_map"):
            _enc = self._district_encoded_map.get(_matched_district, {})
            feats["district_encoded"]            = float(_enc.get("district_encoded",            _GLOBAL_ENC))
            feats["district_type_encoded"]       = float(_enc.get("district_type_encoded",       _GLOBAL_TYPE))
            feats["district_recent_encoded"]     = float(_enc.get("district_recent_encoded",     _GLOBAL_ENC))
            # Always use apartment-specific encoding regardless of property type —
            # matches training where district_apt_encoded is a district-level feature (same value for all rows in district)
            feats["district_apt_encoded"]        = float(_enc.get("district_apt_encoded",        _GLOBAL_ENC))
            feats["district_apt_recent_encoded"] = float(_enc.get("district_apt_recent_encoded", _GLOBAL_ENC))
        else:
            feats.setdefault("district_encoded",            _GLOBAL_ENC)
            feats.setdefault("district_type_encoded",       _GLOBAL_TYPE)
            feats.setdefault("district_apt_encoded",        _GLOBAL_ENC)
            feats.setdefault("district_recent_encoded",     _GLOBAL_ENC)
            feats.setdefault("district_apt_recent_encoded", _GLOBAL_ENC)

        # 7. Connectivity score (recalculate from live spatial if scaler params available)
        # Keep the district median value from step 3 unless live override is possible
        feats.setdefault("riyadh_connectivity_score", 50.0)

        return feats
