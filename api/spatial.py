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
import numpy as np
import pandas as pd
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
        self._load_mortgage_rate()
        print("[SpatialLookup] All data loaded. Ready.")

    # ── Loaders ────────────────────────────────────────────────────────

    def _load_nta_stats(self):
        """Precompute per-NTA medians from features.csv + load NTA boundaries."""
        features = pd.read_csv(os.path.join(PROC, "features.csv"))

        NTA_STAT_COLS = [
            "crime_rate_nta", "noise_density_nta", "livability_complaint_rate",
            "population_2020", "median_income_nta", "borough_income_deviation",
            "school_district", "district_avg_score", "district_school_count",
            "poi_count_500m", "dist_hospital_m", "dist_waterfront_m", "dist_bike_lane_m",
            "builtfar", "residfar", "commfar", "facilfar", "maxallwfar", "far_utilization",
        ]
        available = [c for c in NTA_STAT_COLS if c in features.columns]

        nta_stats = features.groupby("ntacode")[available].median().reset_index()
        self._nta_stats      = nta_stats.set_index("ntacode").to_dict("index")
        self._global_medians = {c: float(features[c].median()) for c in available}

        # Borough-level income for borough_income_deviation fallback
        self._borough_median_income = features.groupby("borough")["median_income_nta"].median().to_dict()

        # NTA GeoDataFrame for point-in-polygon
        self._nta_gdf = gpd.read_file(os.path.join(RAW, "nta_boundaries.geojson"))
        print(f"  NTA stats: {len(self._nta_stats)} NTAs | global medians computed")

    def _load_subway(self):
        subway = pd.read_csv(os.path.join(RAW, "MTA_Subway_Stations_20260308.csv"))
        subway = subway.dropna(subset=["GTFS Latitude", "GTFS Longitude"])
        subway["GTFS Latitude"]  = pd.to_numeric(subway["GTFS Latitude"],  errors="coerce")
        subway["GTFS Longitude"] = pd.to_numeric(subway["GTFS Longitude"], errors="coerce")
        subway = subway.dropna(subset=["GTFS Latitude", "GTFS Longitude"])

        all_coords = subway[["GTFS Latitude", "GTFS Longitude"]].values
        self._subway_tree = cKDTree(all_coords)

        # Express: any station whose Daytime Routes overlap with EXPRESS_ROUTES
        def is_express(routes_str):
            if pd.isna(routes_str):
                return False
            return bool(set(str(routes_str).split()) & EXPRESS_ROUTES)

        express_mask    = subway["Daytime Routes"].apply(is_express)
        express_coords  = subway.loc[express_mask, ["GTFS Latitude", "GTFS Longitude"]].values
        self._express_tree = cKDTree(express_coords) if len(express_coords) > 0 else self._subway_tree
        print(f"  Subway: {len(all_coords)} stations | express: {len(express_coords)}")

    def _load_bus(self):
        bus = pd.read_csv(os.path.join(RAW, "mta_bus_stops.csv"))
        bus = bus.dropna(subset=["latitude", "longitude"])
        bus["latitude"]  = pd.to_numeric(bus["latitude"],  errors="coerce")
        bus["longitude"] = pd.to_numeric(bus["longitude"], errors="coerce")
        bus = bus.dropna(subset=["latitude", "longitude"])
        self._bus_tree = cKDTree(bus[["latitude", "longitude"]].values)
        print(f"  Bus stops: {len(bus)}")

    def _load_schools(self):
        hs   = pd.read_csv(os.path.join(RAW, "schools.csv")).dropna(subset=["latitude", "longitude"])
        elem = pd.read_csv(os.path.join(RAW, "elementary_schools.csv")).dropna(subset=["latitude", "longitude"])
        self._hs_tree   = cKDTree(hs[["latitude",   "longitude"]].values)
        self._elem_tree = cKDTree(elem[["latitude", "longitude"]].values)
        print(f"  Schools: {len(hs)} HS | {len(elem)} elementary")

    def _load_parks(self):
        parks = pd.read_csv(os.path.join(RAW, "parks_with_coords.csv")).dropna(subset=["latitude", "longitude"])
        self._park_tree = cKDTree(parks[["latitude", "longitude"]].values)
        print(f"  Parks: {len(parks)}")

    def _load_airbnb(self):
        airbnb = pd.read_csv(os.path.join(RAW, "airbnb_listings.csv")).dropna(subset=["latitude", "longitude"])
        airbnb["latitude"]  = pd.to_numeric(airbnb["latitude"],  errors="coerce")
        airbnb["longitude"] = pd.to_numeric(airbnb["longitude"], errors="coerce")
        airbnb = airbnb.dropna(subset=["latitude", "longitude"])
        coords_rad = np.radians(airbnb[["latitude", "longitude"]].values)
        self._airbnb_balltree = BallTree(coords_rad, metric="haversine")
        print(f"  Airbnb: {len(airbnb)} listings")

    def _load_mortgage_rate(self):
        mort = pd.read_csv(os.path.join(RAW, "mortgage_rates.csv"))
        self._mortgage_rate = float(mort["mortgage_rate_30yr"].dropna().iloc[-1])
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
            return ntacode if pd.notna(ntacode) else None
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

        # ── NTA-level features ─────────────────────────────────────────
        ntacode  = self._nta_for_point(lat, lon)
        nta_data = self._nta_stats.get(ntacode, {}) if ntacode else {}

        for col in [
            "crime_rate_nta", "noise_density_nta", "livability_complaint_rate",
            "population_2020", "median_income_nta", "borough_income_deviation",
            "school_district", "district_avg_score", "district_school_count",
            "poi_count_500m", "dist_hospital_m", "dist_waterfront_m", "dist_bike_lane_m",
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
