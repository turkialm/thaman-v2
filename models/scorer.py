"""
THAMAN Property Price Scorer
============================
Loads XGBoost v2 (71 features, target-encoded bldgclass).
When thaman_stack.pkl is present, uses XGB + LGB + Ridge meta-learner.

Usage:
    from models.scorer import ThamanScorer
    scorer = ThamanScorer()
    price = scorer.predict_single(
        latitude=40.7128, longitude=-74.0060,
        gross_square_feet=1200, building_age=35,
        bldgclass_encoded=13.39, borough=1, ...
    )
"""

import json, os
import joblib
import numpy as np
import polars as pl
import xgboost as xgb

_DIR = os.path.dirname(os.path.abspath(__file__))


class ThamanScorer:
    # ── Borough / tier constants for adaptive confidence ────────────
    _BOROUGH_INT_TO_NAME = {
        1: "Manhattan", 2: "Bronx", 3: "Brooklyn",
        4: "Queens",    5: "Staten Island",
    }
    _TIER_BINS = [
        (0,           500_000,    "<$500K"),
        (500_000,   1_000_000,   "$500K–1M"),
        (1_000_000,  3_000_000,  "$1M–3M"),
        (3_000_000, 10_000_000,  "$3M–10M"),
    ]

    def __init__(self):
        meta_path   = os.path.join(_DIR, "meta.json")
        model_path  = os.path.join(_DIR, "xgboost_model.json")
        stack_path  = os.path.join(_DIR, "thaman_stack.pkl")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.model = xgb.Booster()
        self.model.load_model(model_path)

        self.feature_names    = self.meta["feature_names"]
        self.winsorize        = self.meta["winsorize_p99"]
        self.acris_medians    = self.meta["acris_medians"]
        self.bldgclass_means    = self.meta["bldgclass_means"]
        self.borough_bldg_means = self.meta["borough_bldg_means"]
        self.global_mean_log    = self.meta["global_mean_log"]

        # Stack (multi-model + Ridge meta) — optional
        self._stack = None
        if os.path.exists(stack_path):
            self._stack = joblib.load(stack_path)
            # v6/v5: xgb_a + xgb_b + lgb + cat; v4: lgb + cat; v3: lgb only
            ver = self._stack.get("version", "v4")
            if ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11"):
                label = f"XGB-A + XGB-B + LGB + CAT + Ridge ({ver})"
            elif "cat" in self._stack:
                label = "XGB + LGB + CAT + Ridge (v4)"
            else:
                label = "XGB + LGB + Ridge (v3)"
            print(f"  [scorer] Stack loaded: {label}")
        else:
            print(f"  [scorer] Stack not found — using XGBoost only")

        # Luxury sub-model (Manhattan $3M+) — optional
        luxury_path = os.path.join(_DIR, "luxury_model.json")
        self._luxury = None
        self._luxury_threshold = self.meta.get("luxury_threshold", 2_000_000)
        if self.meta.get("has_luxury_model") and os.path.exists(luxury_path):
            self._luxury = xgb.Booster()
            self._luxury.load_model(luxury_path)
            print(f"  [scorer] Luxury model loaded (blend ≥ ${self._luxury_threshold/1e6:.0f}M)")

        # Riyadh model (city = 'riyadh')
        riyadh_stack_path = os.path.join(_DIR, "riyadh_stack.pkl")
        riyadh_meta_path  = os.path.join(_DIR, "riyadh_meta.json")
        self._riyadh_stack = None
        self._riyadh_meta  = {}
        if os.path.exists(riyadh_stack_path) and os.path.exists(riyadh_meta_path):
            self._riyadh_stack = joblib.load(riyadh_stack_path)
            with open(riyadh_meta_path) as f:
                self._riyadh_meta = json.load(f)
            print(f"  [scorer] Riyadh stack loaded — {self._riyadh_meta.get('n_features',0)} features")
        self._riyadh_shap_explainer = None   # lazy-built on first SHAP call
        self._nyc_shap_explainer    = None   # lazy-built on first NYC SHAP call

    # ── Fast single-row path: direct numpy, no Polars overhead ─────
    def _prepare_single(self, feat_dict: dict) -> np.ndarray:
        """
        Build float32[1, n_features] directly from a dict — bypasses Polars
        DataFrame creation for single-row inference (~3ms faster per request).
        Applies winsorize clipping and ACRIS median fill identical to _prepare().
        """
        vals = np.empty(len(self.feature_names), dtype=np.float32)
        for i, f in enumerate(self.feature_names):
            v = feat_dict.get(f)
            # Fill null / NaN with ACRIS median if available, else 0
            if v is None or (isinstance(v, float) and v != v):
                v = self.acris_medians.get(f)
                v = 0.0 if (v is None or (isinstance(v, float) and v != v)) else float(v)
            else:
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = 0.0
            # Winsorize
            cap = self.winsorize.get(f)
            if cap is not None and v > cap:
                v = float(cap)
            vals[i] = v
        return vals.reshape(1, -1)

    # ── Internal: preprocess feature matrix ────────────────────────
    def _prepare(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self.feature_names)
        clip_exprs = [
            pl.col(col).clip(upper_bound=cap)
            for col, cap in self.winsorize.items()
            if col in X.columns
        ]
        if clip_exprs:
            X = X.with_columns(clip_exprs)
        fill_exprs = [
            pl.col(col).fill_null(float(med)).fill_nan(float(med))
            for col, med in self.acris_medians.items()
            if col in X.columns
        ]
        if fill_exprs:
            X = X.with_columns(fill_exprs)
        return X.fill_null(0).fill_nan(0.0).to_numpy().astype(np.float32)

    # ── Core prediction from pre-built numpy array ─────────────────
    def _predict_from_array(self, Xv: np.ndarray) -> np.ndarray:
        """
        Run stacked ensemble on a pre-built float32 array.
        Used by both predict() (batch) and predict_single() (fast single-row path).
        """
        dmat = xgb.DMatrix(Xv, feature_names=self.feature_names)
        log_xgb = self.model.predict(dmat)

        if self._stack is not None:
            ver = self._stack.get("version", "v4")
            if ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11"):
                log_xa  = self._stack["xgb_a"].predict(Xv).astype(np.float32)
                log_xb  = self._stack["xgb_b"].predict(Xv).astype(np.float32)
                log_lgb = self._stack["lgb"].predict(Xv).astype(np.float32)
                log_cat = self._stack["cat"].predict(Xv).astype(np.float32)
                S = np.column_stack([log_xa, log_xb, log_lgb, log_cat])
            else:
                log_lgb = self._stack["lgb"].predict(Xv).astype(np.float32)
                cols = [log_xgb, log_lgb]
                if "cat" in self._stack:
                    log_cat = self._stack["cat"].predict(Xv).astype(np.float32)
                    cols.append(log_cat)
                S = np.column_stack(cols)
            log_final = self._stack["meta"].predict(S).astype(np.float32)
        else:
            log_final = log_xgb

        stack_prices = np.expm1(log_final)

        if self._luxury is not None:
            log_lux    = self._luxury.predict(dmat).astype(np.float32)
            lux_prices = np.expm1(log_lux)
            lo         = float(self._luxury_threshold)
            hi         = lo * 2.0
            alpha      = np.clip((stack_prices - lo) / (hi - lo), 0.0, 1.0)
            return (1.0 - alpha) * stack_prices + alpha * lux_prices

        return stack_prices

    # ── Main prediction method ──────────────────────────────────────
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Predict prices for a DataFrame. Returns USD array."""
        return self._predict_from_array(self._prepare(df))

    # ── Single property convenience method ─────────────────────────
    def predict_single(self, **kwargs) -> dict:
        """
        Predict price for one property. Pass feature values as keyword args.
        """
        defaults = {feat: 0.0 for feat in self.feature_names}
        for col in self.acris_medians:
            defaults[col] = None  # polars null
        # Target-encoded features: default to global mean (not 0)
        _gml = self.meta.get("global_mean_log", 0.0)
        for enc_feat in ("nta_encoded", "nta_bldg_encoded", "zip_encoded", "zip_bldg_encoded"):
            if enc_feat in defaults:
                defaults[enc_feat] = _gml
        # Derived interaction features that depend on nta_encoded
        if "nta_rel_price" in defaults:
            defaults["nta_rel_price"] = 1.0           # bldgclass/nta ≈ 1 at global mean
        if "sqft_x_nta_enc" in defaults:
            defaults["sqft_x_nta_enc"] = 0.0          # will be overridden when sqft is passed
        if "bldg_age_x_nta" in defaults:
            defaults["bldg_age_x_nta"] = 0.0
        # Convert any np.nan in kwargs to None
        clean_kwargs = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in kwargs.items()
        }
        defaults.update(clean_kwargs)

        # Fast numpy path — skip Polars DataFrame creation for single-row inference
        Xv    = self._prepare_single(defaults)         # float32[1, n_features]
        price = float(self._predict_from_array(Xv)[0]) # returns array; take scalar

        if self._stack is not None and "stack" in self.meta:
            medape = self.meta["stack"]["medape_holdout"]
            r2     = self.meta["stack"]["r2_holdout"]
            ver    = self._stack.get("version", "v4") if self._stack else "v4"
            model_label = f"Stack {ver} · 4-Model Ensemble" if ver in ("v5","v6","v7","v8","v9","v10","v11") else "XGBoost + LightGBM Stack"
        else:
            medape = self.meta["xgboost"]["medape_test"]
            r2     = self.meta["xgboost"]["r2_test"]
            model_label = "XGBoost v2"

        conf    = self._adaptive_confidence(price, int(kwargs.get("borough", 0)))
        seg_med = conf["segment_medape"]
        mult    = seg_med / 100.0

        result = {
            "predicted_price":    round(price),
            "confidence_low":     round(price * (1.0 - mult)),   # segment-adaptive
            "confidence_high":    round(price * (1.0 + mult)),   # segment-adaptive
            "confidence_score":   conf["confidence_score"],
            "confidence_grade":   conf["confidence_grade"],
            "segment_medape_pct": seg_med,
            "tier_label":         conf["tier_label"],
            "model":              model_label,
            "r2_test":            r2,
            "medape_test_pct":    medape,   # global value kept for backward compat
        }

        # ── NYC SHAP drivers ──────────────────────────────────────────────────
        try:
            stk = self._stack  # the loaded stack dict with "cat" key
            if stk and "cat" in stk:
                if self._nyc_shap_explainer is None:
                    import shap as _shap
                    self._nyc_shap_explainer = _shap.TreeExplainer(stk["cat"])
                feat_names = self.meta.get("feature_names", self.feature_names)
                sv = np.array(
                    self._nyc_shap_explainer.shap_values(Xv), dtype=np.float32
                ).flatten()
                _meta_learner = stk.get("meta")
                meta_coeffs = (
                    _meta_learner.coef_.tolist()
                    if _meta_learner is not None and hasattr(_meta_learner, "coef_")
                    else []
                )
                # For v5+ stacks: [xgb_a, xgb_b, lgb, cat]; cat is index 3
                ver = stk.get("version", "v4")
                if ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11") and len(meta_coeffs) >= 4:
                    w_cat = float(meta_coeffs[3])
                elif len(meta_coeffs) >= 3:
                    w_cat = float(meta_coeffs[2])
                else:
                    w_cat = 0.95
                sv_scaled = sv * w_cat
                top_k = 10
                indices = np.argsort(np.abs(sv_scaled))[::-1][:top_k]
                top_drivers = []
                for i in indices:
                    if i >= len(feat_names) or i >= Xv.shape[1]:
                        continue
                    fname = feat_names[i]
                    top_drivers.append({
                        "feature":     fname,
                        "value":       float(Xv[0, i]),
                        "impact":      float(sv_scaled[i]),
                        "direction":   "positive" if sv_scaled[i] > 0 else "negative",
                        "description": self._NYC_FEAT_LABELS.get(fname, fname.replace("_", " ").title()),
                    })
                result["top_drivers"] = top_drivers
        except Exception as _shap_err:
            result["top_drivers"] = []

        return result

    # ── Segment-adaptive confidence ────────────────────────────────
    def _adaptive_confidence(self, price: float, borough: int) -> dict:
        """
        Returns a confidence score (0–100) and grade (A/B/C/D) based on
        the per-segment MedAPE for the given borough and price tier.
        Uses the WORSE (higher) of the two segment MedAPEs so the interval
        is conservatively wide when two risk factors coincide.
        """
        global_medape = self.meta["stack"]["medape_holdout"]
        borough_name  = self._BOROUGH_INT_TO_NAME.get(borough, "Unknown")

        borough_medape = (
            self.meta.get("segment_by_borough", {})
                .get(borough_name, {}).get("medape", global_medape)
        )

        tier_label,  tier_medape = "$3M–10M", global_medape
        for lo, hi, label in self._TIER_BINS:
            if lo <= price < hi:
                tier_label  = label
                tier_medape = (
                    self.meta.get("segment_by_tier", {})
                        .get(label, {}).get("medape", global_medape)
                )
                break

        segment_medape   = max(borough_medape, tier_medape)
        confidence_score = max(0, min(100, round(100 - segment_medape)))
        if   confidence_score >= 85: grade = "A"
        elif confidence_score >= 75: grade = "B"
        elif confidence_score >= 65: grade = "C"
        else:                         grade = "D"

        return {
            "segment_medape":   round(segment_medape, 2),
            "confidence_score": confidence_score,
            "confidence_grade": grade,
            "tier_label":       tier_label,
            "borough_name":     borough_name,
        }

    # ── SHAP explanation for one property ──────────────────────────
    def explain(self, df: pl.DataFrame, top_n: int = 10) -> pl.DataFrame:
        """
        Returns SHAP-based feature contributions for each row.
        Requires: pip install shap
        """
        import shap
        X = df.select(self.feature_names)
        clip_exprs = [
            pl.col(col).clip(upper_bound=cap)
            for col, cap in self.winsorize.items()
            if col in X.columns
        ]
        if clip_exprs:
            X = X.with_columns(clip_exprs)
        fill_exprs = [
            pl.col(col).fill_null(float(med)).fill_nan(float(med))
            for col, med in self.acris_medians.items()
            if col in X.columns
        ]
        if fill_exprs:
            X = X.with_columns(fill_exprs)
        X_np = X.fill_null(0).fill_nan(0.0).to_numpy()

        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_np)
        return pl.DataFrame(
            {col: shap_values[:, i] for i, col in enumerate(self.feature_names)}
        )


    # Human-readable labels for NYC features used in SHAP display
    _NYC_FEAT_LABELS = {
        "walk_score":            "Walk Score",
        "walk_score_proxy":      "Walk Score",
        "subway_dist_m":         "Distance to Subway",
        "dist_subway_m":         "Distance to Subway",
        "subway_count_500m":     "Subway Stations (500m)",
        "bus_count_500m":        "Bus Stops (500m)",
        "dist_bus_m":            "Distance to Bus Stop",
        "park_dist_m":           "Distance to Park",
        "dist_park_m":           "Distance to Park",
        "school_dist_m":         "Distance to School",
        "dist_school_m":         "Distance to School",
        "dist_elem_school_m":    "Distance to Elem. School",
        "dist_hospital_m":       "Distance to Hospital",
        "dist_waterfront_m":     "Distance to Waterfront",
        "dist_bike_lane_m":      "Distance to Bike Lane",
        "dist_express_subway_m": "Distance to Express Subway",
        "complaint_density":     "311 Complaint Density",
        "crime_density":         "Crime Rate (NTA)",
        "crime_rate_nta":        "Crime Rate (NTA)",
        "noise_density_nta":     "Noise Complaints (NTA)",
        "livability_complaint_rate": "311 Livability Rate",
        "nta_median_income":     "Neighborhood Median Income",
        "median_income_nta":     "Neighborhood Median Income",
        "nta_price_sqft_median": "NTA Median Price/sqft",
        "nta_median_psf":        "NTA Median Price/sqft",
        "bldg_age":              "Building Age (years)",
        "building_age":          "Building Age (years)",
        "floors":                "Number of Floors",
        "numfloors":             "Number of Floors",
        "units_total":           "Total Units in Building",
        "residential_units":     "Residential Units",
        "gross_square_feet":     "Gross Sqft",
        "land_square_feet":      "Land Sqft",
        "commercial_units":      "Commercial Units",
        "mortgage_rate":         "30yr Mortgage Rate",
        "mortgage_rate_30yr":    "30yr Mortgage Rate",
        "log_gross_sqft":        "Log Gross Sqft",
        "prior_sale_price":      "Prior Sale Price",
        "bldgclass_encoded":     "Building Class",
        "borough_bldg_encoded":  "Borough × Building Class",
        "nta_encoded":           "Neighborhood (NTA)",
        "nta_bldg_encoded":      "NTA × Building Class",
        "borough_encoded":       "Borough",
        "borough":               "Borough",
        "sale_year":             "Sale Year",
        "sale_month_sin":        "Sale Season",
        "sale_month_cos":        "Sale Season",
        "waterfront_dist_m":     "Distance to Waterfront",
        "bike_lane_dist_m":      "Distance to Bike Lane",
        "poi_count_500m":        "Points of Interest (500m)",
        "airbnb_count_500m":     "Airbnb Density (500m)",
        "has_elevator":          "Has Elevator",
        "is_condo":              "Is Condo",
        "is_multifamily":        "Is Multi-Family",
        "is_single_fam":         "Is Single-Family",
        "is_mixed_use":          "Is Mixed-Use",
        "is_manhattan":          "Is Manhattan",
        "crime_x_manhattan":     "Crime × Manhattan",
        "nearest_station_is_express": "Nearest Station is Express",
        "district_avg_score":    "School District Score",
        "population_2020":       "NTA Population",
        "builtfar":              "Built FAR",
        "far_utilization":       "FAR Utilization",
        "has_prior_sale":        "Has Prior Sale Record",
        "price_appreciation":    "Price Appreciation",
        "years_since_prior_sale":"Years Since Prior Sale",
        "renovated_since_2018":  "Renovated Since 2018",
        "rat_density_nta":       "Rodent Density (NTA)",
        "heat_density_nta":      "Heat Complaints (NTA)",
        "hpd_severity_score_zip":"HPD Violation Severity (ZIP)",
        "nearest_station_is_cbd":"Nearest Station: CBD",
        "nearest_station_route_count": "Station Route Count",
        "log_dist_subway_m":     "Log Distance to Subway",
        "log_dist_park_m":       "Log Distance to Park",
        "log_dist_school_m":     "Log Distance to School",
        "log_dist_hospital_m":   "Log Distance to Hospital",
        "log_dist_bus_m":        "Log Distance to Bus Stop",
        "log_dist_waterfront_m": "Log Distance to Waterfront",
        "log_dist_bike_lane_m":  "Log Distance to Bike Lane",
    }

    # Human-readable labels for Riyadh features used in SHAP display
    _RIYADH_FEAT_LABELS = {
        "district_median_price_sqm":    "District median price",
        "district_encoded":             "District (target-encoded)",
        "district_price_vs_city_avg":   "Price vs city average",
        "district_price_trend_slope":   "District price trend",
        "district_transaction_volume":  "District transaction volume",
        "district_median_price_apt_sqm":"Apartment median price",
        "district_type_encoded":        "District type (encoded)",
        "rei_residential_qtr_idx":      "Real estate price index",
        "rei_apt_idx":                  "Apartment price index",
        "rei_yoy_change":               "Price index YoY change",
        "rei_qoq_change":               "Price index QoQ change",
        "avg_saudi_salary_yr":          "Average Saudi salary",
        "salary_yoy_change":            "Salary YoY change",
        "sale_year":                    "Sale year",
        "sale_quarter_sin":             "Quarter (seasonal)",
        "sale_quarter_cos":             "Quarter (seasonal)",
        "log_deed_count":               "Transaction deed count",
        "dist_metro_m":                 "Distance to metro station",
        "log_dist_metro_m":             "Distance to metro (log)",
        "metro_stations_1km":           "Metro stations within 1 km",
        "nearest_metro_line_num":       "Nearest metro line",
        "nearest_metro_type_cd":        "Metro station type",
        "dist_metro_line1_m":           "Distance to Metro Line 1",
        "dist_bus_m":                   "Distance to bus stop",
        "log_dist_bus_m":               "Distance to bus stop (log)",
        "bus_stops_500m":               "Bus stops within 500 m",
        "brt_stops_500m":               "BRT stops within 500 m",
        "commercial_count_1km":         "Commercial services (1 km)",
        "commercial_density_score":     "Commercial density score",
        "district_commercial_count":    "District commercial count",
        "district_commercial_mix":      "Commercial mix diversity",
        "hypermarket_count_1km":        "Hypermarkets (1 km)",
        "supermarket_count_1km":        "Supermarkets (1 km)",
        "bank_count_1km":               "Banks (1 km)",
        "restaurant_count_1km":         "Restaurants (1 km)",
        "hotel_count_1km":              "Hotels (1 km)",
        "gas_station_count_1km":        "Gas stations (1 km)",
        "no2_nearest_mean":             "NO₂ air pollution",
        "so2_nearest_mean":             "SO₂ air pollution",
        "pm10_nearest_mean":            "PM10 particulate matter",
        "o3_nearest_mean":              "Ozone (O₃) level",
        "dist_air_station_m":           "Distance to air station",
        "air_quality_score":            "Air quality score",
        "riyadh_connectivity_score":    "Connectivity score",
        "dist_mosque_m":                "Distance to mosque",
        "log_dist_mosque_m":            "Distance to mosque (log)",
        "mosque_count_500m":            "Mosques within 500 m",
        "dist_mall_m":                  "Distance to mall",
        "log_dist_mall_m":              "Distance to mall (log)",
        "mall_count_500m":              "Malls within 500 m",
        "dist_school_m":                "Distance to school",
        "log_dist_school_m":            "Distance to school (log)",
        "school_count_500m":            "Schools within 500 m",
        "dist_hospital_m":              "Distance to hospital",
        "log_dist_hospital_m":          "Distance to hospital (log)",
        "hospital_count_500m":          "Hospitals within 500 m",
        "dist_park_m":                  "Distance to park",
        "log_dist_park_m":              "Distance to park (log)",
        "park_count_500m":              "Parks within 500 m",
        "dist_entertain_m":             "Distance to entertainment",
        "log_dist_entertain_m":         "Distance to entertainment (log)",
        "entertain_count_500m":         "Entertainment venues (500 m)",
        "aqar_median_size_sqm":         "Median rental unit size",
        "aqar_median_bedrooms":         "Median rental bedrooms",
        "aqar_median_property_age":     "Median rental property age",
        "aqar_rent_per_sqm":            "Median rent per sqm",
        # v2 features
        "district_enc_oof":             "District price (historical avg)",
        "district_apt_enc_oof":         "District apartment price (historical)",
        "district_lookback_mean":       "District price history",
        "district_lookback_apt_mean":   "District apartment price history",
        "city_quarter_mean":            "City market level",
        "bayut_asking_psqm":            "Bayut listing price signal",
        "is_apartment":                 "Property: apartment",
        "is_villa":                     "Property: villa",
        "is_residential_plot":          "Property: residential plot",
        "is_building":                  "Property: building",
        "district_lat":                 "District latitude",
        "district_lon":                 "District longitude",
        "dist_major_intersection_m":    "Distance to intersection",
        "log_dist_intersection_m":      "Distance to intersection (log)",
        "intersections_1km":            "Intersections within 1 km",
        "intersections_500m":           "Intersections within 500 m",
    }

    # ── Riyadh prediction ───────────────────────────────────────────
    def predict_riyadh(self, **kwargs) -> dict:
        """
        Predict SAR/sqm for a Riyadh property using the Riyadh stack.
        kwargs should contain all 72 Riyadh model features.
        Returns dict with predicted_price_sqm, medape_pct, r2_test, model.
        """
        if self._riyadh_stack is None:
            raise RuntimeError("Riyadh model not loaded — run train_stack_riyadh_v1.py first.")

        feat_names = self._riyadh_meta.get("feature_names", [])
        X = np.array(
            [float(kwargs.get(f, 0.0) or 0.0) for f in feat_names],
            dtype=np.float32
        ).reshape(1, -1)

        stk  = self._riyadh_stack
        preds = np.column_stack([
            stk["xgb"].predict(X).astype(np.float32),
            stk["lgb"].predict(X).astype(np.float32),
            stk["cat"].predict(X).astype(np.float32),
        ])
        log_pred = stk["meta"].predict(preds)[0]
        price_sqm = float(np.expm1(log_pred))

        # ── SHAP feature drivers (lazy-build CatBoost explainer) ─────────
        top_drivers = []
        try:
            if self._riyadh_shap_explainer is None:
                import shap as _shap
                self._riyadh_shap_explainer = _shap.TreeExplainer(stk["cat"])
            sv = np.array(self._riyadh_shap_explainer.shap_values(X), dtype=np.float32).flatten()
            # Scale by CatBoost meta-weight (≈0.95) so values represent ensemble contribution
            w_cat = float(self._riyadh_meta.get("meta_coefficients", [0.2, -0.15, 0.95])[2])
            sv_scaled = sv * w_cat
            top_k = 10
            indices = np.argsort(np.abs(sv_scaled))[::-1][:top_k]
            top_drivers = [
                {
                    "feature":     feat_names[i],
                    "value":       float(X[0, i]),
                    "impact":      float(sv_scaled[i]),
                    "direction":   "positive" if sv_scaled[i] > 0 else "negative",
                    "description": self._RIYADH_FEAT_LABELS.get(
                        feat_names[i],
                        feat_names[i].replace("_", " ").title()
                    ),
                }
                for i in indices
                if i < len(feat_names)
            ]
        except Exception:
            top_drivers = []

        return {
            "predicted_price_sqm": price_sqm,
            "top_drivers":         top_drivers,
            "medape_pct":  self._riyadh_meta.get("holdout_medape_pct", 23.43),
            "r2_test":     self._riyadh_meta.get("holdout_r2", 0.675),
            "model":       self._riyadh_meta.get("model_version", "riyadh_stack_v1"),
        }


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = ThamanScorer()

    # Target-encode "D4" for Manhattan manually
    bc_enc = scorer.bldgclass_means.get("D4", scorer.global_mean_log)
    bb_enc = scorer.borough_bldg_means.get("1_D", scorer.global_mean_log)

    result = scorer.predict_single(
        latitude=40.7589,
        longitude=-73.9851,
        gross_square_feet=950,
        building_age=40,
        bldgclass_encoded=bc_enc,
        borough_bldg_encoded=bb_enc,
        borough=1,
        numfloors=12,
        dist_subway_m=250,
        dist_park_m=180,
        poi_count_500m=850,
        median_income_nta=120000,
        airbnb_count_500m=45,
    )
    print("Example prediction (Manhattan, 950 sqft D4 apartment):")
    print(f"  Predicted:  ${result['predicted_price']:,}")
    print(f"  Range:      ${result['confidence_low']:,} – ${result['confidence_high']:,}")
    print(f"  Model R²:   {result['r2_test']}")
    print(f"  MedAPE:     {result['medape_test_pct']}%")
