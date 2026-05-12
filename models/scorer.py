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
            if ver in ("v5", "v6", "v7", "v8", "v9", "v10"):
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

    # ── Main prediction method ──────────────────────────────────────
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Predict prices for a DataFrame. Returns USD array.
        Uses stack (XGB + LGB + Ridge) when available, else XGB alone.
        """
        Xv = self._prepare(df)
        dmat = xgb.DMatrix(Xv, feature_names=self.feature_names)
        log_xgb = self.model.predict(dmat)

        if self._stack is not None:
            ver = self._stack.get("version", "v4")
            if ver in ("v5", "v6", "v7", "v8", "v9", "v10"):
                # 4-model diverse stack: XGB-A + XGB-B + LGB + CAT
                log_xa  = self._stack["xgb_a"].predict(Xv).astype(np.float32)
                log_xb  = self._stack["xgb_b"].predict(Xv).astype(np.float32)
                log_lgb = self._stack["lgb"].predict(Xv).astype(np.float32)
                log_cat = self._stack["cat"].predict(Xv).astype(np.float32)
                S = np.column_stack([log_xa, log_xb, log_lgb, log_cat])
            else:
                # Legacy v4 / v3 stack: XGB (base model) + LGB [+ CAT]
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

        # Luxury blend: soft ramp from threshold → threshold*2 for Manhattan $3M+
        if self._luxury is not None:
            log_lux     = self._luxury.predict(dmat).astype(np.float32)
            lux_prices  = np.expm1(log_lux)
            lo          = float(self._luxury_threshold)
            hi          = lo * 2.0
            alpha       = np.clip((stack_prices - lo) / (hi - lo), 0.0, 1.0)
            return (1.0 - alpha) * stack_prices + alpha * lux_prices

        return stack_prices

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

        row   = pl.from_dicts([defaults])
        price = float(self.predict(row)[0])

        if self._stack is not None and "stack" in self.meta:
            medape = self.meta["stack"]["medape_holdout"]
            r2     = self.meta["stack"]["r2_holdout"]
            ver    = self._stack.get("version", "v4") if self._stack else "v4"
            model_label = f"Stack {ver} · 4-Model Ensemble" if ver in ("v5","v6","v7","v8","v9","v10") else "XGBoost + LightGBM Stack"
        else:
            medape = self.meta["xgboost"]["medape_test"]
            r2     = self.meta["xgboost"]["r2_test"]
            model_label = "XGBoost v2"

        conf    = self._adaptive_confidence(price, int(kwargs.get("borough", 0)))
        seg_med = conf["segment_medape"]
        mult    = seg_med / 100.0

        return {
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
