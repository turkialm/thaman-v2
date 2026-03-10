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
import pandas as pd
import xgboost as xgb

_DIR = os.path.dirname(os.path.abspath(__file__))


class ThamanScorer:
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

        # Stack (LGB + Ridge meta) — optional
        self._stack = None
        if os.path.exists(stack_path):
            self._stack = joblib.load(stack_path)
            has_cat = "cat" in self._stack
            label = "XGB + LGB + CAT + Ridge" if has_cat else "XGB + LGB + Ridge"
            print(f"  [scorer] Stack loaded ({label})")
        else:
            print(f"  [scorer] Stack not found — using XGBoost only")

    # ── Internal: preprocess feature matrix ────────────────────────
    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].copy()
        for col, cap in self.winsorize.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=cap)
        for col, med in self.acris_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)
        return X.fillna(0).values.astype(np.float32)

    # ── Main prediction method ──────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices for a DataFrame. Returns USD array.
        Uses stack (XGB + LGB + Ridge) when available, else XGB alone.
        """
        Xv = self._prepare(df)
        dmat = xgb.DMatrix(Xv, feature_names=self.feature_names)
        log_xgb = self.model.predict(dmat)

        if self._stack is not None:
            log_lgb = self._stack["lgb"].predict(Xv).astype(np.float32)
            cols = [log_xgb, log_lgb]
            if "cat" in self._stack:
                log_cat = self._stack["cat"].predict(Xv).astype(np.float32)
                cols.append(log_cat)
            S         = np.column_stack(cols)
            log_final = self._stack["meta"].predict(S).astype(np.float32)
        else:
            log_final = log_xgb

        return np.expm1(log_final)

    # ── Single property convenience method ─────────────────────────
    def predict_single(self, **kwargs) -> dict:
        """
        Predict price for one property. Pass feature values as keyword args.
        """
        defaults = {feat: 0.0 for feat in self.feature_names}
        for col in self.acris_medians:
            defaults[col] = np.nan
        defaults.update(kwargs)

        row   = pd.DataFrame([defaults])
        price = float(self.predict(row)[0])

        if self._stack is not None and "stack" in self.meta:
            medape = self.meta["stack"]["medape_holdout"]
            r2     = self.meta["stack"]["r2_holdout"]
            model_label = "XGBoost + LightGBM Stack"
        else:
            medape = self.meta["xgboost"]["medape_test"]
            r2     = self.meta["xgboost"]["r2_test"]
            model_label = "XGBoost v2"

        mult = medape / 100.0
        return {
            "predicted_price":  round(price),
            "confidence_low":   round(price * (1.0 - mult)),
            "confidence_high":  round(price * (1.0 + mult)),
            "model":            model_label,
            "r2_test":          r2,
            "medape_test_pct":  medape,
        }

    # ── SHAP explanation for one property ──────────────────────────
    def explain(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Returns SHAP-based feature contributions for each row.
        Requires: pip install shap
        """
        import shap
        X = df[self.feature_names].copy()
        for col, cap in self.winsorize.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=cap)
        for col, med in self.acris_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)
        X = X.fillna(0)

        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X.values)
        shap_df     = pd.DataFrame(shap_values, columns=self.feature_names)
        return shap_df


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
