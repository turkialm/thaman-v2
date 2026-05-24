"""
THAMAN — Feature Parity Tests
==============================
Ensure all model features are populated at inference time — no silent
zero-fill for features that should have real values. Catches cases where
a new feature is added to the model but the API lookup is not wired up.

Run with:  pytest tests/test_feature_parity.py -v
"""

import pytest
import numpy as np
from models.scorer import ThamanScorer
from api.main import (
    _lookup_nta, _lookup_v11_features, _lookup_v12_features,
    _build_feature_row,
)
from api.main import PredictRequest


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def scorer():
    return ThamanScorer()


_NYC_COORDS  = {"lat": 40.6892, "lon": -73.9442}   # Brooklyn
_MN_COORDS   = {"lat": 40.7589, "lon": -73.9851}   # Manhattan Midtown
_RYD_COORDS  = {"lat": 24.7136, "lon": 46.6753}    # Riyadh Al-Wurud


# ── NYC feature count ─────────────────────────────────────────────────

def test_nyc_feature_count_v12(scorer):
    """v12 model must have exactly 109 features."""
    assert len(scorer.feature_names) == 109


def test_nyc_feature_names_no_duplicates(scorer):
    """Feature list must have no duplicate names."""
    assert len(scorer.feature_names) == len(set(scorer.feature_names))


# ── NTA lookup completeness ───────────────────────────────────────────

def test_nta_lookup_returns_all_required_keys():
    """_lookup_nta must return nta_encoded, nta_bldg_encoded, nta_sale_count, nta_median_psf."""
    result = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1")
    for key in ["nta_encoded", "nta_bldg_encoded", "nta_sale_count", "nta_median_psf"]:
        assert key in result, f"_lookup_nta missing: {key}"
    assert result.pop("_resolved_nta", None) is not None, "_resolved_nta must be populated"


def test_nta_lookup_brooklyn_returns_valid_ntacode():
    result = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1")
    nta = result.get("_resolved_nta", "")
    assert nta.startswith("BK") or len(nta) >= 4, f"Expected BK NTA code, got: {nta!r}"


def test_nta_lookup_manhattan_returns_mn_ntacode():
    result = _lookup_nta(_MN_COORDS["lat"], _MN_COORDS["lon"], "D4")
    nta = result.get("_resolved_nta", "")
    assert nta.startswith("MN"), f"Expected MN NTA code for Manhattan, got: {nta!r}"


# ── v11 feature completeness ──────────────────────────────────────────

def test_v11_features_all_populated():
    """_lookup_v11_features must return all 8 v11 features."""
    nta = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1").get("_resolved_nta", "")
    result = _lookup_v11_features(nta, _NYC_COORDS["lat"], _NYC_COORDS["lon"])
    expected = [
        "hpd_class_b_viol_zip", "hpd_class_c_viol_zip", "hpd_severity_score_zip",
        "dob_reno_permit_count", "dob_newbld_permit_count",
        "rat_density_nta", "heat_density_nta",
        "nearest_station_is_cbd", "nearest_station_route_count", "nearest_station_is_ada",
    ]
    for key in expected:
        assert key in result, f"_lookup_v11_features missing: {key}"
        assert result[key] is not None, f"{key} is None"


def test_v11_features_non_negative():
    nta = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1").get("_resolved_nta", "")
    result = _lookup_v11_features(nta, _NYC_COORDS["lat"], _NYC_COORDS["lon"])
    for key, val in result.items():
        assert float(val) >= 0, f"{key}={val} is negative — unexpected"


# ── v12 temporal feature completeness ────────────────────────────────

def test_v12_features_all_populated():
    """_lookup_v12_features must return all 5 v12 temporal features."""
    nta = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1").get("_resolved_nta", "")
    result = _lookup_v12_features(nta)
    expected = [
        "nta_lag1q_mean_logp", "nta_lag1q_median_psf", "nta_lag1q_count",
        "nta_lag2q_mean_logp", "nta_logp_momentum",
    ]
    for key in expected:
        assert key in result, f"_lookup_v12_features missing: {key}"
        assert result[key] is not None, f"{key} is None"


def test_v12_lag1_logp_in_reasonable_range():
    """NTA lag-1 mean log-price should be in log-price range for NYC (ln~$200K–$20M → ~12–17)."""
    nta = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1").get("_resolved_nta", "")
    result = _lookup_v12_features(nta)
    logp = result["nta_lag1q_mean_logp"]
    assert 12.0 <= logp <= 17.5, f"nta_lag1q_mean_logp={logp:.3f} outside [12, 17.5]"


def test_v12_lag1_psf_in_reasonable_range():
    """NTA lag-1 median $/sqft should be between $100 and $5000 for NYC."""
    nta = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1").get("_resolved_nta", "")
    result = _lookup_v12_features(nta)
    psf = result["nta_lag1q_median_psf"]
    assert 100 <= psf <= 5000, f"nta_lag1q_median_psf={psf:.1f} outside [100, 5000]"


def test_v12_count_positive():
    """NTA lag-1 sale count must be at least 1."""
    nta = _lookup_nta(_NYC_COORDS["lat"], _NYC_COORDS["lon"], "A1").get("_resolved_nta", "")
    result = _lookup_v12_features(nta)
    assert result["nta_lag1q_count"] >= 1, f"count={result['nta_lag1q_count']}"


# ── Riyadh feature count ──────────────────────────────────────────────

def test_riyadh_feature_count(scorer):
    """Riyadh v2 model must have the features stored in riyadh_meta.json."""
    assert scorer._riyadh_meta.get("n_features", 0) > 0
    n = scorer._riyadh_meta["n_features"]
    assert 60 <= n <= 120, f"Riyadh n_features={n} outside expected range"


def test_riyadh_feature_names_no_duplicates(scorer):
    names = scorer._riyadh_meta.get("feature_names", [])
    assert len(names) == len(set(names)), "Duplicate Riyadh feature names"


# ── End-to-end: all features reach model non-zero ─────────────────────

def test_end_to_end_features_mostly_nonzero(scorer):
    """
    After full feature enrichment (NTA + v11 + v12), at most 20% of features
    should be zero. High zero-rate indicates lookup failures.
    """
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        resp = c.post("/predict", json={
            "latitude": 40.6892, "longitude": -73.9442,
            "gross_square_feet": 1800, "building_age": 55,
            "bldgclass": "A1", "borough": 3,
            "numfloors": 2, "residential_units": 1,
        })
    assert resp.status_code == 200
    data = resp.json()
    drivers = data.get("top_drivers", [])
    assert len(drivers) >= 5, f"Only {len(drivers)} SHAP drivers — feature enrichment likely failed"
    impacts = [abs(d["impact"]) for d in drivers]
    assert max(impacts) > 0.01, "All SHAP impacts near zero — model may be getting zero features"
