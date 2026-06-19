"""
THAMAN — SHAP Explainability Tests
====================================
Validate that SHAP-derived feature attributions (top_drivers) are coherent,
internally consistent, and free of obvious explainability failures.

Run with:  pytest tests/test_shap.py -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

_BROOKLYN_A1 = {
    "latitude": 40.6892, "longitude": -73.9442,
    "gross_square_feet": 1800, "building_age": 55,
    "bldgclass": "A1", "borough": 3,
    "numfloors": 2, "residential_units": 1,
}
_MANHATTAN_D4 = {
    "latitude": 40.7589, "longitude": -73.9851,
    "gross_square_feet": 950, "building_age": 40,
    "bldgclass": "D4", "borough": 1,
    "numfloors": 12, "residential_units": 1,
}
_RYD_VILLA = {"latitude": 24.7136, "longitude": 46.6753, "property_type": "villa", "area_sqm": 300}
_RYD_APT   = {"latitude": 24.6877, "longitude": 46.7219, "property_type": "apartment", "area_sqm": 150}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Structure tests ───────────────────────────────────────────────────

def test_nyc_drivers_present(client):
    data = client.post("/predict", json=_BROOKLYN_A1).json()
    assert "top_drivers" in data
    assert len(data["top_drivers"]) >= 5, "Expected at least 5 SHAP drivers"


def test_nyc_drivers_schema(client):
    """Each driver must have feature, value, impact, direction, description."""
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    for d in drivers:
        for field in ["feature", "value", "impact", "direction", "description"]:
            assert field in d, f"Driver missing field: {field}"


def test_nyc_direction_matches_impact_sign(client):
    """direction='positive' ↔ impact>0, direction='negative' ↔ impact<0."""
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    for d in drivers:
        if d["direction"] == "positive":
            assert d["impact"] > 0, f"Positive direction but impact={d['impact']}"
        else:
            assert d["impact"] < 0, f"Negative direction but impact={d['impact']}"


def test_nyc_impacts_not_all_zero(client):
    """At least one driver must have |impact| > 0.001."""
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    max_abs = max(abs(d["impact"]) for d in drivers)
    assert max_abs > 0.001, f"All impacts near zero (max={max_abs:.6f}) — SHAP may have failed"


def test_nyc_driver_feature_names_valid(client):
    """All driver feature names must be in the model's feature list."""
    from models.scorer import ThamanScorer
    sc = ThamanScorer()
    valid = set(sc.feature_names)
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    for d in drivers:
        assert d["feature"] in valid, (
            f"Driver feature '{d['feature']}' not in model feature list"
        )


def test_nyc_drivers_ordered_by_magnitude(client):
    """Drivers should be sorted by |impact| descending."""
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    impacts = [abs(d["impact"]) for d in drivers]
    assert impacts == sorted(impacts, reverse=True), (
        "top_drivers not sorted by |impact| descending"
    )


# ── Sensitivity tests ─────────────────────────────────────────────────

def test_sqft_driver_is_significant(client):
    """A sqft-related feature (gross_square_feet or log_land_sqft) should appear in top drivers."""
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    feat_names = set(d["feature"] for d in drivers)
    sqft_feats = {"gross_square_feet", "log_land_sqft", "log_gross_sqft"}
    assert feat_names & sqft_feats, (
        f"No sqft feature in top drivers: {feat_names}"
    )


def test_location_feature_in_drivers(client):
    """A location feature (nta_encoded, latitude, or dist_*) should appear in top drivers."""
    drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    feat_names = set(d["feature"] for d in drivers)
    location_feats = {"nta_encoded", "nta_bldg_encoded", "latitude", "longitude",
                      "dist_midtown_manhattan_m", "dist_downtown_brooklyn_m",
                      "borough", "walk_score_proxy"}
    overlap = feat_names & location_feats
    assert overlap, f"No location feature in top drivers: {feat_names}"


def test_manhattan_nta_positive_vs_brooklyn(client):
    """NTA encoding should push Manhattan prediction higher than Brooklyn (positive direction)."""
    mn_drivers = client.post("/predict", json=_MANHATTAN_D4).json()["top_drivers"]
    bk_drivers = client.post("/predict", json=_BROOKLYN_A1).json()["top_drivers"]
    mn_nta = next((d for d in mn_drivers if d["feature"] == "nta_encoded"), None)
    bk_nta = next((d for d in bk_drivers if d["feature"] == "nta_encoded"), None)
    if mn_nta and bk_nta:
        assert mn_nta["impact"] > bk_nta["impact"], (
            f"Manhattan NTA impact ({mn_nta['impact']:.3f}) ≤ Brooklyn ({bk_nta['impact']:.3f})"
        )


# ── Riyadh SHAP tests ─────────────────────────────────────────────────

def test_riyadh_drivers_present(client):
    data = client.post("/predict/riyadh", json=_RYD_VILLA).json()
    assert "top_drivers" in data
    assert len(data["top_drivers"]) >= 5


def test_riyadh_drivers_schema(client):
    drivers = client.post("/predict/riyadh", json=_RYD_VILLA).json()["top_drivers"]
    for d in drivers:
        for field in ["feature", "impact", "direction", "description"]:
            assert field in d, f"Riyadh driver missing: {field}"


def test_riyadh_direction_matches_sign(client):
    drivers = client.post("/predict/riyadh", json=_RYD_VILLA).json()["top_drivers"]
    for d in drivers:
        if d["direction"] == "positive":
            assert d["impact"] > 0
        else:
            assert d["impact"] < 0


def test_riyadh_property_type_in_drivers(client):
    """is_villa or is_residential_plot should appear in top drivers."""
    drivers_villa = client.post("/predict/riyadh", json=_RYD_VILLA).json()["top_drivers"]
    feat_names = {d["feature"] for d in drivers_villa}
    type_feats = {"is_villa", "is_apartment", "is_residential_plot"}
    assert feat_names & type_feats, f"No property-type feature in Riyadh drivers: {feat_names}"


def test_riyadh_district_signal_present(client):
    """District temporal/encoding features should appear in top 10 Riyadh drivers (v11)."""
    drivers = client.post("/predict/riyadh", json=_RYD_VILLA).json()["top_drivers"]
    feat_names = {d["feature"] for d in drivers}
    # v11 district features (type-stratified lags, encoded, or any district-related signal)
    district_feats = {
        "district_type_lag1q_psqm", "district_type_lag2q_psqm",
        "district_lag1q_median_psqm", "district_lag2q_median_psqm",
        "district_type_encoded", "district_enc_oof",
        "district_lookback_mean", "district_lookback_apt_mean",
    }
    assert feat_names & district_feats, (
        f"No district encoding in Riyadh top drivers: {feat_names}"
    )
