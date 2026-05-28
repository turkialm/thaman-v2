"""
THAMAN — Regression Tests
==========================
Pin current model outputs. Any change in model weights, feature engineering,
or preprocessing that shifts these outputs by >5% fails the test, forcing
an explicit acknowledgement of the regression.

Update pins after intentional model retrain:
    python tests/update_regression_pins.py
Run with:  pytest tests/test_regression.py -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# ── Pinned baselines (captured 2026-05-28, NYC v22 / Riyadh v11) ──────
# NYC — raw scorer via API /predict endpoint
_NYC_PINS = [
    {
        "desc": "Brooklyn A1 single-family (1800 sqft, age 55)",
        "req": {
            "latitude": 40.6892, "longitude": -73.9442,
            "gross_square_feet": 1800, "building_age": 55,
            "bldgclass": "A1", "borough": 3,
            "numfloors": 2, "residential_units": 1,
        },
        "expected_price":  1_397_193,  # v22 via API (134 features)
        "tolerance_pct":   5.0,
    },
    {
        "desc": "Manhattan D4 elevator condo (950 sqft, age 40)",
        "req": {
            "latitude": 40.7589, "longitude": -73.9851,
            "gross_square_feet": 950, "building_age": 40,
            "bldgclass": "D4", "borough": 1,
            "numfloors": 12, "residential_units": 1,
        },
        "expected_price":  1_829_702,
        "tolerance_pct":   5.0,
    },
    {
        "desc": "Queens B2 two-family (2400 sqft, age 60)",
        "req": {
            "latitude": 40.7282, "longitude": -73.7949,
            "gross_square_feet": 2400, "building_age": 60,
            "bldgclass": "B2", "borough": 4,
            "numfloors": 2, "residential_units": 2,
        },
        "expected_price":  1_173_523,
        "tolerance_pct":   5.0,
    },
]

# Riyadh — via /predict/riyadh endpoint
_RYD_PINS = [
    {
        "desc": "Al-Wurud villa 300m²",
        "req":  {"latitude": 24.7136, "longitude": 46.6753, "property_type": "villa",     "area_sqm": 300},
        "expected_psqm":  3987,
        "expected_total": 1_196_100,
        "tolerance_pct":  5.0,
    },
    {
        "desc": "Al-Dubbat apartment 150m²",
        "req":  {"latitude": 24.6877, "longitude": 46.7219, "property_type": "apartment", "area_sqm": 150},
        "expected_psqm":  2909,
        "expected_total": 436_350,
        "tolerance_pct":  5.0,
    },
    {
        "desc": "Northern Riyadh plot 600m²",
        "req":  {"latitude": 24.7743, "longitude": 46.7382, "property_type": "plot",      "area_sqm": 600},
        "expected_psqm":  3527,
        "expected_total": 2_116_200,
        "tolerance_pct":  5.0,
    },
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _within_pct(actual: float, expected: float, pct: float) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / expected * 100 <= pct


@pytest.mark.parametrize("pin", _NYC_PINS, ids=[p["desc"] for p in _NYC_PINS])
def test_nyc_regression(client, pin):
    resp = client.post("/predict", json=pin["req"])
    assert resp.status_code == 200, resp.text
    actual = resp.json()["predicted_price"]
    assert _within_pct(actual, pin["expected_price"], pin["tolerance_pct"]), (
        f"{pin['desc']}: got ${actual:,.0f}, expected ${pin['expected_price']:,.0f} "
        f"(±{pin['tolerance_pct']}%)"
    )


@pytest.mark.parametrize("pin", _RYD_PINS, ids=[p["desc"] for p in _RYD_PINS])
def test_riyadh_regression(client, pin):
    resp = client.post("/predict/riyadh", json=pin["req"])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    actual_psqm  = data["predicted_price_sqm"]
    actual_total = data["predicted_total_sar"]
    assert _within_pct(actual_psqm, pin["expected_psqm"], pin["tolerance_pct"]), (
        f"{pin['desc']}: psqm got {actual_psqm}, expected {pin['expected_psqm']} "
        f"(±{pin['tolerance_pct']}%)"
    )
    assert _within_pct(actual_total, pin["expected_total"], pin["tolerance_pct"]), (
        f"{pin['desc']}: total got {actual_total:,.0f}, expected {pin['expected_total']:,.0f} "
        f"(±{pin['tolerance_pct']}%)"
    )
