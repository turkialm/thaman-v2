"""
THAMAN — Golden Dataset Tests
==============================
Fixed set of properties with realistic price ranges derived from public
NYC sales data and Riyadh MOJ transaction reports. Tests that the model
output stays within the expected market range — not exact values, but
economically plausible bounds.

Run with:  pytest tests/test_golden.py -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# ── NYC golden properties (expected range from public sales data) ──────
_NYC_GOLDEN = [
    {
        "desc": "Staten Island A1 (modest single-family)",
        "req":  {"latitude": 40.5795, "longitude": -74.1502,
                 "gross_square_feet": 1400, "building_age": 40,
                 "bldgclass": "A1", "borough": 5, "numfloors": 2, "residential_units": 1},
        "min": 300_000, "max": 900_000,
    },
    {
        "desc": "Bronx B2 two-family (East Tremont)",
        "req":  {"latitude": 40.8448, "longitude": -73.8902,
                 "gross_square_feet": 2000, "building_age": 70,
                 "bldgclass": "B2", "borough": 2, "numfloors": 2, "residential_units": 2},
        "min": 300_000, "max": 900_000,
    },
    {
        "desc": "Brooklyn D4 condo (Williamsburg)",
        "req":  {"latitude": 40.7143, "longitude": -73.9570,
                 "gross_square_feet": 900, "building_age": 10,
                 "bldgclass": "D4", "borough": 3, "numfloors": 6, "residential_units": 1},
        "min": 500_000, "max": 2_500_000,
    },
    {
        "desc": "Queens A1 single-family (Flushing)",
        "req":  {"latitude": 40.7678, "longitude": -73.8330,
                 "gross_square_feet": 1600, "building_age": 50,
                 "bldgclass": "A1", "borough": 4, "numfloors": 2, "residential_units": 1},
        "min": 400_000, "max": 1_200_000,
    },
    {
        "desc": "Manhattan luxury condo (Upper West Side)",
        "req":  {"latitude": 40.7830, "longitude": -73.9800,
                 "gross_square_feet": 1500, "building_age": 20,
                 "bldgclass": "D4", "borough": 1, "numfloors": 30, "residential_units": 1},
        "min": 1_500_000, "max": 8_000_000,
    },
]

# ── Riyadh golden properties ───────────────────────────────────────────
_RYD_GOLDEN = [
    {
        "desc": "Premium villa Al-Malqa (400m²)",
        "req":  {"latitude": 24.7750, "longitude": 46.6380, "property_type": "villa",     "area_sqm": 400},
        "min_total": 1_500_000, "max_total": 8_000_000,
    },
    {
        "desc": "Apartment Al-Olaya district (120m²)",
        "req":  {"latitude": 24.6913, "longitude": 46.6843, "property_type": "apartment", "area_sqm": 120},
        "min_total": 300_000,   "max_total": 1_500_000,
    },
    {
        "desc": "Residential plot Shifa (500m²)",
        "req":  {"latitude": 24.6327, "longitude": 46.7016, "property_type": "plot",      "area_sqm": 500},
        "min_total": 400_000,   "max_total": 3_000_000,
    },
    {
        "desc": "Affordable apartment Al-Aziziyah (100m²)",
        "req":  {"latitude": 24.6169, "longitude": 46.7310, "property_type": "apartment", "area_sqm": 100},
        "min_total": 150_000,   "max_total": 800_000,
    },
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("prop", _NYC_GOLDEN, ids=[p["desc"] for p in _NYC_GOLDEN])
def test_nyc_golden_range(client, prop):
    resp = client.post("/predict", json=prop["req"])
    assert resp.status_code == 200, resp.text
    price = resp.json()["predicted_price"]
    assert prop["min"] <= price <= prop["max"], (
        f"{prop['desc']}: ${price:,.0f} outside [{prop['min']:,.0f}, {prop['max']:,.0f}]"
    )


@pytest.mark.parametrize("prop", _RYD_GOLDEN, ids=[p["desc"] for p in _RYD_GOLDEN])
def test_riyadh_golden_range(client, prop):
    resp = client.post("/predict/riyadh", json=prop["req"])
    assert resp.status_code == 200, resp.text
    total = resp.json()["predicted_total_sar"]
    assert prop["min_total"] <= total <= prop["max_total"], (
        f"{prop['desc']}: SAR {total:,.0f} outside [{prop['min_total']:,.0f}, {prop['max_total']:,.0f}]"
    )


def test_nyc_price_ordering(client):
    """Manhattan luxury > Queens single-family > Staten Island single-family."""
    mn = client.post("/predict", json=_NYC_GOLDEN[4]["req"]).json()["predicted_price"]
    qn = client.post("/predict", json=_NYC_GOLDEN[3]["req"]).json()["predicted_price"]
    si = client.post("/predict", json=_NYC_GOLDEN[0]["req"]).json()["predicted_price"]
    assert mn > qn, f"Manhattan ({mn:,.0f}) should exceed Queens ({qn:,.0f})"
    assert qn > si, f"Queens ({qn:,.0f}) should exceed Staten Island ({si:,.0f})"


def test_riyadh_villa_more_than_apt_same_area(client):
    """Premium villa district (Al-Malqa) psqm should exceed affordable apt district (Al-Aziziyah)."""
    villa_psqm = client.post("/predict/riyadh", json={
        "latitude": 24.7750, "longitude": 46.6380, "property_type": "villa", "area_sqm": 300
    }).json()["predicted_price_sqm"]
    apt_psqm = client.post("/predict/riyadh", json={
        "latitude": 24.6169, "longitude": 46.7310, "property_type": "apartment", "area_sqm": 100
    }).json()["predicted_price_sqm"]
    assert villa_psqm > apt_psqm, (
        f"Al-Malqa villa psqm ({villa_psqm:,.0f}) should exceed Al-Aziziyah apt psqm ({apt_psqm:,.0f})"
    )
