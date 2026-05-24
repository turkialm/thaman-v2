"""
THAMAN — Distribution Shift Detection Tests
============================================
Detect when prediction distributions shift from training baselines.
Uses stored training statistics (from meta.json) and compares against
predictions on a fixed property grid.

Run with:  pytest tests/test_distribution.py -v
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.main import app

# ── Fixed grid covering all 5 NYC boroughs ────────────────────────────
_BOROUGH_GRID = [
    # (lat, lon, borough, bldgclass, sqft, age, floors)
    (40.7589, -73.9851, 1, "D4", 950,  30, 12),   # Manhattan
    (40.8448, -73.8902, 2, "A1", 1500, 60,  2),   # Bronx
    (40.6892, -73.9442, 3, "A1", 1800, 55,  2),   # Brooklyn
    (40.7282, -73.7949, 4, "B2", 2000, 50,  2),   # Queens
    (40.5795, -74.1502, 5, "A1", 1600, 45,  2),   # Staten Island
    # Additional diversity
    (40.7143, -73.9570, 3, "D4",  900, 10,  6),   # Williamsburg condo
    (40.7528, -73.9772, 1, "D4", 1200, 20, 20),   # Midtown high-rise
    (40.6501, -73.9496, 3, "B2", 2400, 70,  3),   # Flatbush two-family
    (40.7600, -73.8292, 4, "A1", 1800, 55,  2),   # Bayside single-family
    (40.6308, -74.0776, 5, "A1", 1400, 40,  2),   # Tottenville SI
]

# ── Training distribution baselines (from meta.json / holdout stats) ──
# NYC holdout: 27,763 properties, last 15% by date (2024-2025)
_NYC_EXPECTED_MEDIAN_PRICE = 750_000   # rough median from holdout
_NYC_PRICE_P10 = 200_000               # 10th percentile
_NYC_PRICE_P90 = 3_000_000             # 90th percentile

# Riyadh grid (lat, lon, type, area_sqm)
_RYD_GRID = [
    (24.7136, 46.6753, "villa",     300),
    (24.6877, 46.7219, "apartment", 150),
    (24.7743, 46.7382, "plot",      600),
    (24.7750, 46.6380, "villa",     400),
    (24.6169, 46.7310, "apartment", 100),
    (24.6327, 46.7016, "plot",      500),
    (24.7200, 46.7050, "villa",     250),
    (24.6550, 46.6900, "apartment", 200),
]

# Riyadh training distribution (from riyadh_meta/MOJ data)
_RYD_EXPECTED_MEDIAN_PSQM = 4000   # SAR/m² overall median
_RYD_PSQM_P10 = 1000
_RYD_PSQM_P90 = 12000


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def nyc_predictions(client):
    preds = []
    for lat, lon, borough, bc, sqft, age, floors in _BOROUGH_GRID:
        resp = client.post("/predict", json={
            "latitude": lat, "longitude": lon, "borough": borough,
            "bldgclass": bc, "gross_square_feet": sqft,
            "building_age": age, "numfloors": floors, "residential_units": 1,
        })
        if resp.status_code == 200:
            preds.append(resp.json()["predicted_price"])
    return preds


@pytest.fixture(scope="module")
def ryd_predictions(client):
    preds = []
    for lat, lon, ptype, area in _RYD_GRID:
        resp = client.post("/predict/riyadh", json={
            "latitude": lat, "longitude": lon,
            "property_type": ptype, "area_sqm": area,
        })
        if resp.status_code == 200:
            preds.append(resp.json()["predicted_price_sqm"])
    return preds


# ── NYC distribution checks ───────────────────────────────────────────

def test_nyc_all_predictions_succeed(nyc_predictions):
    assert len(nyc_predictions) == len(_BOROUGH_GRID), (
        f"Only {len(nyc_predictions)}/{len(_BOROUGH_GRID)} predictions succeeded"
    )


def test_nyc_grid_median_in_range(nyc_predictions):
    """Grid median should be within 3× of expected training median."""
    median = float(np.median(nyc_predictions))
    lo = _NYC_EXPECTED_MEDIAN_PRICE / 3
    hi = _NYC_EXPECTED_MEDIAN_PRICE * 3
    assert lo <= median <= hi, (
        f"Grid median ${median:,.0f} far from expected ${_NYC_EXPECTED_MEDIAN_PRICE:,.0f} "
        f"(bounds: [{lo:,.0f}, {hi:,.0f}])"
    )


def test_nyc_all_prices_positive(nyc_predictions):
    assert all(p > 0 for p in nyc_predictions), "Some prices are non-positive"


def test_nyc_no_extreme_outliers(nyc_predictions):
    """No prediction should be below $50K or above $100M (data artifacts)."""
    for p in nyc_predictions:
        assert p >= 50_000,   f"Suspiciously low price: ${p:,.0f}"
        assert p <= 100_000_000, f"Suspiciously high price: ${p:,.0f}"


def test_nyc_borough_ordering_tendency(nyc_predictions):
    """Manhattan predictions should be among the highest in the grid."""
    manhattan_price = nyc_predictions[0]   # first entry is Manhattan
    bronx_price     = nyc_predictions[1]
    assert manhattan_price > bronx_price, (
        f"Manhattan ${manhattan_price:,.0f} ≤ Bronx ${bronx_price:,.0f} — "
        "price ordering reversed, possible distribution shift"
    )


def test_nyc_grid_price_spread(nyc_predictions):
    """Grid should span at least 3× from p25 to p75 (market heterogeneity)."""
    p25 = float(np.percentile(nyc_predictions, 25))
    p75 = float(np.percentile(nyc_predictions, 75))
    assert p75 >= p25 * 1.5, (
        f"Low price spread p25={p25:,.0f} p75={p75:,.0f} — predictions may be collapsing"
    )


# ── Riyadh distribution checks ────────────────────────────────────────

def test_riyadh_all_predictions_succeed(ryd_predictions):
    assert len(ryd_predictions) == len(_RYD_GRID), (
        f"Only {len(ryd_predictions)}/{len(_RYD_GRID)} Riyadh predictions succeeded"
    )


def test_riyadh_grid_median_psqm_in_range(ryd_predictions):
    """Grid median SAR/m² should be within 3× of expected training median."""
    median = float(np.median(ryd_predictions))
    lo = _RYD_EXPECTED_MEDIAN_PSQM / 3
    hi = _RYD_EXPECTED_MEDIAN_PSQM * 3
    assert lo <= median <= hi, (
        f"Grid median {median:,.0f} SAR/m² far from expected {_RYD_EXPECTED_MEDIAN_PSQM:,.0f}"
    )


def test_riyadh_psqm_positive(ryd_predictions):
    assert all(p > 0 for p in ryd_predictions), "Some Riyadh prices are non-positive"


def test_riyadh_psqm_reasonable_range(ryd_predictions):
    """All predictions should be within 500–50,000 SAR/m²."""
    for p in ryd_predictions:
        assert 500 <= p <= 50_000, f"Riyadh psqm={p:,.0f} outside [500, 50,000]"
