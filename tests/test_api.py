"""
THAMAN API — Unit Tests
========================
Run with:  pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# ── Fixtures ──────────────────────────────────────────────────────────

BROOKLYN_A1 = {
    "latitude":          40.6892,
    "longitude":        -73.9442,
    "gross_square_feet": 1800,
    "building_age":      55,
    "bldgclass":         "A1",
    "borough":           3,
    "numfloors":         2,
    "residential_units": 1,
}

MANHATTAN_D4 = {
    "latitude":          40.7589,
    "longitude":        -73.9851,
    "gross_square_feet": 950,
    "building_age":      40,
    "bldgclass":         "D4",
    "borough":           1,
    "numfloors":         12,
    "residential_units": 1,
}

# ── Info endpoints ────────────────────────────────────────────────────

def test_root_serves_ui(client):
    """GET / should serve the map UI directly (200)."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200


def test_api_info(client):
    """GET /api returns version and endpoint list."""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2.1.0"
    assert "endpoints" in data


def test_health_ok(client):
    """GET /health returns model_loaded and spatial_loaded flags."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded"   in data
    assert "spatial_loaded" in data
    assert "status"         in data


def test_bldgclasses_returns_list(client):
    """GET /bldgclasses returns a non-empty list of codes."""
    response = client.get("/bldgclasses")
    assert response.status_code == 200
    data = response.json()
    assert "bldgclasses" in data
    assert len(data["bldgclasses"]) > 10
    assert "A1" in data["bldgclasses"]


# ── Prediction endpoint ───────────────────────────────────────────────

def test_predict_brooklyn_a1(client):
    """Brooklyn A1 should return a realistic price ($300K–$3M)."""
    response = client.post("/predict", json=BROOKLYN_A1)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price"  in data
    assert "confidence_low"   in data
    assert "confidence_high"  in data
    assert "top_drivers"      in data
    assert 300_000 < data["predicted_price"] < 3_000_000


def test_predict_confidence_range(client):
    """Confidence low < predicted < high."""
    response = client.post("/predict", json=BROOKLYN_A1)
    assert response.status_code == 200
    data = response.json()
    assert data["confidence_low"] < data["predicted_price"] < data["confidence_high"]


def test_predict_manhattan_d4(client):
    """Manhattan D4 elevator building should be > Brooklyn A1."""
    r_bk = client.post("/predict", json=BROOKLYN_A1).json()
    r_mn = client.post("/predict", json=MANHATTAN_D4).json()
    assert r_mn["predicted_price"] > r_bk["predicted_price"]


def test_predict_shap_drivers(client):
    """top_drivers should contain feature, impact, direction."""
    response = client.post("/predict", json=BROOKLYN_A1)
    data = response.json()
    assert len(data["top_drivers"]) > 0
    for d in data["top_drivers"]:
        assert "feature"   in d
        assert "impact"    in d
        assert "direction" in d
        assert d["direction"] in ("positive", "negative")


def test_predict_spatial_features(client):
    """spatial_features should include subway and income keys."""
    response = client.post("/predict", json=BROOKLYN_A1)
    sf = response.json()["spatial_features"]
    assert "dist_subway_m"    in sf
    assert "median_income_nta" in sf
    assert sf["dist_subway_m"] > 0


def test_predict_invalid_out_of_nyc(client):
    """Coordinates outside NYC bounding box should return 422."""
    payload = {**BROOKLYN_A1, "latitude": 51.5074, "longitude": -0.1278}  # London
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field(client):
    """Missing required field should return 422."""
    payload = {k: v for k, v in BROOKLYN_A1.items() if k != "bldgclass"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_sqft(client):
    """Negative gross_square_feet should return 422."""
    payload = {**BROOKLYN_A1, "gross_square_feet": -100}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ── Batch endpoint ────────────────────────────────────────────────────

def test_batch_two_properties(client):
    """Batch endpoint returns results for each property."""
    response = client.post("/batch", json=[BROOKLYN_A1, MANHATTAN_D4])
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert all("predicted_price" in r for r in data["results"])


def test_batch_too_many(client):
    """Batch size > 50 should return 400."""
    payload = [BROOKLYN_A1] * 51
    response = client.post("/batch", json=payload)
    assert response.status_code == 400


# ── Nearby endpoint ───────────────────────────────────────────────────

def test_nearby_returns_sales(client):
    """GET /nearby should return at least 1 sale near Brooklyn."""
    response = client.get("/nearby?lat=40.6892&lon=-73.9442")
    assert response.status_code == 200
    data = response.json()
    assert "nearby" in data
    assert data["count"] > 0


def test_nearby_fields(client):
    """Each nearby sale should have required fields."""
    response = client.get("/nearby?lat=40.6892&lon=-73.9442&limit=3")
    data = response.json()
    for sale in data["nearby"]:
        assert "sale_price"  in sale
        assert "distance_m"  in sale
        assert "bldgclass"   in sale
        assert sale["sale_price"] > 0
        assert sale["distance_m"] >= 0


def test_nearby_invalid_coords(client):
    """Invalid coordinates should return 422."""
    response = client.get("/nearby?lat=999&lon=999")
    assert response.status_code == 422


# ── AVM QC block ──────────────────────────────────────────────────────

def test_predict_returns_avm_qc_block(client):
    """/predict response must include an avm_qc block."""
    data = client.post("/predict", json=BROOKLYN_A1).json()
    assert "avm_qc" in data, "avm_qc block missing from /predict response"
    assert data["avm_qc"] is not None


def test_avm_qc_has_required_fields(client):
    """avm_qc must contain all 2026 AVM standard fields."""
    qc = client.post("/predict", json=BROOKLYN_A1).json()["avm_qc"]
    for field in ["confidence_score", "confidence_grade", "segment_medape_pct",
                  "comparables_found", "comparables_radius_m", "sparse_market", "qc_flags"]:
        assert field in qc, f"Missing avm_qc field: {field}"


def test_avm_qc_confidence_score_range(client):
    """confidence_score must be an integer in [0, 100]."""
    qc = client.post("/predict", json=BROOKLYN_A1).json()["avm_qc"]
    assert isinstance(qc["confidence_score"], int)
    assert 0 <= qc["confidence_score"] <= 100


def test_avm_qc_comparable_count_non_negative(client):
    """comparables_found must be ≥ 0 and radius must be 800."""
    qc = client.post("/predict", json=BROOKLYN_A1).json()["avm_qc"]
    assert qc["comparables_found"] >= 0
    assert qc["comparables_radius_m"] == 800


def test_luxury_flag_for_high_price(client):
    """LUXURY_SEGMENT flag should appear when predicted price exceeds $3M."""
    data = client.post("/predict", json=MANHATTAN_D4).json()
    qc   = data["avm_qc"]
    # If the model predicts above $3M, the flag must be present
    if data["predicted_price"] > 3_000_000:
        assert "LUXURY_SEGMENT" in qc["qc_flags"]
    # qc_flags is always a list (may be empty)
    assert isinstance(qc["qc_flags"], list)
