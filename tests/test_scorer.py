"""
THAMAN Scorer — Unit Tests
===========================
Tests for ThamanScorer inference class.
Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import polars as pl
from models.scorer import ThamanScorer


@pytest.fixture(scope="module")
def scorer():
    """Load scorer once for all tests in this module."""
    return ThamanScorer()


# ── Initialization ────────────────────────────────────────────────────

def test_scorer_loads(scorer):
    """Scorer should load without error."""
    assert scorer is not None


def test_scorer_has_feature_names(scorer):
    """Scorer should have 93 feature names (v9: same v6 feature set, raw NTA encoding)."""
    assert len(scorer.feature_names) == 93


def test_scorer_has_bldgclass_means(scorer):
    """Scorer should have bldgclass target encoding means."""
    assert len(scorer.bldgclass_means) > 0
    assert "A1" in scorer.bldgclass_means


def test_scorer_stack_loaded(scorer):
    """Stack (LGB + CatBoost + Ridge) should be loaded."""
    assert scorer._stack is not None
    assert "lgb"  in scorer._stack
    assert "meta" in scorer._stack


# ── predict_single ────────────────────────────────────────────────────

def test_predict_single_returns_dict(scorer):
    """predict_single should return a dict with price and confidence."""
    result = scorer.predict_single(
        latitude=40.6892, longitude=-73.9442,
        gross_square_feet=1800, building_age=55,
        bldgclass_encoded=scorer.bldgclass_means.get("A1", 13.0),
        borough_bldg_encoded=scorer.borough_bldg_means.get("3_A", 13.0),
        borough=3, numfloors=2, residential_units=1,
    )
    assert "predicted_price"  in result
    assert "confidence_low"   in result
    assert "confidence_high"  in result
    assert "r2_test"          in result
    assert "medape_test_pct"  in result


def test_predict_single_positive_price(scorer):
    """Predicted price should always be positive."""
    result = scorer.predict_single(
        latitude=40.6892, longitude=-73.9442,
        gross_square_feet=1800, building_age=55,
        bldgclass_encoded=scorer.global_mean_log,
        borough_bldg_encoded=scorer.global_mean_log,
        borough=3, numfloors=2, residential_units=1,
    )
    assert result["predicted_price"] > 0


def test_predict_single_confidence_ordering(scorer):
    """confidence_low < predicted_price < confidence_high."""
    result = scorer.predict_single(
        latitude=40.7589, longitude=-73.9851,
        gross_square_feet=950, building_age=40,
        bldgclass_encoded=scorer.bldgclass_means.get("D4", 13.0),
        borough_bldg_encoded=scorer.borough_bldg_means.get("1_D", 13.0),
        borough=1, numfloors=12, residential_units=1,
    )
    assert result["confidence_low"] < result["predicted_price"] < result["confidence_high"]


def test_manhattan_more_expensive_than_staten_island(scorer):
    """Manhattan high-rise condo (D4) should be priced higher than a Staten Island single-family.
    Uses global_mean_log for NTA to avoid leaking borough-specific encoding into the comparison
    — location is captured by lat/lon, borough, and numfloors instead.
    """
    mn = scorer.predict_single(
        latitude=40.7589, longitude=-73.9851,
        gross_square_feet=1200, building_age=20,
        bldgclass_encoded=scorer.bldgclass_means.get("D4", 13.0),
        borough_bldg_encoded=scorer.borough_bldg_means.get("1_D", 13.0),
        borough=1, numfloors=20, residential_units=1,
    )
    si = scorer.predict_single(
        latitude=40.5795, longitude=-74.1502,
        gross_square_feet=1200, building_age=40,
        bldgclass_encoded=scorer.bldgclass_means.get("A1", 13.0),
        borough_bldg_encoded=scorer.borough_bldg_means.get("5_A", 13.0),
        borough=5, numfloors=2, residential_units=1,
    )
    assert mn["predicted_price"] > si["predicted_price"]


# ── predict (batch) ───────────────────────────────────────────────────

def test_predict_batch(scorer):
    """predict() on a polars DataFrame should return array of prices."""
    rows = []
    for _ in range(3):
        defaults = {feat: 0.0 for feat in scorer.feature_names}
        defaults.update({
            "latitude": 40.6892, "longitude": -73.9442,
            "gross_square_feet": 1800, "building_age": 55,
            "borough": 3, "numfloors": 2, "residential_units": 1,
            "bldgclass_encoded": scorer.global_mean_log,
            "borough_bldg_encoded": scorer.global_mean_log,
        })
        rows.append(defaults)

    df     = pl.from_dicts(rows)
    prices = scorer.predict(df)
    assert len(prices) == 3
    assert all(p > 0 for p in prices)


# ── explain (SHAP) ────────────────────────────────────────────────────

def test_explain_returns_shap_df(scorer):
    """explain() should return a polars DataFrame with 81 feature importances."""
    defaults = {feat: 0.0 for feat in scorer.feature_names}
    defaults.update({
        "latitude": 40.6892, "longitude": -73.9442,
        "gross_square_feet": 1800, "building_age": 55,
        "borough": 3, "numfloors": 2, "residential_units": 1,
        "bldgclass_encoded": scorer.global_mean_log,
        "borough_bldg_encoded": scorer.global_mean_log,
    })
    df       = pl.from_dicts([defaults])
    shap_df  = scorer.explain(df)
    assert shap_df.shape == (1, len(scorer.feature_names))
    assert list(shap_df.columns) == scorer.feature_names


# ── Adaptive Confidence ───────────────────────────────────────────────

def test_adaptive_confidence_returns_required_keys(scorer):
    """_adaptive_confidence should return all five required keys."""
    result = scorer._adaptive_confidence(price=750_000, borough=3)  # Brooklyn
    for key in ["segment_medape", "confidence_score", "confidence_grade",
                "tier_label", "borough_name"]:
        assert key in result, f"Missing key: {key}"


def test_adaptive_confidence_score_range(scorer):
    """confidence_score must always be an integer in [0, 100]."""
    for borough in [1, 2, 3, 4, 5]:
        for price in [300_000, 750_000, 2_000_000, 5_000_000]:
            r = scorer._adaptive_confidence(price, borough)
            assert 0 <= r["confidence_score"] <= 100, (
                f"Score out of range for borough={borough}, price={price}: {r['confidence_score']}"
            )


def test_adaptive_confidence_grade_matches_score(scorer):
    """Grade must correspond to the documented score thresholds."""
    r = scorer._adaptive_confidence(price=750_000, borough=4)  # Queens
    score = r["confidence_score"]
    expected = ("A" if score >= 85 else "B" if score >= 75
                else "C" if score >= 65 else "D")
    assert r["confidence_grade"] == expected, (
        f"Grade {r['confidence_grade']} does not match score {score}"
    )


def test_predict_single_returns_confidence_fields(scorer):
    """predict_single must now return confidence_score, confidence_grade, segment_medape_pct."""
    result = scorer.predict_single(
        latitude=40.7589, longitude=-73.9851,
        gross_square_feet=950, building_age=40,
        bldgclass_encoded=scorer.bldgclass_means.get("D4", 13.0),
        borough_bldg_encoded=scorer.borough_bldg_means.get("1_D", 13.0),
        borough=1, numfloors=12, residential_units=1,
    )
    assert "confidence_score"   in result
    assert "confidence_grade"   in result
    assert "segment_medape_pct" in result
    assert result["confidence_grade"] in ("A", "B", "C", "D")


def test_manhattan_lower_confidence_than_staten_island(scorer):
    """Manhattan MedAPE=34.22% > Staten Island=13.79% → lower score at same price."""
    mn = scorer._adaptive_confidence(price=750_000, borough=1)
    si = scorer._adaptive_confidence(price=750_000, borough=5)
    assert mn["confidence_score"] < si["confidence_score"], (
        f"Manhattan score {mn['confidence_score']} should be lower than "
        f"Staten Island {si['confidence_score']}"
    )
