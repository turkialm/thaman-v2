"""
THAMAN Scorer — Unit Tests
===========================
Tests for ThamanScorer inference class.
Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
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
    """Scorer should have 71 feature names."""
    assert len(scorer.feature_names) == 71


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
    """Manhattan property should be priced higher than similar Staten Island property."""
    mn = scorer.predict_single(
        latitude=40.7589, longitude=-73.9851,
        gross_square_feet=1200, building_age=40,
        bldgclass_encoded=scorer.bldgclass_means.get("A1", 13.0),
        borough_bldg_encoded=scorer.borough_bldg_means.get("1_A", 13.0),
        borough=1, numfloors=2, residential_units=1,
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
    """predict() on a DataFrame should return array of prices."""
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

    df     = pd.DataFrame(rows)
    prices = scorer.predict(df)
    assert len(prices) == 3
    assert all(p > 0 for p in prices)


# ── explain (SHAP) ────────────────────────────────────────────────────

def test_explain_returns_shap_df(scorer):
    """explain() should return a DataFrame with feature importances."""
    defaults = {feat: 0.0 for feat in scorer.feature_names}
    defaults.update({
        "latitude": 40.6892, "longitude": -73.9442,
        "gross_square_feet": 1800, "building_age": 55,
        "borough": 3, "numfloors": 2, "residential_units": 1,
        "bldgclass_encoded": scorer.global_mean_log,
        "borough_bldg_encoded": scorer.global_mean_log,
    })
    df       = pd.DataFrame([defaults])
    shap_df  = scorer.explain(df)
    assert shap_df.shape == (1, 71)
    assert list(shap_df.columns) == scorer.feature_names
