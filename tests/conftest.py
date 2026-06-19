"""
Shared pytest fixtures for THAMAN test suite.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    TestClient wrapped in lifespan context so the model and spatial
    data are fully loaded before any test runs.
    """
    with TestClient(app) as c:
        yield c
