"""
THAMAN — API Load / Stress Tests
==================================
Concurrency and latency benchmarks. Uses FastAPI TestClient for
deterministic in-process load (no network overhead). For true network
stress testing, run against a live server with locust or k6.

Benchmarks (pass/fail thresholds):
  • 10 sequential NYC requests    < 30 s total  (3 s each)
  • 10 sequential Riyadh requests < 30 s total  (3 s each)
  • 10 concurrent requests        complete without errors
  • Batch (50 items)              < 60 s

Run with:  pytest tests/test_load.py -v -s
"""

import time
import threading
import pytest
from fastapi.testclient import TestClient
from api.main import app

_NYC_REQ = {
    "latitude": 40.6892, "longitude": -73.9442,
    "gross_square_feet": 1800, "building_age": 55,
    "bldgclass": "A1", "borough": 3,
    "numfloors": 2, "residential_units": 1,
}
_RYD_REQ = {
    "latitude": 24.7136, "longitude": 46.6753,
    "property_type": "villa", "area_sqm": 300,
}

_SEQ_N       = 10
_CONC_N      = 10
_SEQ_LIMIT_S = 30.0   # total for _SEQ_N requests
_REQ_LIMIT_S =  5.0   # per-request cap
_BATCH_LIMIT_S = 60.0


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Sequential throughput ─────────────────────────────────────────────

def test_nyc_sequential_throughput(client):
    """10 NYC predictions complete in under 30 seconds total."""
    t0 = time.perf_counter()
    for _ in range(_SEQ_N):
        resp = client.post("/predict", json=_NYC_REQ)
        assert resp.status_code == 200
    elapsed = time.perf_counter() - t0
    print(f"\n  NYC sequential: {_SEQ_N} requests in {elapsed:.2f}s ({elapsed/_SEQ_N:.2f}s each)")
    assert elapsed < _SEQ_LIMIT_S, f"Too slow: {elapsed:.2f}s > {_SEQ_LIMIT_S}s"


def test_riyadh_sequential_throughput(client):
    """10 Riyadh predictions complete in under 30 seconds total."""
    t0 = time.perf_counter()
    for _ in range(_SEQ_N):
        resp = client.post("/predict/riyadh", json=_RYD_REQ)
        assert resp.status_code == 200
    elapsed = time.perf_counter() - t0
    print(f"\n  Riyadh sequential: {_SEQ_N} requests in {elapsed:.2f}s ({elapsed/_SEQ_N:.2f}s each)")
    assert elapsed < _SEQ_LIMIT_S, f"Too slow: {elapsed:.2f}s > {_SEQ_LIMIT_S}s"


def test_per_request_latency(client):
    """Individual request latency should be under 5 seconds."""
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        client.post("/predict", json=_NYC_REQ)
        latencies.append(time.perf_counter() - t0)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    print(f"\n  p95 latency: {p95:.3f}s  (max allowed {_REQ_LIMIT_S}s)")
    assert p95 < _REQ_LIMIT_S, f"p95 latency {p95:.3f}s exceeds {_REQ_LIMIT_S}s"


# ── Concurrent load ───────────────────────────────────────────────────

def test_concurrent_requests_no_errors(client):
    """10 concurrent NYC predictions all succeed (no 500s, no crashes)."""
    results: list[int] = []
    errors:  list[str] = []
    lock = threading.Lock()

    def _predict():
        try:
            resp = client.post("/predict", json=_NYC_REQ)
            with lock:
                results.append(resp.status_code)
        except Exception as e:
            with lock:
                errors.append(str(e))

    threads = [threading.Thread(target=_predict) for _ in range(_CONC_N)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    elapsed = time.perf_counter() - t0

    print(f"\n  Concurrent: {_CONC_N} threads in {elapsed:.2f}s")
    assert not errors, f"Thread errors: {errors}"
    assert len(results) == _CONC_N, f"Only {len(results)}/{_CONC_N} threads completed"
    failed = [s for s in results if s != 200]
    assert not failed, f"Non-200 responses: {failed}"


def test_concurrent_riyadh_no_errors(client):
    """10 concurrent Riyadh predictions all succeed."""
    results: list[int] = []
    lock = threading.Lock()

    def _predict():
        resp = client.post("/predict/riyadh", json=_RYD_REQ)
        with lock:
            results.append(resp.status_code)

    threads = [threading.Thread(target=_predict) for _ in range(_CONC_N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    failed = [s for s in results if s != 200]
    assert not failed, f"Riyadh concurrent failures: {failed}"
    assert len(results) == _CONC_N


# ── Batch endpoint ─────────────────────────────────────────────────────

def test_batch_50_completes_in_time(client):
    """Batch of 50 properties completes under 60 seconds."""
    payload = [_NYC_REQ] * 50
    t0 = time.perf_counter()
    resp = client.post("/batch", json=payload)
    elapsed = time.perf_counter() - t0
    print(f"\n  Batch 50: {elapsed:.2f}s")
    assert resp.status_code == 200, f"Batch failed: {resp.text[:200]}"
    assert elapsed < _BATCH_LIMIT_S, f"Batch too slow: {elapsed:.2f}s"
    assert resp.json()["count"] == 50


def test_batch_all_results_valid(client):
    """All batch results must have positive prices."""
    payload = [
        {**_NYC_REQ, "latitude": 40.6892 + i * 0.001}
        for i in range(10)
    ]
    data = client.post("/batch", json=payload).json()
    for r in data["results"]:
        assert r.get("predicted_price", 0) > 0


# ── Health endpoint latency ───────────────────────────────────────────

def test_health_endpoint_fast(client):
    """Health endpoint should respond in under 100ms."""
    t0 = time.perf_counter()
    resp = client.get("/health")
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200
    assert elapsed < 0.1, f"Health check took {elapsed:.3f}s (limit: 0.1s)"
