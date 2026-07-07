"""
Re-pin regression baselines after an intentional model retrain.

Prints current model outputs for every pinned case in test_regression.py;
paste the new values into _NYC_PINS / _RYD_PINS.

Run:  python tests/update_regression_pins.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app
from tests.test_regression import _NYC_PINS, _RYD_PINS


def main():
    with TestClient(app) as client:
        print("\n── NYC pins ──────────────────────────────────────")
        for pin in _NYC_PINS:
            resp = client.post("/predict", json=pin["req"])
            resp.raise_for_status()
            price = resp.json()["predicted_price"]
            print(f"{pin['desc']}:")
            print(f"  expected_price: {pin['expected_price']:,} -> {round(price):,}")

        print("\n── Riyadh pins ───────────────────────────────────")
        for pin in _RYD_PINS:
            resp = client.post("/predict/riyadh", json=pin["req"])
            resp.raise_for_status()
            d = resp.json()
            psqm = d["predicted_price_sqm"]
            total = d["predicted_total_sar"]
            print(f"{pin['desc']}:")
            print(f"  expected_psqm:  {pin['expected_psqm']:,} -> {round(psqm):,}")
            print(f"  expected_total: {pin['expected_total']:,} -> {round(total):,}")


if __name__ == "__main__":
    main()
