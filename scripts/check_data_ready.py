#!/usr/bin/env python3
"""Verify THAMAN raw/processed data before pipeline runs. See --help."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW, PROC = BASE / "data/raw", BASE / "data/processed"
NYCCAS_URL = "https://data.cityofnewyork.us/resource/c3uy-2p5r.csv?$limit=500000"


def check(path: Path, min_rows: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.suffix == ".csv":
        n = sum(1 for _ in path.open()) - 1
        return (n >= min_rows, f"{n:,} rows")
    if path.suffix == ".parquet":
        import polars as pl
        n = len(pl.read_parquet(path))
        return (n >= min_rows, f"{n:,} rows")
    if path.suffix == ".geojson":
        n = len(json.load(path.open()).get("features", []))
        return (n >= min_rows, f"{n:,} features")
    return True, f"{path.stat().st_size // 1024} KB"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true")
    p.add_argument("--download-nyccas", action="store_true")
    args = p.parse_args()
    os.chdir(BASE)
    if args.download_nyccas and not (RAW / "nyccas_air_quality.csv").exists():
        urllib.request.urlretrieve(NYCCAS_URL, RAW / "nyccas_air_quality.csv")
    groups = [
        ("NYC raw", ["sales_geocoded.csv", "nta_boundaries.geojson", "nypd_crimes.parquet"]),
        ("Riyadh raw", ["quarter_report SI.xlsx", "sales_riyadh_2025_Q3.csv", "air-quality.csv"]),
    ]
    ok = True
    for title, files in groups:
        print(title)
        for f in files:
            good, msg = check(RAW / f)
            print(f"  [{'OK' if good else 'FAIL'}] {f}: {msg}")
            ok &= good
    if args.strict:
        for f in ["features.csv", "features_v5.csv", "features_riyadh.csv"]:
            good, msg = check(PROC / f, 100)
            print(f"  [{'OK' if good else 'FAIL'}] {f}: {msg}")
            ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
