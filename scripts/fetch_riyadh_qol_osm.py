"""
fetch_riyadh_qol_osm.py
========================
Fetches new QoL POI data for Riyadh from OSM Overpass API.

New categories (not in existing model):
  - gyms / sports centres
  - pharmacies
  - universities / colleges
  - cinemas (Vision 2030 expansion)
  - coffee shops
  - libraries
  - police stations
  - clinics (not full hospitals)
  - kindergartens / nurseries
  - petrol stations (already partially covered, extend)
  - car dealerships (luxury index)

Output: data/raw/riyadh_qol_<category>.csv  per category
        data/raw/riyadh_qol_combined.csv     all combined

Run: python scripts/fetch_riyadh_qol_osm.py
"""

import csv
import json
import time
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data" / "raw"

# Riyadh city bounding box (generous)
BBOX = "24.4,46.4,25.1,47.0"   # south,west,north,east

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM queries — each (output_name, overpass_body)
QUERIES = {
    "gyms": '[out:json][timeout:60];(node["leisure"="fitness_centre"]({bbox});way["leisure"="fitness_centre"]({bbox}););out center;',
    "pharmacies": '[out:json][timeout:60];(node["amenity"="pharmacy"]({bbox});way["amenity"="pharmacy"]({bbox}););out center;',
    "universities": '[out:json][timeout:60];(node["amenity"="university"]({bbox});way["amenity"="university"]({bbox});relation["amenity"="university"]({bbox});node["amenity"="college"]({bbox});way["amenity"="college"]({bbox}););out center;',
    "cinemas": '[out:json][timeout:60];(node["amenity"="cinema"]({bbox});way["amenity"="cinema"]({bbox}););out center;',
    "coffee_shops": '[out:json][timeout:60];(node["amenity"="cafe"]({bbox});way["amenity"="cafe"]({bbox}););out center;',
    "libraries": '[out:json][timeout:60];(node["amenity"="library"]({bbox});way["amenity"="library"]({bbox}););out center;',
    "police": '[out:json][timeout:60];(node["amenity"="police"]({bbox});way["amenity"="police"]({bbox}););out center;',
    "clinics": '[out:json][timeout:60];(node["amenity"="clinic"]({bbox});node["amenity"="doctors"]({bbox});way["amenity"="clinic"]({bbox}););out center;',
    "kindergartens": '[out:json][timeout:60];(node["amenity"="kindergarten"]({bbox});way["amenity"="kindergarten"]({bbox}););out center;',
    "sports_centres": '[out:json][timeout:60];(node["leisure"="sports_centre"]({bbox});way["leisure"="sports_centre"]({bbox}););out center;',
    "swimming_pools": '[out:json][timeout:60];(node["leisure"="swimming_pool"]["access"!="private"]({bbox});way["leisure"="swimming_pool"]["access"!="private"]({bbox}););out center;',
    "stadiums": '[out:json][timeout:60];(node["leisure"="stadium"]({bbox});way["leisure"="stadium"]({bbox}););out center;',
    "supermarkets": '[out:json][timeout:60];(node["shop"="supermarket"]({bbox});way["shop"="supermarket"]({bbox}););out center;',
    "atms": '[out:json][timeout:60];(node["amenity"="atm"]({bbox}););out center;',
    "car_showrooms": '[out:json][timeout:60];(node["shop"="car"]({bbox});way["shop"="car"]({bbox}););out center;',
    "post_offices": '[out:json][timeout:60];(node["amenity"="post_office"]({bbox});way["amenity"="post_office"]({bbox}););out center;',
}

HEADERS = {"User-Agent": "THAMAN-AVM-Research/2.0 (educational project)"}


def extract_coords(element: dict) -> tuple[float, float] | None:
    if element["type"] == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


def fetch_category(name: str, query_template: str) -> list[dict]:
    query = query_template.replace("{bbox}", BBOX)
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=90)
        r.raise_for_status()
        data = r.json()
        elements = data.get("elements", [])
        rows = []
        for el in elements:
            lat, lon = extract_coords(el)
            if lat is None:
                continue
            tags = el.get("tags", {})
            rows.append({
                "category": name,
                "lat": lat,
                "lon": lon,
                "name_ar": tags.get("name:ar", ""),
                "name_en": tags.get("name:en", tags.get("name", "")),
                "osm_id": el.get("id", ""),
                "osm_type": el["type"],
            })
        return rows
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return []


def main():
    all_rows = []
    fields = ["category", "lat", "lon", "name_ar", "name_en", "osm_id", "osm_type"]

    for name, query in QUERIES.items():
        print(f"Fetching {name} ...", end=" ", flush=True)
        rows = fetch_category(name, query)
        print(f"{len(rows)} POIs")

        # Save individual file
        out_path = OUT_DIR / f"riyadh_qol_{name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        all_rows.extend(rows)
        time.sleep(2)   # respect Overpass rate limit

    # Save combined
    combined_path = OUT_DIR / "riyadh_qol_combined.csv"
    with open(combined_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nTotal: {len(all_rows)} POIs across {len(QUERIES)} categories")
    print(f"Saved → {combined_path.name}")

    # Summary
    from collections import Counter
    cnt = Counter(r["category"] for r in all_rows)
    print("\nBreakdown:")
    for cat, n in cnt.most_common():
        print(f"  {cat:20s}: {n}")


if __name__ == "__main__":
    main()
