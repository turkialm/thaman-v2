"""
process_nyc_overture_qol.py
============================
Extracts QoL-relevant POI categories from the existing Overture places GeoJSON
(data/raw/overture_places.geojson, 425K NYC places) and saves per-category CSVs.

New NYC features produced:
  gyms, pharmacies, coffee_shops, grocery_stores, atms,
  cinemas, libraries, childcare, urgent_care, bars, bike_shops

Output: data/raw/nyc_overture_<category>.csv  per category

Run: python scripts/process_nyc_overture_qol.py
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE    = Path(__file__).resolve().parent.parent
IN_GEO  = BASE / "data" / "raw" / "overture_places.geojson"
OUT_DIR = BASE / "data" / "raw"

# Category mapping: overture basic_category -> our label
# Multiple overture categories can map to one label
CATEGORY_MAP = {
    # Gyms / fitness
    "gym":                          "gyms",
    "fitness_center":               "gyms",
    "yoga_studio":                  "gyms",
    "martial_arts_school":          "gyms",
    "sports_club":                  "gyms",
    # Pharmacies
    "pharmacy":                     "pharmacies",
    "drug_store":                   "pharmacies",
    # Coffee
    "coffee_shop":                  "coffee_shops",
    "tea_room":                     "coffee_shops",
    # Grocery
    "grocery_store":                "grocery_stores",
    "supermarket":                  "grocery_stores",
    "convenience_store":            "grocery_stores",
    "wholesale_store":              "grocery_stores",
    # ATMs
    "atm":                          "atms",
    # Cinemas
    "movie_theater":                "cinemas",
    "cinema":                       "cinemas",
    # Libraries
    "library":                      "libraries",
    "public_library":               "libraries",
    # Childcare
    "childcare":                    "childcare",
    "child_care_facility":          "childcare",
    "preschool":                    "childcare",
    # Urgent care / clinics
    "urgent_care":                  "urgent_care",
    "medical_clinic":               "urgent_care",
    "healthcare_location":          "urgent_care",
    # Bars
    "bar":                          "bars",
    "cocktail_bar":                 "bars",
    "pub":                          "bars",
    "wine_bar":                     "bars",
    # Bike shops (cycling infrastructure proxy)
    "bicycle_shop":                 "bike_shops",
    "bicycle_store":                "bike_shops",
    # Beauty (density = commercial vibrancy)
    "beauty_salon":                 "beauty_salons",
    "hair_salon":                   "beauty_salons",
    "nail_salon":                   "beauty_salons",
    # Hotels / hospitality
    "hotel":                        "hotels",
    "hostel":                       "hotels",
    "motel":                        "hotels",
}

FIELDS = ["category", "lat", "lon", "name", "osm_id"]


def main():
    print(f"Loading {IN_GEO.name} ...")
    with open(IN_GEO) as f:
        geo = json.load(f)

    features = geo["features"]
    print(f"  {len(features):,} total features")

    # Collect by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    unmapped = Counter()

    for feat in features:
        props = feat.get("properties", {})
        basic_cat = props.get("basic_category") or (props.get("categories") or {}).get("primary", "")
        label = CATEGORY_MAP.get(basic_cat)
        if label is None:
            unmapped[basic_cat] += 1
            continue

        coords = feat.get("geometry", {}).get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]

        name = ""
        names = props.get("names")
        if isinstance(names, dict):
            name = names.get("primary", "")
        elif isinstance(names, str):
            name = names

        by_cat[label].append({
            "category": label,
            "lat": lat,
            "lon": lon,
            "name": name,
            "osm_id": props.get("id", ""),
        })

    # Save per category
    print(f"\nExtracted categories:")
    for cat, rows in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        out_path = OUT_DIR / f"nyc_overture_{cat}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  {cat:20s}: {len(rows):6,} → {out_path.name}")

    print(f"\nTotal POIs: {sum(len(v) for v in by_cat.values()):,}")
    print(f"Categories extracted: {len(by_cat)}")


if __name__ == "__main__":
    main()
