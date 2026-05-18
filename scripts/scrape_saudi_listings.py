"""
scrape_saudi_listings.py - Playwright scraper for Haraj.com.sa Saudi property listings
Usage: python scripts/scrape_saudi_listings.py [--max-pages N] [--type all|apartment|villa|plot|building]
Output: data/raw/saudi_listings_haraj_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterator

# ── third-party (must be installed) ──────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout
except ImportError as exc:
    sys.exit(
        f"[ERROR] Missing dependency: {exc}. "
        "Run: pip install playwright && playwright install chromium"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
TODAY        = date.today().strftime("%Y%m%d")

CSV_COLUMNS = [
    "source", "listing_id", "district", "lat", "lon",
    "property_type_ar", "property_type_en",
    "price_sar", "area_sqm", "price_per_sqm",
    "bedrooms", "bathrooms", "age_years", "listing_date", "url",
]

# Haraj tag URLs for each property type (Riyadh-specific)
HARAJ_URLS: dict[str, str] = {
    "apartment": (
        "https://haraj.com.sa/tags/"
        "%D8%A7%D9%84%D8%B1%D9%8A%D8%A7%D8%B6_%D8%B4%D9%82%D9%82%20%D9%84%D9%84%D8%A8%D9%8A%D8%B9"
    ),
    "villa": (
        "https://haraj.com.sa/tags/"
        "%D8%A7%D9%84%D8%B1%D9%8A%D8%A7%D8%B6_%D9%81%D9%84%D9%84%20%D9%84%D9%84%D8%A8%D9%8A%D8%B9"
    ),
    "plot": (
        "https://haraj.com.sa/tags/"
        "%D8%A7%D9%84%D8%B1%D9%8A%D8%A7%D8%B6_%D8%A7%D8%B1%D8%A7%D8%B6%D9%8A%20%D9%84%D9%84%D8%A8%D9%8A%D8%B9"
    ),
    "building": (
        "https://haraj.com.sa/tags/"
        "%D8%A7%D9%84%D8%B1%D9%8A%D8%A7%D8%B6_%D8%B9%D9%85%D8%A7%D8%B1%D8%A9%20%D9%84%D9%84%D8%A8%D9%8A%D8%B9"
    ),
}

# Property type normalization (Arabic and English → canonical English slug)
TYPE_MAP: list[tuple[str, str]] = [
    # Apartment
    ("Apartment",          "apartment"),
    ("apartment",          "apartment"),
    ("شقة",                "apartment"),
    ("شقق",                "apartment"),
    # Villa
    ("Villa",              "villa"),
    ("villa",              "villa"),
    ("فيلا",               "villa"),
    ("فلل",                "villa"),
    # Plot / Land
    ("Residential_Land",   "plot"),
    ("Land",               "plot"),
    ("land",               "plot"),
    ("أرض",                "plot"),
    ("ارض",                "plot"),
    ("أراضي",              "plot"),
    # Building
    ("Building",           "building"),
    ("building",           "building"),
    ("عمارة",              "building"),
    ("عمائر",              "building"),
]

# Description field patterns (Arabic labels)
_RE_PRICE    = re.compile(r"سعر(?:\s*الوحدة)?\s*[:\-–]\s*([\d,\.]+)", re.UNICODE)
_RE_AREA     = re.compile(r"مساحة(?:\s*العقار)?\s*[:\-–]\s*([\d,\.]+)", re.UNICODE)
_RE_ROOMS    = re.compile(r"عدد\s*الغرف\s*[:\-–]\s*([\d]+)", re.UNICODE)
_RE_BATHS    = re.compile(r"(?:عدد\s*)?(?:الحمامات?|دورات?\s*المياه)\s*[:\-–]\s*([\d]+)", re.UNICODE)
_RE_TYPE     = re.compile(r"نوع\s*العقار\s*[:\-–]\s*([^\n\r,،]+)", re.UNICODE)
_RE_AGE      = re.compile(r"عمر\s*العقار\s*[:\-–]\s*([^\n\r,،]+)", re.UNICODE)
_RE_CITY     = re.compile(r"المدينة\s*[:\-–]\s*([^\n\r,،]+)", re.UNICODE)
_RE_DISTRICT = re.compile(r"الحي\s*[:\-–]\s*([^\n\r,،]+)", re.UNICODE)
_RE_MAPS     = re.compile(r"maps\.google\.com/\?q=([\-\d\.]+),([\-\d\.]+)", re.UNICODE)
_RE_LISTING_ID = re.compile(r"haraj\.com\.sa/(\d+)/")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_type(raw: str) -> tuple[str, str]:
    """Return (type_ar, type_en) tuple. type_ar is the raw value, type_en is canonical."""
    raw = raw.strip()
    for token, slug in TYPE_MAP:
        if token in raw:
            return raw, slug
    return raw, "other"


def _parse_age(raw: str) -> int | None:
    """Convert age string to integer years. New/جديد → 0."""
    raw = raw.strip()
    if raw in ("New", "new", "جديد", "جديدة"):
        return 0
    m = re.search(r"(\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def _parse_number(raw: str) -> float | None:
    """Strip commas/spaces and parse as float."""
    cleaned = re.sub(r"[,\s]", "", raw.strip())
    try:
        v = float(cleaned)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def _extract_listing_id(url: str) -> str:
    """Extract numeric listing ID from haraj.com.sa URL."""
    m = _RE_LISTING_ID.search(url)
    return m.group(1) if m else url.strip("/").split("/")[-1]


def _parse_description(desc: str) -> dict:
    """Parse structured fields from Haraj listing description text."""
    result: dict = {}

    m = _RE_PRICE.search(desc)
    result["price_sar"] = _parse_number(m.group(1)) if m else None

    m = _RE_AREA.search(desc)
    result["area_sqm"] = _parse_number(m.group(1)) if m else None

    m = _RE_ROOMS.search(desc)
    result["bedrooms"] = int(m.group(1)) if m else None

    m = _RE_BATHS.search(desc)
    result["bathrooms"] = int(m.group(1)) if m else None

    m = _RE_TYPE.search(desc)
    if m:
        ar, en = _map_type(m.group(1))
        result["property_type_ar"] = ar
        result["property_type_en"] = en
    else:
        result["property_type_ar"] = None
        result["property_type_en"] = None

    m = _RE_AGE.search(desc)
    result["age_years"] = _parse_age(m.group(1)) if m else None

    m = _RE_CITY.search(desc)
    result["city"] = m.group(1).strip() if m else None

    m = _RE_DISTRICT.search(desc)
    result["district"] = m.group(1).strip() if m else None

    m = _RE_MAPS.search(desc)
    if m:
        result["lat"] = float(m.group(1))
        result["lon"] = float(m.group(2))
    else:
        result["lat"] = None
        result["lon"] = None

    return result


def _parse_json_ld_page(json_ld: dict, prop_type_hint: str) -> list[dict]:
    """
    Parse a Haraj JSON-LD page blob and return a list of CSV row dicts.
    Filters out non-Riyadh listings and listings with missing price or area.
    """
    rows: list[dict] = []

    try:
        items = json_ld["mainEntity"]["itemListElement"]
    except (KeyError, TypeError):
        return rows

    for element in items:
        try:
            listing = element.get("item", {})
            if listing.get("@type") != "RealEstateListing":
                continue

            # Address filter — Riyadh only
            address = listing.get("address", {})
            locality = address.get("addressLocality", "")
            if locality != "الرياض":
                continue

            url = listing.get("url", "")
            desc = listing.get("description", "")
            date_posted = listing.get("datePosted", "")
            listing_date = date_posted[:10] if date_posted else None

            parsed = _parse_description(desc)

            price_sar = parsed.get("price_sar")
            area_sqm  = parsed.get("area_sqm")

            # Skip if price or area is zero / missing
            if not price_sar or not area_sqm:
                continue

            price_per_sqm = round(price_sar / area_sqm, 2)

            type_ar = parsed.get("property_type_ar") or ""
            type_en = parsed.get("property_type_en") or prop_type_hint

            # If we couldn't parse the type from description, use the URL hint
            if type_en == "other" or not type_en:
                type_en = prop_type_hint

            row = {
                "source":            "haraj",
                "listing_id":        _extract_listing_id(url),
                "district":          parsed.get("district") or "",
                "lat":               parsed.get("lat"),
                "lon":               parsed.get("lon"),
                "property_type_ar":  type_ar,
                "property_type_en":  type_en,
                "price_sar":         price_sar,
                "area_sqm":          area_sqm,
                "price_per_sqm":     price_per_sqm,
                "bedrooms":          parsed.get("bedrooms"),
                "bathrooms":         parsed.get("bathrooms"),
                "age_years":         parsed.get("age_years"),
                "listing_date":      listing_date,
                "url":               url,
            }
            rows.append(row)

        except Exception as exc:
            print(f"    [WARN] item parse error: {exc}")
            continue

    return rows


def _fetch_json_ld(page: Page, url: str) -> dict | None:
    """
    Navigate to a Haraj tag page and extract the JSON-LD blob.
    Returns parsed dict or None if element not found.
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=30_000)
    except PWTimeout:
        print(f"  [WARN] page.goto timeout for {url} — retrying once")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except PWTimeout:
            print(f"  [WARN] second timeout — skipping page")
            return None

    page.wait_for_timeout(1500)

    # Extract JSON-LD element
    handle = page.query_selector("script#json-ld-posts-list")
    if handle is None:
        return None

    raw = handle.inner_text()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [WARN] JSON decode error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core scraper
# ─────────────────────────────────────────────────────────────────────────────

def scrape_haraj(
    prop_types: list[str],
    max_pages: int = 50,
    out_path: Path | None = None,
) -> int:
    """
    Scrape Haraj.com.sa for the given property types.
    Writes (appends) results to out_path.
    Returns total number of rows written.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = DATA_RAW / f"saudi_listings_haraj_{TODAY}.csv"

    file_exists = out_path.exists() and out_path.stat().st_size > 0
    total_written = 0

    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
            fh.flush()
        print(f"[INFO] Appending to {out_path}")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                locale="ar-SA",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            try:
                for prop_type in prop_types:
                    base_url = HARAJ_URLS[prop_type]
                    print(f"\n[{prop_type.upper()}] base URL: {base_url}")
                    consecutive_empty = 0

                    for page_num in range(1, max_pages + 1):
                        url = base_url if page_num == 1 else f"{base_url}?page={page_num}"
                        print(f"  [haraj/{prop_type}] page {page_num}/{max_pages} — {url}")

                        json_ld = _fetch_json_ld(page, url)

                        # Stop if JSON-LD element not found (end of pagination)
                        if json_ld is None:
                            print(f"  [haraj/{prop_type}] JSON-LD not found — end of pagination at page {page_num}")
                            break

                        rows = _parse_json_ld_page(json_ld, prop_type)
                        riyadh_count = len(rows)

                        if riyadh_count == 0:
                            consecutive_empty += 1
                            print(f"  [haraj/{prop_type}] 0 Riyadh listings (consecutive empty: {consecutive_empty}/3)")
                            if consecutive_empty >= 3:
                                print(f"  [haraj/{prop_type}] 3 consecutive empty pages — stopping")
                                break
                        else:
                            consecutive_empty = 0

                        # Checkpoint: write rows immediately
                        for row in rows:
                            clean = {col: row.get(col) for col in CSV_COLUMNS}
                            writer.writerow(clean)
                        total_written += riyadh_count
                        fh.flush()
                        print(f"  [haraj/{prop_type}] wrote {riyadh_count} rows (total so far: {total_written})")

                        time.sleep(1.5)

            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Partial data saved.")
            finally:
                context.close()
                browser.close()

    return total_written


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Playwright scraper for Haraj.com.sa Saudi property listings (Riyadh)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages to scrape per property type (default: 50, ~21 listings/page)",
    )
    parser.add_argument(
        "--type",
        dest="prop_type",
        choices=["all", "apartment", "villa", "plot", "building"],
        default="all",
        help="Property type to scrape (default: all)",
    )
    args = parser.parse_args()

    if args.prop_type == "all":
        prop_types = list(HARAJ_URLS.keys())
    else:
        prop_types = [args.prop_type]

    out_path = DATA_RAW / f"saudi_listings_haraj_{TODAY}.csv"

    print(f"[INFO] Property types : {prop_types}")
    print(f"[INFO] Max pages/type : {args.max_pages}")
    print(f"[INFO] Est. max rows  : {args.max_pages * len(prop_types) * 21} (≈21 listings/page)")
    print(f"[INFO] Output         : {out_path}")
    print(f"[INFO] Date           : {TODAY}\n")

    total = scrape_haraj(prop_types=prop_types, max_pages=args.max_pages, out_path=out_path)
    print(f"\n[DONE] Total rows written: {total} → {out_path}")


if __name__ == "__main__":
    main()
