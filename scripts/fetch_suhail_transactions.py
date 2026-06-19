"""
fetch_suhail_transactions.py
============================
Fetches individual MOJ deed-transfer transactions from Suhail.ai API
for all Riyadh-region neighbourhoods (IDs 1003000–1004999).

Covers roughly May 2025 → May 2026 (last 12 months from API).

Output:
  data/raw/suhail_riyadh_tx_raw.csv        — individual transactions
  data/raw/suhail_nh_id_map.csv            — neighbourhood ID → name mapping

Run: python scripts/fetch_suhail_transactions.py
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE    = Path(__file__).resolve().parent.parent
OUT_TX  = BASE / "data" / "raw" / "suhail_riyadh_tx_raw.csv"
OUT_MAP = BASE / "data" / "raw" / "suhail_nh_id_map.csv"

API_BASE = "https://api2.suhail.ai/transactions/neighbourhood"
REGION_ID = 10
PAGE_SIZE = 500
ID_RANGE  = range(1003000, 1005000)   # 2 000 IDs to scan
MAX_WORKERS = 25
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.suhail.ai/",
}


def fetch_page(nh_id: int, page: int = 1) -> dict | None:
    url = (
        f"{API_BASE}?regionId={REGION_ID}"
        f"&neighbourhoodId={nh_id}&page={page}&pageSize={PAGE_SIZE}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_nh(nh_id: int) -> list[dict]:
    """Return all transactions for one neighbourhood."""
    first = fetch_page(nh_id, 1)
    if not first or not first.get("status"):
        return []
    pagination = first.get("meta", {}).get("pagination", {})
    total = pagination.get("total", 0)
    if total == 0:
        return []

    rows = list(first.get("data", []))
    page_count = (total + PAGE_SIZE - 1) // PAGE_SIZE
    for p in range(2, page_count + 1):
        page_data = fetch_page(nh_id, p)
        if page_data and page_data.get("data"):
            rows.extend(page_data["data"])
        time.sleep(0.05)
    return rows


def parse_tx(tx: dict) -> dict:
    return {
        "nh_id":         tx.get("neighborhoodId", ""),
        "district_ar":   tx.get("neighborhood", ""),
        "province_name": tx.get("provinceName", ""),
        "date":          (tx.get("transactionDate") or "")[:10],
        "price_sar":     tx.get("transactionPrice", ""),
        "psqm":          tx.get("priceOfMeter", ""),
        "area_sqm":      tx.get("totalArea", ""),
        "property_type": tx.get("propertyType", ""),
        "land_use":      tx.get("landUseGroup", ""),
        "parcel_no":     tx.get("parcelNo", ""),
        "block_no":      tx.get("blockNo", ""),
        "tx_number":     tx.get("transactionNumber", ""),
    }


def main():
    print(f"Scanning {len(ID_RANGE):,} neighbourhood IDs with {MAX_WORKERS} workers …")

    nh_map   = {}   # id -> name
    all_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_nh, nh_id): nh_id for nh_id in ID_RANGE}
        done = 0
        for fut in as_completed(futures):
            nh_id = futures[fut]
            try:
                txs = fut.result()
            except Exception:
                txs = []

            if txs:
                name = txs[0].get("neighborhood", "")
                nh_map[nh_id] = name
                all_rows.extend(parse_tx(t) for t in txs if (t.get("priceOfMeter") or 0) > 0)

            done += 1
            if done % 100 == 0:
                print(f"  {done:,}/{len(ID_RANGE):,} IDs scanned | "
                      f"{len(nh_map)} neighbourhoods | {len(all_rows):,} transactions")

    print(f"\nDone: {len(nh_map)} neighbourhoods, {len(all_rows):,} transactions")

    # ── Save neighbourhood map ────────────────────────────────────────────────
    with open(OUT_MAP, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["nh_id", "district_ar"])
        w.writeheader()
        for nh_id, name in sorted(nh_map.items()):
            w.writerow({"nh_id": nh_id, "district_ar": name})
    print(f"Saved neighbourhood map → {OUT_MAP.name}")

    # ── Save transactions ─────────────────────────────────────────────────────
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(OUT_TX, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Saved {len(all_rows):,} transactions → {OUT_TX.name}")

        # Quick summary
        from collections import Counter
        months = Counter(r["date"][:7] for r in all_rows if r["date"])
        print("\nTransactions by month:")
        for m in sorted(months):
            print(f"  {m}: {months[m]:,}")


if __name__ == "__main__":
    main()
