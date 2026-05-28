"""
aggregate_suhail_quarterly.py
==============================
Aggregates individual Suhail transaction records into district-quarter rows
compatible with the Riyadh training data format (features_riyadh.csv).

Input:  data/raw/suhail_riyadh_tx_raw.csv
Output: data/raw/suhail_riyadh_quarterly.csv   — district × quarter aggregates
        data/raw/suhail_riyadh_q_summary.txt   — summary printout

Quarter mapping (Gregorian):
  20253 = 2025 Q3 (Jul–Sep 2025)
  20254 = 2025 Q4 (Oct–Dec 2025)   ← new data!
  20261 = 2026 Q1 (Jan–Mar 2026)   ← new data!
  20262 = 2026 Q2 (Apr–Jun 2026)   ← partial

Run: python scripts/aggregate_suhail_quarterly.py
"""

import csv
from pathlib import Path
from collections import defaultdict
import statistics

BASE   = Path(__file__).resolve().parent.parent
IN_TX  = BASE / "data" / "raw" / "suhail_riyadh_tx_raw.csv"
OUT_Q  = BASE / "data" / "raw" / "suhail_riyadh_quarterly.csv"
OUT_SUM = BASE / "data" / "raw" / "suhail_riyadh_q_summary.txt"


def date_to_quarter_id(date_str: str) -> int | None:
    """'2025-10-15' → 20254 (2025 Q4)."""
    if not date_str or len(date_str) < 7:
        return None
    try:
        year = int(date_str[:4])
        month = int(date_str[5:7])
        q = (month - 1) // 3 + 1
        return year * 10 + q
    except ValueError:
        return None


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def main():
    if not IN_TX.exists():
        print(f"ERROR: {IN_TX} not found. Run fetch_suhail_transactions.py first.")
        return

    # ── Load raw transactions ────────────────────────────────────────────────
    bucket: dict[tuple, list[float]] = defaultdict(list)  # (district_ar, quarter_id) -> [psqm]
    deed_counts: dict[tuple, int] = defaultdict(int)
    total_values: dict[tuple, float] = defaultdict(float)

    residential_types = {"سكني", "مختلط"}

    skipped = 0
    loaded = 0
    with open(IN_TX, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            district = row.get("district_ar", "").strip()
            date     = row.get("date", "").strip()
            psqm_str = row.get("psqm", "").strip()
            price_str = row.get("price_sar", "").strip()
            land_use = row.get("land_use", "").strip()

            if not district or not date or not psqm_str:
                skipped += 1
                continue

            try:
                psqm  = float(psqm_str)
                price = float(price_str) if price_str else 0.0
            except ValueError:
                skipped += 1
                continue

            if psqm <= 100 or psqm > 100_000:   # sanity bounds
                skipped += 1
                continue

            qid = date_to_quarter_id(date)
            if qid is None:
                skipped += 1
                continue

            key = (district, qid)
            # Only residential
            if land_use in residential_types or not land_use:
                bucket[key].append(psqm)
                deed_counts[key] += 1
                total_values[key] += price
                loaded += 1

    print(f"Loaded {loaded:,} residential transactions | skipped {skipped:,}")
    print(f"District-quarter combinations: {len(bucket)}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    results = []
    for (district, qid), psqms in bucket.items():
        if len(psqms) < 3:   # skip tiny samples
            continue
        year = qid // 10
        q    = qid % 10
        results.append({
            "district_ar":            district,
            "quarter_id":             qid,
            "sale_year":              year,
            "sale_quarter":           q,
            "deed_count":             deed_counts[(district, qid)],
            "total_value_sar":        round(total_values[(district, qid)], 2),
            "sale_price_sar_sqm":     round(statistics.median(psqms), 2),
            "mean_psqm":              round(statistics.mean(psqms), 2),
            "p25_psqm":               round(percentile(psqms, 25), 2),
            "p75_psqm":               round(percentile(psqms, 75), 2),
            "n_transactions":         len(psqms),
        })

    results.sort(key=lambda r: (r["district_ar"], r["quarter_id"]))

    # ── Save quarterly aggregates ─────────────────────────────────────────────
    fields = list(results[0].keys()) if results else []
    with open(OUT_Q, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"Saved {len(results)} district-quarter rows → {OUT_Q.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    qid_counts = Counter(r["quarter_id"] for r in results)
    lines = [
        "Suhail Quarterly Aggregation Summary",
        "=" * 45,
        f"Total district-quarter rows: {len(results)}",
        f"Quarters covered:",
    ]
    for qid in sorted(qid_counts):
        yr, q = qid // 10, qid % 10
        n = qid_counts[qid]
        total_deeds = sum(r["deed_count"] for r in results if r["quarter_id"] == qid)
        lines.append(f"  {yr} Q{q} (id={qid}): {n} districts, {total_deeds:,} deeds")

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT_SUM.write_text(summary, encoding="utf-8")

    # Check overlap with training districts
    import sys
    try:
        feat_path = BASE / "data" / "processed" / "features_riyadh.csv"
        if feat_path.exists():
            with open(feat_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                train_districts = {row["district_ar"] for row in reader}
            suhail_districts = {r["district_ar"] for r in results}
            overlap = train_districts & suhail_districts
            print(f"\nDistrict overlap with training data:")
            print(f"  Training districts: {len(train_districts)}")
            print(f"  Suhail districts:   {len(suhail_districts)}")
            print(f"  Overlap:            {len(overlap)} ({100*len(overlap)/len(suhail_districts):.0f}% of Suhail)")
    except Exception as e:
        print(f"Could not check overlap: {e}")


if __name__ == "__main__":
    main()
