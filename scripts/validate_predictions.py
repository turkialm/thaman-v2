"""
validate_predictions.py
=======================
Load scraped listing CSVs from data/raw/saudi_listings_*.csv,
call the local Thaman API (/predict/riyadh) for each valid listing,
compute prediction errors vs asking prices, and produce:

  - docs/prediction_vs_market.csv   — summary table by type + district
  - docs/validation_scatter.png     — scatter: predicted vs asking (SAR/sqm)

Usage:
    python scripts/validate_predictions.py
    python scripts/validate_predictions.py --api-url http://localhost:8000 --max-rows 500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    import requests
    import matplotlib
    matplotlib.use("Agg")            # headless
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(f"[ERROR] Missing dependency: {exc}. Run: pip install pandas numpy requests matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DOCS_DIR     = PROJECT_ROOT / "docs"

# Riyadh bounding box (must match API validation in models.py)
LAT_MIN, LAT_MAX = 23.5, 26.0
LON_MIN, LON_MAX = 45.5, 48.0

# Arabic → API property_type string (must match RiyadhPredictRequest)
TYPE_API_MAP = {
    "apartment": "شقة",
    "villa":     "فيلا",
    "plot":      "قطعة أرض-سكنى",
    "building":  "عمارة",
    "duplex":    "شقة",    # closest equivalent
    "studio":    "شقة",
    "office":    "عمارة",
    "other":     "شقة",   # default fallback
}

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ─────────────────────────────────────────────────────────────────────────────
# Load scraped data
# ─────────────────────────────────────────────────────────────────────────────

def load_scraped(data_raw: Path) -> pd.DataFrame:
    """
    Load all saudi_listings_*.csv files from data/raw/ and concatenate.
    Filters out rows missing lat, lon, area_sqm, price_sar.
    """
    pattern = str(data_raw / "saudi_listings_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"[ERROR] No scraped files found matching {pattern}. Run scrape_saudi_listings.py first.")

    print(f"[INFO] Found {len(files)} scraped file(s):")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            print(f"       {Path(f).name}: {len(df)} rows")
            dfs.append(df)
        except Exception as exc:
            print(f"  [WARN] Could not read {f}: {exc}")

    if not dfs:
        sys.exit("[ERROR] All scraped files failed to load.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows before filter: {len(combined)}")

    # Coerce numeric columns
    for col in ["lat", "lon", "area_sqm", "price_sar", "price_per_sqm"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Drop duplicates by (source, listing_id)
    if "listing_id" in combined.columns:
        combined.drop_duplicates(subset=["source", "listing_id"], inplace=True)

    # Filter: need lat, lon, area_sqm, price_sar
    mask = (
        combined["lat"].notna() & combined["lon"].notna() &
        combined["area_sqm"].notna() & combined["area_sqm"].gt(0) &
        combined["price_sar"].notna() & combined["price_sar"].gt(0)
    )
    # Riyadh bounding box
    mask &= (
        combined["lat"].between(LAT_MIN, LAT_MAX) &
        combined["lon"].between(LON_MIN, LON_MAX)
    )

    valid = combined[mask].copy()
    valid.reset_index(drop=True, inplace=True)
    print(f"[INFO] Valid rows after filter: {len(valid)}")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# API caller
# ─────────────────────────────────────────────────────────────────────────────

def check_api(api_url: str) -> bool:
    """Check that the API is reachable at /health."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=5,
                            headers={"User-Agent": CHROME_UA})
        if resp.status_code == 200:
            print(f"[INFO] API healthy at {api_url}")
            return True
        print(f"[WARN] API returned {resp.status_code} at {api_url}/health")
        return False
    except requests.RequestException as exc:
        print(f"[ERROR] Cannot reach API at {api_url}: {exc}")
        return False


def predict_one(api_url: str, row: pd.Series) -> dict | None:
    """
    Call /predict/riyadh for one listing row.
    Returns the JSON response dict or None on failure.
    """
    prop_en = str(row.get("property_type_en", "") or "other").lower()
    prop_ar = TYPE_API_MAP.get(prop_en, "شقة")

    payload = {
        "latitude":      float(row["lat"]),
        "longitude":     float(row["lon"]),
        "property_type": prop_ar,
        "area_sqm":      float(row["area_sqm"]),
    }

    try:
        resp = requests.post(
            f"{api_url}/predict/riyadh",
            json=payload,
            timeout=15,
            headers={"Content-Type": "application/json", "User-Agent": CHROME_UA},
        )
        if resp.status_code == 200:
            return resp.json()
        # Log once per error type, not every row
        return None
    except requests.RequestException:
        return None


def batch_predict(df: pd.DataFrame, api_url: str, max_rows: int,
                  delay: float = 0.05) -> pd.DataFrame:
    """
    Run predictions for all valid rows, add result columns to df.
    Returns df with columns: predicted_sqm, predicted_total, ape, abs_error_sqm.
    """
    if max_rows and len(df) > max_rows:
        print(f"[INFO] Sampling {max_rows} rows from {len(df)} for validation")
        df = df.sample(n=max_rows, random_state=42).copy()

    predicted_sqm    = []
    predicted_total  = []
    api_district     = []
    errors_count     = 0

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 50 == 0:
            print(f"  [{i}/{total}] calling API …")
        result = predict_one(api_url, row)
        if result:
            predicted_sqm.append(result.get("predicted_price_sqm"))
            predicted_total.append(result.get("predicted_total_sar"))
            api_district.append(result.get("district_ar"))
        else:
            predicted_sqm.append(None)
            predicted_total.append(None)
            api_district.append(None)
            errors_count += 1
        time.sleep(delay)

    df = df.copy()
    df["predicted_sqm"]    = predicted_sqm
    df["predicted_total"]  = predicted_total
    df["api_district"]     = api_district

    # Compute price_per_sqm from scraped data if missing
    mask = df["price_per_sqm"].isna() & df["price_sar"].notna() & df["area_sqm"].gt(0)
    df.loc[mask, "price_per_sqm"] = df.loc[mask, "price_sar"] / df.loc[mask, "area_sqm"]

    # Absolute percentage error
    valid_mask = df["predicted_sqm"].notna() & df["price_per_sqm"].notna() & df["price_per_sqm"].gt(0)
    df["ape"] = None
    df.loc[valid_mask, "ape"] = (
        (df.loc[valid_mask, "predicted_sqm"] - df.loc[valid_mask, "price_per_sqm"]).abs()
        / df.loc[valid_mask, "price_per_sqm"]
        * 100
    )
    df["abs_error_sqm"] = None
    df.loc[valid_mask, "abs_error_sqm"] = (
        (df.loc[valid_mask, "predicted_sqm"] - df.loc[valid_mask, "price_per_sqm"]).abs()
    )

    print(f"[INFO] API errors: {errors_count}/{total} ({errors_count/total*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + outputs
# ─────────────────────────────────────────────────────────────────────────────

def compute_medape(df: pd.DataFrame) -> float:
    """Overall median APE (%) against asking price."""
    ape = pd.to_numeric(df["ape"], errors="coerce").dropna()
    if len(ape) == 0:
        return float("nan")
    return float(ape.median())


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: MedAPE by property_type_en and district.
    """
    df2 = df.copy()
    df2["ape"] = pd.to_numeric(df2["ape"], errors="coerce")
    df2["price_per_sqm"]  = pd.to_numeric(df2["price_per_sqm"],  errors="coerce")
    df2["predicted_sqm"]  = pd.to_numeric(df2["predicted_sqm"],  errors="coerce")

    mask = df2["ape"].notna()
    df2  = df2[mask]

    group_cols = []
    if "property_type_en" in df2.columns:
        group_cols.append("property_type_en")
    if "district" in df2.columns:
        group_cols.append("district")

    if not group_cols:
        return pd.DataFrame()

    agg = (
        df2.groupby(group_cols, dropna=False)
        .agg(
            n_listings          = ("ape",          "count"),
            medape_pct          = ("ape",          "median"),
            mean_asking_sqm     = ("price_per_sqm", "mean"),
            mean_predicted_sqm  = ("predicted_sqm", "mean"),
            mean_abs_error_sqm  = ("abs_error_sqm", "mean"),
        )
        .reset_index()
        .sort_values("medape_pct")
    )
    agg = agg.round({"medape_pct": 2, "mean_asking_sqm": 0,
                     "mean_predicted_sqm": 0, "mean_abs_error_sqm": 0})
    return agg


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter plot: predicted SAR/sqm vs asking SAR/sqm, coloured by property type.
    Saves PNG to out_path.
    """
    df2 = df.copy()
    for col in ["price_per_sqm", "predicted_sqm"]:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=["price_per_sqm", "predicted_sqm"])

    if len(df2) == 0:
        print("[WARN] No rows with both price_per_sqm and predicted_sqm — skipping scatter.")
        return

    # Remove extreme outliers (top/bottom 1%)
    lo_ask  = df2["price_per_sqm"].quantile(0.01)
    hi_ask  = df2["price_per_sqm"].quantile(0.99)
    lo_pred = df2["predicted_sqm"].quantile(0.01)
    hi_pred = df2["predicted_sqm"].quantile(0.99)
    df2 = df2[
        df2["price_per_sqm"].between(lo_ask, hi_ask) &
        df2["predicted_sqm"].between(lo_pred, hi_pred)
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    types  = df2["property_type_en"].fillna("other").unique()
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, ptype in enumerate(sorted(types)):
        sub = df2[df2["property_type_en"].fillna("other") == ptype]
        ax.scatter(sub["price_per_sqm"], sub["predicted_sqm"],
                   label=ptype, alpha=0.55, s=18,
                   color=colors[i % len(colors)])

    # 45° line
    lims = [
        min(df2["price_per_sqm"].min(), df2["predicted_sqm"].min()),
        max(df2["price_per_sqm"].max(), df2["predicted_sqm"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")

    overall_medape = compute_medape(df)
    n_pts          = len(df2)

    ax.set_xlabel("Asking Price  (SAR / m²)", fontsize=12)
    ax.set_ylabel("Predicted Price  (SAR / m²)", fontsize=12)
    ax.set_title(
        f"Thaman Model vs Market Asking Prices\n"
        f"n={n_pts:,} listings  |  Overall MedAPE = {overall_medape:.1f}%",
        fontsize=13,
    )
    ax.legend(title="Property type", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Source breakdown annotation
    if "source" in df2.columns:
        src_counts = df2["source"].value_counts().to_dict()
        src_text   = "  ".join(f"{k}: {v}" for k, v in sorted(src_counts.items()))
        ax.annotate(f"Sources: {src_text}", xy=(0.02, 0.97), xycoords="axes fraction",
                    fontsize=8, va="top", color="gray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Scatter plot saved: {out_path}")


def print_report(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print a concise validation report to stdout."""
    overall_medape = compute_medape(df)
    n_valid = df["ape"].notna().sum()

    print("\n" + "=" * 60)
    print("  THAMAN MODEL VALIDATION vs MARKET ASKING PRICES")
    print("=" * 60)
    print(f"  Valid comparisons  : {n_valid:,}")
    print(f"  Overall MedAPE     : {overall_medape:.2f}%")
    if "source" in df.columns:
        for src, grp in df[df["ape"].notna()].groupby("source"):
            medape = compute_medape(grp)
            print(f"    {src:10s}       : {medape:.2f}%  ({len(grp)} listings)")
    print()

    if len(summary) > 0:
        # Top 10 best-predicted (lowest MedAPE)
        print("  Best predicted segments (lowest MedAPE):")
        top10 = summary.head(10)
        for _, r in top10.iterrows():
            tag = " | ".join(str(r[c]) for c in summary.columns if c in ["property_type_en", "district"])
            print(f"    {tag:40s}  MedAPE={r['medape_pct']:.1f}%  n={int(r['n_listings'])}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Thaman model predictions against scraped Saudi listing prices."
    )
    parser.add_argument("--api-url",  default="http://localhost:8000",
                        help="Thaman API base URL (default: http://localhost:8000)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max listing rows to validate (0 = all, default: 0)")
    parser.add_argument("--data-dir", default=str(DATA_RAW),
                        help="Directory containing saudi_listings_*.csv files")
    args = parser.parse_args()

    data_raw = Path(args.data_dir)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load scraped data
    df = load_scraped(data_raw)
    if len(df) == 0:
        sys.exit("[ERROR] No valid listings found.")

    # 2. Check API
    if not check_api(args.api_url):
        print("[WARN] API not reachable. Check that `uvicorn api.main:app --port 8000` is running.")
        print("       Continuing — all predictions will be empty.")

    # 3. Run predictions
    max_rows = args.max_rows if args.max_rows > 0 else len(df)
    df = batch_predict(df, args.api_url, max_rows=max_rows)

    # 4. Print report
    summary = make_summary_table(df)
    print_report(df, summary)

    # 5. Save summary CSV
    summary_path = DOCS_DIR / "prediction_vs_market.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[INFO] Summary table saved: {summary_path}")

    # 6. Save scatter plot
    scatter_path = DOCS_DIR / "validation_scatter.png"
    make_scatter(df, scatter_path)

    # 7. Save full detail CSV (optional, useful for deep dives)
    detail_path = DOCS_DIR / "prediction_vs_market_detail.csv"
    save_cols   = [c for c in df.columns if c in [
        "source", "listing_id", "district", "lat", "lon",
        "property_type_en", "property_type_ar",
        "price_sar", "area_sqm", "price_per_sqm",
        "predicted_sqm", "predicted_total", "ape", "abs_error_sqm",
        "bedrooms", "listing_date", "url",
    ]]
    df[save_cols].to_csv(detail_path, index=False, encoding="utf-8")
    print(f"[INFO] Full detail saved: {detail_path}")


if __name__ == "__main__":
    main()
