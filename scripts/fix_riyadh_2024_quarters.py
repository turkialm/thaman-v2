"""
Fix quarter_id / sale_quarter / REI columns for 2024 rows in features_riyadh.csv.

Bug: quarter_id was computed as year*10 + sale_quarter where sale_quarter was already
in YYYYQ format (e.g. 20241), producing values like 40481 instead of 20241.
Effect: all 2024 rows landed in holdout (cutoff 20251) and got wrong REI values.

Mappings applied to rows where sale_year == 2024:
  sale_quarter:            {20241→1, 20243→3, 20244→4}
  quarter_id:              {40481→20241, 40483→20243, 40484→20244}
  sale_quarter_sin:        {1→1.0, 3→-1.0, 4→0.0}
  sale_quarter_cos:        {1→0.0, 3→0.0,  4→1.0}
  rei_residential_qtr_idx: {20241→99.18, 20243→101.35, 20244→102.35}
  rei_yoy_change:          {20241→-1.10, 20243→1.60,   20244→3.10}
  rei_qoq_change:          {20241→-0.10, 20243→0.20,   20244→1.00}
  rei_apt_idx:             {20241→99.18, 20243→101.35,  20244→102.35}
"""

import math
import polars as pl
from pathlib import Path

BASE = Path(__file__).parent.parent
CSV  = BASE / "data" / "processed" / "features_riyadh.csv"

print(f"Reading {CSV} ...")
df = pl.read_csv(CSV, encoding="utf-8-sig")
print(f"  Shape: {df.shape}")

# ── Snapshot before ───────────────────────────────────────────────────────────
n_2024 = df.filter(pl.col("sale_year") == 2024).shape[0]
bad_qids_before = df.filter(pl.col("quarter_id").is_in([40481, 40483, 40484])).shape[0]
print(f"\nBefore patch:")
print(f"  2024 rows:             {n_2024}")
print(f"  Rows with bad qid:     {bad_qids_before}")
print(f"  All quarter_ids:       {sorted(df['quarter_id'].unique().to_list())}")

# ── REI lookup keyed by CORRECT quarter_id (YYYYQ format) ────────────────────
REI = {
    20241: dict(res=99.18,  yoy=-1.10, qoq=-0.10, apt=99.18),
    20243: dict(res=101.35, yoy=1.60,  qoq=0.20,  apt=101.35),
    20244: dict(res=102.35, yoy=3.10,  qoq=1.00,  apt=102.35),
}

# Trig values for sale_quarter_sin / cos  (sin/cos of 2π*q/4)
def sq_sin(q):
    return round(math.sin(2 * math.pi * q / 4), 10)

def sq_cos(q):
    return round(math.cos(2 * math.pi * q / 4), 10)

QTR_SIN = {1: sq_sin(1), 3: sq_sin(3), 4: sq_sin(4)}
QTR_COS = {1: sq_cos(1), 3: sq_cos(3), 4: sq_cos(4)}

print(f"\nTrig values:")
for q in [1, 3, 4]:
    print(f"  Q{q}: sin={QTR_SIN[q]:.6f}  cos={QTR_COS[q]:.6f}")

# ── Build mapping from bad quarter_id → corrected values ────────────────────
# bad → good:  40481→20241, 40483→20243, 40484→20244
BAD_TO_GOOD = {40481: 20241, 40483: 20243, 40484: 20244}
# bad quarter_id → correct sale_quarter (1,3,4)
BAD_TO_SQ = {40481: 1, 40483: 3, 40484: 4}

# ── Apply patches using polars when/then expressions ─────────────────────────
mask_2024 = pl.col("sale_year") == 2024

df = df.with_columns([
    # Fix sale_quarter: map YYYYQ back to Q (for 2024 rows only)
    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(1))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(3))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(4))
      .otherwise(pl.col("sale_quarter"))
      .alias("sale_quarter"),

    # Fix quarter_id
    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(20241))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(20243))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(20244))
      .otherwise(pl.col("quarter_id"))
      .alias("quarter_id"),

    # Fix sale_quarter_sin
    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(QTR_SIN[1]))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(QTR_SIN[3]))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(QTR_SIN[4]))
      .otherwise(pl.col("sale_quarter_sin"))
      .alias("sale_quarter_sin"),

    # Fix sale_quarter_cos
    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(QTR_COS[1]))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(QTR_COS[3]))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(QTR_COS[4]))
      .otherwise(pl.col("sale_quarter_cos"))
      .alias("sale_quarter_cos"),

    # Fix REI columns — keyed by bad quarter_id (before it's patched above,
    # so use original column state via lazy evaluation of 40481/40483/40484)
    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(REI[20241]["res"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(REI[20243]["res"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(REI[20244]["res"]))
      .otherwise(pl.col("rei_residential_qtr_idx"))
      .alias("rei_residential_qtr_idx"),

    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(REI[20241]["yoy"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(REI[20243]["yoy"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(REI[20244]["yoy"]))
      .otherwise(pl.col("rei_yoy_change"))
      .alias("rei_yoy_change"),

    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(REI[20241]["qoq"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(REI[20243]["qoq"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(REI[20244]["qoq"]))
      .otherwise(pl.col("rei_qoq_change"))
      .alias("rei_qoq_change"),

    pl.when(mask_2024 & (pl.col("quarter_id") == 40481)).then(pl.lit(REI[20241]["apt"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40483)).then(pl.lit(REI[20243]["apt"]))
      .when(mask_2024 & (pl.col("quarter_id") == 40484)).then(pl.lit(REI[20244]["apt"]))
      .otherwise(pl.col("rei_apt_idx"))
      .alias("rei_apt_idx"),
])

# ── Verification ──────────────────────────────────────────────────────────────
bad_qids_after = df.filter(pl.col("quarter_id").is_in([40481, 40483, 40484])).shape[0]
qids_2024 = sorted(df.filter(pl.col("sale_year") == 2024)["quarter_id"].unique().to_list())
sq_2024   = sorted(df.filter(pl.col("sale_year") == 2024)["sale_quarter"].unique().to_list())

print(f"\nAfter patch:")
print(f"  Bad quarter_ids remaining: {bad_qids_after}  (expected 0)")
print(f"  2024 quarter_id values:    {qids_2024}  (expected [20241, 20243, 20244])")
print(f"  2024 sale_quarter values:  {sq_2024}   (expected [1, 3, 4])")

# Show REI sanity
rei_check = (
    df.filter(pl.col("sale_year") == 2024)
    .select(["quarter_id", "sale_quarter", "rei_residential_qtr_idx",
             "rei_yoy_change", "rei_qoq_change", "rei_apt_idx"])
    .unique()
    .sort("quarter_id")
)
print(f"\nREI check for 2024:")
print(rei_check)

assert bad_qids_after == 0, "ERROR: bad quarter_ids still present!"
assert qids_2024 == [20241, 20243, 20244], f"ERROR: unexpected quarter_ids: {qids_2024}"

# Overall split preview (using training script's 80/20 logic)
all_qids = sorted(df["quarter_id"].unique().to_list())
cutoff_idx = int(len(all_qids) * 0.80)
cutoff_qid = all_qids[cutoff_idx]
work_n = df.filter(pl.col("quarter_id") < cutoff_qid).shape[0]
hold_n = df.filter(pl.col("quarter_id") >= cutoff_qid).shape[0]
print(f"\nSplit preview (80/20 on {len(all_qids)} unique quarter_ids):")
print(f"  Cutoff quarter_id: {cutoff_qid}")
print(f"  Work set: {work_n} rows")
print(f"  Holdout:  {hold_n} rows")
print(f"  All quarter_ids: {all_qids}")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {CSV} ...")
df.write_csv(CSV)
print("Done. features_riyadh.csv patched successfully.")
