"""
analyze_listings.py
Descriptive analysis of Haraj.com.sa Riyadh property listings (2026-05-18).
Outputs: docs/price_by_type.png, docs/price_by_district.png,
         docs/price_heatmap_district.png, docs/district_price_summary.csv
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = "/Users/totam/Desktop/new_try"
CSV_IN = os.path.join(BASE, "data/raw/saudi_listings_haraj_20260518.csv")
DOCS   = os.path.join(BASE, "docs")
os.makedirs(DOCS, exist_ok=True)

DPI = 150

# ── load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)

# Recalculate price_per_sqm where possible (guards against bad source values)
mask_recalc = (df["price_sar"] > 0) & (df["area_sqm"] > 0)
df.loc[mask_recalc, "price_per_sqm"] = df.loc[mask_recalc, "price_sar"] / df.loc[mask_recalc, "area_sqm"]

# Filter out non-positive prices
df = df[(df["price_sar"] > 0) & (df["price_per_sqm"] > 0)].copy()
df["property_type_en"] = df["property_type_en"].str.strip().str.lower()

# Normalise type labels
type_map = {"apartment": "Apartment", "villa": "Villa", "plot": "Plot",
            "building": "Building", "land": "Plot"}
df["type_label"] = df["property_type_en"].map(type_map).fillna(df["property_type_en"].str.title())

n_total = len(df)

# ── console output ─────────────────────────────────────────────────────────────
print("=" * 60)
print("THAMAN — Haraj Riyadh Listings: Descriptive Analysis")
print("=" * 60)
print(f"\nTotal listings (after filter): {n_total:,}")
print("\nListings by type:")
for t, cnt in df["type_label"].value_counts().items():
    print(f"  {t:<12} {cnt:>4}  ({cnt/n_total*100:.1f}%)")

ppsm = df["price_per_sqm"]
print(f"\nOverall price_per_sqm (SAR/sqm):")
print(f"  Min    : {ppsm.min():>12,.0f}")
print(f"  Median : {ppsm.median():>12,.0f}")
print(f"  Mean   : {ppsm.mean():>12,.0f}")
print(f"  Max    : {ppsm.max():>12,.0f}")

print("\nMedian price_per_sqm by type:")
for t, med in df.groupby("type_label")["price_per_sqm"].median().sort_values(ascending=False).items():
    print(f"  {t:<12} {med:>10,.0f} SAR/sqm")

dist_med = (df.groupby("district")["price_per_sqm"]
              .agg(["median", "count"])
              .query("count >= 3")
              .sort_values("median", ascending=False))

print("\nTop 5 most expensive districts (median SAR/sqm, n≥3):")
for d, row in dist_med.head(5).iterrows():
    print(f"  {d:<25} {row['median']:>9,.0f}  (n={int(row['count'])})")

print("\nTop 5 cheapest districts (median SAR/sqm, n≥3):")
for d, row in dist_med.tail(5).iterrows():
    print(f"  {d:<25} {row['median']:>9,.0f}  (n={int(row['count'])})")

bed_pct = df["bedrooms"].notna().mean() * 100
print(f"\n% listings with bedrooms data: {bed_pct:.1f}%")
print("Average bedrooms per type (where available):")
for t, avg in df.groupby("type_label")["bedrooms"].mean().items():
    print(f"  {t:<12} {avg:.1f}")

# ── Chart 1: Box plots — price_per_sqm by type (log y-axis) ───────────────────
type_order = (df.groupby("type_label")["price_per_sqm"]
                .median()
                .sort_values(ascending=False)
                .index.tolist())

groups = [df.loc[df["type_label"] == t, "price_per_sqm"].values for t in type_order]

fig, ax = plt.subplots(figsize=(9, 6))
bp = ax.boxplot(groups, labels=type_order, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor="steelblue", markeredgecolor="none"),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.5))

colors = plt.cm.Set2(np.linspace(0, 1, len(type_order)))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_yscale("log")
ax.set_ylabel("Price per sqm (SAR) — log scale", fontsize=11)
ax.set_xlabel("Property Type", fontsize=11)
ax.set_title("Asking Price Distribution by Property Type — Riyadh 2026", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")

# Median labels
medians = [df.loc[df["type_label"] == t, "price_per_sqm"].median() for t in type_order]
for i, (med, grp) in enumerate(zip(medians, groups), start=1):
    ax.text(i, med * 1.12, f"{med:,.0f}", ha="center", va="bottom",
            fontsize=9, color="black", fontweight="bold")

# Count labels below x-axis
for i, (t, grp) in enumerate(zip(type_order, groups), start=1):
    ax.text(i, ax.get_ylim()[0], f"n={len(grp)}", ha="center", va="top",
            fontsize=8, color="gray", transform=ax.get_xaxis_transform())

fig.tight_layout()
out1 = os.path.join(DOCS, "price_by_type.png")
fig.savefig(out1, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\n[saved] {out1}")

# ── Chart 2: Horizontal bar — top 25 districts (apartments, n≥3) ───────────────
apts = df[df["type_label"] == "Apartment"].copy()
dist_apt = (apts.groupby("district")
                .agg(median_ppsm=("price_per_sqm", "median"),
                     n=("price_per_sqm", "count"))
                .query("n >= 3")
                .sort_values("median_ppsm", ascending=False)
                .head(25))

fig, ax = plt.subplots(figsize=(10, 9))
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=dist_apt["median_ppsm"].min(),
                          vmax=dist_apt["median_ppsm"].max())
bar_colors = [cmap(norm(v)) for v in dist_apt["median_ppsm"]]

bars = ax.barh(range(len(dist_apt)), dist_apt["median_ppsm"], color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(dist_apt)))
ax.set_yticklabels(dist_apt.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Median Price per sqm (SAR)", fontsize=11)
ax.set_title("Median Asking Price by District (Apartments, SAR/sqm)", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")

# Value + n labels
for i, (val, n) in enumerate(zip(dist_apt["median_ppsm"], dist_apt["n"])):
    ax.text(val + dist_apt["median_ppsm"].max() * 0.01, i,
            f"{val:,.0f}  (n={n})", va="center", fontsize=8.5)

ax.set_xlim(0, dist_apt["median_ppsm"].max() * 1.22)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Median SAR/sqm", fontsize=9)

fig.tight_layout()
out2 = os.path.join(DOCS, "price_by_district.png")
fig.savefig(out2, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out2}")

# ── Chart 3: Scatter map — lat/lon colored by price_per_sqm ───────────────────
map_df = df.dropna(subset=["lat", "lon", "price_per_sqm"]).copy()
map_df = map_df[(map_df["lat"].between(24.0, 25.5)) & (map_df["lon"].between(46.0, 47.5))]

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    map_df["lon"], map_df["lat"],
    c=map_df["price_per_sqm"],
    s=(map_df["area_sqm"].fillna(100) / 50).clip(10, 200),
    cmap="RdYlBu_r",
    alpha=0.5,
    edgecolors="none",
    norm=mcolors.LogNorm(vmin=map_df["price_per_sqm"].quantile(0.02),
                          vmax=map_df["price_per_sqm"].quantile(0.98))
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label("Price per sqm (SAR) — log scale", fontsize=9)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Riyadh Property Price Heatmap — Asking Prices", fontsize=13, fontweight="bold")
ax.set_facecolor("#f0f0f0")
ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

# Legend for dot size
for area_ex, label in [(50, "50 sqm"), (150, "150 sqm"), (500, "500 sqm")]:
    ax.scatter([], [], s=area_ex/50, c="gray", alpha=0.5, label=label)
ax.legend(title="Area", fontsize=8, title_fontsize=8, loc="lower right")

fig.tight_layout()
out3 = os.path.join(DOCS, "price_heatmap_district.png")
fig.savefig(out3, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out3}")

# ── CSV: district_price_summary.csv ───────────────────────────────────────────
def types_list(s):
    return ", ".join(sorted(s.dropna().unique()))

summary = (df.groupby("district")
             .agg(
                 count=("price_per_sqm", "count"),
                 median_price_sqm=("price_per_sqm", "median"),
                 mean_price_sqm=("price_per_sqm", "mean"),
                 median_area_sqm=("area_sqm", "median"),
                 median_price_sar=("price_sar", "median"),
                 property_types=("type_label", types_list),
             )
             .sort_values("median_price_sqm", ascending=False)
             .reset_index())

summary["median_price_sqm"] = summary["median_price_sqm"].round(0).astype(int)
summary["mean_price_sqm"]   = summary["mean_price_sqm"].round(0).astype(int)
summary["median_area_sqm"]  = summary["median_area_sqm"].round(1)
summary["median_price_sar"] = summary["median_price_sar"].round(0).astype(int)

out4 = os.path.join(DOCS, "district_price_summary.csv")
summary.to_csv(out4, index=False, encoding="utf-8-sig")
print(f"[saved] {out4}  ({len(summary)} districts)")

print("\nDone.")
