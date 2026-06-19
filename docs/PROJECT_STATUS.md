# THAMAN — Project Status
> Dual-City AI-Powered AVM (NYC + Riyadh)
> Last updated: 2026-05-18

---

## ✅ Project Complete — All Phases Done

The system is fully deployed at:
- **HuggingFace:** https://huggingface.co/spaces/Turki-Almurahhem/thaman
- **GitHub:** https://github.com/turkialm/thaman-v2

### Launch Locally

```bash
cd /Users/totam/Desktop/new_try
uvicorn api.main:app --port 8000
# Open: http://localhost:8000/ui
```

---

## Phase 1 — NYC Data Pipeline ✅

| Item | Detail |
|---|---|
| Training rows | 185,092 NYC sales (2022–2026) |
| Feature matrix | `data/processed/features_v4.csv` — 185K × 104 cols |
| Key sources | NYC Open Data sales, PLUTO, MTA, NYPD, 311, NTA, FRED, Airbnb |
| Feature groups | Structural, zoning/FAR, transit, QoL amenities, NTA encoding, ACRIS price history, building health (HPD/DOB), macro |

---

## Phase 2 — Riyadh Data Pipeline ✅

| Item | Detail |
|---|---|
| Training rows | 6,910 district-quarter observations (2018–2025) |
| Districts | 163 unique Riyadh districts |
| Feature matrix | `data/processed/features_riyadh.csv` — 6,910 × 87 cols |
| Key sources | Saudi Open Data Portal quarterly reports, SA_Aqar rentals, OSM Overpass polygons |
| Feature groups | Location, property type, metro/bus transit, traffic, commercial, air quality (NO₂/SO₂/PM₁₀/O₃), macro (REI + salary), district aggregates, QoL POIs (6 types), rental signals, temporal |

---

## Phase 3 — NYC Model (Stack v11) ✅

| Metric | Value |
|---|---|
| R² (holdout) | 0.6450 |
| MedAPE (holdout) | 20.24% |
| Holdout rows | 27,763 |
| Features | 104 |
| CV | 5-fold Spatial GroupKFold (by NTA) |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |
| Model files | `models/xgboost_model.json`, `models/thaman_stack.pkl`, `models/meta.json` |

Key v11 additions: HPD violation severity by ZIP, DOB permit activity, 311 rodent/heat density by NTA, MTA station quality (CBD connectivity, route count, ADA).

---

## Phase 4 — Riyadh Model (Stack v1) ✅

| Metric | Value |
|---|---|
| OOF R² | 0.9427 |
| OOF MedAPE | 8.28% |
| Holdout R² | 0.6747 |
| Holdout MedAPE | 23.45% |
| Holdout MAE | 1,206 SAR/sqm |
| Holdout period | 2025 Q1–Q3 (fully out-of-sample) |
| Features | 76 |
| Training rows | 4,664 |
| Holdout rows | 2,246 |
| CV | 5-fold Spatial GroupKFold (by district_ar) |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |
| Model files | `models/riyadh_stack.pkl`, `models/riyadh_meta.json` |

Note: large OOF-to-holdout gap explained by temporal distribution shift (2025 post-Metro market) + district-aggregate granularity inflating OOF R².

---

## Phase 5 — Backend API ✅

| Endpoint | Description |
|---|---|
| `GET /health` | Model + spatial status |
| `GET /bldgclasses` | NYC building class codes |
| `POST /predict` | NYC price (USD) + SHAP drivers + QC flags |
| `POST /predict/riyadh` | Riyadh price (SAR/sqm + total) + spatial features |
| `GET /layers/nta` | NYC NTA choropleth GeoJSON (21 layers) |
| `GET /layers/district` | Riyadh district polygon GeoJSON (12 layers, 133 polygons) |
| `GET /nearby` | Nearby comparable NYC sales |
| `GET /sales/tile` | Tile-based NYC sales bubble layer |

NYC spatial features: KD-tree lookups for subway, bus, parks, schools, hospital, waterfront, bike lanes; BallTree for Airbnb density; NTA polygon lookup for income/crime/noise.

Riyadh spatial features: KD-tree lookups for metro, bus, commercial POIs, air quality stations, mosques, malls, schools, hospitals, parks, entertainment; composite connectivity score.

---

## Phase 6 — Frontend ✅

**Dual-city UI** with city toggle (🗽 NYC ↔ 🕌 Riyadh):

| Feature | NYC | Riyadh |
|---|---|---|
| Map center | New York City | Riyadh (24.71°N, 46.68°E) |
| Pin validation | NYC boundary GeoJSON | Riyadh bbox |
| Form fields | type, size, age, floors, units, borough | type, area_sqm |
| Price display | USD total | SAR/sqm + SAR total |
| Result panel | SHAP drivers + spatial grid | Spatial grid (12 metrics) |
| Choropleth | 21-layer NTA choropleth | 12-layer district polygon choropleth |
| Geocoder | NYC-constrained Nominatim | Riyadh-constrained Nominatim |
| Header badges | Stack v11, R²=0.647, MedAPE, Analytics | Stack v1, R²=0.675, MedAPE 23.45% |
| Language | EN / عربي | EN / عربي |

Performance fixes: NTA simplified 4.4 MB → 436 KB; sales debounce 250 ms → 600 ms; Riyadh mode skips NYC sales fetch; layers lazy-loaded on first click.

---

## Phase 7 — Deployment ✅

| Item | Detail |
|---|---|
| Platform | Hugging Face Spaces (Docker) |
| Port | 7860 (HF) / 8000 (local) |
| Dockerfile | `python:3.11-slim` + GDAL + all model + data files |
| GitHub | https://github.com/turkialm/thaman-v2 |
| Paper | `docs/thaman_paper.txt` — 1,334 lines, dual-city BSc paper |

---

## Current File Tree

```
thaman-v2/
├── api/
│   ├── main.py          # FastAPI (NYC + Riyadh endpoints)
│   ├── spatial.py       # SpatialLookup + RiyadhSpatialLookup
│   └── models.py        # Pydantic schemas
├── models/
│   ├── scorer.py
│   ├── xgboost_model.json      # NYC XGBoost
│   ├── thaman_stack.pkl        # NYC stack
│   ├── meta.json               # NYC meta
│   ├── luxury_model.json       # NYC luxury segment
│   ├── riyadh_stack.pkl        # Riyadh stack
│   └── riyadh_meta.json        # Riyadh meta
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── charts.html
│   └── nyc_boundary.geojson
├── training/
│   ├── train_stack_v2.py
│   └── train_stack_riyadh_v1.py
├── scripts/
│   ├── riyadh_feature_engineering.py
│   └── download_more_sales.py
├── data/
│   ├── processed/
│   │   ├── features_v4.csv
│   │   ├── features_riyadh.csv
│   │   ├── nta_simplified.geojson
│   │   ├── riyadh_district_polygons.geojson
│   │   └── district_centroids.csv
│   └── raw/  (spatial reference files)
├── docs/
│   ├── thaman_paper.txt
│   ├── DATA_CATALOG.md
│   └── PROJECT_STATUS.md
├── tests/
├── Dockerfile
└── requirements.txt
```
