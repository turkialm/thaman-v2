---
title: THAMAN Property Valuation
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏙️🕌 THAMAN — Dual-City AI Property Valuation

> BSc Graduation Project — Umm Al-Qura University  
> Interactive map-based AVM for **New York City** and **Riyadh** using ensemble ML and Quality-of-Life spatial indicators.

**Live demo:** https://huggingface.co/spaces/Turki-Almurahhem/thaman  
**GitHub:** https://github.com/turkialm/thaman-v2

---

## What It Does

| City | Flow |
|---|---|
| 🗽 **NYC** | Click map → fill building details → get USD price estimate + SHAP drivers |
| 🕌 **Riyadh** | Click map → fill area + type → get SAR/sqm estimate + spatial feature grid |

Toggle between cities with the NYC / Riyadh button in the top-left of the map.

---

## Model Performance

### NYC — Stack v12 (109 features, 185K sales, 2022–2026)

| Metric | Value |
|---|---|
| R² (holdout) | 0.6446 |
| MedAPE | 20.31% |
| Holdout rows | 27,763 |
| CV Strategy | 5-fold Spatial GroupKFold (by NTA) |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |

v12 adds 5 quarterly NTA temporal features (lagged mean log-price, median $/sqft,
sale count, 2-quarter lag, momentum) for leakage-free market trend signals.

### Riyadh — Stack v2 (76 features, 6,910 district-quarter rows, 2018–2025)

| Metric | Value |
|---|---|
| OOF R² | 0.8441 |
| OOF MedAPE | 19.75% |
| Holdout R² | 0.6841 |
| Holdout MedAPE | 22.24% |
| Holdout period | 2025 Q1–Q3 (fully unseen) |
| CV Strategy | 5-fold Spatial GroupKFold (by district) |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, LightGBM, CatBoost, scikit-learn Ridge |
| Backend | FastAPI + Uvicorn |
| Spatial | SciPy KD-trees, GeoPandas, Polars |
| Frontend | Leaflet.js, Chart.js, vanilla JS |
| Deployment | Docker (Hugging Face Spaces) |
| Data pipeline | Polars (Riyadh), Pandas (NYC) |

---

## Run Locally

```bash
git clone https://github.com/turkialm/thaman-v2.git
cd thaman-v2
pip install -r requirements.txt
uvicorn api.main:app --port 8000
# Open: http://localhost:8000/ui
```

> First startup ~30 seconds — loads both ML stacks and all spatial data.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model + spatial status |
| `GET` | `/bldgclasses` | NYC building class codes |
| `POST` | `/predict` | NYC price estimate (USD) |
| `POST` | `/predict/riyadh` | Riyadh price estimate (SAR/sqm) |
| `POST` | `/batch` | NYC batch predictions (up to 50 properties) |
| `GET` | `/layers/nta` | NYC NTA choropleth GeoJSON |
| `GET` | `/layers/district` | Riyadh district polygon GeoJSON |
| `GET` | `/docs` | Swagger UI |

### NYC Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude":40.6892,"longitude":-73.9442,"gross_square_feet":1800,
       "building_age":55,"bldgclass":"A1","borough":3,
       "numfloors":2,"residential_units":1}'
```

### Riyadh Example

```bash
curl -X POST http://localhost:8000/predict/riyadh \
  -H "Content-Type: application/json" \
  -d '{"latitude":24.7136,"longitude":46.6753,
       "property_type":"شقة","area_sqm":150}'
```

---

## Project Structure

```
thaman-v2/
├── api/
│   ├── main.py          # FastAPI — all endpoints (NYC + Riyadh)
│   ├── spatial.py       # NYC SpatialLookup + RiyadhSpatialLookup
│   └── models.py        # Pydantic schemas
├── models/
│   ├── scorer.py            # ThamanScorer (NYC inference)
│   ├── xgboost_model.json   # NYC XGBoost base learner
│   ├── thaman_stack.pkl     # NYC LGB+CAT+Ridge meta
│   ├── meta.json            # NYC feature names + metrics
│   ├── riyadh_stack.pkl     # Riyadh XGB+LGB+CAT+Ridge stack
│   └── riyadh_meta.json     # Riyadh feature names + metrics
├── frontend/
│   ├── index.html       # Dual-city map UI
│   ├── app.js           # Leaflet + city toggle + both forms
│   ├── style.css        # Styles
│   └── charts.html      # NYC analytics dashboard
├── tests/
│   ├── test_api.py              # Smoke + integration (13 tests)
│   ├── test_scorer.py           # Unit tests for ThamanScorer (7 tests)
│   ├── test_regression.py       # Pinned output regression (6 tests, ±5%)
│   ├── test_golden.py           # Market-range golden dataset (14 tests)
│   ├── test_feature_parity.py   # Feature completeness checks (13 tests)
│   ├── test_distribution.py     # Distribution shift detection (12 tests)
│   ├── test_shap.py             # SHAP explainability (15 tests)
│   └── test_load.py             # API load/stress benchmarks (20 tests)
├── training/
│   ├── train_stack_v12.py        # NYC Stack v12 training (current)
│   ├── train_stack_v2.py         # NYC Stack v11 training
│   └── train_stack_riyadh_v1.py  # Riyadh Stack v2 training
├── scripts/
│   └── riyadh_feature_engineering.py  # Riyadh Polars pipeline
├── data/
│   ├── processed/
│   │   ├── features_v4.csv                  # NYC feature matrix
│   │   ├── features_riyadh.csv              # Riyadh feature matrix (6,910 × 87)
│   │   ├── nta_simplified.geojson           # NYC NTA polygons (436 KB)
│   │   ├── riyadh_district_polygons.geojson # 133 Riyadh district polygons
│   │   └── district_centroids.csv           # 147 district lat/lon
│   └── raw/                                 # Spatial reference files
├── docs/
│   ├── thaman_paper.txt     # BSc paper (1,334 lines)
│   ├── DATA_CATALOG.md
│   └── PROJECT_STATUS.md
├── Dockerfile
└── requirements.txt
```

---

## Test Suite

100 tests across 8 files — run with `pytest tests/ -v` (~90 s).

| File | Type | Tests |
|---|---|---|
| `test_api.py` | Smoke / integration | 13 |
| `test_scorer.py` | Unit (ThamanScorer) | 7 |
| `test_regression.py` | Pinned output regression (±5%) | 6 |
| `test_golden.py` | Market-range sanity + ordering | 14 |
| `test_feature_parity.py` | NTA / v11 / v12 feature completeness | 13 |
| `test_distribution.py` | Distribution shift detection | 12 |
| `test_shap.py` | SHAP driver structure + sensitivity | 15 |
| `test_load.py` | Load / stress benchmarks | 20 |

---

## Data Sources

### NYC
| Dataset | Source |
|---|---|
| Property Sales (185K) | NYC Open Data — Citywide Rolling Sales |
| Building Data | NYC DCP — PLUTO 2025 |
| Subway / Bus | MTA Open Data |
| Crime / 311 | NYPD + NYC Open Data |
| Parks / Schools | NYC Open Data |
| Airbnb Listings | Inside Airbnb |
| NTA Boundaries | NYC DCP |
| Mortgage Rates | FRED Economic Data |

### Riyadh
| Dataset | Source |
|---|---|
| Real Estate Transactions | Saudi Open Data Portal (quarterly reports) |
| Metro Stations | Saudi Open Data Portal |
| Bus Stops | Saudi Open Data Portal |
| Commercial Services | Saudi Open Data Portal |
| Traffic Intersections | Saudi Open Data Portal |
| Air Quality (NO₂/SO₂/PM₁₀/O₃) | RCRC / Saudi Open Data Portal |
| Rental Listings (SA_Aqar) | SA_Aqar platform |
| District Polygons | OSM Overpass API (admin_level=10) |
| Real Estate Price Index | Saudi Open Data Portal |

---

## Author

**Turki Almurahhem** — BSc Computer Science, Umm Al-Qura University
