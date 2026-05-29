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
> Interactive map-based AVM for **New York City** and **Riyadh** using stacked ensemble ML, SHAP explainability, and Quality-of-Life spatial indicators.

**Live demo:** https://huggingface.co/spaces/Turki-Almurahhem/thaman  
**GitHub:** https://github.com/turkialm/thaman-v2

---

## What It Does

| City | Flow |
|---|---|
| 🗽 **NYC** | Click map → fill building details → get USD price estimate + confidence interval + SHAP drivers + spatial grid |
| 🕌 **Riyadh** | Click map → select property type + area → get SAR/sqm estimate + AVM grade badge + Bayut asking overlay |

Toggle between cities with the NYC / Riyadh button on the map. Full bilingual (EN/AR) with RTL support.

---

## Model Performance

### NYC — Stack v22 (134 features, 157K sales, 2022–2026)

| Metric | Value |
|---|---|
| R² (holdout) | 0.6495 |
| MedAPE | 20.32% |
| CV Strategy | 10-fold Spatial GroupKFold (by NTA) |
| Stack | XGB-A + XGB-B + LightGBM + CatBoost + Ridge meta |

**By borough:** Manhattan 35.16% · Bronx 21.08% · Brooklyn 20.64% · Queens 17.43% · Staten Island 14.41%

Key v22 features: NTA × building-type temporal lag (2 lags + momentum), BBL building-level LOO $/sqft history.

### Riyadh — Stack v11 (140 features, 7,258 transactions, 2018–2025)

| Metric | Value |
|---|---|
| OOF R² | 0.9343 |
| OOF MedAPE | 8.28% |
| Holdout R² | 0.8003 |
| Holdout MedAPE | 15.56% |
| Holdout MAE | 980 SAR/m² |
| Holdout period | 2025 Q1+ (1,727 rows, fully unseen) |
| CV Strategy | 5-fold Spatial GroupKFold (by district) |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |

**By type:** Apartment 12.70% · Villa 12.83% · Residential Plot 21.20% · Building 21.45%

Key v11 features: type-stratified district lag prices + price volatility + Suhail transaction density.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, LightGBM, CatBoost, scikit-learn Ridge |
| Backend | FastAPI + Uvicorn |
| Spatial | SciPy BallTree/cKDTree, GeoPandas, Polars |
| Explainability | SHAP (TreeExplainer), custom waterfall chart |
| Frontend | Leaflet.js, Chart.js, vanilla JS (bilingual EN/AR) |
| Deployment | Docker (Hugging Face Spaces) |
| Data pipeline | Polars (training), Pandas (feature scripts) |

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
| `POST` | `/predict/riyadh` | Riyadh price estimate (SAR/m²) |
| `POST` | `/batch` | NYC batch predictions (up to 50 properties) |
| `GET` | `/layers/nta` | NYC NTA choropleth GeoJSON |
| `GET` | `/layers/district` | Riyadh district polygon GeoJSON |
| `GET` | `/riyadh/stats` | Riyadh model statistics |
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
new_try/
├── api/
│   ├── main.py          # FastAPI — all endpoints (NYC + Riyadh)
│   ├── spatial.py       # SpatialLookup + RiyadhSpatialLookup (BallTree/cKDTree)
│   └── models.py        # Pydantic request/response schemas
├── models/
│   ├── scorer.py            # ThamanScorer — stacked ensemble inference
│   ├── thaman_stack.pkl     # NYC Stack v22 (134 features, 52 MB)
│   ├── xgboost_model.json   # NYC XGBoost base learner (JSON)
│   ├── meta.json            # NYC feature names + NTA lookups + BBL history
│   ├── riyadh_stack.pkl     # Riyadh Stack v11 (140 features)
│   └── riyadh_meta.json     # Riyadh feature names + district lag maps
├── frontend/
│   ├── index.html       # Dual-city map UI (bilingual EN/AR)
│   ├── app.js           # Leaflet + city toggle + predict + i18n
│   ├── style.css        # Responsive CSS (dark header, RTL-aware)
│   └── charts.html      # Analytics dashboard (NYC + Riyadh)
├── training/
│   ├── train_stack_v12.py        # NYC Stack training (current, v22+)
│   └── train_stack_riyadh_v2.py  # Riyadh Stack training (current, v11+)
├── scripts/
│   ├── generate_scatter.py       # Holdout scatter plot data
│   └── prepare_v12_features.py   # NYC feature engineering pipeline
├── data/
│   ├── processed/
│   │   ├── features_riyadh_v2.csv           # Riyadh feature matrix (7,258 × 140)
│   │   ├── nta_simplified.geojson           # NYC NTA polygons
│   │   └── riyadh_district_polygons.geojson # Riyadh district polygons
│   └── raw/                                 # Source data files
├── tests/
│   └── test_api.py              # API smoke + integration tests
├── docs/
│   └── DATA_CATALOG.md
├── Dockerfile
└── requirements.txt
```

---

## Test Suite

Run with `pytest tests/ -v`.

| File | Tests |
|---|---|
| `test_api.py` | Smoke / integration (NYC + Riyadh endpoints) |

---

## Data Sources

### NYC
| Dataset | Source |
|---|---|
| Property Sales (185K) | NYC Open Data — Citywide Rolling Sales 2022–2026 |
| Building Data | NYC DCP — PLUTO 2025 |
| Subway / Bus | MTA Open Data |
| Crime / 311 | NYPD + NYC Open Data |
| Parks / Schools | NYC Open Data |
| Airbnb Listings | Inside Airbnb |
| NTA Boundaries | NYC DCP (2020 NTAs, 207 codes) |
| Mortgage Rates | FRED Economic Data |
| Historic Districts / Flood Zones | NYC LPC + FEMA 2015 |
| Census Income | ACS 5-Year (2020), 2,327 tracts |
| Overture POIs | Overture Maps (57,669 POIs, 12 categories) |

### Riyadh
| Dataset | Source |
|---|---|
| Real Estate Transactions | Suhail / Saudi Open Data Portal (2018–2026) |
| Asking Prices | Haraj listings (May 2026, 1,824 listings) |
| Metro Stations | REGA / Saudi Open Data (94 stations, 6 lines) |
| Bus Stops / POIs | OSM Overpass API |
| Air Quality | RCRC / Saudi Open Data Portal |
| District Polygons | OSM (admin_level=10) |
| Real Estate Price Index | Saudi Open Data Portal (REI by type, 19 quarters) |

---

## Author

**Turki Almurahhem** — BSc Computer Science, Umm Al-Qura University
