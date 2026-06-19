---
title: THAMAN Property Valuation
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏙️🕌 THAMAN — Dual-City AI Property Valuation

> Interactive AVM for **New York City** and **Riyadh** using stacked ensemble ML, SHAP explainability, and geospatial enrichment.

**Live demo:** https://huggingface.co/spaces/Turki-Almurahhem/thaman  
**GitHub:** https://github.com/turkialm/thaman-v2

---

## What It Does

| City | Flow |
|---|---|
| 🗽 **NYC** | Click map → fill building details → USD estimate + confidence interval + SHAP drivers + 19-factor spatial grid |
| 🕌 **Riyadh** | Click map → select property type + area → SAR/m² estimate + AVM grade + Bayut asking overlay + weerate benchmarks |

- Toggle cities with the NYC / Riyadh button
- Full bilingual EN/AR with RTL support
- Share any estimate via URL — restores form + auto-runs on open
- Install as PWA on mobile (iOS + Android)
- Batch valuations via CSV upload (up to 50 properties)
- Embeddable iframe widget

---

## Pages

| URL | Description |
|---|---|
| `/` | Landing page |
| `/ui` | Interactive map app |
| `/ui/charts.html` | Analytics dashboard — model history, borough/district breakdown, weerate Jun 2026 |
| `/ui/batch.html` | Batch CSV valuation — NYC or Riyadh, download results |
| `/ui/embed.html` | Lightweight embeddable widget |
| `/docs` | Swagger API docs |

---

## Model Performance

### NYC — Stack v22 (134 features · 157K sales · 2022–2026)

| Metric | Value |
|---|---|
| R² holdout | 0.6495 |
| MedAPE | 20.32% |
| CV | 10-fold Spatial GroupKFold by NTA |
| Stack | XGB-A + XGB-B + LightGBM + CatBoost + Ridge meta |

Borough MedAPE: Manhattan 35.2% · Bronx 21.1% · Brooklyn 20.6% · Queens 17.4% · Staten Island 14.4%

Key features: NTA × building-type temporal lag (2 lags + momentum), BBL building-level LOO $/sqft history, 57K Overture POIs (12 categories).

### Riyadh — Stack v12 (149 features · 7,261 district-quarter records · 2018–2026)

| Metric | Value |
|---|---|
| OOF R² | 0.9348 |
| OOF MedAPE | 8.25% |
| Holdout R² | 0.8014 |
| Holdout MedAPE | 15.59% |
| CV | 5-fold Spatial GroupKFold by district |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |

Type MedAPE: Apartment 12.8% · Villa 12.4% · Residential Plot 20.8% · Building 18.6%

Key features: type-stratified district lag prices, Suhail transaction density, Riyadh Metro proximity (94 stations), Haraj asking prices, REI indices, Bayut villa medians, Haraj structural (type-stratified area + age).

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, LightGBM, CatBoost, scikit-learn Ridge |
| Backend | FastAPI + Uvicorn |
| Spatial | SciPy BallTree/cKDTree, GeoPandas, Polars |
| Explainability | SHAP TreeExplainer — bar + waterfall views |
| Frontend | Leaflet.js, Chart.js, vanilla JS (bilingual EN/AR) |
| Mobile | PWA (manifest + service worker), CSS bottom-drawer |
| Deployment | Docker (Hugging Face Spaces) |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model load status |
| `POST` | `/predict` | NYC single estimate (USD) |
| `POST` | `/predict/riyadh` | Riyadh single estimate (SAR/m²) |
| `POST` | `/batch` | NYC batch — up to 50 properties |
| `POST` | `/batch/riyadh` | Riyadh batch — up to 50 properties |
| `GET` | `/riyadh/stats` | Riyadh analytics + weerate Jun 2026 benchmarks |
| `GET` | `/metrics` | Live model metrics |
| `GET` | `/scatter?city=nyc\|riyadh` | Predicted-vs-actual scatter data |
| `GET` | `/layers/nta` | NYC NTA choropleth GeoJSON |
| `GET` | `/layers/nyc-heatmap` | NTA-level sales heatmap |
| `GET` | `/robots.txt` | SEO |
| `GET` | `/sitemap.xml` | Dynamic sitemap |

### NYC Example
```bash
curl -X POST https://huggingface.co/spaces/Turki-Almurahhem/thaman/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude":40.6892,"longitude":-73.9442,"gross_square_feet":1800,
       "building_age":55,"bldgclass":"A1","borough":3,"numfloors":2,"residential_units":1}'
```

### Riyadh Example
```bash
curl -X POST https://huggingface.co/spaces/Turki-Almurahhem/thaman/predict/riyadh \
  -H "Content-Type: application/json" \
  -d '{"latitude":24.7136,"longitude":46.6753,"property_type":"شقة","area_sqm":150}'
```

### Embed Widget
```html
<iframe src="https://huggingface.co/spaces/Turki-Almurahhem/thaman/ui/embed.html"
        width="340" height="560" frameborder="0"></iframe>
```

---

## Run Locally

```bash
git clone https://github.com/turkialm/thaman-v2.git
cd thaman-v2
pip install -r requirements.txt
uvicorn api.main:app --port 8000
# Open: http://localhost:8000
```

First startup ~30 sec — loads both ML stacks + spatial data.

---

## Project Structure

```
├── api/
│   ├── main.py           # All FastAPI endpoints
│   ├── models.py         # Pydantic schemas
│   └── schemas.py
├── models/
│   ├── scorer.py         # ThamanScorer — stacked ensemble inference
│   ├── thaman_stack.pkl  # NYC Stack v22 (134 features, ~50 MB)
│   ├── riyadh_stack.pkl  # Riyadh Stack v12 (149 features)
│   ├── meta.json         # NYC lookups (NTA, BBL, income)
│   └── riyadh_meta.json  # Riyadh lookups (district lags, metro, Haraj)
├── frontend/
│   ├── landing.html      # Marketing landing page
│   ├── index.html        # Dual-city map UI (bilingual EN/AR)
│   ├── app.js            # Leaflet + city toggle + predict + share + i18n
│   ├── style.css         # Responsive CSS (mobile drawer, RTL)
│   ├── charts.html       # Analytics dashboard
│   ├── batch.html        # Batch CSV valuation
│   ├── embed.html        # Embeddable iframe widget
│   ├── manifest.json     # PWA manifest
│   └── sw.js             # Service worker (cache-first static)
├── training/
│   ├── train_stack_v12.py        # NYC v22+ training pipeline
│   └── train_stack_riyadh_v4.py  # Riyadh v11+ training pipeline
├── scripts/
│   ├── fetch_suhail_transactions.py   # Pull fresh Suhail MOJ data
│   ├── aggregate_suhail_quarterly.py  # District-quarter aggregation
│   ├── riyadh_structural_features.py  # Aqar/Bayut/Haraj enrichment
│   └── generate_scatter.py            # Holdout scatter data
├── data/
│   ├── processed/        # Feature matrices (git-ignored, large)
│   └── raw/
│       └── weerate_riyadh_jun2026.json  # Market benchmarks (Bayut/KF/Maxwell/GASTAT)
├── tests/                # 109 tests across 8 files
├── Dockerfile
└── requirements.txt
```

---

## Data Sources

### NYC
- **Property Sales** (157K) — NYC Open Data Rolling Calendar 2022–2026
- **Building Data** — NYC DCP PLUTO 2025
- **Transit** — MTA subway, bus stops, LIRR/Metro-North
- **POIs** — Overture Maps (57K POIs, 12 categories)
- **Census** — ACS 5-Year income (2,327 tracts)
- **Other** — NYPD crime, HPD violations, DOB permits, flood zones, historic districts

### Riyadh
- **Transactions** (33K) — Suhail / MOJ deed transfers 2018–2026
- **Asking Prices** — Haraj listings (May 2026, 1,824 listings)
- **Metro** — REGA (94 stations, 6 lines)
- **Market Benchmarks** — weerate Jun 2026 (Bayut · Knight Frank · Cavendish Maxwell · GASTAT · JLL)
- **POIs** — OSM (mosques, malls, schools, hospitals, parks)
- **Price Index** — Saudi Open Data REI (19 quarters, by type)

---

## Tests

```bash
python -m pytest tests/ -v   # 109 tests
```

| File | Tests | Coverage |
|---|---|---|
| `test_api.py` | 31 | Endpoints, validation, Riyadh batch, robots, sitemap, 404 |
| `test_scorer.py` | 17 | Features, predict, SHAP, confidence |
| `test_feature_parity.py` | 15 | Feature counts, NTA lookup |
| `test_shap.py` | 14 | SHAP structure, sensitivity |
| `test_distribution.py` | 12 | Grid median, borough ordering |
| `test_golden.py` | 9 | Known price ranges |
| `test_regression.py` | 6 | Pinned output regression (±5%) |
| `test_load.py` | 5 | Concurrent load (10 threads) |

---

## Author

**Turki Almurahhem**
