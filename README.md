---
title: THAMAN Property Valuation
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏙️ THAMAN — AI-Powered NYC Property Valuation

> BSc Graduation Project — Umm Al-Qura University
> An interactive map-based web app that estimates NYC property prices using machine learning and Quality-of-Life spatial indicators.

---

## What It Does

1. **Click** anywhere on an NYC map
2. **Fill in** building details (type, size, age, floors, units)
3. **Get** an instant price estimate with confidence range + top SHAP feature drivers

---

## Model Performance

| Metric | Value |
|---|---|
| R² (test set) | 0.5912 |
| MedAPE | 18.83% |
| Features | 70 |
| CV Strategy | Spatial GroupKFold |
| Stack | XGBoost + LightGBM + CatBoost + Ridge meta |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, LightGBM, CatBoost, scikit-learn |
| Backend | FastAPI + Uvicorn |
| Spatial | GeoPandas, SciPy KD-trees, BallTree |
| Frontend | Leaflet.js, Chart.js, vanilla JS |
| Deployment | Docker (Hugging Face Spaces) |

---

## Run Locally

### Requirements
- Python 3.10+
- Git

### Step 1 — Clone the repo

```bash
git clone https://github.com/turkialm/thaman-v2.git
cd thaman-v2
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you get a geopandas error on macOS:
> ```bash
> brew install gdal
> pip install geopandas
> ```

### Step 3 — Start the API

```bash
uvicorn api.main:app --port 8000
```

> First startup takes ~25 seconds — it loads the ML models and spatial data.

### Step 4 — Open the app

| Page | URL |
|---|---|
| Map UI | http://localhost:8000/ui |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Redirects to map UI |
| `GET` | `/health` | Model + spatial data status |
| `GET` | `/bldgclasses` | All valid NYC building class codes |
| `POST` | `/predict` | Predict price for one property |
| `POST` | `/batch` | Predict up to 50 properties |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.6892,
    "longitude": -73.9442,
    "gross_square_feet": 1800,
    "building_age": 55,
    "bldgclass": "A1",
    "borough": 3,
    "numfloors": 2,
    "residential_units": 1
  }'
```

### Example Response

```json
{
  "predicted_price": 1307229,
  "confidence_low": 1061969,
  "confidence_high": 1552489,
  "model": "XGBoost + LightGBM Stack",
  "r2_test": 0.5912,
  "medape_pct": 18.83,
  "borough_name": "Brooklyn",
  "top_drivers": [
    { "feature": "bldgclass", "impact": 0.149, "direction": "positive" },
    { "feature": "airbnb_count_500m", "impact": 0.118, "direction": "positive" }
  ]
}
```

---

## Project Structure

```
thaman-v2/
├── api/
│   ├── main.py          # FastAPI app — all endpoints
│   ├── spatial.py       # GIS lookups (subway, parks, income, crime...)
│   └── models.py        # Pydantic request/response schemas
├── models/
│   ├── scorer.py        # ThamanScorer inference class
│   ├── xgboost_model.json   # Trained XGBoost (917 trees)
│   ├── thaman_stack.pkl     # LGB + CatBoost + Ridge meta-learner
│   └── meta.json            # Feature names, encodings, metrics
├── frontend/
│   ├── index.html       # Interactive map UI
│   ├── app.js           # Leaflet map + API calls + result rendering
│   ├── style.css        # Styles
│   └── charts.html      # Analytics dashboard
├── data/
│   ├── processed/features.csv    # Feature matrix (LFS tracked)
│   └── raw/                      # Spatial reference files (runtime only)
├── Dockerfile           # For Hugging Face Spaces deployment
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Source |
|---|---|
| Property Sales | NYC Open Data — Citywide Rolling Sales |
| Building Data | NYC DCP — PLUTO 2025 |
| Subway Stations | MTA Open Data |
| Bus Stops | MTA Open Data |
| Crime Complaints | NYPD Complaint Data |
| 311 Noise | NYC Open Data |
| Parks | NYC Parks Properties |
| Schools | NYC Open Data |
| Airbnb Listings | Inside Airbnb |
| NTA Boundaries | NYC DCP — NTA 2020 |
| Mortgage Rates | FRED Economic Data |

---

## Spatial Features Auto-Computed from lat/lng

When you click a point on the map, the system automatically computes:

- Distance to nearest subway, express subway, bus stop
- Distance to nearest park, school, hospital
- Airbnb listing density within 500m
- Neighborhood crime rate, noise level, income
- School district + average district score
- Current 30-year mortgage rate
- Urban gravity distances (Midtown, Downtown, LIC)

---

## Authors

- **Turki Almurahhem** — BSc Computer Science, Umm Al-Qura University
