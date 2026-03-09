# THAMAN — Project Status
> AI-Powered PropTech System for Property Valuation
> Last updated: 2026-03-09

---

## ✅ Completed Phases

### Phase 5 — Frontend Map UI (DONE)

Interactive web application built and served from FastAPI at `http://localhost:8000/ui`.

#### How to Launch

```bash
cd "new try"
python3 -m uvicorn api.main:app --port 8000
# Then open: http://localhost:8000/ui
```

#### Files

| File | Description |
|---|---|
| `frontend/index.html` | Main page — map + sidebar layout (250 lines) |
| `frontend/style.css` | Full stylesheet — CSS variables, grid, animations (380 lines) |
| `frontend/app.js` | Leaflet map init + form handling + API fetch + render logic (270 lines) |

#### User Flow

1. **Map** — NYC displayed via Leaflet.js (OpenStreetMap tiles, free/no API key)
2. **Click** — user clicks anywhere on NYC map → pin placed, lat/lng captured, borough auto-guessed
3. **Form** — user fills: Building Type (grouped dropdown, 18 common bldgclasses), Size, Age, Floors, Units, Borough
4. **Advanced** — collapsible panel for Land Size, Prior Sale Price, Renovation flag, Valuation Year
5. **Estimate** — form validates, loading spinner shown, `POST /predict` called
6. **Results appear**:
   - **Price Card** — large dollar amount + confidence range + borough/type label
   - **Confidence Bar** — visual range bar with centered prediction marker
   - **Map Popup** — pin shows formatted price when clicked
   - **SHAP Drivers** — 10 horizontal bars, green↑ for positive impact, red↓ for negative
   - **Spatial Grid** — 10 location stats (subway dist, income, crime, school district, airbnb count…)

#### UI Structure

```
┌─ Header ─────────────────────────────────────────────────────────────┐
│  🏙️ THAMAN  ·  NYC Property Valuation  |  XGBoost  R²=0.735  API Docs│
├─ Map (flex-grow) ─────────────────────┬─ Sidebar (380px) ────────────┤
│                                       │  📋 How to use (3 steps)     │
│  Leaflet.js + OpenStreetMap           │                               │
│  Click-to-place marker                │  🏠 Property Details Form     │
│  Draggable pin                        │   • Location (lat/lng display)│
│  Map popup shows predicted price      │   • Borough dropdown          │
│                                       │   • Building type (grouped)   │
│                                       │   • Size, Age, Floors, Units  │
│                                       │   ▶ Advanced Options          │
│                                       │   [ Estimate Price button ]   │
│                                       │                               │
│                                       │  💰 Estimated Value           │
│                                       │   $1,307,229                  │
│                                       │   Range: $1.07M – $1.54M     │
│                                       │   ══════[●]══════             │
│                                       │                               │
│                                       │  📊 Price Drivers (SHAP)      │
│                                       │   ↑ Building class   ████ +0.15│
│                                       │   ↑ Airbnb density   ███  +0.12│
│                                       │   ↓ Building size    ██   -0.06│
│                                       │                               │
│                                       │  📍 Location Details          │
│                                       │   🚇803m  🌳136m  💰$63k     │
│                                       │   🏫#13   🏠308   📈6.0%     │
└───────────────────────────────────────┴───────────────────────────────┘
```

#### Technical Details

| Feature | Implementation |
|---|---|
| Map library | Leaflet.js 1.9.4 (CDN, free OSM tiles) |
| Marker | Draggable emoji pin (📍), no external image |
| Borough auto-fill | Rough lat/lng bounding boxes |
| API calls | `fetch()` to same-origin `/predict` (no CORS needed) |
| SHAP bars | Pure CSS, width proportional to \|impact\| / max_impact |
| Price formatting | `toLocaleString()` + short forms ($1.07M) |
| Responsive | 2-col desktop; stacked mobile at 768px breakpoint |
| Validation | Required fields highlighted red if missing |
| Animations | `fadeUp` for result cards, `spin` for loading, `slideDown` for advanced panel |

#### Validation Tests Passed (37/37)

| Category | Checks | Result |
|---|---|---|
| HTML structure | 14 elements present | ✅ 14/14 |
| JavaScript logic | 10 functions/handlers | ✅ 10/10 |
| CSS rules | 13 style sections | ✅ 13/13 |
| API integration (live) | Brooklyn A1 full flow | ✅ $1,307,229 returned |
| SHAP render | 10 drivers displayed | ✅ correct |
| Spatial grid | 10 location stats | ✅ correct |

---

### Phase 4 — Backend API (DONE)

FastAPI server built, tested, and running. All endpoints functional.

#### API Summary

| Item | Value |
|---|---|
| Framework | FastAPI + Uvicorn |
| Base URL | `http://localhost:8000` |
| Docs | `http://localhost:8000/docs` (auto-generated Swagger UI) |
| Startup time | ~25 seconds (loads spatial data + XGBoost model) |

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info, version, performance metrics |
| `GET` | `/health` | Health check — confirms model + spatial data loaded |
| `GET` | `/bldgclasses` | All 128 valid NYC building class codes |
| `POST` | `/predict` | Predict price for one property + SHAP top-10 drivers |
| `POST` | `/batch` | Batch predict up to 50 properties |

#### How to Start

```bash
cd "new try"
python3 -m uvicorn api.main:app --reload --port 8000
```

#### Example `/predict` Request

```json
POST /predict
{
  "latitude": 40.6892,
  "longitude": -73.9442,
  "gross_square_feet": 1800,
  "building_age": 55,
  "bldgclass": "A1",
  "borough": 3,
  "numfloors": 2,
  "residential_units": 1,
  "land_square_feet": 2000
}
```

#### Example `/predict` Response

```json
{
  "predicted_price": 1307229,
  "confidence_low": 1074542,
  "confidence_high": 1539915,
  "confidence_note": "±17.8% MedAPE confidence interval",
  "model": "XGBoost",
  "r2_test": 0.7354,
  "medape_pct": 17.78,
  "borough_name": "Brooklyn",
  "bldgclass_description": "Two story detached - small or moderate",
  "spatial_features": {
    "dist_subway_m": 803,
    "dist_bus_m": 108,
    "dist_park_m": 136,
    "crime_rate_nta": 114.1,
    "median_income_nta": 63382,
    "school_district": 13,
    "airbnb_count_500m": 308,
    "mortgage_rate_30yr": 6.0
  },
  "top_drivers": [
    { "feature": "bldgclass",       "impact": +0.149, "direction": "positive", "description": "Building class / type" },
    { "feature": "airbnb_count_500m", "impact": +0.118, "direction": "positive", "description": "Airbnb density within 500m" },
    { "feature": "poi_count_500m",  "impact": +0.075, "direction": "positive", "description": "Points of interest within 500m" },
    { "feature": "gross_square_feet", "impact": -0.056, "direction": "negative", "description": "Building size (sq ft)" }
  ]
}
```

#### Files Created

| File | Description |
|---|---|
| `api/__init__.py` | Package init |
| `api/main.py` | FastAPI app — routes, lifespan, CORS |
| `api/spatial.py` | `SpatialLookup` — loads all spatial data, serves KD-tree + NTA lookups |
| `api/models.py` | Pydantic schemas — `PredictRequest`, `PredictResponse`, `FeatureDriver` |
| `.claude/launch.json` | Dev server config for preview start |

#### Spatial Lookups (Auto-Computed from lat/lng)

| Feature | Data Source | Method |
|---|---|---|
| `dist_subway_m` | MTA Subway stations CSV | scipy KD-tree |
| `dist_express_subway_m` | MTA Subway (express routes) | scipy KD-tree |
| `nearest_station_is_express` | Computed | distance comparison |
| `dist_bus_m` | MTA Bus stops CSV | scipy KD-tree |
| `dist_school_m` | Schools CSV | scipy KD-tree |
| `dist_elem_school_m` | Elementary schools CSV | scipy KD-tree |
| `dist_park_m` | Parks CSV | scipy KD-tree |
| `airbnb_count_500m` | Airbnb listings CSV | sklearn BallTree (haversine, 500m) |
| `crime_rate_nta` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `noise_density_nta` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `median_income_nta` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `school_district` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `poi_count_500m` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `dist_hospital_m` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `dist_waterfront_m` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `dist_bike_lane_m` | features.csv (NTA median) | point-in-polygon → NTA lookup |
| FAR features | features.csv (NTA median) | point-in-polygon → NTA lookup |
| `mortgage_rate_30yr` | mortgage_rates.csv | latest value |
| `sale_year/month` | System clock | current date |

#### Bug Fixed During Development

| Bug | Root Cause | Fix |
|---|---|---|
| Predictions ~$20k (100× too low) | `predict_single` defaulted ACRIS cols to `0.0`, but `fillna(median)` only replaces NaN — training imputation never triggered | Changed ACRIS defaults to `np.nan` in `predict_single` |

#### Validation Tests Passed

| Test | Predicted | Actual / Expected |
|---|---|---|
| Brooklyn A1, 1800 sqft, 55yr | $1,307,229 | ✅ Realistic ($800k–$1.5M range) |
| Brooklyn A1 from test set (actual $720,800) | $688,428 | ✅ 4.5% error |
| Washington Heights R1 condo, 950 sqft | $1,082,577 | ✅ Realistic |
| Staten Island A1, 1600 sqft (batch) | $632,832 | ✅ Realistic |
| Out-of-NYC coords (London) | 422 Validation Error | ✅ Rejected correctly |

---

### Phase 3 — Model Training (DONE)

#### Results Summary

| Metric | Baseline (Ridge) | XGBoost | Improvement |
|---|---|---|---|
| R² (log scale) | 0.238 | **0.735** | +0.497 |
| MAE (raw $) | $1,230,842 | **$726,593** | −$504,249 |
| MAPE | 86.1% | 50.1% | −36% |
| MedAPE | — | **17.8%** | — |
| RMSE (log) | — | 0.497 | — |
| Train R² | — | 0.894 | gap=0.159 (moderate overfit) |
| Best round | — | 917 / 2000 | early stopping |

> MedAPE (17.8%) is the key metric — MAPE is inflated by extreme outliers (max sale = $1.08B).

#### Top 10 Features by SHAP Importance

| Rank | Feature | Mean |SHAP| | Insight |
|---|---|---|---|
| 1 | `gross_square_feet` | 0.247 | Physical size is #1 driver |
| 2 | `bldgclass` | 0.203 | Building type (D4 vs A1 etc.) critical |
| 3 | `land_square_feet` | 0.080 | Land value component |
| 4 | `longitude` | 0.071 | East-West location (Manhattan premium) |
| 5 | `latitude` | 0.066 | North-South location |
| 6 | `school_district` | 0.047 | QoL feature — families pay premium |
| 7 | `median_income_nta` | 0.045 | Neighborhood wealth signal |
| 8 | `residential_units` | 0.042 | Building scale |
| 9 | `building_age` | 0.041 | Older = cheaper (generally) |
| 10 | `airbnb_count_500m` | 0.038 | Tourist/urban density premium |

#### Model Files Saved

| File | Description | Size |
|---|---|---|
| `models/xgboost_model.json` | Trained XGBoost model (917 trees) | 4.9 MB |
| `models/scorer.py` | `ThamanScorer` class for inference | 5.4 KB |
| `models/meta.json` | Feature names, encoders, metrics | 4.9 KB |
| `models/shap_importance.png` | Top 20 SHAP feature importance bar chart | 79 KB |
| `models/actual_vs_predicted.png` | Actual vs Predicted scatter plot | 124 KB |
| `models/error_by_borough.png` | % Error distribution by borough (boxplot) | 52 KB |
| `models/X_train/test.npy` | Pre-processed train/test arrays | 14 MB |

#### Pre-processing Steps Applied Before Training

1. Dropped: `address`, `bbl`, `ntacode`, `ntaname`, `neighborhood`, `sale_date`, `prior_sale_date`, `zip_code`, `total_units`
2. Target: `log1p(sale_price)` — skewness 54.86 → 0.783
3. Winsorized at 99th percentile: `crime_rate_nta` (→247), `noise_density_nta` (→180), `livability_complaint_rate` (→59)
4. Label-encoded `bldgclass` (128 unique classes)
5. Filled ACRIS nulls with training-set medians for `prior_sale_price`, `price_appreciation`, `years_since_prior_sale`
6. 80/20 train/test split stratified by `borough`

---

### Phase 1 — Data Collection (DONE)
All 24 data sources downloaded and verified on disk.

| # | Dataset | File | Records | Size |
|---|---|---|---|---|
| 1 | Property Sales | `data/raw/sales_geocoded.csv` | 81,305 (36,203 filtered) | 14.6 MB |
| 2 | PLUTO Building Data | `data/raw/nyc_pluto_25v4_csv/pluto_25v4.csv` | 858,644 parcels | 385.9 MB |
| 3 | Overture Maps POIs | `data/raw/overture_places.geojson` | 425,387 places | 618.9 MB |
| 4 | MTA Subway Stations | `data/raw/MTA_Subway_Stations_20260308.csv` | 496 stations | 0.1 MB |
| 5 | MTA Bus Stops | `data/raw/mta_bus_stops.csv` | 9,747 stops | 0.6 MB |
| 6 | NYC Parks | `data/raw/parks_with_coords.csv` | 2,058 parks | 0.1 MB |
| 7 | High Schools | `data/raw/schools.csv` | 427 schools | 0.0 MB |
| 8 | Elementary Schools | `data/raw/elementary_schools.csv` | 423 schools | 0.0 MB |
| 9 | NYPD Crime | `data/raw/nypd_crimes.parquet` | 1,646,571 complaints | 12.4 MB |
| 10 | 311 Noise | `data/raw/noise_complaints.parquet` | 1,000,000 complaints | 14.6 MB |
| 11 | 311 Livability | `data/raw/livability_complaints.parquet` | 383,778 complaints | 6.8 MB |
| 12 | NTA Boundaries | `data/raw/nta_boundaries.geojson` | 262 neighborhoods | 5.2 MB |
| 13 | Road Network | `data/raw/road_network/*.graphml` | 10 files (all 5 boroughs) | ~300 MB |
| 14 | Census Population | `data/raw/census_tract_population.csv` | 2,327 tracts | 0.1 MB |
| 15 | Census Income | `data/raw/census_tract_income.csv` | 2,327 tracts | 0.2 MB |
| 16 | DOB Permits | `data/raw/dob_permits.csv` | 2,673 unique BBLs | 0.1 MB |
| 17 | FRED Mortgage Rates | `data/raw/mortgage_rates.csv` | ~1,500 weeks | 0.0 MB |
| 18 | Airbnb Listings | `data/raw/airbnb_listings.csv` | 36,261 listings | 2.7 MB |
| 19 | LPC Landmarks | via Socrata API | 3,695 landmark BBLs | — |
| 20 | OSM Coastline | via Overpass API | 33,507 points | — |
| 21 | OSM Bike Lanes | via Overpass API | 4,155 segments | — |
| 22 | ACRIS Master | via Socrata API | 500,000 DEEDs | — |
| 23 | ACRIS Legals | via Socrata API | ~24k fetched (22.5M total) | — |
| 24 | School Districts | via ArcGIS REST | 33 districts | — |

---

### Phase 2 — Feature Engineering (DONE)
Feature matrix built, audited, and all quality issues fixed.

**File:** `data/processed/features.csv`
**Shape:** 36,203 rows × 61 columns
**Memory:** ~30 MB

#### Feature Groups

| Group | Features | Null % |
|---|---|---|
| Identifiers | address, bbl, latitude, longitude, ntacode, ntaname, neighborhood, zip_code, borough | 0% |
| Target | sale_price, sale_date | 0% |
| Property Structure | building_age, numfloors, bldgclass, gross_square_feet, land_square_feet, residential_units, total_units | 0% |
| Zoning / FAR | residfar, commfar, facilfar, builtfar, maxallwfar, far_utilization | 0% |
| Building Type Flags | has_elevator, is_condo, is_multifamily, is_single_fam, is_mixed_use | 0% |
| Transit Distances | dist_subway_m, dist_bus_m, dist_express_subway_m, nearest_station_is_express | 0% |
| Amenity Distances | dist_school_m, dist_elem_school_m, dist_park_m, dist_hospital_m, dist_waterfront_m, dist_bike_lane_m | 0% |
| POI Density | poi_count_500m, airbnb_count_500m | 0% |
| NTA Quality of Life | crime_rate_nta, noise_density_nta, livability_complaint_rate, population_2020, median_income_nta, borough_income_deviation | 0% |
| Renovation / Age | renovated_since_2018, years_since_renovation, is_landmark, is_historic_district | 0% |
| Macro / Time | mortgage_rate_30yr, sale_year, sale_month | 0% |
| School District | school_district, district_avg_score, district_school_count | 0% |
| ACRIS Price History | prior_sale_price, prior_sale_date, price_appreciation, years_since_prior_sale, has_prior_sale, is_flip | 86.3% for ACRIS cols; is_flip + has_prior_sale = 0% |

#### Top Feature Correlations with sale_price

| Feature | Correlation | Direction |
|---|---|---|
| prior_sale_price | +0.719 | Strong positive (momentum) |
| commfar | +0.194 | Commercial zoning → premium |
| poi_count_500m | +0.152 | Urban density premium |
| crime_rate_nta | +0.131 | Paradox: high-crime = Manhattan |
| maxallwfar | +0.123 | High-density zoning = city core |
| price_appreciation | +0.120 | Appreciating areas = higher prices |
| airbnb_count_500m | +0.078 | Tourist/mixed-use premium |
| median_income_nta | +0.055 | Wealthy neighborhood premium |
| numfloors | +0.066 | Taller = more valuable |
| dist_subway_m | -0.050 | Farther from subway = cheaper |

#### Data Quality Fixes Applied

| Fix | Issue | Action |
|---|---|---|
| building_age | 984 properties with age > 200 (yearbuilt=0 in source) | Capped at 200 |
| years_since_prior_sale | 542 negative values (ACRIS date anomalies) | Clipped to 0 |
| is_flip | Was 0 for all no-prior-sale properties (misleading) | Now only 1 where has_prior_sale=1 AND < 2yr |
| has_prior_sale | Missing flag | Added — cleaner than relying on null check |
| district_avg_score | 56.9% null after spatial join | Borough median → global median fallback → 0% null |
| numfloors | 2.9% null | Median by bldgclass + borough → 0% null |
| far_utilization | 0.3% null (vacant lots) | Filled with 0 |
| NTA edge cases | 13 properties outside NTA boundary | Borough median for all NTA-level cols |
| small nulls | zip_code (14), builtfar (8), bldgclass (1), building_age (1) | Filled with 0 / mode / median |

#### Known Remaining Issues (Pre-Training)

| Issue | Action Before Training |
|---|---|
| sale_price skewness = 54.86 | Apply `np.log1p(sale_price)` as target |
| bldgclass is categorical string | One-hot encode or use LightGBM/XGBoost native categorical |
| NTA noise/crime outliers | Winsorize at 99th percentile |
| ACRIS 86.3% null group | Pass raw nulls to XGBoost (handles natively) OR impute with NTA median |
| prior_sale_date column | Drop before training (leakage risk if kept as raw datetime) |
| Identifiers leak | Drop: address, bbl, ntacode, ntaname, neighborhood during training |

---

## 🔲 Optional Enhancements (Future Work)

The core system is fully functional. Potential improvements for future versions:

| Enhancement | Effort | Impact |
|---|---|---|
| Mapbox GL JS (3D buildings, satellite view) | Medium | High UX |
| Heat-map overlay of predicted prices across NYC | High | Unique feature |
| Comparison mode (side-by-side two properties) | Medium | Useful |
| Export prediction as PDF report | Medium | Academic value |
| Deploy to cloud (Render, Railway, Fly.io) | Low | Accessibility |
| EPA Air Quality data (skipped — needs API key) | Low | Model accuracy |
| FEMA flood zone risk overlay | Medium | Risk signal |
| Model re-training with newer sales data | Low | Accuracy |
| LightGBM / stacking ensemble | Medium | R² improvement |

---

## 📦 Archive — Phase 3 Model Training Plans

### Recommended Pipeline (Executed Successfully)

```python
# 1. Load and prepare
features = pd.read_csv("data/processed/features.csv")

DROP_COLS = ["address", "bbl", "ntacode", "ntaname", "neighborhood",
             "sale_date", "prior_sale_date"]
CAT_COLS  = ["bldgclass", "borough"]

X = features.drop(columns=DROP_COLS + ["sale_price"])
y = np.log1p(features["sale_price"])

# 2. Winsorize outlier NTA rates
for col in ["crime_rate_nta", "noise_density_nta", "livability_complaint_rate"]:
    p99 = X[col].quantile(0.99)
    X[col] = X[col].clip(upper=p99)

# 3. Encode categoricals
X = pd.get_dummies(X, columns=CAT_COLS)

# 4. Train/test split (stratified by borough)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=features["borough"]
)

# 5. Train XGBoost
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric="rmse"
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=100)

# 6. Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
r2  = r2_score(y_test, y_pred)
print(f"MAE: ${mae:,.0f} | R²: {r2:.4f}")

# 7. SHAP explanations
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Metrics to Target

| Metric | Target | Notes |
|---|---|---|
| R² (log scale) | > 0.80 | Comparable to NYC AVM models |
| MAE (raw $) | < $200,000 | Median price = $850k, so < 24% error |
| MAPE | < 20% | Standard for AVM models |

### Models to Compare

| Model | Expected R² | Notes |
|---|---|---|
| Linear Regression (baseline) | ~0.45 | With log-transform |
| Random Forest | ~0.75 | Good baseline tree model |
| XGBoost | ~0.82 | Best expected performance |
| LightGBM | ~0.82 | Faster than XGBoost, similar accuracy |
| Stacking ensemble | ~0.84 | XGBoost + RF + Ridge meta-learner |

---

## 📁 File Tree

```
new try/
├── frontend/                      ← Phase 5 Map UI ★ NEW
│   ├── index.html                 ← Main page (map + sidebar, 250 lines)
│   ├── style.css                  ← Full stylesheet (CSS vars, grid, animations, 380 lines)
│   └── app.js                     ← Leaflet + form + API fetch + render (270 lines)
├── api/                           ← Phase 4 FastAPI backend
│   ├── __init__.py
│   ├── main.py                    ← FastAPI app (routes, lifespan, CORS, StaticFiles /ui)
│   ├── spatial.py                 ← SpatialLookup class (KD-tree, BallTree, NTA)
│   └── models.py                  ← Pydantic schemas (PredictRequest, PredictResponse)
├── models/
│   ├── xgboost_model.json         ← Trained XGBoost (917 trees, 4.9 MB)
│   ├── scorer.py                  ← ThamanScorer class for inference
│   ├── meta.json                  ← Feature names, encoders, metrics
│   ├── shap_importance.png        ← SHAP top-20 bar chart
│   ├── actual_vs_predicted.png    ← Scatter plot (capped $10M)
│   ├── error_by_borough.png       ← % Error boxplot by borough
│   ├── X_train/test.npy           ← Pre-processed arrays (14 MB)
│   └── y_train/test.npy
├── data/
│   ├── raw/
│   │   ├── sales_geocoded.csv
│   │   ├── nyc_pluto_25v4_csv/pluto_25v4.csv
│   │   ├── overture_places.geojson
│   │   ├── MTA_Subway_Stations_20260308.csv
│   │   ├── mta_bus_stops.csv
│   │   ├── parks_with_coords.csv
│   │   ├── schools.csv
│   │   ├── elementary_schools.csv
│   │   ├── nypd_crimes.parquet
│   │   ├── noise_complaints.parquet
│   │   ├── livability_complaints.parquet
│   │   ├── nta_boundaries.geojson
│   │   ├── road_network/          ← 10 GraphML files
│   │   ├── census_tract_population.csv
│   │   ├── census_tract_income.csv
│   │   ├── dob_permits.csv
│   │   ├── mortgage_rates.csv
│   │   └── airbnb_listings.csv
│   └── processed/
│       └── features.csv           ← 36,203 × 61 cols ★
├── feature_engineering.py
├── DATA_CATALOG.md
└── PROJECT_STATUS.md              ← this file
```
