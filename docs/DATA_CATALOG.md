# THAMAN — Data Catalog
> AI-Powered PropTech System for Property Valuation
> Last updated: March 2026

---

## Riyadh Data Sources

| # | Dataset | File | Records | Notes |
|---|---|---|---|---|
| R1 | Real Estate Transactions (2018–2023) | `data/raw/quarter_report SI.xlsx` | ~6,500 rows | Saudi Open Data baseline; district-level quarterly aggregates |
| R2 | RE Transactions 2024 Q1 | `data/raw/` (Saudi Open Data CSV) | ~300 rows | Property transactions by type + district |
| R3 | RE Transactions 2024 Q3 | `data/raw/` (Saudi Open Data CSV) | ~300 rows | |
| R4 | RE Transactions 2024 Q4 | `data/raw/` (Saudi Open Data CSV) | ~300 rows | |
| R5 | RE Transactions 2025 Q1–Q3 | `data/raw/` (Saudi Open Data CSVs) | ~600 rows | Holdout period |
| R6 | Metro Stations | `data/raw/metro-stations-in-riyadh-*.geojson` | 85 stations | 6 lines; opened 2024 |
| R7 | Bus Stops | `data/raw/bus-stops-in-riyadh-*.geojson` | ~2,500 stops | Includes BRT stops |
| R8 | Traffic Intersections | `data/raw/traffic-intersections-*.geojson` | ~8,000 intersections | Major street crossings |
| R9 | Commercial Services | `data/raw/commercial-services-*.geojson` | ~12,000 POIs | 10 categories (hypermarkets, banks, restaurants…) |
| R10 | Air Quality Stations | `data/raw/air-quality-stations-*.geojson` | 12 stations | NO₂, SO₂, PM₁₀, O₃ measurements |
| R11 | Air Quality Readings | `data/raw/air-quality.csv` | ~500 rows | Station-level pollutant means |
| R12 | Mosques | `data/raw/riyadh_mosques.csv` | ~3,000 | OSM-derived |
| R13 | Malls | `data/raw/riyadh_malls.csv` | ~80 | OSM-derived |
| R14 | Schools | `data/raw/riyadh_schools.csv` | ~600 | OSM-derived |
| R15 | Hospitals | `data/raw/riyadh_hospitals.csv` | ~120 | OSM-derived |
| R16 | Parks | `data/raw/riyadh_parks.csv` | ~200 | OSM-derived |
| R17 | Entertainment Venues | `data/raw/rcrc_entertainment.csv` | ~150 | RCRC leisure venues |
| R18 | Rental Listings (SA_Aqar) | `data/raw/SA_Aqar.csv` | 960 listings | District-level medians: size, bedrooms, age, rent/sqm |
| R19 | Real Estate Price Index | `data/raw/real-estate-indices.csv` | ~200 rows | REI residential + apartment quarterly (2019–2025) |
| R20 | District Polygons | `data/processed/riyadh_district_polygons.geojson` | 133 polygons | OSM Overpass admin_level=10; 107/133 enriched |
| R21 | District Centroids | `data/processed/district_centroids.csv` | 147 centroids | Derived from OSM polygon centroids |

**Processed output:** `data/processed/features_riyadh.csv` — 6,910 rows × 87 columns  
**Pipeline:** `scripts/riyadh_feature_engineering.py` (Polars-native)

---

## Changelog

| Date | File | Change |
|---|---|---|
| 2026-03-08 | `feature_engineering.py` | Initial pipeline created — 26 columns: property sales, NTA assignment, KD-tree distances (subway, school, park, hospital), BallTree POI count, crime/noise rates, building age |
| 2026-03-08 | `data/raw/overture_places.geojson` | Replaced broken OSM POI export (12,886 records) with Overture Maps via leafmap (425,387 records) |
| 2026-03-08 | `data/raw/nta_boundaries.geojson` | Added `population_2020` (Census 2020 Decennial) and `median_income_nta` (ACS 5-year 2020) enriched into NTA file via census tract spatial join |
| 2026-03-08 | `data/raw/mta_bus_stops.csv` | Added bus stop locations from MTA GTFS S3 feeds (4 boroughs, 9,747 stops); added `dist_bus_m` feature |
| 2026-03-08 | `feature_engineering.py` | Added grouped imputation for `gross_square_feet`, `land_square_feet`, `residential_units` (35.8% null → 0% null) using median by `bldgclass` + `borough` |
| 2026-03-08 | `feature_engineering.py` | Added `renovated_since_2018` and `years_since_renovation` from DOB A1/A2 permits via NYC Open Data |
| 2026-03-08 | `feature_engineering.py` | Added `mortgage_rate_30yr` from FRED MORTGAGE30US; joined to sales via `merge_asof` on sale date |
| 2026-03-08 | `feature_engineering.py` | Added `is_landmark` and `is_historic_district` from LPC dataset (endpoint: ncre-qhxs); matched via BBL |
| 2026-03-08 | `feature_engineering.py` | Added `dist_waterfront_m` from OSM Overpass coastline query (33,507 points) via KD-tree |
| 2026-03-08 | `feature_engineering.py` | Added `dist_bike_lane_m` from OSM Overpass cycleway query (4,155 segments) via KD-tree |
| 2026-03-08 | `feature_engineering.py` | Added `dist_elem_school_m` from NYC elementary school directory (423 schools) via KD-tree |
| 2026-03-08 | `feature_engineering.py` | Added `dist_express_subway_m` and `nearest_station_is_express` from MTA GTFS Daytime Routes column (380 express stations) |
| 2026-03-08 | `feature_engineering.py` | Added `livability_complaint_rate` from 311 heat/rodent/dirty complaints (383,778 records); normalized per 1k NTA residents |
| 2026-03-08 | `feature_engineering.py` | Added `borough_income_deviation` (NTA median income minus borough median income) |
| 2026-03-08 | `feature_engineering.py` | Added `sale_year` and `sale_month` for temporal/seasonality signals |
| 2026-03-08 | `DATA_CATALOG.md` | Added raw data source entries #14–20 (DOB permits, FRED, elementary schools, 311 livability, LPC landmarks, OSM coastline, OSM bike lanes) |
| 2026-03-08 | `DATA_CATALOG.md` | Expanded "What Could Further Improve the Model" from 7 → 15 items with difficulty/notes columns |
| 2026-03-08 | `data/processed/features.csv` | Added FAR/zoning fields from PLUTO: `residfar`, `commfar`, `facilfar`, `builtfar`, `maxallwfar`, `far_utilization` (40 → 46 cols). Fix: actual column is `commfar` not `comfar`; `maxallwfar` computed as max(residfar, commfar, facilfar) |
| 2026-03-08 | `data/processed/features.csv` | Added building type flags from `bldgclass`: `has_elevator`, `is_condo`, `is_multifamily`, `is_single_fam`, `is_mixed_use` (46 → 51 cols). Fix: fill NaN bldgclass before `.str.startswith()` |
| 2026-03-08 | `data/raw/airbnb_listings.csv` | Downloaded Inside Airbnb NYC December 2025 snapshot (36,261 listings). URL required `.csv.gz` format + browser User-Agent header; earlier dated snapshot URLs returned 403 |
| 2026-03-08 | `data/processed/features.csv` | Added `airbnb_count_500m` via BallTree — 92.9% of properties have ≥1 Airbnb within 500m; median=18. corr with sale_price: +0.078 (51 → 52 cols) |
| 2026-03-08 | `data/processed/features.csv` | Added ACRIS prior sale features: `prior_sale_price`, `prior_sale_date`, `price_appreciation`, `years_since_prior_sale`, `is_flip` (52 → 57 cols). Source: ACRIS Master (bnx9-e6tj) + Legals (8h5j-fqxa) joined on document_id. 50 concurrent batches by borough+block in 31s. Match rate: 13.7%; prior_sale_price corr with sale_price: +0.719 |
| 2026-03-08 | `data/processed/features.csv` | Added school district quality composite: `school_district`, `district_avg_score`, `district_school_count` (57 → 60 cols). District boundaries from ArcGIS REST (NYC Open Data GeoJSON endpoint returned 400). 272 HS schools across 19 of 33 districts; 56.9% null (districts without HS data) |
| 2026-03-09 | `models/xgboost_model.json` | XGBoost trained: 917 trees, R²=0.735, MedAPE=17.8%, MAE=$726,593 on test set. Ridge baseline R²=0.238. Pre-processing: log1p target, winsorize 3 cols, label-encode bldgclass, 80/20 split by borough |
| 2026-03-09 | `models/scorer.py` | ThamanScorer class created — loads model, encodes inputs, returns predicted price + confidence range ±17.8% |
| 2026-03-09 | `models/shap_importance.png` | SHAP top-20 chart: gross_sqft #1, bldgclass #2, land_sqft #3, longitude #4, latitude #5, school_district #6 |
| 2026-03-09 | `models/actual_vs_predicted.png` | Actual vs Predicted scatter (properties < $10M, R²=0.735) |
| 2026-03-09 | `models/error_by_borough.png` | % error by borough boxplot |
| 2026-03-08 | `data/processed/features.csv` | Full audit + 6 data quality fixes: (1) building_age capped at 200 — 984 outliers fixed; (2) years_since_prior_sale clipped to 0 — 542 negatives fixed; (3) added `has_prior_sale` binary flag + corrected `is_flip` logic; (4) district_avg_score imputed with borough→global median — 0% null; (5) numfloors imputed by bldgclass+borough — 0% null; (6) far_utilization nulls filled with 0. Final: 36,203 × 61 cols, all non-ACRIS cols 0% null |
| 2026-03-09 | `api/main.py` | FastAPI backend created — `POST /predict` returns predicted price + SHAP top-10 drivers; `POST /batch` handles up to 50 properties; CORS enabled for frontend |
| 2026-03-09 | `api/spatial.py` | SpatialLookup class — loads 7 spatial datasets at startup; KD-tree for transit/school/park distances; BallTree for Airbnb density; NTA point-in-polygon for crime/income/school-district features |
| 2026-03-09 | `api/models.py` | Pydantic schemas — PredictRequest (8 required, 20+ optional fields), PredictResponse (price, range, SHAP drivers, spatial summary) |
| 2026-03-09 | `models/scorer.py` | Bug fix: ACRIS defaults changed from `0.0` to `np.nan` in `predict_single` — training median imputation now correctly applies (was causing ~100× underpricing) |
| 2026-03-09 | `frontend/index.html` | Map UI created — Leaflet.js NYC map + property form + result/SHAP/spatial panels served at `/ui` |
| 2026-03-09 | `frontend/style.css` | Full UI stylesheet — CSS variables, 2-col layout, animated cards, SHAP bars, responsive at 768px |
| 2026-03-09 | `frontend/app.js` | Browser JS — map click, borough auto-detect, form validation, fetch `/predict`, SHAP bar + spatial grid render |
| 2026-03-09 | `api/main.py` | Added `StaticFiles` mount at `/ui` → `frontend/`; root `/` now serves `index.html` directly; API info moved to `/api` |
| 2026-03-09 | `data/processed/features.csv` | Expanded dataset 5× — from 36,203 rows (2025 only) to 185,092 rows (2022–2026) via `scripts/download_more_sales.py`. Added 1 new column: `assesstot` (PLUTO total assessed value) for prior_sale_price imputation (62 cols total) |
| 2026-03-09 | `data/processed/features.csv` | Prior sale coverage improved from 13.7% → ~99% using `assesstot` ratios to estimate prior_sale_price for properties without ACRIS history |
| 2026-03-09 | `models/xgboost_model.json` | Retrained XGBoost base learner on 185K rows with Spatial GroupKFold CV (5 folds by NTA) |
| 2026-03-09 | `models/thaman_stack.pkl` | Trained stacking ensemble v2.1: XGBoost + LightGBM + CatBoost base learners + Ridge meta-learner. R²=0.6509, MedAPE=20.29% on spatial holdout |
| 2026-03-09 | `models/meta.json` | Updated to 71 feature names (added log-dist + target-encoded cols), stack metrics, `bldgclass_means`, `borough_bldg_means`, `walk_score_scaler` |
| 2026-03-09 | `api/main.py` | Fixed 500 error on `/predict` — `int(NaN)` on `school_district` spatial lookup. Added `_safe_int()` and `_safe_round()` helpers. Updated metrics to v2.1 |
| 2026-03-10 | `tests/conftest.py` | Created module-scoped pytest fixture — `with TestClient(app) as c` triggers FastAPI lifespan (model + spatial data load) for test suite |
| 2026-03-10 | `tests/test_api.py` | Fixed 17 test functions to use `client` fixture; renamed `test_root_redirects` → `test_root_serves_ui` (root now returns 200, not redirect) |
| 2026-03-10 | `tests/test_scorer.py` | Fixed hardcoded feature counts: 70 → 71 (2 assertions) |

---

## Feature Matrix (Model Input)
**File:** `data/processed/features.csv`
**Rows:** 185,092 properties | **Columns:** 62
**Target variable:** `sale_price` (log-transform required — raw skewness: 54.9)
**Sale date range:** 2022 – 2026 (NYC Rolling Sales, 4 years)

| Column | Type | Null % | Notes |
|---|---|---|---|
| `sale_price` | float | 0% | Target. Filter: > $10,000. Log-transform before training |
| `building_age` | int | 0% | Current year minus `yearbuilt` |
| `numfloors` | float | 2.9% | From PLUTO |
| `bldgclass` | str | 0% | NYC building class code (D4=elevator apt, A1=single family, etc.) |
| `gross_square_feet` | float | 35.8% | High null rate in Manhattan (condos) — impute with median by bldgclass |
| `land_square_feet` | float | 35.8% | Same |
| `residential_units` | float | 35.8% | Same |
| `dist_subway_m` | float | 0% | KD-tree distance to nearest subway station |
| `dist_bus_m` | float | 0% | KD-tree distance to nearest bus stop |
| `dist_express_subway_m` | float | 0% | Distance to nearest express subway station |
| `nearest_station_is_express` | int (0/1) | 0% | 1 if nearest subway station is an express stop |
| `dist_elem_school_m` | float | 0% | Distance to nearest elementary school |
| `dist_waterfront_m` | float | 0% | KD-tree distance to NYC coastline (OSM) |
| `dist_bike_lane_m` | float | 0% | Distance to nearest bike lane/cycleway |
| `is_landmark` | int (0/1) | 0% | 1 if property is an NYC Landmark (LPC) |
| `is_historic_district` | int (0/1) | 0% | 1 if in an LPC Historic District |
| `renovated_since_2018` | int (0/1) | 0% | 1 if DOB A1/A2 permit issued since 2018 |
| `years_since_renovation` | int | 0% | Years since last major permit (999 = never) |
| `mortgage_rate_30yr` | float | 0% | 30yr fixed mortgage rate at time of sale (FRED) |
| `sale_year` | int | 0% | Year of sale |
| `sale_month` | int | 0% | Month of sale (1–12, captures seasonality) |
| `livability_complaint_rate` | float | 0% | Heat/rodent/dirty complaints per 1k NTA residents |
| `borough_income_deviation` | float | 0% | NTA median income minus borough median income |
| `dist_school_m` | float | 0% | KD-tree distance to nearest school |
| `dist_park_m` | float | 0% | KD-tree distance to nearest park centroid |
| `dist_hospital_m` | float | 0% | KD-tree distance to nearest health POI (Overture) |
| `poi_count_500m` | int | 0% | BallTree count of Overture places within 500m |
| `crime_rate_nta` | float | 0% | Crimes per 1,000 residents in NTA (2022–2024) |
| `noise_density_nta` | float | 0% | Noise complaints per 1,000 residents in NTA (2022–2024) |
| `median_income_nta` | float | 0% | Median household income in NTA (ACS 2020) |
| `population_2020` | int | 0% | NTA population from 2020 Census |
| `residfar` | float | 0% | Residential Floor Area Ratio allowed by zoning (PLUTO) |
| `commfar` | float | 0% | Commercial FAR allowed by zoning (PLUTO) |
| `facilfar` | float | 0% | Facility FAR allowed by zoning (PLUTO) |
| `builtfar` | float | 0% | Actual built FAR (PLUTO) |
| `maxallwfar` | float | 0% | Max allowed FAR = max(residfar, commfar, facilfar) |
| `far_utilization` | float | 0.3% | builtfar / maxallwfar — how fully the lot is developed (capped at 5) |
| `has_elevator` | int (0/1) | 0% | 1 if building class is elevator-type (D class, select R class) |
| `is_condo` | int (0/1) | 0% | 1 if bldgclass starts with R (condo/co-op) |
| `is_multifamily` | int (0/1) | 0% | 1 if bldgclass starts with D (multifamily elevator) |
| `is_single_fam` | int (0/1) | 0% | 1 if bldgclass starts with A (single family) |
| `is_mixed_use` | int (0/1) | 0% | 1 if bldgclass starts with S (mixed use) |
| `airbnb_count_500m` | int | 0% | Airbnb listings within 500m (Inside Airbnb Dec 2025); median=18, corr=+0.078 |
| `prior_sale_price` | float | 86.3% | Price of second most recent DEED for same BBL (ACRIS); corr with sale_price: +0.719 |
| `prior_sale_date` | datetime | 86.3% | Date of prior sale |
| `price_appreciation` | float | 86.3% | (sale_price - prior_sale_price) / prior_sale_price; clipped to [-1, 10] |
| `years_since_prior_sale` | float | 86.3% | Years between prior deed and current sale |
| `is_flip` | int (0/1) | 0% | 1 if prior sale exists AND years_since_prior_sale < 2 |
| `has_prior_sale` | int (0/1) | 0% | 1 if ACRIS prior deed found for this BBL (13.7% coverage) |
| `school_district` | int | 0.03% | NYC school district number (1–32) |
| `district_avg_score` | float | 56.9% | Average graduation rate of HS schools in the district |
| `district_school_count` | int | 56.9% | Number of high schools in the district |

---

## Raw Data Sources

### 1. Property Sales
**File:** `data/raw/sales_geocoded.csv`
**Records:** 185,092 filtered (price > $10k) across 2022–2026 | originally 81,305 (2025 only)
**Description:** NYC property sale transactions across all 5 boroughs. Includes address, sale price, sale date, building class, and structural attributes joined from PLUTO via BBL. Expanded from 1 year (2025) to 4 years (2022–2026) using `scripts/download_more_sales.py`.
**Source:** [NYC Rolling Calendar Sales — NYC Open Data](https://www.nyc.gov/site/finance/property/property-rolling-sales-data.page)

---

### 2. PLUTO (Building Attributes)
**File:** `data/raw/nyc_pluto_25v4_csv/pluto_25v4.csv`
**Records:** 858,644 parcels | 92 columns
**Description:** Parcel-level building data for all NYC lots. Provides building age (yearbuilt), floor count, building class, zoning, lot area, and lat/lng coordinates used for geocoding the sales data.
**Source:** [NYC MapPLUTO — NYC Department of City Planning](https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page)

---

### 3. Overture Maps Places (POIs)
**File:** `data/raw/overture_places.geojson`
**Records:** 425,387 places
**Description:** Comprehensive point-of-interest dataset for NYC covering restaurants, shops, healthcare locations, services, and more. Used for `poi_count_500m` and `dist_hospital_m`. Replaced a broken OSM export that had only 12,886 records.
**Source:** [Overture Maps Foundation](https://overturemaps.org) via `leafmap.get_overture_data(type='place', bbox=NYC_BBOX)`

---

### 4. MTA Subway Stations
**File:** `data/raw/MTA_Subway_Stations_20260308.csv`
**Records:** 496 stations
**Description:** All NYC subway station locations with GTFS coordinates, line names, and ADA accessibility info. Used to compute `dist_subway_m`.
**Source:** [MTA Subway Stations — NYC Open Data](https://data.cityofnewyork.us/Transportation/Subway-Stations/arq3-7z49)

---

### 5. MTA Bus Stops
**File:** `data/raw/mta_bus_stops.csv`
**Records:** 9,747 stops (Manhattan + Brooklyn + Queens + Bronx)
**Description:** Bus stop locations from MTA GTFS feeds across all NYC boroughs. Used to compute `dist_bus_m` for the transit accessibility score.
**Source:** [MTA GTFS Static Feeds — rrgtfsfeeds.s3.amazonaws.com](https://rrgtfsfeeds.s3.amazonaws.com)
- Manhattan: `gtfs_m.zip`
- Brooklyn: `gtfs_b.zip`
- Queens: `gtfs_q.zip`
- Bronx: `gtfs_bx.zip`

---

### 6. Parks
**File:** `data/raw/parks_with_coords.csv`
**Records:** 2,058 parks (with centroids extracted from polygon geometry)
**Description:** NYC parks properties with WKT polygon geometry. Centroids were extracted using GeoPandas to compute `dist_park_m`.
**Source:** [Parks Properties — NYC Open Data](https://data.cityofnewyork.us/Recreation/Parks-Properties/enfh-gkve)

---

### 7. Schools
**File:** `data/raw/schools.csv`
**Records:** 427 schools
**Description:** NYC high school locations with quality metrics (graduation rate, attendance rate). Joined from two datasets: school directory (locations) and school quality report (metrics). Used for `dist_school_m`.
**Source (locations):** [NYC School Locations — NYC Open Data](https://data.cityofnewyork.us/resource/23z9-6uk9.csv)
**Source (quality):** [NYC School Quality Reports — NYC Open Data](https://data.cityofnewyork.us/resource/dnpx-dfnc.csv)

---

### 8. NYPD Crime Complaint Data
**File:** `data/raw/nypd_crimes.parquet`
**Records:** 1,646,571 complaints (2022–2024)
**Description:** All NYPD crime complaints with offense type, borough, date, and lat/lng. Used to compute `crime_rate_nta` (crimes per 1,000 NTA residents).
**Source:** [NYPD Complaint Data Historic — NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i)

---

### 9. NYC 311 Noise Complaints
**File:** `data/raw/noise_complaints.parquet`
**Records:** 1,000,000 complaints (2022–2024, filtered to Noise types)
**Description:** NYC 311 noise service requests with complaint type, date, and lat/lng. Types include Residential, Street/Sidewalk, Commercial, Vehicle. Used to compute `noise_density_nta`.
**Source:** [311 Service Requests — NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)

---

### 10. NTA Boundaries + Population + Income
**File:** `data/raw/nta_boundaries.geojson`
**Records:** 262 neighborhoods
**Description:** 2020 Neighborhood Tabulation Area boundaries enriched with:
- `population_2020` — from 2020 Decennial Census (P1_001N by tract, aggregated to NTA)
- `median_income_nta` — from ACS 5-year 2020 (B19013_001E by tract, aggregated to NTA)
Used to normalize crime and noise rates per 1,000 residents.
**Source (boundaries):** [NYC NTA 2020 — ArcGIS REST API](https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Neighborhood_Tabulation_Areas_2020/FeatureServer/0)
**Source (population):** [US Census 2020 Decennial API](https://api.census.gov/data/2020/dec/pl)
**Source (income):** [US Census ACS 5-year 2020 API](https://api.census.gov/data/2020/acs/acs5)

---

### 11. Road Network
**Files:** `data/raw/road_network/*.graphml` (10 files)
**Records:** Drive + Walk networks for all 5 boroughs
**Description:** Full road and pedestrian network graphs downloaded via OSMnx from OpenStreetMap. Can be used for accurate network-distance routing (vs straight-line KD-tree approximation) and road type analysis.

| Borough | Drive nodes | Drive edges | Walk nodes | Walk edges |
|---|---|---|---|---|
| Manhattan | 4,628 | 9,916 | 36,631 | 116,868 |
| Brooklyn | 12,224 | 30,350 | 69,677 | 224,888 |
| Queens | 21,491 | 55,972 | 123,096 | 400,104 |
| Bronx | 7,723 | 19,259 | 39,392 | 120,888 |
| Staten Island | 9,209 | 23,464 | 59,469 | 169,418 |

**Source:** [OpenStreetMap](https://www.openstreetmap.org) via [OSMnx](https://osmnx.readthedocs.io)

---

### 12. Census Tract Population
**File:** `data/raw/census_tract_population.csv`
**Records:** 2,327 NYC tracts
**Description:** 2020 Decennial Census population (P1_001N) for every census tract in NYC's 5 counties. Used as an intermediate step to compute NTA-level population via spatial join.
**Source:** [US Census Bureau 2020 Decennial API](https://api.census.gov/data/2020/dec/pl)

---

### 13. Census Tract Income
**File:** `data/raw/census_tract_income.csv`
**Records:** 2,327 NYC tracts
**Description:** ACS 5-year 2020 median household income (B19013_001E) by census tract. Used as an intermediate step to compute NTA-level median income via spatial join.
**Source:** [US Census Bureau ACS 5-year 2020 API](https://api.census.gov/data/2020/acs/acs5)

---

### 14. DOB Renovation Permits
**File:** `data/raw/dob_permits.csv`
**Records:** 2,673 unique BBLs with A1/A2 permits since 2018
**Description:** NYC Department of Buildings alteration permits (job types A1/A2) filed since 2018. Used to compute `renovated_since_2018` (binary flag) and `years_since_renovation`. BBL constructed from borough + block + lot columns matching PLUTO format.
**Source:** [DOB Permit Issuance — NYC Open Data](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a)

---

### 15. FRED Mortgage Rates
**File:** `data/raw/mortgage_rates.csv`
**Records:** ~1,500 weekly observations (1971–2026)
**Description:** Weekly 30-year fixed mortgage rate from the Federal Reserve Economic Data (FRED). Joined to each property sale via `merge_asof` on sale date. Used as `mortgage_rate_30yr` feature capturing macro credit conditions at time of sale.
**Source:** [FRED MORTGAGE30US — St. Louis Federal Reserve](https://fred.stlouisfed.org/series/MORTGAGE30US)

---

### 16. Elementary Schools
**File:** `data/raw/elementary_schools.csv`
**Records:** 423 elementary schools
**Description:** NYC public elementary school (K–5/PK) locations filtered from the NYC school directory. Used to compute `dist_elem_school_m` via KD-tree. Separate from the high school dataset to capture proximity to schools relevant to families with young children.
**Source:** [NYC School Locations — NYC Open Data](https://data.cityofnewyork.us/resource/23z9-6uk9.csv)

---

### 17. NYC 311 Livability Complaints
**File:** `data/raw/livability_complaints.parquet`
**Records:** 383,778 complaints (2022–2024)
**Description:** NYC 311 service requests filtered to livability issue types: heat/hot water, rodent, and dirty conditions. Aggregated to NTA level and normalized per 1,000 residents to compute `livability_complaint_rate`. Complements noise complaints as a separate QoL dimension.
**Source:** [311 Service Requests — NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)

---

### 18. LPC Landmarks
**File:** Used directly from Socrata API (not cached)
**Records:** 39,385 landmark records → 3,695 unique landmark BBLs
**Description:** New York City Landmarks Preservation Commission (LPC) designated landmarks and historic district properties. Matched to sales via BBL to compute `is_landmark` and `is_historic_district`. Landmarks can affect both value (premium) and renovation flexibility.
**Source:** [LPC Individual Landmarks — NYC Open Data](https://data.cityofnewyork.us/Housing-Development/LPC-Individual-Landmarks/ncre-qhxs)

---

### 19. NYC Coastline (OSM Overpass)
**File:** Used in-memory during pipeline (not saved separately)
**Records:** 33,507 coastline points
**Description:** NYC shoreline geometry retrieved via OpenStreetMap Overpass API (`natural=coastline` within NYC bounding box). Used to compute `dist_waterfront_m` via KD-tree. Waterfront proximity is a major price driver in NYC, particularly for properties in Battery Park, DUMBO, Red Hook, and Williamsburg.
**Source:** [OpenStreetMap Overpass API](https://overpass-api.de) — query: `natural=coastline` within NYC bbox

---

### 20. Bike Lanes (OSM Overpass)
**File:** Used in-memory during pipeline (not saved separately)
**Records:** 4,155 bike lane segments → coordinate points extracted
**Description:** NYC bike lane and cycleway geometries from OpenStreetMap via Overpass API. Segment coordinates extracted and used for KD-tree nearest-distance calculation (`dist_bike_lane_m`). Captures micro-mobility infrastructure as a livability and urban quality indicator.
**Source:** [OpenStreetMap Overpass API](https://overpass-api.de) — query: `highway=cycleway` + `cycleway` tags within NYC bbox

---

### 21. Inside Airbnb NYC Listings
**File:** `data/raw/airbnb_listings.csv`
**Records:** 36,261 active listings (December 2025 snapshot)
**Description:** Short-term rental listings across all 5 boroughs. Used to compute `airbnb_count_500m` — density of Airbnb units within 500m radius via BallTree. High Airbnb density correlates with tourist-heavy areas and mixed-use neighborhoods. Correlation with sale_price: +0.078.
**Source:** [Inside Airbnb — New York City](https://insideairbnb.com/get-the-data/) (free, no key). Note: use `.csv.gz` URL format + browser User-Agent header; plain CSV URLs return 403.

---

### 22. ACRIS Real Property Master
**File:** Used directly from Socrata API (not cached)
**Records:** 500,000 most recent DEED transactions (2011–2026)
**Description:** NYC property deed transfer records from the Automated City Register Information System. Contains `document_id`, sale date, and sale amount. **Does not contain BBL** — must be joined with ACRIS Legals (source #23) on `document_id` to obtain the BBL.
**Source:** [ACRIS Real Property Master — NYC Open Data](https://data.cityofnewyork.us/City-Government/ACRIS-Real-Property-Master/bnx9-e6tj)

---

### 23. ACRIS Real Property Legals
**File:** Used directly from Socrata API (not cached)
**Records:** 22.5M total; ~24k fetched via 50 batched borough+block queries
**Description:** Maps each ACRIS document_id to its parcel (borough, block, lot). Used in combination with ACRIS Master to build `prior_sale_price`, `price_appreciation`, `years_since_prior_sale`, and `is_flip`. Downloaded in 50 concurrent batches (300 blocks per batch) in ~31 seconds.
**Source:** [ACRIS Real Property Legals — NYC Open Data](https://data.cityofnewyork.us/City-Government/ACRIS-Real-Property-Legals/8h5j-fqxa)

---

### 24. NYC School Districts
**File:** Used in-memory during pipeline (not saved separately)
**Records:** 33 school districts
**Description:** NYC school district boundary polygons. Used for spatial join to assign each property to a district, then joined with school quality metrics to compute `district_avg_score` and `district_school_count`. Note: NYC Open Data GeoJSON export endpoint returned 400 — used ArcGIS REST API instead.
**Source:** [NYC School Districts — ArcGIS REST API](https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_School_Districts/FeatureServer/0)

---

## Data Quality Notes

> **Last audit:** 2026-03-10 — All 24 raw sources present on disk ✓ | Feature matrix: 185,092 × 62 cols

| Issue | Detail | Status | Action Taken |
|---|---|---|---|
| Price skewness | Raw skewness = 54.86 | ⚠️ Requires transform | Apply `log1p(sale_price)` before training — log skewness = 0.78 ✓ |
| gross_sqft nulls | 35.8% null (Manhattan condos) | ✅ Fixed | Imputed with median by `bldgclass` + `borough` → 0% null |
| NTA outliers | BX1203 noise=2,711/1k, MN0502 crime=460/1k | ⚠️ Requires transform | Winsorize or log-transform at 99th percentile before training |
| Borough encoding | Stored as integers 1–5 | ✅ OK | Map: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island |
| building_age outlier | 984 properties with age > 200 (yearbuilt=0 in source) | ✅ Fixed | Capped at 200 years |
| years_since_prior_sale negatives | 542 properties with negative values (ACRIS ordering anomaly) | ✅ Fixed | Clipped to 0 minimum |
| is_flip ambiguity | Was 0 for no-prior-sale properties (misleading) | ✅ Fixed | Added `has_prior_sale` flag; is_flip only set 1 where prior sale exists |
| district_avg_score nulls | 56.9% null (districts with no HS data) | ✅ Fixed | Imputed with borough median → global median fallback → 0% null |
| numfloors nulls | 2.9% null | ✅ Fixed | Imputed with median by `bldgclass` + `borough` → 0% null |
| far_utilization nulls | 0.3% null (vacant lots with no FAR) | ✅ Fixed | Filled with 0 (no development) |
| ACRIS prior sale coverage | 86.3% null — window limited to 2011–2026 | ⚠️ Known limitation | Use `has_prior_sale` as binary feature; model imputes missing with tree splits |
| NTA edge properties | 13 properties fell outside NTA boundary | ✅ Fixed | Filled with borough median for all NTA-level features |
| dist_bus_m outlier | Max 26,807m (Staten Island, no bus service) | ✅ Valid | Real data — Staten Island bus coverage is sparse. Keep as-is |
| dist_express_subway_m outlier | Max 27,592m (Staten Island — no express service) | ✅ Valid | Real data — Staten Island has no express subway. Keep as-is |

---

## Model Readiness Checklist

| Check | Status | Notes |
|---|---|---|
| All raw files on disk | ✅ 19/19 files present | Road network = 10 GraphML files |
| Feature matrix shape | ✅ 185,092 × 62 cols | Expanded to 4 years of sales (2022–2026) |
| Non-ACRIS nulls | ✅ 0% null | All columns except ACRIS prior-sale group fully populated |
| ACRIS prior sale coverage | ✅ ~99% | `prior_sale_price` coverage improved using `assesstot` ratios for properties without ACRIS history |
| Target variable | ✅ Ready | Apply `log1p()` before training (skewness 54.86 → 0.78) |
| Categorical encoding | ✅ Done | `bldgclass` → target-encoded `bldgclass_encoded` + `borough_bldg_encoded` (cross-feature) |
| NTA outliers | ✅ Done | Winsorized at 99th percentile (`crime_rate_nta`, `noise_density_nta`, `livability_complaint_rate`) |
| Train/test split | ✅ Done | Spatial GroupKFold CV (5 folds by NTA) — prevents geographic leakage |
| Model | ✅ Done | Stack v2.1: XGB + LGB + CatBoost + Ridge meta | R²=0.6509 | MedAPE=20.29% |

---

## What Could Further Improve the Model

| Data | Impact | Difficulty | Notes |
|---|---|---|---|
| Floor level (condo unit floor #) | Very High — strongest predictor for apartments | Very Hard | Not in public data; ACRIS deeds sometimes mention unit/floor but inconsistent |
| View / floor premium (condos) | High | Very Hard | Would require scraping StreetEasy/Zillow — not in public records |
| Interior sqft (condos) | High — 35.8% null in PLUTO for condos | Medium | ACRIS mortgage docs sometimes contain unit sizes; complex to parse |
| Elementary school quality ratings | High — distance added, quality not yet scored | Medium | NYC DOE K-8 quality reports; same long-format as HS dataset; join on `dbn` |
| ACRIS prior sale — expand window | Medium — only 13.7% matched | Easy | Add more 500k batches: `$offset=500000`, `$offset=1000000` to reach pre-2011 deeds |
| Air quality index by NTA | Medium — pollution negatively affects prices | Medium | EPA AQS API (free key at `aqs.epa.gov`); param 88101=PM2.5; IDW interpolation from monitors |
| Flood risk score | Medium — FEMA zones suppress coastal prices | Medium | FEMA NFHL API had SSL errors; download GDB locally from msc.fema.gov |
| Foreclosure / lis pendens rate | Medium — financial distress signal | Medium | ACRIS `doc_type='LIS PEN'`; same pipeline as sources #22–23 |
| Walk Score proxy | Medium — composite pedestrian signal | Easy | Compute from existing features: weighted(dist_subway, dist_bus, poi_count_500m) |
| Shadow / sunlight hours | Low–Medium | Hard | Requires NYC 3D Building Model + solar angle computation (pysolar) |
