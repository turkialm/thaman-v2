FROM python:3.11-slim

# System dependencies required by geopandas / GDAL
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and runtime data
COPY api/        api/
COPY models/scorer.py           models/scorer.py
COPY models/meta.json           models/meta.json
COPY models/xgboost_model.json  models/xgboost_model.json
COPY models/thaman_stack.pkl    models/thaman_stack.pkl
COPY models/luxury_model.json   models/luxury_model.json
COPY models/riyadh_stack.pkl    models/riyadh_stack.pkl
COPY models/riyadh_meta.json    models/riyadh_meta.json
COPY frontend/   frontend/

# Runtime data files only (training data excluded via .gitignore)
# features_v4.csv preferred (v4 NTA stats); spatial.py falls back to features.csv
COPY data/processed/features_v4.csv              data/processed/features_v4.csv
COPY data/processed/features.csv                 data/processed/features.csv
COPY data/raw/nta_boundaries.geojson             data/raw/nta_boundaries.geojson
COPY data/raw/airbnb_listings.csv                data/raw/airbnb_listings.csv
COPY data/raw/MTA_Subway_Stations_20260308.csv   data/raw/MTA_Subway_Stations_20260308.csv
COPY data/raw/mta_bus_stops.csv                  data/raw/mta_bus_stops.csv
COPY data/raw/parks_with_coords.csv              data/raw/parks_with_coords.csv
COPY data/raw/schools.csv                        data/raw/schools.csv
COPY data/raw/elementary_schools.csv             data/raw/elementary_schools.csv
COPY data/raw/mortgage_rates.csv                 data/raw/mortgage_rates.csv
# v4 additions: waterfront & bike-lane distance features
COPY data/raw/nyc_coastline_pts.npy              data/raw/nyc_coastline_pts.npy
COPY data/raw/nyc_bike_lanes.geojson             data/raw/nyc_bike_lanes.geojson
# Riyadh spatial + district layer
COPY data/processed/features_riyadh.csv          data/processed/features_riyadh.csv
COPY data/raw/metro-stations-in-riyadh-by-metro-line-and-station-type-2024.geojson  data/raw/metro-stations-in-riyadh-by-metro-line-and-station-type-2024.geojson
COPY data/raw/bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024.geojson  data/raw/bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024.geojson
COPY data/raw/traffic-intersections-by-main-street-and-cross-street-2024.geojson  data/raw/traffic-intersections-by-main-street-and-cross-street-2024.geojson
COPY data/raw/commercial-services-by-category-sub-municipality-and-district-2024.geojson  data/raw/commercial-services-by-category-sub-municipality-and-district-2024.geojson
COPY data/raw/air-quality.csv                    data/raw/air-quality.csv
COPY data/raw/riyadh_mosques.csv                 data/raw/riyadh_mosques.csv
COPY data/raw/riyadh_malls.csv                   data/raw/riyadh_malls.csv
COPY data/raw/riyadh_schools.csv                 data/raw/riyadh_schools.csv
COPY data/raw/riyadh_hospitals.csv               data/raw/riyadh_hospitals.csv
COPY data/raw/riyadh_parks.csv                   data/raw/riyadh_parks.csv
COPY data/raw/rcrc_entertainment.csv             data/raw/rcrc_entertainment.csv
# Note: overture_places.geojson (590MB) is gitignored; POI counts degrade to 0 on HF

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
