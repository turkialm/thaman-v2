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
COPY api/             api/
COPY models/scorer.py models/scorer.py
COPY frontend/        frontend/
COPY download_models.py download_models.py
COPY app.py           app.py

# Static data files (small — not on HF Hub)
COPY data/raw/nta_boundaries.geojson             data/raw/nta_boundaries.geojson
COPY data/raw/airbnb_listings.csv                data/raw/airbnb_listings.csv
COPY data/raw/MTA_Subway_Stations_20260308.csv   data/raw/MTA_Subway_Stations_20260308.csv
COPY data/raw/mta_bus_stops.csv                  data/raw/mta_bus_stops.csv
COPY data/raw/parks_with_coords.csv              data/raw/parks_with_coords.csv
COPY data/raw/schools.csv                        data/raw/schools.csv
COPY data/raw/elementary_schools.csv             data/raw/elementary_schools.csv
COPY data/raw/mortgage_rates.csv                 data/raw/mortgage_rates.csv
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
COPY data/raw/saudi_listings_haraj_20260518.csv  data/raw/saudi_listings_haraj_20260518.csv
COPY data/processed/features_riyadh.csv          data/processed/features_riyadh.csv
COPY data/processed/features_riyadh_v2.csv       data/processed/features_riyadh_v2.csv
COPY data/processed/riyadh_district_polygons.geojson  data/processed/riyadh_district_polygons.geojson
COPY data/processed/nta_simplified.geojson        data/processed/nta_simplified.geojson
COPY data/processed/district_centroids.csv        data/processed/district_centroids.csv

# Create model dirs so download_models.py can write into them
RUN mkdir -p models data/raw

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# app.py downloads models from HF Hub at startup, then launches uvicorn
CMD ["python", "app.py"]
