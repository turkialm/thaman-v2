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
COPY models/scorer.py      models/scorer.py
COPY models/meta.json      models/meta.json
COPY models/xgboost_model.json  models/xgboost_model.json
COPY models/thaman_stack.pkl    models/thaman_stack.pkl
COPY frontend/   frontend/

# Runtime data files only (training data excluded via .gitignore)
COPY data/processed/features.csv          data/processed/features.csv
COPY data/raw/nta_boundaries.geojson      data/raw/nta_boundaries.geojson
COPY data/raw/airbnb_listings.csv         data/raw/airbnb_listings.csv
COPY data/raw/MTA_Subway_Stations_20260308.csv  data/raw/MTA_Subway_Stations_20260308.csv
COPY data/raw/mta_bus_stops.csv           data/raw/mta_bus_stops.csv
COPY data/raw/parks_with_coords.csv       data/raw/parks_with_coords.csv
COPY data/raw/schools.csv                 data/raw/schools.csv
COPY data/raw/elementary_schools.csv      data/raw/elementary_schools.csv
COPY data/raw/mortgage_rates.csv          data/raw/mortgage_rates.csv

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
