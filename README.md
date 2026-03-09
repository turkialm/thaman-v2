---
title: THAMAN Property Valuation
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# THAMAN — AI-Powered NYC Property Valuation

BSc Graduation Project — Umm Al-Qura University

An interactive map-based web application that estimates NYC property prices using machine learning and Quality-of-Life (QoL) spatial indicators.

## Features

- **XGBoost + LightGBM + CatBoost** stacking ensemble (R²=0.59, MedAPE=18.83%)
- **70 features** — structural attributes + GIS spatial lookups
- **Auto-computed** transit distances, crime rate, income, school district from lat/lng
- **SHAP explanations** — top 10 drivers per prediction
- **Interactive Leaflet.js map** — click anywhere in NYC to estimate

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **ML**: XGBoost, LightGBM, CatBoost, scikit-learn
- **Spatial**: GeoPandas, SciPy KD-trees, BallTree
- **Frontend**: Leaflet.js, vanilla JS

## Usage

Open the app and:
1. Click anywhere on the NYC map
2. Fill in building details (type, size, age, floors, units)
3. Click **Estimate Price**

## API

- `GET /api` — API info
- `POST /predict` — predict one property
- `GET /docs` — Swagger UI
