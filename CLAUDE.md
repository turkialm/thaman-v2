# THAMAN — Developer Guide for Claude

## Project
AI-powered property valuation (AVM) for NYC + Riyadh.
BSc graduation project — Umm Al-Qura University, 2026.

## Quick Start
```bash
cd /Users/totam/Desktop/new_try
uvicorn api.main:app --port 8000          # start API
python -m pytest tests/ -v               # run all 100 tests
open frontend/index.html                  # open UI (or use API /ui)
```

## Architecture

```
api/main.py          FastAPI v2 — /predict (NYC), /predict/riyadh, /nearby, /batch, /health
models/scorer.py     ThamanScorer — loads NYC stack (v22, 134 features) + Riyadh stack (v11, 140 features)
models/spatial.py    SpatialLookup — NTA, subway, parks, waterfront, commuter rail, income, etc.
frontend/            index.html + app.js + style.css — bilingual (EN/AR) map UI
frontend/charts.html Analytics dashboard — model versions, borough breakdown, feature importance
training/            train_stack_v12.py (NYC), train_stack_riyadh_v2.py (Riyadh)
tests/               100 tests across 8 files — api, scorer, feature parity, shap, regression, golden, distribution, load
```

## Models

### NYC Stack v22
- R² = 0.6495, MedAPE = 20.32%, 134 features
- Stack: XGB-A + XGB-B + LGB + CatBoost + Ridge meta
- Key features: NTA encoding, subway proximity, school scores, park distance, waterfront, BBL history, income, flood zone, landmarks, NTA×bldgtype temporal lags
- Spatial 10-fold GroupKFold CV by NTA

### Riyadh Stack v11
- Holdout R² = 0.8003, MedAPE = 15.56%, 140 features
- OOF R² = 0.9343, MedAPE = 8.28%
- Stack: XGB-A + XGB-B + LGB + CatBoost + Ridge meta
- Key features: district encoding, type-stratified temporal lags, metro proximity, hub distances, Haraj asking prices, Suhail transaction density
- 5-fold GroupKFold CV by district

## Key Files
| File | Purpose |
|---|---|
| `models/thaman_stack.pkl` | NYC v22 trained stack (134 features) |
| `models/riyadh_stack.pkl` | Riyadh v11 trained stack (140 features) |
| `models/meta.json` | NYC feature names, NTA encodings, BBL lookup, income lookup |
| `models/riyadh_meta.json` | Riyadh feature names, district encodings, metro lookup, Haraj lookup |
| `api/main.py` | FastAPI endpoints — feature engineering at inference time |
| `models/scorer.py` | ThamanScorer.predict(), explain(), adaptive_confidence() |
| `models/spatial.py` | SpatialLookup — all geospatial feature enrichment (NYC) |

## Adding New Model Versions

1. Update `training/train_stack_v12.py` — add feature computation block, update version string, update `meta.update({})` with new lookup maps.
2. Run training: `python training/train_stack_v12.py` (~70 min).
3. Update `models/scorer.py` — add new version string to BOTH version lists in `__init__` and `_predict_raw`.
4. Update `api/main.py` — add feature injection block for new features.
5. Update `frontend/index.html` — model badges, disclaimer, compare modal.
6. Update `frontend/app.js` — version strings in modelTag chain, metrics in copyResults.
7. Update `frontend/charts.html` — NYC_VERSIONS array, baseline chart, stack compare chart.
8. Update `tests/test_regression.py` — re-pin expected_price values.
9. Update `tests/test_scorer.py` and `tests/test_feature_parity.py` — feature count assertions.
10. Run `pytest tests/ -v` — confirm 100/100.

## Inference Pipeline (NYC)

```
POST /predict → feature_vector() → SpatialLookup enrichment → ThamanScorer.predict()
                                 → NTA lookup → subway/park/income → SHAP explain
                                 → adaptive_confidence() → AVM QC → response JSON
```

## i18n
- `setLang(lang)` in app.js — switches EN↔AR, updates RTL, re-renders all text
- `TR` dict in app.js — all UI strings with `en`/`ar` keys
- Arabic card titles set explicitly in `renderRiyadhResults()` and `setLang()`

## Tests
```
tests/test_api.py           22 tests — endpoints, validation, QC block, luxury flag
tests/test_scorer.py        17 tests — feature names, predict, SHAP, confidence
tests/test_feature_parity.py 15 tests — feature counts, NTA lookup, v11/v12/v22 features
tests/test_shap.py          14 tests — SHAP drivers structure, sensitivity, Riyadh
tests/test_regression.py     6 tests — pinned output regression (±5%)
tests/test_golden.py         9 tests — known price ranges (broad ±40%)
tests/test_distribution.py  12 tests — grid median, borough ordering, Riyadh range
tests/test_load.py           5 tests — concurrent load (10 threads)
```

## Data Sources
- NYC: Citywide Rolling Calendar Sales, PLUTO, ACS Census (income), NYC Open Data (311/NYPD/Parks/Schools), Overture Maps POIs, LIRR/Metro-North stations, MTA subway, Citi Bike, NYC coastline
- Riyadh: Suhail transaction data (33,837 tx May 2025–May 2026), Haraj listings (May 2026, 1,824 listings), OSM POIs, Riyadh Metro (94 stations), REI indices
