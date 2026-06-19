"""
THAMAN API — Pydantic request/response schemas
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List


# ── Borough mapping ───────────────────────────────────────────────────
# 1 = Manhattan, 2 = Bronx, 3 = Brooklyn, 4 = Queens, 5 = Staten Island

BOROUGH_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

BLDGCLASS_DESCRIPTIONS = {
    "A0": "Cape Cod",
    "A1": "Two story detached - small or moderate",
    "A2": "One story - permanent living quarters",
    "A3": "Large suburban residence",
    "A4": "City residence - one family",
    "A5": "City residence - attached (rowhouse)",
    "A6": "Summer residence",
    "A7": "Mansion type or town house",
    "A8": "Bungalow colony",
    "A9": "Single family residence NEC",
    "B1": "Two family brick",
    "B2": "Two family frame",
    "B3": "Two family conversion",
    "B9": "Two family NEC",
    "C0": "Three families",
    "C1": "Over six families without stores",
    "C2": "Five to six families",
    "C3": "Four families",
    "C4": "Old law tenements",
    "C5": "Converted dwellings - rooming houses",
    "C6": "Cooperative",
    "C7": "Walk-up apartments with stores",
    "C8": "Walk-up apartment with offices",
    "C9": "Walk-up apartment NEC",
    "D0": "Elevator cooperative",
    "D1": "Elevator apartment (semi-fireproof)",
    "D2": "Elevator apartment (artists)",
    "D3": "Elevator apartment (fireproof - subsidized)",
    "D4": "Elevator apartment building (full building)",
    "D5": "Converted elevator building",
    "D6": "Elevator apartment with stores",
    "D7": "Elevator apartment with offices",
    "D8": "Elevator apartment NEC",
    "D9": "Elevator apartment NEC",
    "R1": "Condo residential unit in elevator building",
    "S0": "Primarily one-family with two stores or offices",
    "S1": "Primarily one-family with one store or office",
    "S2": "Primarily two-family with one store or office",
    "S4": "Primarily mixed use (3+ families + commercial)",
    "S5": "Converted from one/two family",
}


# ── Request model ─────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    # Required: Location
    latitude: float = Field(..., ge=40.4, le=40.95, description="Property latitude (NYC: 40.4–40.95)")
    longitude: float = Field(..., ge=-74.3, le=-73.7, description="Property longitude (NYC: -74.3 – -73.7)")

    # Required: Property basics
    gross_square_feet: float = Field(..., gt=0, description="Total building gross square footage")
    building_age: int = Field(..., ge=0, le=200, description="Age of building in years")
    bldgclass: str = Field(..., description="NYC building class code (e.g. 'A1', 'D4', 'R1')")
    borough: int = Field(..., ge=1, le=5, description="Borough: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island")
    numfloors: float = Field(..., gt=0, description="Number of floors")
    residential_units: int = Field(..., ge=0, description="Number of residential units")

    # Optional property attributes
    land_square_feet: Optional[float] = Field(None, ge=0, description="Land area in sq ft")

    # Optional: building type flags (auto-inferred from bldgclass if not set)
    has_elevator: Optional[int] = Field(None, ge=0, le=1, description="1 if building has elevator")
    is_condo: Optional[int] = Field(None, ge=0, le=1, description="1 if condo (R-class)")
    is_multifamily: Optional[int] = Field(None, ge=0, le=1, description="1 if multi-family elevator (D-class)")
    is_single_fam: Optional[int] = Field(None, ge=0, le=1, description="1 if single family (A-class)")
    is_mixed_use: Optional[int] = Field(None, ge=0, le=1, description="1 if mixed-use (S-class)")

    # Optional: FAR/Zoning (from PLUTO; defaults to 0 if unknown)
    builtfar: Optional[float] = Field(None, ge=0, description="Built Floor Area Ratio")
    residfar: Optional[float] = Field(None, ge=0, description="Residential allowable FAR")
    commfar: Optional[float] = Field(None, ge=0, description="Commercial allowable FAR")
    facilfar: Optional[float] = Field(None, ge=0, description="Community facility allowable FAR")
    maxallwfar: Optional[float] = Field(None, ge=0, description="Maximum allowable FAR")
    far_utilization: Optional[float] = Field(None, ge=0, description="FAR utilization ratio (builtfar/maxallwfar)")

    # Optional: ACRIS prior sale data
    prior_sale_price: Optional[float] = Field(None, ge=0, description="Prior recorded sale price (from ACRIS)")
    price_appreciation: Optional[float] = Field(None, description="Price appreciation ratio since prior sale")
    years_since_prior_sale: Optional[float] = Field(None, ge=0, description="Years since prior sale")
    has_prior_sale: Optional[int] = Field(0, ge=0, le=1, description="1 if property has a prior sale record")
    is_flip: Optional[int] = Field(0, ge=0, le=1, description="1 if resold within 2 years")

    # Optional: Renovation / landmark status
    renovated_since_2018: Optional[int] = Field(0, ge=0, le=1, description="1 if major permit since 2018")
    years_since_renovation: Optional[float] = Field(0.0, ge=0, description="Years since last major renovation")

    # Optional: Time context (defaults to current year/month)
    sale_year: Optional[int] = Field(None, ge=2010, le=2030, description="Year of hypothetical sale")
    sale_month: Optional[int] = Field(None, ge=1, le=12, description="Month of hypothetical sale")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "latitude": 40.6892,
            "longitude": -73.9442,
            "gross_square_feet": 1800,
            "building_age": 55,
            "bldgclass": "A1",
            "borough": 3,
            "numfloors": 2,
            "residential_units": 1,
            "land_square_feet": 2000,
        }
    })


# ── Response models ───────────────────────────────────────────────────

class FeatureDriver(BaseModel):
    feature: str = Field(description="Feature name")
    value: float = Field(description="Feature value for this property")
    impact: float = Field(description="SHAP value (log-price units)")
    direction: str = Field(description="'positive' or 'negative'")
    description: str = Field(description="Human-readable feature description")


class AvmQc(BaseModel):
    """2026 AVM Quality Control block — confidence score, hit rate, and QC flags."""
    confidence_score:     int   = Field(description="0–100 AVM reliability score (100 – segment MedAPE)")
    confidence_grade:     str   = Field(description="Letter grade: A (≥85) / B (≥75) / C (≥65) / D (<65)")
    segment_medape_pct:   float = Field(description="Segment-specific MedAPE used for this prediction")
    comparables_found:    int   = Field(description="Training-set sales within 800m (hit rate signal)")
    comparables_radius_m: int   = Field(default=800, description="Radius used for comparable count")
    sparse_market:        bool  = Field(description="True if fewer than 5 comparables found within 800m")
    qc_flags:             List[str] = Field(default_factory=list,
                              description="SPARSE_MARKET | LUXURY_SEGMENT | HIGH_UNCERTAINTY | METRO_CORE")


class SpatialFeatures(BaseModel):
    dist_subway_m: float
    dist_bus_m: float
    dist_express_subway_m: float
    nearest_station_is_express: int
    dist_school_m: float
    dist_elem_school_m: float
    dist_park_m: float
    dist_hospital_m: float
    dist_waterfront_m: float
    dist_bike_lane_m: float
    poi_count_500m: float
    airbnb_count_500m: float
    crime_rate_nta: float
    noise_density_nta: float
    livability_complaint_rate: float
    population_2020: float
    median_income_nta: float
    school_district: float
    district_avg_score: float
    mortgage_rate_30yr: float


class PredictResponse(BaseModel):
    # Core prediction
    predicted_price: int
    confidence_low: int
    confidence_high: int
    confidence_note: str = "±18.98% MedAPE confidence interval"

    # Model metadata
    model: str
    r2_test: float
    medape_pct: float

    # Context
    borough_name: str
    bldgclass_description: str

    # Auto-computed spatial features
    spatial_features: dict

    # SHAP explanations
    top_drivers: List[FeatureDriver]

    # AVM Quality Control (2026 standard) — always present
    avm_qc: Optional[AvmQc] = None

    # Resolved NTA code (for transparency / debugging)
    nta_code: Optional[str] = None

# ── Riyadh predict schemas ────────────────────────────────────────────

RIYADH_TYPE_LABELS = {
    "شقة":           "Apartment / شقة",
    "فيلا":          "Villa / فيلا",
    "قطعة أرض-سكنى": "Residential Plot / قطعة أرض سكنية",
    "عمارة":         "Building / عمارة",
}


class RiyadhPredictRequest(BaseModel):
    latitude:      float = Field(..., ge=23.5, le=26.0, description="Property latitude (Riyadh bbox)")
    longitude:     float = Field(..., ge=45.5, le=48.0, description="Property longitude (Riyadh bbox)")
    property_type: str   = Field(..., description="Property type: شقة | فيلا | قطعة أرض-سكنى | عمارة")
    area_sqm:      float = Field(..., gt=0, description="Property area in square meters")
    year:          Optional[int]   = Field(None, ge=2018, le=2030)
    quarter:       Optional[int]   = Field(None, ge=1, le=4)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "latitude": 24.7500,
            "longitude": 46.6800,
            "property_type": "شقة",
            "area_sqm": 150.0,
            "year": 2025,
            "quarter": 2,
        }
    })


class RiyadhPredictResponse(BaseModel):
    predicted_price_sqm:  int   = Field(description="Predicted price per sqm (SAR/m²)")
    predicted_total_sar:  int   = Field(description="Estimated total price (SAR) for the given area")
    confidence_low_sqm:   int
    confidence_high_sqm:  int
    confidence_low_sar:   int
    confidence_high_sar:  int
    area_sqm:             float
    property_type:        str
    district_ar:          Optional[str] = None
    model:                str
    r2_test:              float
    medape_pct:           float
    spatial_features:     dict
    top_drivers:          List[FeatureDriver] = Field(default_factory=list, description="Top SHAP feature drivers")
    # Asking-price overlay (Bayut listing median for matched district)
    asking_price_psqm:    Optional[int]   = Field(None, description="Bayut median asking price per sqm (SAR/m²)")
    asking_price_total:   Optional[int]   = Field(None, description="Estimated Bayut asking total for given area (SAR)")
    asking_spread_pct:    Optional[float] = Field(None, description="Asking-price premium over THAMAN transaction estimate (%)")
    asking_price_source:  Optional[str]   = Field(None, description="Source platform for asking-price data")
