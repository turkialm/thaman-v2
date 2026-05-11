"""
repair_meta_v7.py — rebuilds meta.json after train_stack_v7 corrupted it.
Run:  cd /Users/totam/Desktop/new_try && python training/repair_meta_v7.py
"""
import os, sys, json, warnings, joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC      = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")

SMOOTH_K = 30

# ── Stats from training log ───────────────────────────────────────────
STATS = {
    "r2_xa": 0.6214, "mae_xa": 1_116_066, "mp_xa": 21.35,
    "r2_xb": 0.6157, "mae_xb": 1_102_473, "mp_xb": 21.96,
    "r2_lg": 0.6152, "mae_lg": 1_107_890, "mp_lg": 21.57,
    "r2_ct": 0.6329, "mae_ct": 1_103_777, "mp_ct": 22.51,
    "r2_stk": 0.6331, "mae_stk": 1_098_686, "mp_stk": 21.45,
    "meta_type": "ridge",
    "segment_by_borough": {
        "Manhattan":    {"n": 5411,  "medape": 43.00},
        "Bronx":        {"n": 2735,  "medape": 22.70},
        "Brooklyn":     {"n": 7032,  "medape": 21.71},
        "Queens":       {"n": 9533,  "medape": 17.87},
        "Staten Island":{"n": 3052,  "medape": 15.13},
    }
}

print("="*60); print("  THAMAN v7 — meta.json repair"); print("="*60)

def smooth_target_encode_pl(df_src, df_apply, key_col, target_col, alias, gm, k=SMOOTH_K):
    stats = (df_src.group_by(key_col)
             .agg([pl.col(target_col).mean().alias("_mu"), pl.col(target_col).count().alias("_n")])
             .with_columns(((pl.col("_n")*pl.col("_mu")+k*gm)/(pl.col("_n")+k)).alias(alias))
             .select([key_col, alias]))
    df_apply = df_apply.join(stats, on=key_col, how="left").with_columns(pl.col(alias).fill_null(gm))
    return df_apply, {r[key_col]: r[alias] for r in stats.iter_rows(named=True)}

_FLOAT_OVERRIDES = {
    "dist_waterfront_m": pl.Float64, "dist_bike_lane_m": pl.Float64,
    "school_district": pl.Float64, "district_avg_score": pl.Float64,
    "district_school_count": pl.Float64, "prior_sale_price": pl.Float64,
    "years_since_prior_sale": pl.Float64, "price_appreciation": pl.Float64,
    "is_flip": pl.Float64, "zip_code": pl.Float64,
}
print("\n[1] Loading features_v4.csv …")
df = (pl.read_csv(os.path.join(PROC,"features_v4.csv"), schema_overrides=_FLOAT_OVERRIDES)
      .with_columns(pl.col("sale_date").str.to_datetime(format=None,strict=False))
      .drop_nulls(subset=["sale_date","sale_price","latitude","longitude"]))
print(f"  Rows: {len(df):,}")

_PLUTO_PATH = os.path.join(BASE,"data","raw","nyc_pluto_25v4_csv","pluto_25v4.csv")
if os.path.exists(_PLUTO_PATH):
    print("  Joining PLUTO …")
    _pluto = (pl.read_csv(_PLUTO_PATH,columns=["bbl","assesstot","assessland"])
              .with_columns(pl.col("bbl").cast(pl.Float64,strict=False))
              .drop_nulls(subset=["bbl"])
              .with_columns(pl.col("bbl").cast(pl.Int64).alias("_bbl_int")).drop("bbl")
              .rename({"assesstot":"_at","assessland":"_al"}))
    df = (df.with_columns(pl.col("bbl").cast(pl.Float64,strict=False).fill_null(0).cast(pl.Int64).alias("_bbl_int"))
          .join(_pluto,on="_bbl_int",how="left").drop("_bbl_int"))
    for c,p in [("assesstot","_at"),("assessland","_al")]:
        if c not in df.columns: df=df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
        df=df.with_columns(pl.coalesce([pl.col(c),pl.col(p)]).alias(c)).drop(p)
else:
    for c in ["assesstot","assessland"]:
        if c not in df.columns: df=df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

print("\n[2] Feature engineering …")
dist_cols=[c for c in df.columns if c.startswith("dist_")]
df=df.with_columns([pl.col(c).cast(pl.Float64,strict=False).alias(c) for c in dist_cols])
df=df.with_columns([pl.col(c).clip(lower_bound=0).log1p().alias(f"log_{c}") for c in dist_cols])
GRAVITY={"midtown_manhattan":(40.7549,-73.9840),"downtown_manhattan":(40.7074,-74.0113),
          "downtown_brooklyn":(40.6928,-73.9903),"long_island_city":(40.7447,-73.9485)}
df=df.with_columns([(((pl.col("latitude")-clat)**2+(pl.col("longitude")-clon)**2)**0.5*111_000).alias(f"dist_{nm}_m")
                    for nm,(clat,clon) in GRAVITY.items()])
df=df.with_columns([(pl.col("borough")==1).cast(pl.Int32).alias("is_manhattan"),
                    (pl.col("crime_rate_nta")*(pl.col("borough")==1).cast(pl.Int32)).alias("crime_x_manhattan"),
                    (pl.col("crime_rate_nta")*(1-(pl.col("borough")==1).cast(pl.Int32))).alias("crime_x_non_manhattan")])
eps=1e-6
walk_cols_order=["transit","bus","amenities","bike","park"]
walk_comps_np=np.column_stack([1.0/df["dist_subway_m"].clip(lower_bound=eps).to_numpy(),
                               1.0/df["dist_bus_m"].clip(lower_bound=eps).to_numpy(),
                               df["poi_count_500m"].to_numpy(),
                               1.0/df["dist_bike_lane_m"].clip(lower_bound=eps).to_numpy(),
                               1.0/df["dist_park_m"].clip(lower_bound=eps).to_numpy()])
ws_scaler=MinMaxScaler(); walk_normed_np=ws_scaler.fit_transform(walk_comps_np)
walk_score_np=np.clip(walk_normed_np@np.array([0.35,0.15,0.30,0.10,0.10])*100,0,100)
df=df.with_columns(pl.Series("walk_score_proxy",walk_score_np))
walk_score_scaler_params={col:{"data_min":float(ws_scaler.data_min_[i]),"data_max":float(ws_scaler.data_max_[i]),"scale":float(ws_scaler.scale_[i])} for i,col in enumerate(walk_cols_order)}
df=df.with_columns([(pl.col("sale_month")*(2*np.pi/12)).sin().alias("sale_month_sin"),(pl.col("sale_month")*(2*np.pi/12)).cos().alias("sale_month_cos")])
df=df.with_columns([(pl.col("gross_square_feet")/pl.col("numfloors").clip(lower_bound=1)).alias("sqft_per_floor"),
                    (pl.col("median_income_nta")/(pl.col("crime_rate_nta")+1.0)).alias("income_over_crime"),
                    (pl.col("residential_units")/pl.col("gross_square_feet").clip(lower_bound=1)*1000.0).alias("density_index"),
                    ((pl.col("gross_square_feet").clip(lower_bound=1).log1p())*(pl.col("numfloors").clip(lower_bound=1).log1p())).alias("log_sqft_x_floors")])
df=df.with_columns([pl.col("land_square_feet").clip(lower_bound=1).log1p().alias("log_land_sqft"),
                    (pl.col("gross_square_feet")/pl.col("land_square_feet").clip(lower_bound=1)).clip(upper_bound=10).alias("lot_coverage"),
                    (pl.col("gross_square_feet")*pl.col("numfloors").clip(lower_bound=1)).log1p().alias("bldg_vol_proxy"),
                    (pl.col("prior_sale_price")/pl.col("gross_square_feet").clip(lower_bound=1)).alias("prior_price_psf")])
if "zip_code" in df.columns:
    df=df.with_columns(pl.col("zip_code").cast(pl.Int64,strict=False).cast(pl.Utf8).fill_null("00000").alias("zip_str"))

df_sorted=df.sort("sale_date"); n_hold=int(len(df_sorted)*0.15)
df_work=df_sorted[:-n_hold]; df_hold=df_sorted[-n_hold:]

# Impute
_has_both=df_work.filter((pl.col("prior_sale_price")>0)&pl.col("assesstot").is_not_null()&(pl.col("assesstot")>0)).with_columns((pl.col("prior_sale_price")/pl.col("assesstot")).alias("_pr"))
_ratio_df=_has_both.group_by("borough").agg(pl.col("_pr").median().alias("ratio"))
_ratio_map={int(r["borough"]):float(r["ratio"]) for r in _ratio_df.iter_rows(named=True)}
_glob_r=float(_has_both["_pr"].median()) if len(_has_both) else 10.0
_rl=pl.DataFrame({"borough":list(_ratio_map.keys()),"_ir":list(_ratio_map.values())}).with_columns(pl.col("borough").cast(df_work.schema["borough"]))
def _impute(fr):
    fr=fr.join(_rl,on="borough",how="left").with_columns(pl.col("_ir").fill_null(_glob_r))
    return fr.with_columns(pl.when((pl.col("prior_sale_price").is_null()|(pl.col("prior_sale_price")==0))&pl.col("assesstot").is_not_null()&(pl.col("assesstot")>0)).then(pl.col("assesstot")*pl.col("_ir")).otherwise(pl.col("prior_sale_price")).alias("prior_sale_price")).drop("_ir")
df_work=_impute(df_work); df_hold=_impute(df_hold)

print("\n[3] Computing encoding maps (Bayesian smoothed k=30) …")
LOG_TARGET="log_price"
df_work=df_work.with_columns(pl.col("sale_price").log1p().alias(LOG_TARGET))
global_mean_log=float(df_work[LOG_TARGET].mean())

df_work, bm = smooth_target_encode_pl(df_work,df_work,"bldgclass",LOG_TARGET,"bldgclass_encoded",global_mean_log)
df_work=df_work.with_columns((pl.col("borough").cast(pl.Utf8)+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_bbk"))
df_work, bb = smooth_target_encode_pl(df_work,df_work,"_bbk",LOG_TARGET,"borough_bldg_encoded",global_mean_log)
df_work=df_work.drop("_bbk")

nta_map_save={}; nta_features=[]; zip_map_save={}; zip_features=[]
if "ntacode" in df_work.columns:
    df_work, nta_map = smooth_target_encode_pl(df_work,df_work,"ntacode",LOG_TARGET,"nta_encoded",global_mean_log)
    nta_map_save={k:round(float(v),6) for k,v in nta_map.items()}
    df_work=df_work.with_columns((pl.col("ntacode")+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_ntab"))
    df_work, _ = smooth_target_encode_pl(df_work,df_work,"_ntab",LOG_TARGET,"nta_bldg_encoded",global_mean_log)
    df_work=df_work.drop("_ntab")
    nta_stats=(df_work.with_columns((pl.col("sale_price")/pl.col("gross_square_feet").clip(lower_bound=1)).alias("_psf"))
               .group_by("ntacode").agg([pl.count("sale_price").alias("nta_sale_count"),pl.col("_psf").median().alias("nta_median_psf")]))
    df_work=df_work.join(nta_stats,on="ntacode",how="left")
    for c in ["nta_sale_count","nta_median_psf"]:
        med=float(df_work[c].drop_nulls().median() or 0)
        df_work=df_work.with_columns(pl.col(c).fill_null(med))
    nta_trend=(df_work.group_by(["ntacode","sale_year"]).agg(pl.col(LOG_TARGET).mean().alias("_yr_price")).sort("sale_year"))
    nta_slopes={}
    for ntac in nta_trend["ntacode"].unique().to_list():
        sub=nta_trend.filter(pl.col("ntacode")==ntac).sort("sale_year")
        if len(sub)<2: continue
        yrs=sub["sale_year"].to_numpy().astype(float); prs=sub["_yr_price"].to_numpy().astype(float)
        yrs-=yrs.mean()
        nta_slopes[ntac]=round(float(np.dot(yrs,prs)/(np.dot(yrs,yrs)+1e-12)),6)
    slope_df=pl.DataFrame({"ntacode":list(nta_slopes.keys()),"nta_price_trend":list(nta_slopes.values())})
    global_slope=float(np.median(list(nta_slopes.values()))) if nta_slopes else 0.0
    df_work=df_work.join(slope_df,on="ntacode",how="left").with_columns(pl.col("nta_price_trend").fill_null(global_slope))
    nta_features=["nta_encoded","nta_bldg_encoded","nta_sale_count","nta_median_psf","nta_price_trend"]
    print(f"  NTA codes: {len(nta_map_save)}")

if "zip_str" in df_work.columns:
    df_work, zip_map = smooth_target_encode_pl(df_work,df_work,"zip_str",LOG_TARGET,"zip_encoded",global_mean_log)
    zip_map_save={k:round(float(v),6) for k,v in zip_map.items()}
    df_work=df_work.with_columns((pl.col("zip_str")+"_"+pl.col("bldgclass").str.slice(0,1)).alias("_zbk"))
    df_work, _ = smooth_target_encode_pl(df_work,df_work,"_zbk",LOG_TARGET,"zip_bldg_encoded",global_mean_log)
    df_work=df_work.drop("_zbk")
    zip_features=["zip_encoded","zip_bldg_encoded"]
    print(f"  ZIP codes: {len(zip_map_save)}")

if "nta_encoded" in df_work.columns:
    df_work=df_work.with_columns([(pl.col("bldgclass_encoded")/pl.col("nta_encoded").clip(lower_bound=0.1)).alias("nta_rel_price"),
                                   (pl.col("gross_square_feet").clip(lower_bound=1).log1p()*pl.col("nta_encoded")).alias("sqft_x_nta_enc"),
                                   (pl.col("building_age")*pl.col("nta_encoded")).alias("bldg_age_x_nta")])
v7_interact=["nta_rel_price","sqft_x_nta_enc","bldg_age_x_nta"] if "nta_encoded" in df_work.columns else []

V4_BASE=["latitude","longitude","borough","building_age","numfloors","gross_square_feet","land_square_feet","residential_units","dist_subway_m","dist_school_m","dist_park_m","dist_hospital_m","poi_count_500m","crime_rate_nta","noise_density_nta","population_2020","median_income_nta","dist_bus_m","renovated_since_2018","years_since_renovation","dist_waterfront_m","dist_bike_lane_m","dist_elem_school_m","dist_express_subway_m","nearest_station_is_express","livability_complaint_rate","borough_income_deviation","sale_year","sale_month_sin","sale_month_cos","mortgage_rate_30yr","builtfar","residfar","commfar","facilfar","far_utilization","has_elevator","is_condo","is_multifamily","is_single_fam","is_mixed_use","airbnb_count_500m","prior_sale_price","price_appreciation","years_since_prior_sale","is_flip","school_district","district_avg_score","district_school_count","has_prior_sale","assesstot","assessland","poi_cafe_500m","poi_restaurant_500m","poi_gym_500m","poi_grocery_500m","poi_bar_500m","poi_pharmacy_500m",*[f"log_{c}" for c in dist_cols],"dist_midtown_manhattan_m","dist_downtown_manhattan_m","dist_downtown_brooklyn_m","dist_long_island_city_m","is_manhattan","crime_x_manhattan","crime_x_non_manhattan","walk_score_proxy","bldgclass_encoded","borough_bldg_encoded","tree_count_200m","pm25_mean","no2_mean","hpd_viol_rate_nta"]
V5_FEATS=["sqft_per_floor","income_over_crime","density_index","log_sqft_x_floors"]
V6_FEATS=["log_land_sqft","lot_coverage","bldg_vol_proxy","prior_price_psf"]+nta_features
V7_FEATS=zip_features+v7_interact
FEATURE_NAMES=[f for f in (V4_BASE+V5_FEATS+V6_FEATS+V7_FEATS) if f in df_work.columns]
print(f"  Feature count: {len(FEATURE_NAMES)}")

acris_cols=["prior_sale_price","price_appreciation","years_since_prior_sale"]
acris_medians={c:float(df_work.filter(pl.col(c).is_not_null()&(pl.col(c)!=0))[c].median() or 0) for c in acris_cols}
qol_cols=["crime_rate_nta","noise_density_nta","livability_complaint_rate"]
winsorize_p99={c:float(np.percentile(df_work[c].drop_nulls().to_numpy(),99)) for c in qol_cols}

print("\n[4] Loading stack pkl …")
stack=joblib.load(os.path.join(MODEL_DIR,"thaman_stack.pkl"))
print(f"  version={stack.get('version')}  meta_type={stack.get('meta_type')}  "
      f"n_features={stack['xgb_a'].n_features_in_}")

print("\n[5] Writing meta.json …")
# Start from git version (last known good base)
import subprocess
result=subprocess.run(["git","show","HEAD:models/meta.json"],capture_output=True,text=True,cwd=BASE)
meta=json.loads(result.stdout)

meta.update({
    "feature_names":      FEATURE_NAMES,
    "n_features":         len(FEATURE_NAMES),
    "n_train":            len(df_work),
    "n_holdout":          len(df_hold),
    "walk_score_scaler":  walk_score_scaler_params,
    "bldgclass_means":    {k:round(float(v),6) for k,v in bm.items()},
    "borough_bldg_means": {k:round(float(v),6) for k,v in bb.items()},
    "nta_means":          nta_map_save,
    "zip_means":          zip_map_save,
    "global_mean_log":    round(global_mean_log,6),
    "luxury_threshold":   2_500_000,
    "has_luxury_model":   True,
    "acris_medians":      {k:round(float(v),6) for k,v in acris_medians.items()},
    "winsorize_p99":      {k:round(float(v),6) for k,v in winsorize_p99.items()},
    "segment_by_borough": {k:{"n":v["n"],"medape":v["medape"]} for k,v in STATS["segment_by_borough"].items()},
    "stack": {
        "version":         "v7",
        "base_learners":   ["xgb_a","xgb_b","lightgbm","catboost"],
        "meta_learner":    STATS["meta_type"],
        "r2_holdout":      round(STATS["r2_stk"],4),
        "mae_holdout":     round(STATS["mae_stk"],0),
        "medape_holdout":  round(STATS["mp_stk"],2),
        "xgb_a":    {"r2_holdout":STATS["r2_xa"],"medape_holdout":STATS["mp_xa"]},
        "xgb_b":    {"r2_holdout":STATS["r2_xb"],"medape_holdout":STATS["mp_xb"]},
        "lightgbm": {"r2_holdout":STATS["r2_lg"],"medape_holdout":STATS["mp_lg"]},
        "catboost": {"r2_holdout":STATS["r2_ct"],"medape_holdout":STATS["mp_ct"]},
        "r2_improvement":     round(STATS["r2_stk"]-0.6454,4),
        "medape_improvement": round(20.19-STATS["mp_stk"],2),
    },
})
meta_path=os.path.join(MODEL_DIR,"meta.json")
with open(meta_path,"w") as f: json.dump(meta,f,indent=2)
print("  meta.json written ✓")
print(f"\n  Stack v7: R²={STATS['r2_stk']:.4f}  MedAPE={STATS['mp_stk']:.2f}%  ({len(FEATURE_NAMES)} features)")
print("  NOTE: v7 is WORSE than v6 (R²=0.6454, MedAPE=20.19%) — Bayesian smoothing degraded top features.")
print("  Recommend retraining v8 with raw encoding + ZIP + NTA trend (no interactions).\n")
