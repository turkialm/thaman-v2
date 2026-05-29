"""
Download large model/data files from HuggingFace Hub at startup.
Run automatically by app.py before uvicorn starts.
"""
import os
from pathlib import Path

REPO_ID = "Turki-Almurahhem/thaman-models"

FILES = [
    "models/thaman_stack.pkl",
    "models/riyadh_stack.pkl",
    "models/xgboost_model.json",
    "models/luxury_model.json",
    "models/meta.json",
    "models/riyadh_meta.json",
    "data/raw/nyc_coastline_pts.npy",
    "data/raw/nyc_bike_lanes.geojson",
]


def download_if_missing():
    from huggingface_hub import hf_hub_download
    for path in FILES:
        if os.path.exists(path):
            print(f"  [hub] exists: {path}")
            continue
        print(f"  [hub] downloading: {path} ...", flush=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type="model",
            local_dir=".",
        )
        print(f"  [hub] ready: {path}")


if __name__ == "__main__":
    download_if_missing()
