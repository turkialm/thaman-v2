"""
HuggingFace Space entrypoint.
Downloads large model files from Hub, then starts the FastAPI server.
"""
import subprocess
import sys

print("=== THAMAN Space Startup ===")
print("[1/2] Downloading model files from HF Hub...")
import download_models
download_models.download_if_missing()

print("[2/2] Starting API server...")
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "api.main:app",
    "--host", "0.0.0.0",
    "--port", "7860",
], check=True)
