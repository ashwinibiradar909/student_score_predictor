# app/main.py
"""
Flask inference app for Student Score Predictor.

This file ensures the project root and src/ are on sys.path so `from src.model import StudentNet`
works whether you run `python app/main.py` or `python -m app.main`.
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, Any
from flask import render_template

# Ensure project root and src/ are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now safe imports
from flask import Flask, request, jsonify
import joblib
import torch
import pandas as pd
import numpy as np

# Try to import the model class from src.model; give a helpful error if not possible
try:
    from src.model import StudentNet
except Exception as e:
    # Print details and re-raise a clear error
    print("Error importing StudentNet from src.model.")
    print("Current working directory:", Path.cwd())
    print("sys.path (first entries):", sys.path[:5])
    print("Files in project root:", [p.name for p in ROOT.iterdir()])
    print("Files in src/:", [p.name for p in SRC_DIR.iterdir()] if SRC_DIR.exists() else "src/ missing")
    traceback.print_exc()
    raise e

# Paths and config
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_FULL = PROCESSED_DIR / "raw_full.csv"   # created in preprocessing
PREPROC_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_STATE_PATH = MODELS_DIR / "student_state_dict.pt"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names.csv"

# Validate existence of required files
missing = []
for p in (PREPROC_PATH, MODEL_STATE_PATH, RAW_FULL, FEATURE_NAMES_PATH):
    if not p.exists():
        missing.append(str(p.relative_to(ROOT)))
if missing:
    msg = "Missing required files: " + ", ".join(missing)
    raise FileNotFoundError(msg + ". Make sure you ran preprocessing and training scripts.")

# Load preprocessor, feature names, raw df
preprocessor = joblib.load(PREPROC_PATH)
feat_names = pd.read_csv(FEATURE_NAMES_PATH, header=None)[0].tolist()
raw_df = pd.read_csv(RAW_FULL)

# Determine input dim and load model
INPUT_DIM = len(feat_names)
device = "cpu"
model = StudentNet(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device))
model.to(device)
model.eval()

# Feature columns expected in input JSON (original raw feature order)
feature_cols = [c for c in raw_df.columns.tolist() if c != "G3"]



app = Flask(__name__)

@app.route("/home/", methods=["GET"])
def home():
    # renders the simple HTML form
    return render_template("index.html")




def preprocess_single(features: Dict[str, Any]) -> np.ndarray:
    # Build single-row DataFrame with same columns as raw training X
    row = {col: features.get(col, None) for col in feature_cols}
    df_row = pd.DataFrame([row], columns=feature_cols)
    X_proc = preprocessor.transform(df_row)
    return X_proc.astype(np.float32)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "input_dim": INPUT_DIM, "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload or "features" not in payload:
        return jsonify({"error": "Send JSON with key 'features' mapping to a feature dict."}), 400
    try:
        X_proc = preprocess_single(payload["features"])
        x_tensor = torch.from_numpy(X_proc)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy().flatten()[0]
        return jsonify({"predicted_score": float(pred)})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    payload = request.get_json()
    if not payload or "rows" not in payload:
        return jsonify({"error": "Send JSON with key 'rows' (list of feature dicts)."}), 400
    try:
        rows = payload["rows"]
        df_rows = pd.DataFrame(rows, columns=feature_cols)
        X_proc = preprocessor.transform(df_rows).astype(np.float32)
        x_tensor = torch.from_numpy(X_proc)
        with torch.no_grad():
            preds = model(x_tensor).cpu().numpy().flatten().tolist()
        out = []
        for i, row in enumerate(rows):
            item = {"predicted_score": float(preds[i])}
            if "G3" in row:
                item["error"] = float(preds[i]) - float(row["G3"])
            out.append(item)
        return jsonify({"predictions": out})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    print("Starting Flask app. model_loaded:", MODEL_STATE_PATH.exists(), "preprocessor:", PREPROC_PATH.exists())
    app.run(host="0.0.0.0", port=5000, debug=True)
