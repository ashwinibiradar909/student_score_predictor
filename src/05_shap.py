# src/05_shap.py
import sys
from pathlib import Path

# Add project root and src/ to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))





# src/05_shap.py
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
import torch
from src.model import StudentNet

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# load processed arrays and feature names
train_np = np.load(PROCESSED_DIR / "train.npz")
test_np = np.load(PROCESSED_DIR / "test.npz")
X_train = train_np["X"]
X_test = test_np["X"]
feature_names = pd.read_csv(PROCESSED_DIR / "feature_names.csv", header=None)[0].tolist()

# load model
INPUT_DIM = X_train.shape[1]
model = StudentNet(INPUT_DIM)
model.load_state_dict(torch.load(MODELS_DIR / "student_state_dict.pt", map_location="cpu"))
model.eval()

# wrapper prediction function for SHAP (expects 2D numpy -> returns 1D preds)
def predict_fn(x_np):
    with torch.no_grad():
        x_t = torch.from_numpy(x_np.astype(np.float32))
        preds = model(x_t).numpy().reshape(-1)
    return preds

# pick background & sample sizes
bg_size = min(100, X_train.shape[0])
sample_size = min(200, X_test.shape[0])

rng = np.random.default_rng(42)
background = X_train[rng.choice(X_train.shape[0], bg_size, replace=False)]
X_sample = X_test[rng.choice(X_test.shape[0], sample_size, replace=False)]

print("Running SHAP KernelExplainer (can be slow). Background size:", background.shape[0], "Sample size:", X_sample.shape[0])
explainer = shap.KernelExplainer(predict_fn, background, link="identity")
shap_values = explainer.shap_values(X_sample, nsamples=100)

# Save wide-format SHAP values (one column per feature)
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.insert(0, "sample_index", range(len(shap_df)))
shap_df.to_csv(PROCESSED_DIR / "shap_values_wide.csv", index=False)
print("Saved:", PROCESSED_DIR / "shap_values_wide.csv")

# Save summary (mean absolute SHAP per feature)
shap_summary = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
shap_summary.to_csv(PROCESSED_DIR / "shap_summary.csv", index=False)
print("Saved:", PROCESSED_DIR / "shap_summary.csv")
print("Top features:\n", shap_summary.head(10).to_string(index=False))
