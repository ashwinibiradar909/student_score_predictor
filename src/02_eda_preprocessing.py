# src/02_eda_preprocessing.py
"""
EDA + Preprocessing script for Student Score Predictor.

What it does:
- Loads raw UCI student CSV (student-mat.csv)
- Prints basic EDA summaries
- Detects numeric & categorical features (keeps G3 as target)
- Builds a ColumnTransformer:
    - numeric: median imputer + StandardScaler
    - categorical: most_frequent imputer + OneHotEncoder (handles sklearn version differences)
- Fits preprocessor on training set, transforms train/test
- Saves preprocessor (.joblib), processed arrays (.npz) and feature_names.csv
- Prints processed shapes and sample processed rows
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "student-mat.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) Load data ---
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=';')
print("Loaded shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# --- 2) Quick EDA prints ---
print("\n--- df.info() ---")
print(df.info())
print("\n--- df.describe() ---")
print(df.describe(include='all').transpose())

print("\nMissing value counts:")
print(df.isna().sum())

# --- 3) Identify numeric vs categorical (keep G3 as target) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'G3' in numeric_cols:
    numeric_cols.remove('G3')

categorical_cols = [c for c in df.columns if c not in numeric_cols + ['G3']]

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# --- 4) Build the preprocessing pipelines ---
# Create a OneHotEncoder that works across sklearn versions
try:
    # newer sklearn: use sparse_output
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    # older sklearn: use sparse
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', ohe_encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ],
    remainder='drop'
)

# --- 5) Prepare X, y and train/test split ---
X = df.drop(columns=['G3'])
y = df['G3'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")

# --- 6) Fit preprocessor and transform ---
print("\nFitting preprocessor on training data...")
preprocessor.fit(X_train)

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# --- 7) Save preprocessor ---
preproc_path = MODELS_DIR / "preprocessor.joblib"
joblib.dump(preprocessor, preproc_path)
print("Saved preprocessor to:", preproc_path)

# --- 8) Build feature names after transformation ---
feature_names = list(numeric_cols)  # start with numeric names

# Extract OHE feature names in a sklearn-version-safe way
try:
    # sklearn >= 1.0+ often supports get_feature_names_out
    ohe_step = preprocessor.named_transformers_['cat'].named_steps['onehot']
    try:
        cat_feature_names = ohe_step.get_feature_names_out(categorical_cols).tolist()
    except AttributeError:
        # older sklearn: fallback to get_feature_names
        cat_feature_names = ohe_step.get_feature_names(categorical_cols).tolist()
except Exception as e:
    # If something unexpected happens, create generic names based on columns and categories
    print("Warning: couldn't auto-extract OHE names (sklearn version issue). Falling back to generic names.", e)
    # try to get categories if available
    cat_feature_names = []
    try:
        cat_cats = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
        for col, cats in zip(categorical_cols, cat_cats):
            cat_feature_names.extend([f"{col}__{str(c)}" for c in cats])
    except Exception:
        # last fallback: placeholder names
        cat_feature_names = [f"cat_{i}" for i in range(X_train_proc.shape[1] - len(numeric_cols))]

feature_names.extend(cat_feature_names)
print("\nNumber of final features (after OHE):", len(feature_names))

# --- 9) Save processed arrays and feature names ---
np.savez_compressed(PROCESSED_DIR / "train.npz", X=X_train_proc, y=y_train)
np.savez_compressed(PROCESSED_DIR / "test.npz", X=X_test_proc, y=y_test)
pd.Series(feature_names).to_csv(PROCESSED_DIR / "feature_names.csv", index=False, header=False)

print("\nProcessed shapes:")
print(" X_train_proc:", X_train_proc.shape)
print(" X_test_proc:", X_test_proc.shape)
print(" y_train:", y_train.shape)
print(" y_test:", y_test.shape)

# --- 10) Print small sample of processed X (first 5 rows) ---
try:
    proc_df = pd.DataFrame(X_train_proc, columns=feature_names)
    print("\nSample processed X (first 5 rows):")
    print(proc_df.head().to_string())
except Exception as e:
    # If column names count doesn't match (rare), just print numeric array sample
    print("\nCould not create DataFrame with feature names (mismatch). Showing numeric values instead.")
    print("Sample processed X (first 5 rows):")
    print(X_train_proc[:5, :])
    print("Error:", e)

# --- 11) Save raw full CSV for reference (optional) ---
df.to_csv(PROCESSED_DIR / "raw_full.csv", index=False)

print("\nAll done for Step 2 — EDA & Preprocessing.")
