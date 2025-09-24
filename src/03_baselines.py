# src/03_baselines.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED = np.load(PROCESSED_DIR / "train.npz")
X_train = PROCESSED["X"]
y_train = PROCESSED["y"]
TEST = np.load(PROCESSED_DIR / "test.npz")
X_test = TEST["X"]
y_test = TEST["y"]

def eval_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # compute RMSE without using 'squared' kwarg for sklearn compatibility
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n{name} results:")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE:  {mae:.4f}")
    print(f" R2:   {r2:.4f}")
    return preds, {"rmse": rmse, "mae": mae, "r2": r2}

if __name__ == "__main__":
    # Linear Regression
    lr = LinearRegression()
    lr_preds, lr_metrics = eval_model("LinearRegression", lr, X_train, y_train, X_test, y_test)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_preds, rf_metrics = eval_model("RandomForestRegressor", rf, X_train, y_train, X_test, y_test)

    # Save baseline predictions (with original test rows)
    raw_df = pd.read_csv(ROOT / "data" / "raw" / "student-mat.csv", sep=';')
    X = raw_df.drop(columns=['G3'])
    y = raw_df['G3'].values.astype(float)
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=42)
    out_df = X_test_df.copy().reset_index(drop=True)
    out_df["G3"] = y_test_df
    out_df["lr_pred"] = lr_preds
    out_df["rf_pred"] = rf_preds
    out_df["rf_error"] = out_df["rf_pred"] - out_df["G3"]
    out_df.to_csv(ROOT / "data" / "processed" / "baselines_predictions.csv", index=False)
    print("\nSaved baseline predictions to data/processed/baselines_predictions.csv")
