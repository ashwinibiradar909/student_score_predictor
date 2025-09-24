# src/check_predictions.py
import pandas as pd

PATH = "data/processed/ann_predictions.csv"
df = pd.read_csv(PATH)

print("File:", PATH)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 8 rows:")
print(df.head(8).to_string(index=False))

print("\n\nBasic stats for actual vs predicted:")
print(df[["G3","ann_pred","ann_error"]].describe().to_string())

# show top 5 worst predictions by absolute error
df["abs_error"] = df["ann_error"].abs()
print("\nTop 5 largest absolute errors:")
print(df.sort_values("abs_error", ascending=False).head(5).to_string(index=False))
