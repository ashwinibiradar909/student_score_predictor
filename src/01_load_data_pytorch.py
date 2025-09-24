import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "student-mat.csv"

df = pd.read_csv(DATA_PATH, sep=';')   # UCI CSV uses semicolon
print("rows, cols:", df.shape)
print(df.head().to_string())
