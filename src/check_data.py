from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nFirst three examples:")
print(df[["instruction", "category"]].head(3).to_string(index=False))

print("\nCategory counts:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df[["instruction", "category"]].isna().sum())

print("\nDuplicate input texts:")
print(df["instruction"].duplicated().sum())