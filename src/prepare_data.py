from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "tickets_clean.csv"

df = pd.read_csv(DATA_PATH)[["instruction", "category"]].copy()

# Normalize case and whitespace for duplicate checking.
# Keep punctuation and spelling: these can matter in real messages.
df["text_key"] = (
    df["instruction"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["category"] = df["category"].str.strip()

# Exclude missing or blank inputs and labels.
valid_rows = (
    df["text_key"].notna()
    & df["text_key"].ne("")
    & df["category"].notna()
    & df["category"].ne("")
)
df = df.loc[valid_rows].copy()

# Find identical normalized text assigned to different categories.
label_counts = df.groupby("text_key")["category"].nunique()
conflicting_keys = label_counts[label_counts > 1].index

print("Conflicting normalized messages:", len(conflicting_keys))

# Exclude conflicts rather than guessing their correct label.
conflict_mask = df["text_key"].isin(conflicting_keys)
print("Rows excluded due to conflicting labels:", conflict_mask.sum())
df = df.loc[~conflict_mask].copy()

rows_before = len(df)
df = df.drop_duplicates(subset="text_key").reset_index(drop=True)

print("Duplicate rows removed:", rows_before - len(df))
print("Clean dataset shape:", df.shape)

print("\nClean category counts:")
print(df["category"].value_counts())

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")