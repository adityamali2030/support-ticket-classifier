from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tickets_clean.csv"

df = pd.read_csv(DATA_PATH)

# Only message text is used as input.
X = df["instruction"]
y = df["category"]

# Reserve 20% for final testing.
train_val, test = train_test_split(
    df,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Reserve another 20% of the full dataset for validation.
train, validation = train_test_split(
    train_val,
    test_size=0.25,
    stratify=train_val["category"],
    random_state=42,
)

# Verify normalized messages do not overlap between sets.
train_keys = set(train["text_key"])
validation_keys = set(validation["text_key"])
test_keys = set(test["text_key"])

assert train_keys.isdisjoint(validation_keys)
assert train_keys.isdisjoint(test_keys)
assert validation_keys.isdisjoint(test_keys)

X_train = train["instruction"]
y_train = train["category"]

X_val = validation["instruction"]
y_val = validation["category"]

X_test = test["instruction"]
y_test = test["category"]

for name, split in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(f"\n{name}: {len(split)} messages")
    print(split["category"].value_counts().sort_index())

print("\nNo normalized-text overlap between splits.")


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score

nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )),
    ("classifier", MultinomialNB(alpha=1.0)),
])

print("\nTraining TF-IDF + Naive Bayes...")
nb_model.fit(X_train, y_train)

nb_val_predictions = nb_model.predict(X_val)

print("\nNaive Bayes validation report:")
print(classification_report(
    y_val,
    nb_val_predictions,
    digits=4,
    zero_division=0,
))

nb_macro_f1 = f1_score(
    y_val,
    nb_val_predictions,
    average="macro",
    zero_division=0,
)

print(f"Validation macro F1: {nb_macro_f1:.4f}")

from sklearn.linear_model import LogisticRegression

lr_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nTraining TF-IDF + Logistic Regression...")
lr_model.fit(X_train, y_train)

lr_val_predictions = lr_model.predict(X_val)

print("\nLogistic Regression validation report:")
print(classification_report(
    y_val,
    lr_val_predictions,
    digits=4,
    zero_division=0,
))

lr_macro_f1 = f1_score(
    y_val,
    lr_val_predictions,
    average="macro",
    zero_division=0,
)

print(f"Validation macro F1: {lr_macro_f1:.4f}")

print("\nValidation model comparison:")
print(f"Naive Bayes macro F1:         {nb_macro_f1:.4f}")
print(f"Logistic Regression macro F1: {lr_macro_f1:.4f}")

# Selected using validation macro F1 only.
final_model = lr_model

test_predictions = final_model.predict(X_test)

print("\nFINAL TEST RESULTS — Logistic Regression")
print(classification_report(
    y_test,
    test_predictions,
    digits=4,
    zero_division=0,
))

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0,
)

print(f"Test macro F1: {test_macro_f1:.4f}")

# Save per-category metrics and errors for inspection.
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

test_report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(test_report).transpose().to_csv(
    REPORT_DIR / "test_metrics.csv",
    index_label="label",
)

test_results = pd.DataFrame({
    "text": X_test.to_numpy(),
    "actual_category": y_test.to_numpy(),
    "predicted_category": test_predictions,
})

errors = test_results.loc[
    test_results["actual_category"]
    != test_results["predicted_category"]
]

# Keep dataset-derived messages in the ignored data folder.
errors.to_csv(
    PROJECT_ROOT / "data" / "test_errors.csv",
    index=False,
)

print(f"\nMisclassified test messages: {len(errors)}")
print("Metrics saved to reports/test_metrics.csv")
print("Errors saved to data/test_errors.csv")

import joblib

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_bundle = {
    "pipeline": final_model,
    "categories": final_model.classes_.tolist(),
    "model_name": "TF-IDF + Logistic Regression",
}

MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"
joblib.dump(model_bundle, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

print("\nFirst five test errors:")
print(errors.head(5).to_string(index=False))