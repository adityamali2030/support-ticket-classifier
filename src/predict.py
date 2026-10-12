from pathlib import Path
import argparse
import json

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"

parser = argparse.ArgumentParser(
    description="Classify a customer-support message."
)
parser.add_argument("text", help="Support message to classify")
args = parser.parse_args()

text = args.text.strip()

if not text:
    parser.error("Message must not be blank.")

if len(text) > 5000:
    parser.error("Message must not exceed 5,000 characters.")

# Load only the trusted model created by this project.
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]

scores = pipeline.predict_proba([text])[0]
ranked_indices = scores.argsort()[::-1][:3]

result = {
    "predicted_category": str(
        pipeline.classes_[ranked_indices[0]]
    ),
    "top_categories": [
        {
            "category": str(pipeline.classes_[index]),
            "score": round(float(scores[index]), 4),
        }
        for index in ranked_indices
    ],
    "note": (
        "Scores are model estimates, not calibrated confidence. "
        "The model always selects a known category, even for "
        "unrelated messages."
    ),
}

print(json.dumps(result, indent=2))