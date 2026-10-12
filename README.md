# Support Ticket Classification API

A local NLP application that suggests customer-support categories
using TF-IDF and Logistic Regression, with a FastAPI backend and
a simple browser interface.

No Ollama, LLM or paid API is required.

## Features
- Dataset inspection and normalized-text deduplication
- Naive Bayes and Logistic Regression comparison
- Validation-based model selection
- Held-out test evaluation and error export
- Saved TF-IDF and classifier pipeline
- Command-line predictions
- FastAPI endpoint returning three ranked categories
- Browser interface with human-review warnings
- 10 automated API and homepage tests

## Technologies
Python, Pandas, Scikit-learn, Joblib, FastAPI, Uvicorn,
HTTPX2, HTML, CSS and JavaScript.

## Dataset

Source:
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

License: CDLA-Sharing-1.0.
See the source for licensing terms and attribution information.
This repository does not redistribute the dataset or trained model.

Download:
Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv

Rename it to tickets.csv and place it in data/.

The downloaded CSV contains 26,872 hybrid synthetic examples.
Only instruction is used as input and category as the target.
Intent and response are excluded from model inputs.

The downloaded version contains 11 categories:
ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE,
ORDER, PAYMENT, REFUND, SHIPPING and SUBSCRIPTION.

## Data preparation
- Normalize case and whitespace for duplicate checking.
- Check for conflicting labels on identical normalized messages.
- No conflicting labels were found in this run.
- Remove 2,598 normalized duplicate rows.
- Retain 24,274 messages.
- Use stratified 60/20/20 training, validation and test splits
  with random_state=42.
- Verify no identical normalized text crosses splits.
- Fit TF-IDF only on training data through a pipeline.

Similar synthetic templates may still occur across splits.

## Results

| Model | Validation macro F1 |
|---|---:|
| TF-IDF + Naive Bayes | 0.9953 |
| TF-IDF + Logistic Regression | 0.9962 |

Logistic Regression was selected using validation macro F1.

Final held-out test:
- 4,855 messages
- Macro F1: 0.9976
- Accuracy: approximately 0.9975
- Misclassified messages: 12

Per-category metrics are in reports/test_metrics.csv.

These are synthetic-dataset benchmark results, not measured
real-world ticket-routing performance.

## Known failure examples

An independently written message:
"My card was charged twice for the same purchase."

was incorrectly classified as DELIVERY with a top score of 0.1583.

The unrelated message:
"What is the weather tomorrow?"

also received DELIVERY, with a top score of 0.2659.

These exploratory examples are not an independent accuracy benchmark.
They demonstrate domain-shift and out-of-scope limitations.

## Setup

Run from the project root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Download the dataset as described above, then run:

    python src/check_data.py
    python src/prepare_data.py
    python src/train_model.py

Training saves models/ticket_classifier.joblib.
The dataset and model are excluded from Git.

The API and automated tests require this model, so complete training
before starting the server or running tests.

## Command-line prediction

    python src/predict.py "I need a copy of my invoice."

## Run the application

    python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Interface: http://127.0.0.1:8000/
API documentation: http://127.0.0.1:8000/docs

## Endpoints
- GET /: browser interface
- GET /health: service health
- POST /predict: suggested category and top three model scores

Example request:

    {"text": "I need a copy of my invoice."}

Input must be a nonblank string of at most 5,000 characters.
Invalid input returns HTTP 422.

Predictions always require human review.
Scores are not calibrated confidence.

## Tests

    python -m unittest discover -s tests -v

The current suite has 10 passing API and homepage tests.
It does not measure real-world classification accuracy or
automatically test browser JavaScript interactions.

## Limitations and safety
- Synthetic templates may inflate random-split benchmark scores.
- New phrasing, typos and joined words can cause errors.
- There is no unknown category or validated abstention threshold.
- Unrelated text is forced into a known category.
- Do not use predictions for unattended routing.
- Only load trusted Joblib files; loading can execute code.
- Intended for local development, not public production deployment.
- Public deployment would require authentication, request limits,
  monitoring and further evaluation.
- Re-running training also re-runs evaluation. Do not tune settings
  based on the final test results.

## Project status
Working local educational prototype.