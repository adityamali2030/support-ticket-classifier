from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load our trusted model once when the server starts.
    bundle = joblib.load(MODEL_PATH)
    app.state.pipeline = bundle["pipeline"]
    yield


app = FastAPI(
    title="Support Ticket Classification API",
    description=(
        "Suggests ticket categories using a local ML model. "
        "Predictions require human review."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: TicketRequest):
    pipeline = app.state.pipeline
    scores = pipeline.predict_proba([request.text])[0]
    ranked_indices = scores.argsort()[::-1][:3]

    return {
        "suggested_category": str(
            pipeline.classes_[ranked_indices[0]]
        ),
        "top_categories": [
            {
                "category": str(pipeline.classes_[index]),
                "score": round(float(scores[index]), 4),
            }
            for index in ranked_indices
        ],
        "requires_human_review": True,
        "note": (
            "Scores are not calibrated confidence. "
            "Unrelated messages also receive a known category."
        ),
    }


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(
        PROJECT_ROOT / "app" / "static" / "index.html"
    )