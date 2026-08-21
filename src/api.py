"""FastAPI application for Customer Support Ticket Intelligence."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prediction.predictor import TicketPredictor


app = FastAPI(
    title="Customer Support Ticket Intelligence API",
    description="Multilingual ticket classification for Type, Queue, and Priority.",
    version="1.0.0",
)


# Load models once when the API starts.
predictor: Optional[TicketPredictor] = None


class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ...,
        min_length=1,
        description="Customer support ticket text.",
    )

    language: str = "en"
    business_type: str = "IT_Services"


    tag_1: str = ""
    tag_2: str = ""
    tag_3: str = ""
    tag_4: str = ""
    tag_5: str = ""
    tag_6: str = ""
    tag_7: str = ""
    tag_8: str = ""
    tag_9: str = ""


@app.on_event("startup")
def load_models():
    """Load all ML models once when the API starts."""

    global predictor

    print("Loading ML models...")
    predictor = TicketPredictor()
    print("All ML models loaded successfully.")


@app.get("/")
def root():
    """API health/status endpoint."""

    return {
        "name": "Customer Support Ticket Intelligence API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    """Health check endpoint."""

    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
    }


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    """Predict Type, Queue, and Priority for a support ticket."""

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML models are not loaded.",
        )

    try:
        ticket_data = ticket.model_dump()

        result = predictor.predict(ticket_data)

        return {
            "status": "success",
            "prediction": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc