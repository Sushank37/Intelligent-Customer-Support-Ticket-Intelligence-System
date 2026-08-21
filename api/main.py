"""FastAPI Application for Real-Time Ticket Intelligence."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.inference.predict import InferenceEngine

app = FastAPI(
    title="Customer Support Ticket Intelligence API",
    description="API for ticket categorization, priority assessment, and sentiment analysis",
    version="1.0.0"
)

engine = InferenceEngine()


class TicketRequest(BaseModel):
    ticket_id: Optional[str] = Field(None, example="TCK-10023")
    subject: str = Field(..., example="Unable to access billing dashboard")
    description: str = Field(..., example="I received an unexpected charge on my credit card and cannot download the invoice.")


class TicketResponse(BaseModel):
    ticket_id: Optional[str]
    predicted_category: str
    priority: str
    confidence_score: float
    cleaned_text: str


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "online", "service": "Customer Support Ticket Intelligence API"}


@app.post("/predict", response_model=TicketResponse)
def predict_ticket(request: TicketRequest):
    """Predict category and priority for an incoming support ticket."""
    full_text = f"{request.subject} {request.description}"
    result = engine.predict_ticket(full_text)

    return TicketResponse(
        ticket_id=request.ticket_id,
        predicted_category=result["predicted_category"],
        priority=result["priority"],
        confidence_score=result["confidence_score"],
        cleaned_text=result["cleaned_text"]
    )
