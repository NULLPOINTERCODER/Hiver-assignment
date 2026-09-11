"""FastAPI Request and Response schemas."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    """Incoming customer query payload."""
    message: str = Field(..., min_length=2, description="Customer query text", example="My iPhone battery is draining very fast.")

class EvidenceResponseItem(BaseModel):
    conversation_id: str
    intent: str
    customer_issue: str
    brand_resolution: str
    relevance_score: float

class PredictResponse(BaseModel):
    """Structured response from the AI support agent."""
    intent: str = Field(..., description="Predicted intent class")
    intent_confidence: float = Field(..., description="Intent classifier probability")
    decision: Literal["auto", "escalate"] = Field(..., description="Routing decision")
    reason: str = Field(..., description="Decision reasoning")
    reply: str = Field(..., description="Grounded response draft")
    confidence: float = Field(..., description="Composite system confidence")
    evidence: List[EvidenceResponseItem] = Field(default_factory=list, description="Historical precedent evidence")

class HealthResponse(BaseModel):
    status: str
    brand: str
    version: str
