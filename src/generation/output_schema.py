"""Pydantic schemas for structured LLM generation, validation, and evaluation."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    """Represents a piece of historical evidence retrieved for grounding."""
    conversation_id: str = Field(..., description="Conversation ID of the historical interaction")
    intent: str = Field(..., description="Historical issue intent")
    customer_issue: str = Field(..., description="Customer query summary")
    brand_resolution: str = Field(..., description="Authoritative resolution provided by brand support")
    relevance_score: float = Field(default=0.0, description="Reranker / similarity relevance score")

class AgentOutput(BaseModel):
    """Standardized production response schema for the AI Support Agent."""
    intent: str = Field(..., description="Predicted intent classification")
    reply: str = Field(..., description="Grounded customer support response draft")
    decision: Literal["auto", "escalate"] = Field(..., description="Routing decision: auto-handle vs human escalation")
    reason: str = Field(..., description="Actionable rationale for the decision and resolution")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall calibrated system confidence score")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="List of grounding historical support cases")

class JudgeScore(BaseModel):
    """Rubric scoring evaluation output from LLM-as-a-Judge."""
    groundedness: int = Field(..., ge=1, le=5, description="Response is grounded solely in retrieved evidence")
    correctness: int = Field(..., ge=1, le=5, description="Resolution is technically accurate for the brand")
    helpfulness: int = Field(..., ge=1, le=5, description="Provides clear next steps and addresses the customer need")
    actionability: int = Field(..., ge=1, le=5, description="Includes clear actionable links, settings paths, or steps")
    tone: int = Field(..., ge=1, le=5, description="Professional, empathetic, and concise tone matching brand standard")
    reasoning: str = Field(..., description="Detailed explanation supporting the score rubric")
