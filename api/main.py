"""FastAPI application for AI Customer Support Agent inference."""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from api.schemas import EvidenceResponseItem, HealthResponse, PredictRequest, PredictResponse
from src.agent.agent import SupportAgent
from src.utils.logger import setup_logger

logger = setup_logger("api_main")

app = FastAPI(
    title="Hiver AI Customer Support Agent API",
    description="Production REST API for intent classification, historical resolution retrieval, and conservative escalation routing for Apple Support.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent singleton
support_agent: Optional[SupportAgent] = None

def get_support_agent() -> SupportAgent:
    global support_agent
    if support_agent is None:
        logger.info("Initializing SupportAgent singleton...")
        support_agent = SupportAgent.from_config()
    return support_agent

@app.on_event("startup")
def startup_event():
    get_support_agent()

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "brand": "AppleSupport",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Processes customer support query and returns intent, reply, routing decision, and historical evidence."""
    agent = get_support_agent()

    try:
        output = agent.process_message(request.message)
        
        evidence_resp = [
            EvidenceResponseItem(
                conversation_id=e.conversation_id,
                intent=e.intent,
                customer_issue=e.customer_issue,
                brand_resolution=e.brand_resolution,
                relevance_score=e.relevance_score
            )
            for e in output.evidence
        ]

        return PredictResponse(
            intent=output.intent,
            intent_confidence=output.confidence,
            decision=output.decision,
            reason=output.reason,
            reply=output.reply,
            confidence=output.confidence,
            evidence=evidence_resp
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
