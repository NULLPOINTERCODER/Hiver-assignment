"""Response generator coordinating prompts, LLM invocation, and output parsing."""
import json
import re
from typing import Any, Dict, List, Optional

from src.generation.llm import LLMClient
from src.generation.output_schema import AgentOutput, EvidenceItem
from src.generation.prompts import GENERATION_USER_PROMPT_TEMPLATE, SYSTEM_PROMPT
from src.retrieval.documents import KnowledgeDocument
from src.utils.logger import setup_logger

logger = setup_logger("generator")

class ResponseGenerator:
    """Generates grounded support replies using retrieved evidence and LLM reasoning."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def format_evidence_text(self, docs: List[KnowledgeDocument]) -> str:
        """Formats retrieved documents into structured prompt context."""
        if not docs:
            return "No historical support evidence found."
        lines = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"[{i}] Issue: {doc.customer_issue}")
            lines.append(f"    Resolution: {doc.brand_resolution}")
            lines.append(f"    Intent: {doc.intent} | Relevance Score: {doc.reranker_score or doc.similarity_score or 0.0:.2f}")
        return "\n".join(lines)

    def generate_response(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_docs: List[KnowledgeDocument],
        escalation_directives: str = "Escalate if high risk or insufficient evidence."
    ) -> AgentOutput:
        """Constructs prompt, executes LLM generation, and returns validated AgentOutput."""
        evidence_text = self.format_evidence_text(evidence_docs)
        user_prompt = GENERATION_USER_PROMPT_TEMPLATE.format(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            evidence_text=evidence_text,
            escalation_directives=escalation_directives
        )

        raw_output = self.llm_client.generate(user_prompt, system_prompt=SYSTEM_PROMPT)

        # Parse JSON
        try:
            # Extract json block if surrounded by markdown code fences
            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_output.strip())
            clean_json = re.sub(r"\s*```$", "", clean_json)
            parsed = json.loads(clean_json)
        except Exception as e:
            logger.warning(f"Failed to parse LLM output as JSON ({e}). Raw: '{raw_output[:100]}...' Escalating safely.")
            return AgentOutput(
                intent=predicted_intent,
                reply="We'd love to help take a closer look into this for you. Please DM us so our team can assist directly.",
                decision="escalate",
                reason="Malformed generator response; escalated for safety.",
                confidence=0.0,
                evidence=[EvidenceItem(
                    conversation_id=d.conversation_id,
                    intent=d.intent,
                    customer_issue=d.customer_issue,
                    brand_resolution=d.brand_resolution,
                    relevance_score=d.reranker_score or d.similarity_score or 0.0
                ) for d in evidence_docs]
            )

        evidence_items = [
            EvidenceItem(
                conversation_id=d.conversation_id,
                intent=d.intent,
                customer_issue=d.customer_issue,
                brand_resolution=d.brand_resolution,
                relevance_score=d.reranker_score or d.similarity_score or 0.0
            ) for d in evidence_docs
        ]

        return AgentOutput(
            intent=parsed.get("intent", predicted_intent),
            reply=parsed.get("reply", ""),
            decision=parsed.get("decision", "escalate"),
            reason=parsed.get("reason", "Standard resolution policy."),
            confidence=float(parsed.get("confidence", intent_confidence)),
            evidence=evidence_items
        )
