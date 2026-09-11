"""Composite confidence calculation module."""
from typing import List, Optional
from src.retrieval.documents import KnowledgeDocument
from src.utils.logger import setup_logger

logger = setup_logger("confidence_calculator")

class CompositeConfidenceCalculator:
    """Calculates grounded composite confidence from classifier, retrieval, and reranking signals."""

    def __init__(
        self,
        w_intent: float = 0.40,
        w_retrieval: float = 0.30,
        w_rerank: float = 0.20,
        w_evidence_count: float = 0.10
    ):
        self.w_intent = w_intent
        self.w_retrieval = w_retrieval
        self.w_rerank = w_rerank
        self.w_evidence_count = w_evidence_count

    def calculate(
        self,
        intent_confidence: float,
        evidence_docs: List[KnowledgeDocument],
        has_risk_penalty: bool = False
    ) -> float:
        """Calculates normalized composite confidence score in [0.0, 1.0]."""
        # Retrieval similarity component
        top_sim = evidence_docs[0].similarity_score if (evidence_docs and evidence_docs[0].similarity_score is not None) else 0.0
        norm_sim = max(0.0, min(1.0, float(top_sim)))

        # Reranker score component
        top_rerank = evidence_docs[0].reranker_score if (evidence_docs and evidence_docs[0].reranker_score is not None) else 0.0
        norm_rerank = max(0.0, min(1.0, float(top_rerank)))

        # Evidence count saturation
        count_factor = min(1.0, len(evidence_docs) / 3.0)

        # Composite sum
        score = (
            (self.w_intent * intent_confidence) +
            (self.w_retrieval * norm_sim) +
            (self.w_rerank * norm_rerank) +
            (self.w_evidence_count * count_factor)
        )

        if has_risk_penalty:
            score -= 0.30

        confidence = max(0.0, min(1.0, round(score, 4)))
        return confidence
