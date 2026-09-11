"""Escalation policy and rule evaluation engine."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.retrieval.documents import KnowledgeDocument
from src.utils.io import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger("escalation_engine")

class EscalationEngine:
    """Evaluates multi-layered escalation policies (security, legal, hardware, low confidence)."""

    def __init__(
        self,
        rules_path: Optional[Union[str, Path]] = None,
        min_intent_confidence: float = 0.65,
        min_retrieval_similarity: float = 0.45,
        min_reranker_score: float = -1.5,
        min_evidence_count: int = 1
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity
        self.min_reranker_score = min_reranker_score
        self.min_evidence_count = min_evidence_count
        self.rules: List[Dict[str, Any]] = []

        if rules_path:
            rules_cfg = load_yaml(rules_path)
            self.rules = rules_cfg.get("rules", [])
            logger.info(f"Loaded {len(self.rules)} deterministic escalation rules.")

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_docs: List[KnowledgeDocument]
    ) -> Tuple[bool, str, Optional[str]]:
        """Evaluates customer message and system state against escalation policies.
        
        Returns:
            (should_escalate, reason, rule_id)
        """
        msg_lower = customer_message.lower()

        # 1. Deterministic Rule Matching (Security, Legal, Hardware damage)
        for rule in self.rules:
            keywords = rule.get("keywords", [])
            for kw in keywords:
                if kw in msg_lower:
                    rule_id = rule.get("id", "rule_match")
                    reason = rule.get("reason", f"Triggered escalation rule: {rule.get('name')}")
                    logger.info(f"Escalation triggered by rule '{rule_id}' (matched keyword: '{kw}').")
                    return True, reason, rule_id

        # 2. Intent Confidence Check
        if intent_confidence < self.min_intent_confidence:
            reason = f"Intent confidence ({intent_confidence:.2f}) is below safe threshold ({self.min_intent_confidence:.2f}); escalating to human agent."
            logger.info(reason)
            return True, reason, "low_intent_confidence"

        # 3. Evidence Grounding Availability Check
        if len(evidence_docs) < self.min_evidence_count:
            reason = "No authoritative historical resolution found; escalating to prevent hallucinated policy advice."
            logger.info(reason)
            return True, reason, "zero_evidence"

        # 4. Retrieval Score Check
        top_sim = evidence_docs[0].similarity_score if evidence_docs[0].similarity_score is not None else 0.5
        top_rerank = evidence_docs[0].reranker_score if evidence_docs[0].reranker_score is not None else 0.0
        
        if top_sim < self.min_retrieval_similarity:
            reason = f"Top historical resolution similarity ({top_sim:.2f}) below threshold ({self.min_retrieval_similarity:.2f}); routing to human specialist."
            logger.info(reason)
            return True, reason, "low_retrieval_similarity"

        return False, "Passed all safety and confidence checks for auto-handling.", None
