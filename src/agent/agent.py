"""Master AI Customer Support Agent orchestrator."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agent.confidence import CompositeConfidenceCalculator
from src.agent.escalation import EscalationEngine
from src.agent.validator import OutputValidator
from src.data.cleaner import TextCleaner
from src.generation.generator import ResponseGenerator
from src.generation.llm import LLMClient
from src.generation.output_schema import AgentOutput, EvidenceItem
from src.intent.classifier import SemanticIntentClassifier
from src.intent.predictor import IntentPredictor
from src.retrieval.documents import KnowledgeDocument
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import TwoStageRetriever
from src.retrieval.vector_store import HybridVectorStore
from src.utils.config import load_config
from src.utils.io import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger("support_agent")

class SupportAgent:
    """Production End-to-End AI Support Agent."""

    def __init__(
        self,
        cleaner: TextCleaner,
        intent_predictor: IntentPredictor,
        retriever: TwoStageRetriever,
        generator: ResponseGenerator,
        escalation_engine: EscalationEngine,
        confidence_calc: CompositeConfidenceCalculator,
        validator: OutputValidator
    ):
        self.cleaner = cleaner
        self.intent_predictor = intent_predictor
        self.retriever = retriever
        self.generator = generator
        self.escalation_engine = escalation_engine
        self.confidence_calc = confidence_calc
        self.validator = validator

    def process_message(self, raw_message: str) -> AgentOutput:
        """Processes an incoming customer message end-to-end and returns a validated AgentOutput."""
        logger.info(f"Processing customer message: '{raw_message[:60]}...'")

        # 1. Text Preprocessing & PII Masking
        clean_query = self.cleaner.clean_text(raw_message)

        # 2. Intent Classification
        pred_res = self.intent_predictor.predict_one(clean_query)
        predicted_intent = pred_res["intent"]
        intent_conf = pred_res["confidence"]
        logger.info(f"Classified intent: '{predicted_intent}' (confidence: {intent_conf:.2f})")

        # 3. Two-Stage Historical Retrieval
        evidence_docs = self.retriever.retrieve(
            query=clean_query,
            predicted_intent=predicted_intent,
            intent_confidence=intent_conf
        )

        # 4. Escalation Policy Evaluation
        should_escalate, esc_reason, rule_id = self.escalation_engine.evaluate(
            customer_message=clean_query,
            predicted_intent=predicted_intent,
            intent_confidence=intent_conf,
            evidence_docs=evidence_docs
        )

        # 5. Response Generation
        if should_escalate:
            logger.info(f"Routing to human escalation: {esc_reason}")
            evidence_items = [
                EvidenceItem(
                    conversation_id=d.conversation_id,
                    intent=d.intent,
                    customer_issue=d.customer_issue,
                    brand_resolution=d.brand_resolution,
                    relevance_score=d.reranker_score or d.similarity_score or 0.0
                ) for d in evidence_docs
            ]
            composite_conf = self.confidence_calc.calculate(
                intent_confidence=intent_conf,
                evidence_docs=evidence_docs,
                has_risk_penalty=True
            )
            raw_output = AgentOutput(
                intent=predicted_intent,
                reply="We'd love to help take a closer look into this for you. Please DM us your device details so our support team can assist directly.",
                decision="escalate",
                reason=esc_reason,
                confidence=composite_conf,
                evidence=evidence_items
            )
        else:
            raw_output = self.generator.generate_response(
                customer_message=clean_query,
                predicted_intent=predicted_intent,
                intent_confidence=intent_conf,
                evidence_docs=evidence_docs,
                escalation_directives="Auto-handle standard issue grounded in evidence."
            )
            composite_conf = self.confidence_calc.calculate(
                intent_confidence=intent_conf,
                evidence_docs=evidence_docs,
                has_risk_penalty=False
            )
            raw_output.confidence = composite_conf

        # 6. Final Safety & Grounding Validation
        final_output = self.validator.enforce_safety(raw_output)
        return final_output

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "SupportAgent":
        """Factory method to initialize complete support agent from config and model weights."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        config = load_config(config_path)

        cleaner = TextCleaner(mask_pii=True)

        # Load Intent Model
        model_path = root_dir / config["intent"]["classifier_model_path"]
        if not model_path.exists():
            model_path = root_dir / "models" / "intent_classifier" / "semantic_intent_model.joblib"
        classifier = SemanticIntentClassifier.load(model_path)
        intent_predictor = IntentPredictor(classifier, confidence_threshold=config["intent"].get("confidence_threshold", 0.65))

        # Load Vector Store
        vector_dir = root_dir / config["retrieval"]["vectorstore_dir"]
        vector_store = HybridVectorStore.load(persist_dir=vector_dir, collection_name=config["retrieval"].get("collection_name", "brand_support_kb"))
        reranker = CrossEncoderReranker(score_threshold=config["reranker"].get("score_threshold", -2.0))
        retriever = TwoStageRetriever(
            vector_store=vector_store,
            reranker=reranker,
            top_k_retrieve=config["retrieval"].get("top_k_retrieval", 10),
            top_k_rerank=config["reranker"].get("top_k_rerank", 3)
        )

        # Generator & LLM
        llm_client = LLMClient(
            provider=config["generation"].get("provider"),
            model=config["generation"].get("model"),
            temperature=config["generation"].get("temperature", 0.2)
        )
        generator = ResponseGenerator(llm_client=llm_client)

        # Escalation Engine
        rules_path = root_dir / config["escalation"]["rules_config"]
        escalation_engine = EscalationEngine(
            rules_path=rules_path,
            min_intent_confidence=config["escalation"].get("min_intent_confidence", 0.65),
            min_retrieval_similarity=config["retrieval"].get("similarity_threshold", 0.45)
        )

        confidence_calc = CompositeConfidenceCalculator()
        
        # Taxonomy intents
        tax_path = root_dir / config["intent"]["taxonomy_path"]
        tax_data = load_yaml(tax_path)
        valid_intents = [item["id"] for item in tax_data.get("intents", [])]
        validator = OutputValidator(valid_intents=valid_intents)

        return cls(
            cleaner=cleaner,
            intent_predictor=intent_predictor,
            retriever=retriever,
            generator=generator,
            escalation_engine=escalation_engine,
            confidence_calc=confidence_calc,
            validator=validator
        )
