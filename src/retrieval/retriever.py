"""Two-stage retrieval pipeline (Vector search + Reranking)."""
from typing import Any, Dict, List, Optional
from src.retrieval.documents import KnowledgeDocument
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import HybridVectorStore
from src.utils.logger import setup_logger

logger = setup_logger("retriever")

class TwoStageRetriever:
    """Orchestrates candidate retrieval from vector store and subsequent Cross-Encoder reranking."""

    def __init__(
        self,
        vector_store: HybridVectorStore,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 3,
        use_metadata_filter: bool = True
    ):
        self.vector_store = vector_store
        self.reranker = reranker or CrossEncoderReranker()
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.use_metadata_filter = use_metadata_filter

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        intent_confidence: float = 1.0,
        min_confidence_for_filter: float = 0.65
    ) -> List[KnowledgeDocument]:
        """Retrieves and reranks top historical support resolutions."""
        # Use metadata filter only if intent confidence is sufficient
        filter_intent = predicted_intent if (self.use_metadata_filter and intent_confidence >= min_confidence_for_filter) else None

        logger.info(f"Retrieving candidate documents (query='{query[:50]}...', intent_filter={filter_intent})...")
        candidates = self.vector_store.query(
            query_text=query,
            top_k=self.top_k_retrieve,
            intent_filter=filter_intent
        )

        if not candidates and filter_intent:
            logger.info("Filtered query returned 0 documents; executing unfiltered fallback retrieval...")
            candidates = self.vector_store.query(query_text=query, top_k=self.top_k_retrieve, intent_filter=None)

        # Rerank to top_k_rerank
        reranked_docs = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
        return reranked_docs
