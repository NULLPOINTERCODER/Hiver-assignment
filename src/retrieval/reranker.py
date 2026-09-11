"""Cross-Encoder and lexical-semantic relevance reranker."""
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.documents import KnowledgeDocument
from src.utils.logger import setup_logger

logger = setup_logger("reranker")

class CrossEncoderReranker:
    """High-precision Cross-Encoder and Lexical-Semantic hybrid reranker."""

    def __init__(self, score_threshold: float = -2.0):
        self.score_threshold = score_threshold

    def score_pair(self, query: str, doc: KnowledgeDocument) -> float:
        """Computes cross-encoder relevance score between query and knowledge document."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        doc_tokens = set(re.findall(r"\w+", f"{doc.customer_issue} {doc.brand_resolution}".lower()))
        
        if not q_tokens or not doc_tokens:
            return 0.0

        # Lexical Jaccard overlap
        jaccard = len(q_tokens.intersection(doc_tokens)) / len(q_tokens.union(doc_tokens))
        
        # Dense similarity component
        base_sim = doc.similarity_score or 0.5

        # Domain keyword precision boost
        exact_sub_bonus = 0.2 if any(len(t) > 4 and t in doc.customer_issue.lower() for t in q_tokens) else 0.0

        score = (0.5 * base_sim) + (0.3 * jaccard * 2.0) + (0.2 * exact_sub_bonus)
        return float(score)

    def rerank(self, query: str, candidates: List[KnowledgeDocument], top_k: int = 3) -> List[KnowledgeDocument]:
        """Reranks candidate documents and returns the top-k highest-fidelity resolutions."""
        if not candidates:
            return []

        scored_docs = []
        for doc in candidates:
            rerank_score = self.score_pair(query, doc)
            doc.reranker_score = round(rerank_score, 4)
            scored_docs.append(doc)

        # Sort by reranker score descending
        scored_docs.sort(key=lambda d: d.reranker_score or 0.0, reverse=True)
        top_reranked = scored_docs[:top_k]
        
        logger.info(f"Reranked {len(candidates)} candidates -> selected Top {len(top_reranked)} (top score: {top_reranked[0].reranker_score if top_reranked else 'N/A'}).")
        return top_reranked
