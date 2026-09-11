"""Vector Store integration supporting semantic search and metadata filtering."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.documents import KnowledgeDocument
from src.retrieval.embeddings import SemanticEmbedder
from src.utils.io import load_json, save_json
from src.utils.logger import setup_logger

logger = setup_logger("vector_store")

class HybridVectorStore:
    """Production vector store supporting dense semantic indexing, cosine search, and metadata filtering."""

    def __init__(
        self,
        persist_dir: Union[str, Path] = "vectorstore/chroma_db",
        collection_name: str = "brand_support_kb"
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedder = SemanticEmbedder(n_components=250, random_state=42)
        self.documents: List[KnowledgeDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.doc_index: Dict[str, int] = {}

    def build_from_documents(self, docs: List[KnowledgeDocument]) -> "HybridVectorStore":
        """Builds index from list of KnowledgeDocument objects."""
        logger.info(f"Building vector index for {len(docs)} knowledge base documents...")
        self.documents = docs
        self.doc_index = {doc.doc_id: i for i, doc in enumerate(docs)}

        # Fit semantic embedder and compute embeddings
        corpus_texts = [f"{doc.customer_issue} -> {doc.brand_resolution}" for doc in docs]
        self.embedder.fit(corpus_texts)
        self.embeddings = self.embedder.encode(corpus_texts)
        logger.info(f"Computed embeddings matrix of shape {self.embeddings.shape}.")
        return self

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        intent_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[KnowledgeDocument]:
        """Queries vector index with optional metadata intent filtering."""
        if self.embeddings is None or not self.documents:
            logger.warning("Vector store is empty.")
            return []

        query_vec = self.embedder.encode([query_text])
        sim_scores = cosine_similarity(query_vec, self.embeddings)[0]

        ranked_indices = sim_scores.argsort()[::-1]
        results = []

        for idx in ranked_indices:
            score = float(sim_scores[idx])
            if score < min_similarity:
                continue

            doc = self.documents[idx]
            # Apply metadata filter if specified
            if intent_filter and doc.intent.lower() != intent_filter.lower():
                continue

            doc_copy = KnowledgeDocument(
                doc_id=doc.doc_id,
                conversation_id=doc.conversation_id,
                brand=doc.brand,
                intent=doc.intent,
                customer_issue=doc.customer_issue,
                brand_resolution=doc.brand_resolution,
                conversation_context=doc.conversation_context,
                similarity_score=round(score, 4)
            )
            results.append(doc_copy)

            if len(results) >= top_k:
                break

        # Fallback: If metadata filtering returned too few results, relax the filter
        if intent_filter and len(results) < 2:
            logger.info(f"Metadata filter '{intent_filter}' yielded only {len(results)} results; executing broad semantic fallback...")
            return self.query(query_text, top_k=top_k, intent_filter=None, min_similarity=min_similarity)

        return results

    def save(self) -> None:
        """Persists index and document metadata to disk."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.persist_dir / f"{self.collection_name}_meta.json"
        emb_path = self.persist_dir / f"{self.collection_name}_emb.npy"

        doc_dicts = [d.to_dict() for d in self.documents]
        save_json(doc_dicts, meta_path)
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
        logger.info(f"Vector store persisted to {self.persist_dir}")

    @classmethod
    def load(cls, persist_dir: Union[str, Path] = "vectorstore/chroma_db", collection_name: str = "brand_support_kb") -> "HybridVectorStore":
        """Loads persisted vector store from disk."""
        persist_dir = Path(persist_dir)
        meta_path = persist_dir / f"{collection_name}_meta.json"
        emb_path = persist_dir / f"{collection_name}_emb.npy"

        if not meta_path.exists() or not emb_path.exists():
            raise FileNotFoundError(f"Persisted vector store files not found in {persist_dir}")

        store = cls(persist_dir=persist_dir, collection_name=collection_name)
        raw_docs = load_json(meta_path)
        store.documents = [KnowledgeDocument(**d) for d in raw_docs]
        store.doc_index = {doc.doc_id: i for i, doc in enumerate(store.documents)}
        store.embeddings = np.load(emb_path)

        # Refit embedder on corpus texts
        corpus_texts = [f"{doc.customer_issue} -> {doc.brand_resolution}" for doc in store.documents]
        store.embedder.fit(corpus_texts)
        logger.info(f"Loaded vector store with {len(store.documents)} documents.")
        return store
