"""Knowledge Base Document data model."""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

@dataclass
class KnowledgeDocument:
    """Represents a historical support interaction document in the RAG knowledge base."""
    doc_id: str
    conversation_id: str
    brand: str
    intent: str
    customer_issue: str
    brand_resolution: str
    conversation_context: Optional[str] = None
    similarity_score: Optional[float] = None
    reranker_score: Optional[float] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Converts to flat metadata dictionary for vector store storage."""
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "intent": self.intent,
            "brand_resolution": self.brand_resolution[:500],
            "customer_issue": self.customer_issue[:300]
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
