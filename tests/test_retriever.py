"""Unit tests for Vector Store, Reranker, and Two-Stage Retriever."""
import pytest
from src.retrieval.documents import KnowledgeDocument
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import TwoStageRetriever
from src.retrieval.vector_store import HybridVectorStore

@pytest.fixture
def sample_docs():
    return [
        KnowledgeDocument(
            doc_id="doc_1",
            conversation_id="conv_1",
            brand="AppleSupport",
            intent="battery_power_issue",
            customer_issue="Battery draining fast on iOS 11",
            brand_resolution="Please check Settings > Battery to see app usage."
        ),
        KnowledgeDocument(
            doc_id="doc_2",
            conversation_id="conv_2",
            brand="AppleSupport",
            intent="software_update_issue",
            customer_issue="iOS update failed to verify",
            brand_resolution="Restart your Wi-Fi and re-download the update installer."
        )
    ]

def test_vector_store_query(sample_docs, tmp_path):
    store = HybridVectorStore(persist_dir=tmp_path)
    store.build_from_documents(sample_docs)
    
    results = store.query("battery dying fast", top_k=1)
    assert len(results) == 1
    assert results[0].intent == "battery_power_issue"

    # Test metadata filtering
    filtered = store.query("update problem", top_k=1, intent_filter="software_update_issue")
    assert len(filtered) == 1
    assert filtered[0].intent == "software_update_issue"

def test_reranker(sample_docs):
    reranker = CrossEncoderReranker()
    sample_docs[0].similarity_score = 0.8
    sample_docs[1].similarity_score = 0.5

    reranked = reranker.rerank("battery draining", sample_docs, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].intent == "battery_power_issue"
    assert reranked[0].reranker_score is not None
