"""Script to index historical support resolutions into HybridVectorStore."""
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.retrieval.documents import KnowledgeDocument
from src.retrieval.vector_store import HybridVectorStore
from src.utils.logger import setup_logger

logger = setup_logger("build_vectorstore")

def main():
    kb_path = ROOT_DIR / "data" / "processed" / "knowledge_base.parquet"
    if not kb_path.exists():
        kb_path = ROOT_DIR / "data" / "sample" / "knowledge_base_sample.parquet"

    df = pd.read_parquet(kb_path)
    logger.info(f"Loaded {len(df)} knowledge base items from {kb_path}")

    docs = []
    for i, row in df.iterrows():
        doc = KnowledgeDocument(
            doc_id=f"kb_doc_{i:05d}",
            conversation_id=str(row.get("conversation_id", f"kb_conv_{i}")),
            brand=str(row.get("brand", "AppleSupport")),
            intent=str(row.get("intent", "general_inquiry_advice")),
            customer_issue=str(row.get("clean_customer_message", "")),
            brand_resolution=str(row.get("clean_brand_response", "")),
            conversation_context=str(row.get("history", ""))
        )
        docs.append(doc)

    vector_dir = ROOT_DIR / "vectorstore" / "chroma_db"
    vector_store = HybridVectorStore(persist_dir=vector_dir, collection_name="brand_support_kb")
    vector_store.build_from_documents(docs)
    vector_store.save()

    logger.info("Vector store successfully built and persisted to disk.")

if __name__ == "__main__":
    main()
