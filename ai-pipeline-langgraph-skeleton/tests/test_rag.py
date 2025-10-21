import os
from src.config import Settings
from src.rag.index import rag_retrieve

def test_rag_retrieve_works_with_empty_collection(monkeypatch):
    # Point to a non-existent collection to validate graceful empty result (won't raise)
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection_empty")
    settings = Settings()
    # No data inserted; should return list (possibly empty), not raise
    res = rag_retrieve("dummy query", settings, k=3)
    assert isinstance(res, list)
