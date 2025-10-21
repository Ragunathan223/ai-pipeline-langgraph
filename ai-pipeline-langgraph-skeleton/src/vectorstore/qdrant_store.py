from __future__ import annotations
from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import os

def _try_http_client():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key, timeout=10.0)
    client.get_collections()  # probe
    return client

def _embedded_client():
    data_path = os.getenv("QDRANT_LOCAL_PATH", "./.qdrant")
    os.makedirs(data_path, exist_ok=True)
    return QdrantClient(path=data_path)

def get_qdrant() -> QdrantClient:
    if os.getenv("QDRANT_EMBEDDED") == "1":
        return _embedded_client()
    try:
        return _try_http_client()
    except Exception:
        return _embedded_client()

def _get_existing_dim(client: QdrantClient, name: str) -> Optional[int]:
    try:
        info = client.get_collection(name)
    except Exception:
        return None
    try:
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return vectors.size
        if isinstance(vectors, dict):
            for v in vectors.values():
                if hasattr(v, "size"):
                    return v.size
    except Exception:
        pass
    return None

def ensure_collection(client: QdrantClient, name: str, dim: int = 384) -> None:
    existing_dim = _get_existing_dim(client, name)
    if existing_dim is None or existing_dim != dim:
        client.recreate_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

def upsert_texts(client: QdrantClient, collection: str, embeddings: List[List[float]], payloads: List[dict]):
    # EXTRA safety: (re)create with correct dim right here too
    dim = len(embeddings[0])
    ensure_collection(client, collection, dim=dim)
    points = [qm.PointStruct(id=i, vector=embeddings[i], payload=payloads[i]) for i in range(len(embeddings))]
    client.upsert(collection_name=collection, points=points)

def search(client: QdrantClient, collection: str, query: List[float], k: int = 5) -> List[Tuple[float, dict]]:
    res = client.search(collection_name=collection, query_vector=query, limit=k)
    return [(r.score, r.payload) for r in res]
