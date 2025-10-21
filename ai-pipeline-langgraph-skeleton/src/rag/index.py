from __future__ import annotations
import os, tempfile
from typing import List, Callable, Optional

# Where we keep a TF-IDF vectorizer so queries match the index
def _vectorizer_path(settings) -> str:
    base = os.getenv("QDRANT_LOCAL_PATH", "./.qdrant")
    os.makedirs(base, exist_ok=True)
    name = getattr(settings, "QDRANT_COLLECTION", "assignment_docs")
    return os.path.join(base, f"{name}_tfidf.joblib")

def _fastembed_embedder(model_name: Optional[str]) -> Optional[Callable[[List[str]], List[List[float]]]]:
    try:
        from qdrant_client.fastembed import TextEmbedding
        te = TextEmbedding(model_name or "BAAI/bge-small-en-v1.5")
        def _embed(texts: List[str]) -> List[List[float]]:
            return [vec.tolist() for vec in te.embed(texts)]
        return _embed
    except Exception:
        return None

def _tfidf_fit_transform(texts: List[str], save_path: str, max_features: int = 384) -> List[List[float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    mat = vec.fit_transform(texts)
    joblib.dump(vec, save_path)
    dense = mat.toarray().astype("float32")
    return dense.tolist()

def _tfidf_transform(texts: List[str], load_path: str) -> List[List[float]]:
    import joblib
    vec = joblib.load(load_path)
    mat = vec.transform(texts)
    dense = mat.toarray().astype("float32")
    return dense.tolist()

from .pdf_loader import load_and_chunk_pdf
from ..vectorstore.qdrant_store import get_qdrant
from qdrant_client.http import models as qm

def index_pdf_into_qdrant(uploaded_file, settings):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    docs = load_and_chunk_pdf(tmp_path)
    os.unlink(tmp_path)
    if not docs:
        return

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # Embedding strategy (torchless)
    embedder = _fastembed_embedder(getattr(settings, "EMBEDDINGS_MODEL", None))
    if embedder is not None:
        vectors = embedder(texts)
        dim = len(vectors[0])
        # Remove any old TF-IDF vectorizer if switching to FastEmbed
        vec_file = _vectorizer_path(settings)
        if os.path.exists(vec_file):
            try: os.remove(vec_file)
            except Exception: pass
    else:
        # FIXED TF-IDF size = 384 so we never conflict later
        vec_file = _vectorizer_path(settings)
        vectors = _tfidf_fit_transform(texts, vec_file, max_features=384)
        dim = len(vectors[0])  # 384

    # Recreate collection with the exact dim we’re about to insert
    client = get_qdrant()
    client.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    # Upsert points directly
    payloads = [{"text": t, **(m or {})} for t, m in zip(texts, metadatas)]
    points = [qm.PointStruct(id=i, vector=vectors[i], payload=payloads[i]) for i in range(len(vectors))]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

def rag_retrieve(query: str, settings, k: int = 5) -> List[str]:
    vec_file = _vectorizer_path(settings)
    embedder = _fastembed_embedder(getattr(settings, "EMBEDDINGS_MODEL", None))

    if os.path.exists(vec_file) and embedder is None:
        qvec = _tfidf_transform([query], vec_file)[0]  # 384-dim
    else:
        if embedder is None:
            raise RuntimeError(
                "No embedding backend available. Install qdrant-client[fastembed] or index a PDF first."
            )
        qvec = embedder([query])[0]

    client = get_qdrant()
    res = client.search(collection_name=settings.QDRANT_COLLECTION, query_vector=qvec, limit=k)
    return [hit.payload.get("text", "") for hit in res]
