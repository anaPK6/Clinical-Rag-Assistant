"""BGE embeddings via sentence-transformers.

IMPORTANT — BGE is *asymmetric*:
  - Passages (the note chunks we store) are embedded with NO prefix.
  - Queries (a user's question) get a retrieval prefix.
Getting this wrong silently hurts recall, so the two paths are separate
methods and never mixed.

The model is loaded lazily (first use) so importing this module is cheap
and the API process starts fast.
"""
from __future__ import annotations

from typing import List, Optional

from app.config import EMBED_MODEL_NAME, BGE_QUERY_PREFIX

_model = None  # lazily initialized SentenceTransformer


def _get_model():
    global _model
    if _model is None:
        # Imported here so `import app.embeddings` doesn't pull torch eagerly.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_passages(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed note chunks for storage. No prefix (BGE passage convention).
    Vectors are L2-normalized so cosine == dot product in Qdrant."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(query: str) -> List[float]:
    """Embed a single user query with the BGE retrieval prefix."""
    model = _get_model()
    vec = model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.tolist()
