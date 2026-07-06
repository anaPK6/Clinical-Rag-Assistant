"""Retrieval: question -> most relevant note chunks.

This is the first half of the RAG loop (the "R"). It:
  1. Embeds the user's question with the BGE *query* prefix.
  2. Searches Qdrant for the nearest chunk vectors.
  3. Optionally filters by note scope (multi-note corpus) so the user can
     ask over one note, a set of notes, or everything.

It returns lightweight RetrievedChunk objects that carry everything the
answer/citation steps need: the text, its section, and — critically — the
char span into the original note (so citations can be verified later).

No LLM is involved here; retrieval is fully testable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qdrant_client.http import models as qm

from app.config import QDRANT_COLLECTION
from app.embeddings import embed_query
from app.vector_store import get_client


@dataclass
class RetrievedChunk:
    note_id: str
    section: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    score: float  # cosine similarity (higher = more relevant)


def _note_filter(note_ids: Optional[List[str]]) -> Optional[qm.Filter]:
    """Build a Qdrant filter restricting search to the given note_ids.
    None or empty => no filter (search the whole corpus)."""
    if not note_ids:
        return None
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="note_id",
                match=qm.MatchAny(any=list(note_ids)),
            )
        ]
    )


def retrieve(
    query: str,
    top_k: int = 5,
    note_ids: Optional[List[str]] = None,
    client=None,
) -> List[RetrievedChunk]:
    """Return the top_k most relevant chunks for a query.

    Args:
        query:    the user's natural-language question.
        top_k:    how many chunks to return.
        note_ids: restrict to these notes; None/empty = search everything.
    """
    client = client or get_client()
    qvec = embed_query(query)

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        query_filter=_note_filter(note_ids),
        limit=top_k,
        with_payload=True,
    )

    results: List[RetrievedChunk] = []
    for h in hits:
        p = h.payload
        results.append(
            RetrievedChunk(
                note_id=p["note_id"],
                section=p["section"],
                chunk_index=p["chunk_index"],
                text=p["text"],
                char_start=p["char_start"],
                char_end=p["char_end"],
                score=h.score,
            )
        )
    return results
