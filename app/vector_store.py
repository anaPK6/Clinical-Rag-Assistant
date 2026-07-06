"""Qdrant vector store wrapper.

Multi-note corpus: all chunks from all notes live in ONE collection.
`note_id` is stored in each point's payload and used as a *filter* at query
time (Week 2) so the user can scope search to one note, a set, or everything.

Point IDs are deterministic (uuid5 of note_id + chunk_index) so re-ingesting
a note overwrites its chunks instead of duplicating them.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import QDRANT_URL, QDRANT_COLLECTION, EMBED_DIM
from app.chunking import Chunk

_NAMESPACE = uuid.UUID("c11d1ca1-0000-4000-8000-000000000001")  # stable ns


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: Optional[QdrantClient] = None) -> None:
    """Create the collection if it doesn't exist (cosine distance)."""
    client = client or get_client()
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
    )
    # Index note_id so filtered search (scope to a note) is fast.
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="note_id",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )


def _point_id(note_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{note_id}:{chunk_index}"))


def upsert_chunks(
    chunks: List[Chunk],
    vectors: List[List[float]],
    client: Optional[QdrantClient] = None,
) -> int:
    """Upsert chunk vectors + payloads. Returns the number upserted."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) != vectors ({len(vectors)})")
    client = client or get_client()
    points = [
        qm.PointStruct(
            id=_point_id(c.note_id, c.chunk_index),
            vector=vec,
            payload=c.to_payload(),
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def delete_note(note_id: str, client: Optional[QdrantClient] = None) -> None:
    """Remove all chunks for a note (e.g. before re-ingesting)."""
    client = client or get_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="note_id", match=qm.MatchValue(value=note_id))]
            )
        ),
    )


def count(client: Optional[QdrantClient] = None) -> int:
    client = client or get_client()
    return client.count(collection_name=QDRANT_COLLECTION, exact=True).count
