"""The RAG pipeline: question -> retrieve -> generate -> verify citations.

Faithful citations are the point of this project, so the flow is:

  1. Retrieve top-k chunks (retrieval.py). Number them [1..k].
  2. Ask the LLM to answer using ONLY those passages and to cite [n] markers.
  3. Parse the [n] markers out of the answer.
  4. VERIFY each marker: it must map to a real retrieved chunk. Markers the
     model invented (out of range) are dropped. Each surviving citation is
     resolved to its true provenance: note_id, section, char span, and the
     exact snippet text (which we re-confirm exists in the original note in
     Week-2 tests).

This guarantees a cited [n] always points at real, retrieved evidence — the
model cannot fabricate a source that isn't in the context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional

from app.retrieval import retrieve, RetrievedChunk
from app.llm import generate, generate_json
from app.prompts import RELEVANCE_GATE_SYSTEM, RELEVANCE_GATE_USER
from app.prompts import (
    QA_SYSTEM,
    QA_USER_TEMPLATE,
    format_context,
    SUMMARY_SYSTEM,
    SUMMARY_USER_TEMPLATE,
)
from app.vector_store import get_client
from app.config import QDRANT_COLLECTION
from qdrant_client.http import models as qm

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    marker: int          # the [n] number as it appears in the answer
    note_id: str
    section: str
    char_start: int
    char_end: int
    snippet: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: List[Citation]
    retrieved: List[dict]      # all retrieved chunks (for transparency/debug)
    dropped_markers: List[int] # markers the model cited that were invalid


def _parse_markers(text: str) -> List[int]:
    """Unique citation numbers appearing in the answer, in first-seen order."""
    seen = []
    for m in _CITE_RE.findall(text):
        n = int(m)
        if n not in seen:
            seen.append(n)
    return seen


def _verify_citations(
    markers: List[int], chunks: List[RetrievedChunk]
) -> tuple:
    """Map cited markers to real chunks. Returns (valid_citations,
    dropped_markers). A marker is valid iff 1 <= marker <= len(chunks)."""
    citations: List[Citation] = []
    dropped: List[int] = []
    for n in markers:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            citations.append(
                Citation(
                    marker=n,
                    note_id=c.note_id,
                    section=c.section,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    snippet=c.text,
                )
            )
        else:
            dropped.append(n)  # model invented a passage number
    return citations, dropped


def answer_question(
    question: str,
    top_k: int = 5,
    note_ids: Optional[List[str]] = None,
    use_relevance_gate: bool = False,
) -> RAGAnswer:
    """Run the full RAG loop with citation verification.

    use_relevance_gate: if True, run a cheap LLM yes/no check that the retrieved
    context actually contains the answer BEFORE generating — refusing early when
    it doesn't. Reduces over-answering on loosely-related retrieval at the cost
    of one extra (small) LLM call.
    """
    chunks = retrieve(question, top_k=top_k, note_ids=note_ids)

    if not chunks:
        return RAGAnswer(
            question=question,
            answer="I couldn't find anything relevant in the available notes.",
            citations=[],
            retrieved=[],
            dropped_markers=[],
        )

    context = format_context(chunks)

    if use_relevance_gate and not _context_answers(question, context):
        return RAGAnswer(
            question=question,
            answer="I cannot find the answer to that in this note.",
            citations=[],
            retrieved=[asdict_chunk(c) for c in chunks],
            dropped_markers=[],
        )

    prompt = QA_USER_TEMPLATE.format(context=context, question=question)
    raw_answer = generate(prompt, system=QA_SYSTEM)

    markers = _parse_markers(raw_answer)
    citations, dropped = _verify_citations(markers, chunks)

    return RAGAnswer(
        question=question,
        answer=raw_answer,
        citations=citations,
        retrieved=[asdict_chunk(c) for c in chunks],
        dropped_markers=dropped,
    )


def _context_answers(question: str, context: str) -> bool:
    """LLM relevance gate: does the context directly contain the answer?
    Fail-open (return True) on any error so the gate never silently blocks."""
    try:
        prompt = RELEVANCE_GATE_USER.format(question=question, context=context)
        res = generate_json(prompt, system=RELEVANCE_GATE_SYSTEM)
        return bool(res.get("answerable", True))
    except Exception:
        return True


def asdict_chunk(c: RetrievedChunk) -> dict:
    return {
        "note_id": c.note_id,
        "section": c.section,
        "chunk_index": c.chunk_index,
        "score": c.score,
        "text": c.text,
    }


def _load_note_chunks(note_id: str) -> List[RetrievedChunk]:
    """Fetch ALL chunks for one note, ordered by chunk_index, straight from
    Qdrant (no vector search). Used for whole-note summarization."""
    client = get_client()
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="note_id", match=qm.MatchValue(value=note_id))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    chunks = [
        RetrievedChunk(
            note_id=p.payload["note_id"],
            section=p.payload["section"],
            chunk_index=p.payload["chunk_index"],
            text=p.payload["text"],
            char_start=p.payload["char_start"],
            char_end=p.payload["char_end"],
            score=1.0,
        )
        for p in points
    ]
    return sorted(chunks, key=lambda c: c.chunk_index)


def summarize_note(note_id: str) -> str:
    """Generate a structured summary of one full note."""
    chunks = _load_note_chunks(note_id)
    if not chunks:
        return f"No note found with id {note_id!r}."
    context = format_context(chunks)
    prompt = SUMMARY_USER_TEMPLATE.format(context=context)
    return generate(prompt, system=SUMMARY_SYSTEM)
