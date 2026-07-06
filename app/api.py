"""FastAPI backend for the clinical RAG assistant.

Wraps the existing pipeline (retrieval, RAG QA, summarization, extraction)
behind HTTP endpoints so the Streamlit frontend (or any client) can call it.

Endpoints:
  GET  /health            liveness + backing-service status
  GET  /notes             curated note menu (number, id, title)
  POST /ask               question over ONE note (default) or all notes (explicit)
  POST /summarize         structured summary of one note
  POST /extract-entities  5-category entity extraction for one note
  POST /upload-note       ingest a new note (text) into the corpus

Design note: the single-patient-scope safety model is preserved — /ask
requires a note_id by default; searching across all notes is an explicit,
clearly-flagged opt-in (search_all=true).
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_pipeline import answer_question, summarize_note
from app.extraction import extract_entities
from app.note_catalog import CURATED_NOTES, title_for, resolve
from app.config import QDRANT_URL, OLLAMA_HOST, LLM_MODEL

app = FastAPI(title="Clinical RAG Assistant", version="1.0.0")


# ── request/response models ──
class AskRequest(BaseModel):
    question: str
    note_id: Optional[str] = None       # which note to search
    search_all: bool = False            # explicit cross-note (cross-patient) search
    top_k: int = 5
    strict_grounding: bool = True       # relevance gate: refuse if note lacks the answer


class SummarizeRequest(BaseModel):
    note_id: str


class ExtractRequest(BaseModel):
    note_id: str


class UploadRequest(BaseModel):
    note_id: str
    text: str


# ── endpoints ──
@app.get("/health")
def health():
    """Liveness + whether backing services are reachable."""
    import requests

    def up(url: str) -> bool:
        try:
            return requests.get(url, timeout=2).ok
        except Exception:
            return False

    return {
        "status": "ok",
        "qdrant": up(f"{QDRANT_URL}/readyz"),
        "ollama": up(f"{OLLAMA_HOST}/api/tags"),
        "llm_model": LLM_MODEL,
    }


@app.get("/notes")
def notes():
    """Curated note menu for the UI's selector."""
    return {
        "notes": [
            {"number": i + 1, "note_id": nid, "title": title}
            for i, (nid, title) in enumerate(CURATED_NOTES)
        ]
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")

    # Enforce scope: one note by default; all-notes only if explicitly requested.
    if req.search_all:
        note_ids = None
    else:
        nid = resolve(req.note_id) if req.note_id else None
        if not nid:
            raise HTTPException(
                status_code=400,
                detail="note_id required (or set search_all=true to search every note).",
            )
        note_ids = [nid]

    r = answer_question(req.question, top_k=req.top_k, note_ids=note_ids,
                        use_relevance_gate=req.strict_grounding)
    return {
        "question": r.question,
        "answer": r.answer,
        "search_all": req.search_all,
        "citations": [
            {
                "marker": c.marker,
                "note_id": c.note_id,
                "note_title": title_for(c.note_id),
                "section": c.section,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "snippet": c.snippet,
            }
            for c in r.citations
        ],
        "dropped_markers": r.dropped_markers,
    }


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    nid = resolve(req.note_id)
    if not nid:
        raise HTTPException(status_code=400, detail="Unknown note_id.")
    return {"note_id": nid, "title": title_for(nid), "summary": summarize_note(nid)}


@app.post("/extract-entities")
def extract(req: ExtractRequest):
    nid = resolve(req.note_id)
    if not nid:
        raise HTTPException(status_code=400, detail="Unknown note_id.")
    result = extract_entities(nid)
    return result.model_dump() | {"title": title_for(nid), "counts": result.counts()}


@app.post("/upload-note")
def upload_note(req: UploadRequest):
    """Ingest a new note (raw text) into the corpus."""
    from app.chunking import chunk_note
    from app.embeddings import embed_passages
    from app.vector_store import ensure_collection, upsert_chunks, delete_note, get_client

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty note text.")

    client = get_client()
    ensure_collection(client)
    chunks = chunk_note(req.note_id, req.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from note.")
    vectors = embed_passages([c.text for c in chunks])
    delete_note(req.note_id, client=client)
    n = upsert_chunks(chunks, vectors, client=client)
    return {"note_id": req.note_id, "chunks_ingested": n,
            "sections": sorted({c.section for c in chunks})}
