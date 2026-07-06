# Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (:8501)                        │
│   note selector · Ask / Summary / Extract tabs · cited answers      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP (decoupled)
┌───────────────────────────────▼─────────────────────────────────────┐
│                        FastAPI backend  (:8000)                     │
│   /ask  /summarize  /extract-entities  /upload-note  /notes  /health │
└───────┬───────────────────────┬───────────────────────┬─────────────┘
        │                       │                       │
   retrieval               generation              extraction
        │                       │                       │
┌───────▼────────┐     ┌────────▼─────────┐    ┌────────▼──────────┐
│ BGE embeddings │     │  Ollama (local)  │    │ schema-constrained│
│ bge-base-en    │     │  Llama 3.1 8B    │    │ JSON + grounding  │
└───────┬────────┘     └────────┬─────────┘    └────────┬──────────┘
        │                       │                       │
┌───────▼───────────────────────▼───────────────────────▼─────────────┐
│                 Qdrant vector DB  (:6333, Docker)                   │
│   one collection · payload: note_id · section · char_start/end      │
│   note_id indexed → single-patient filtering (safety)               │
└─────────────────────────────────────────────────────────────────────┘

Ingestion (offline):
  Clinical note ──▶ section-aware chunking ──▶ BGE embed ──▶ Qdrant upsert
                    (char spans preserved → verifiable citations)

Everything runs locally. No clinical text leaves the machine.
```

## The faithfulness pipeline (why this isn't a toy RAG demo)

**Verifiable citations.** The LLM cites `[n]` markers; the pipeline verifies each
maps to a real retrieved chunk and *drops fabricated ones*. A citation always
points at real, retrieved evidence — with the exact note, section, and character
span. Proven adversarially: a fabricated `[9]` (when only 5 chunks were
retrieved) is rejected while the real `[1]` is kept.

**Honest refusal.** When the answer isn't in the note, the system says so and
returns zero citations — it does not fabricate.

**Grounded extraction.** Every extracted entity carries a verbatim `source_text`
verified to exist in the note; entities that can't be verified are flagged
`⚠ unverified` rather than silently trusted.

**Single-patient safety.** Retrieval defaults to one selected note; cross-note
search is an explicit, clearly-warned opt-in — preventing answers that blend
multiple patients.
```
