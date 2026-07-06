# 🩺 Clinical RAG Assistant

**A local-first RAG assistant for clinical notes — that won't lie to a clinician, and proves it.**

Upload a clinical note, then **ask questions**, **generate summaries**, and
**extract structured data** (diagnoses, medications, allergies, procedures,
follow-ups) — with **every answer backed by a verifiable citation** into the
source note. Runs **entirely locally**: no clinical text ever leaves the machine.

> Built on public **MTSamples** transcription reports (no PHI, no credentialing).
> The pipeline is data-agnostic — real corpora flow through unchanged.

---

## Why this isn't another RAG chatbot

Most RAG demos wire an LLM to a vector DB and call it done. In a clinical
setting, an LLM that *sounds* confident but invents a medication or dose is
worse than useless — it's dangerous. This project is engineered for the thing
that actually matters in clinical AI: **trust**.

- **✅ Verifiable citations.** The LLM cites `[n]` markers; the pipeline verifies
  each maps to a real retrieved chunk and **drops fabricated ones**. Every
  citation resolves to an exact note → section → character span.
  *Proven adversarially:* a fabricated `[9]` (when only 5 chunks were retrieved)
  is rejected while the real `[1]` is kept.
- **✅ Honest refusal.** When the answer isn't in the note, it says so and
  returns zero citations — it does not fabricate. Backed by a relevance gate
  (see Evaluation).
- **✅ Grounded extraction.** Every extracted entity carries a verbatim
  `source_text` verified to exist in the note; unverifiable ones are flagged
  `⚠ unverified` rather than silently trusted.
- **✅ Single-patient safety.** Answers default to one selected note; cross-note
  search is an explicit, clearly-warned opt-in — so answers can't blend patients.

---

## Evaluation

Measured on a hand-labeled gold set of 24 Q&A pairs (19 answerable + 5 designed
refusal cases) across the demo notes. Local Llama 3.1 8B.

| Metric | Gate OFF | **Gate ON** (default) |
|---|---|---|
| Answer correctness | 84.2% | 84.2% |
| Citation accuracy | 89.5% | 89.5% |
| Retrieval recall@k | 89.5% | 89.5% |
| **Refusal on unanswerable Qs** | 60% | **100%** |
| False-refusal rate | 0% | 10.5% |
| Latency (mean) | 3.7s | 5.5s |

**The interesting finding — a measured safety/recall tradeoff.** Out of the box
the system over-answered questions it should decline (e.g. answering "family
*history*?" from "completed *family*"). I first tried a similarity-score
threshold to fix it, but measured that the score distributions of answerable vs.
unanswerable questions **overlap too much** (an unanswerable question scored
0.60, higher than most real ones) — so a threshold would wreck the perfect
false-refusal rate. Instead I added an **LLM relevance gate** (a yes/no "does
this context contain the answer?" check before answering). It lifts refusal on
unanswerable questions from 60% → **100%**, at the cost of ~10% false refusals
and ~2s latency. For a clinical tool, *refusing a real question is safer than
fabricating an answer to a fake one*, so the gate is **on by default**.

Runs are logged to **MLflow** (`./mlruns`). Reproduce:

```bash
python -m evals.rag_eval          # gate off (baseline)
python -m evals.rag_eval --gate   # gate on (default behavior)
mlflow ui                         # view runs
```

---

## Architecture

```
┌──────────────────────── Streamlit UI (:8501) ───────────────────────┐
│  note selector · Ask / Summary / Extract · cited answers · themes    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP (decoupled)
┌──────────────────────────────▼───────────────────────────────────────┐
│                       FastAPI backend (:8000)                        │
│  /ask · /summarize · /extract-entities · /upload-note · /notes · /health │
└─────┬──────────────────────┬───────────────────────┬──────────────────┘
   retrieval             generation              extraction
      │                      │                       │
┌─────▼──────┐      ┌────────▼────────┐     ┌────────▼─────────┐
│ BGE embed  │      │ Ollama (local)  │     │ schema JSON +    │
│ bge-base   │      │ Llama 3.1 8B    │     │ span grounding   │
└─────┬──────┘      └────────┬────────┘     └────────┬─────────┘
      │                      │                       │
┌─────▼──────────────────────▼───────────────────────▼──────────────────┐
│              Qdrant vector DB (:6333, Docker)                         │
│  payload: note_id · section · char_start/end  ·  note_id indexed →    │
│  single-patient filtering                                            │
└───────────────────────────────────────────────────────────────────────┘

Ingestion: note ─▶ section-aware chunking ─▶ BGE embed ─▶ Qdrant
                   (char spans preserved → citations are verifiable)
```

## Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit (light/dark) |
| Vector DB | Qdrant (local, Docker) |
| Embeddings | `BAAI/bge-base-en-v1.5` (local) |
| LLM | Ollama — Llama 3.1 8B |
| Eval | MLflow (local) |
| Deploy | Docker Compose |

---

## Quickstart

```bash
# 0. prerequisites: Docker Desktop + Ollama installed
ollama pull llama3.1:8b

# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. services
docker compose up -d qdrant            # Qdrant on :6333
open -a Ollama                         # or: ollama serve

# 3. ingest demo notes
python -m app.ingest                              # synthetic demo notes
python -m app.load_mtsamples --n 30               # + real MTSamples sample

# 4. run the app
uvicorn app.api:app --port 8000                   # backend  (:8000/docs = Swagger)
streamlit run frontend/streamlit_app.py           # UI       (http://localhost:8501)
```

Or try it in the terminal: `python chat.py`

---

## How it works (the details worth knowing)

- **Section-aware chunking** ([app/chunking.py](app/chunking.py)) splits notes on
  clinical section headers — handling *both* standalone-line headers and
  MTSamples' inline `HEADER:, content` format. Every chunk stores its exact
  `char_start`/`char_end`, which is what makes citations *verifiable* (re-slice
  the note and confirm the cited text is really there).
- **BGE asymmetry handled** ([app/embeddings.py](app/embeddings.py)): queries get
  the retrieval prefix, passages don't — a detail that quietly hurts recall if
  you get it wrong.
- **Qdrant over FAISS** because the app needs metadata filtering (scope to one
  patient's `note_id`) — a filtered vector query FAISS can't do natively.
- **Faithful citations** ([app/rag_pipeline.py](app/rag_pipeline.py)): parse `[n]`
  → verify against retrieved chunks → drop fabricated → resolve to provenance.

See [DEVLOG.md](DEVLOG.md) for the full build history and every bug found along
the way, and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for current limitations.

---

## ⚠️ Data hygiene

Built to be safe with clinical data under DUAs. **Never committed:** notes,
chunks, embeddings, logs, screenshots, model outputs — enforced by `.gitignore`
from commit #1. The public repo ships **code + a few synthetic demo notes only**.

## Roadmap / honest status

- ✅ Ingestion, RAG QA + citations, summarization, entity extraction
- ✅ FastAPI + Streamlit UI (host-run, verified)
- ✅ Evaluation + MLflow + relevance-gate tuning
- 📝 Docker Compose authored for the full stack; container images not yet built
  (host-run is the verified path). See DEVLOG.
- Not a medical device — a portfolio project on public sample data.

---

*Built by Anagha.*
