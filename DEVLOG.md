# Development Log

A running record of what was built, the decisions made, and the problems hit
along the way. Kept deliberately honest — the bugs and dead-ends are as much
part of the story as the features.

---

## Design decisions (locked before coding)

| Decision | Choice | Why |
|---|---|---|
| Dataset | **MTSamples** (dropped MIMIC & emrQA) | Free, public, no credentialing/DUA. MIMIC/emrQA both gated + leak-risk. Same pipeline works on real data later if wanted. |
| LLM | **Local (Ollama, Llama 3.1 8B)** | Privacy: no clinical text leaves the machine. Rules out cloud APIs. Trade-off: weaker model, handled with tighter prompts + grounding. |
| Embeddings | **BAAI/bge-base-en-v1.5** | Strong open retrieval model, runs locally. Handled its query/passage prefix asymmetry. |
| Vector DB | **Qdrant** | Needed metadata filtering (scope to a note) + local + free. FAISS lacks easy filtering; Pinecone is cloud. |
| UI | **Streamlit** (React as stretch) | Fastest path to a demo surface. Backend is HTTP-decoupled so UI is swappable. |
| Retrieval scope | **Multi-note, but single-note-by-default** (revised) | Originally multi-note; changed after a patient-mixing bug (see below). |
| Audience | **Portfolio piece** | Prioritize README, demo surface, faithfulness story, honest eval. |

---

## Build timeline

### Week 1 — Ingestion pipeline
- Built section-aware chunking, BGE embeddings, Qdrant store, ingestion runner.
- PHI-safe `.gitignore` from commit #1.
- **Verified:** 14 chunks from 2 synthetic notes; char-span verification 14/14
  (every chunk re-sliceable from the original — the basis for faithful citations).
- **Bug caught:** malformed namespace UUID crashed first ingest run; fixed.

### Week 2 — RAG QA + faithful citations + summarization
- Built retrieval, LLM client (Ollama), prompts, full RAG pipeline, summarization.
- **The differentiator — faithful citations:** LLM cites `[n]` markers → we verify
  each maps to a real retrieved chunk → drop fabricated ones.
- **Verified adversarially:** fed an answer citing a fabricated `[9]` (only 5 chunks
  retrieved) → system correctly kept real `[1]`, rejected `[9]`.
- **Verified refusal:** out-of-note question → "cannot find it", 0 citations (no hallucination).

### Real dataset integration (mid-project)
- Downloaded real MTSamples (4999 rows) from a public mirror.
- **Major bug caught — the reason we test on real data:** the chunker (built on
  synthetic notes with standalone-line headers) *completely failed* on real
  MTSamples, which uses **inline** headers (`HEADER:, content` separated by `.,`).
  Every real note collapsed into one "PREAMBLE" chunk.
- **Fix:** rewrote header detection to match headers at start-of-text / after
  newline / after `.,` / after `. `; rewrote `split_sections` as a whole-text
  regex scan. Verified: real notes now split into all sections, char-span 23/23;
  synthetic notes still work.
- Ingested 30 real notes (138 chunks).

### Week 3 — Entity extraction
- Built Pydantic schemas + extraction pipeline for 5 types (diagnoses,
  medications, allergies, procedures, follow-ups), each with source grounding.
- **Grounding = the extraction analog of citations:** each entity's verbatim
  `source_text` is verified to exist in the note; entities that don't match are
  flagged `grounded=False` rather than silently trusted.
- **Verified:** synthetic discharge note → all 5 categories, 21/21 grounded,
  fields parsed (dose/route/freq, allergen/reaction, dates, timeframes).
- **Minor fix:** grounding initially missed follow-ups where the LLM re-numbered
  list items ("2. Repeat BMP...") — added leading-list-marker stripping before
  matching. Back to 21/21.

### Patient-mixing bug + scope redesign
- **Bug found (user testing):** "summarize this note" / "is patient 3 on meds?"
  with no note selected → system searched ALL notes and blended multiple
  patients into one incoherent answer. **Worst-case error for a clinical tool.**
- Root cause: multi-note search with no enforced scope; "this note" was undefined.
- **Fix (hybrid model):** work one note at a time by default. `use <n>` selects a
  chart; questions/summary/extract operate on it; `all <q>` is an explicit,
  clearly-warned cross-patient search. No-selection questions are refused.
- **Verified:** no-selection refused; single-note answers no longer blend patients;
  `all` mode works with its ⚠ warning.
- **Portfolio angle:** found a safety bug in my own design and engineered
  safety-by-default around it.

### Friendly note naming
- Added `note_catalog.py`: display-only map of internal `note_id` → friendly
  title (e.g. `mt_0018_vasectomy-4` → "Urology — Vasectomy Op Note").
- Numbered menu; `use 1` selects by number; citations show friendly titles.
- Internal IDs unchanged everywhere (Qdrant/citations/spans intact).
- 8 curated well-structured notes = the tested demo path.

---

### Week 4 — FastAPI backend + Streamlit UI + Docker
- Built `app/api.py` (FastAPI): `/health`, `/notes`, `/ask`, `/summarize`,
  `/extract-entities`, `/upload-note`. Wraps the existing pipeline; uses the note
  catalog for friendly titles. **Single-patient scope enforced at the API level**:
  `/ask` requires a `note_id` unless `search_all=true` (returns 400 otherwise).
- Built `frontend/streamlit_app.py`: sidebar note-selector dropdown (the safety
  mechanism — you must pick one patient), explicit "search all notes" checkbox,
  live backend health, and three tabs (Ask with expandable citations / Summary /
  Extract with grounded ✓ vs ⚠ tables). Themed via `.streamlit/config.toml`.
- Decoupled architecture: UI holds no ML logic, talks to the API over HTTP.
- **Verified end-to-end (host-run):** API tested via curl (all endpoints, incl.
  the 400 on missing note_id); UI driven with a headless browser — asked "what
  medications is the patient on?" → answer with inline `[n]` citations + expandable
  Sources showing note/section/char-span. Screenshots captured.
- **Docker:** wrote `Dockerfile` + expanded `docker-compose.yml` (qdrant + api + ui;
  Ollama stays on host, reached via `host.docker.internal`). **Honest status:** the
  compose is authored but the container images were NOT built/tested this session
  (multi-GB torch build). The **host-run** stack (uvicorn + streamlit directly) is
  what's verified. Building/verifying the containers is a remaining task.

### Week 5 — Evaluation + relevance-gate tuning
- Built a hand-labeled gold set (`evals/gold_set.py`, 24 Q&A pairs: 19 answerable
  + 5 refusal cases incl. the "family history" trap) and an eval harness
  (`evals/rag_eval.py`) measuring correctness, citation accuracy, retrieval
  recall@k, refusal rate, false-refusal rate, latency. Logs to MLflow.
- **Baseline:** correctness 84.2%, citation acc 89.5%, recall 89.5%,
  **refusal-on-negatives 60%**, false-refusal 0%, latency 3.7s.
- **The over-answering fix — measured, not guessed:**
  - First tried a similarity-score threshold. Pulled the score distributions and
    found answerable (0.32–0.70) and unanswerable (0.43–0.60) **overlap heavily**
    — an unanswerable Q scored 0.60, above most real ones. A threshold would
    destroy the 0% false-refusal rate. **Rejected it.**
  - Instead added an **LLM relevance gate** (`_context_answers` in rag_pipeline.py
    + prompts): yes/no "does this context contain the answer?" before generating.
  - **Gate ON:** refusal-on-negatives 60% → **100%** (both traps fixed), but
    false-refusal 0% → 10.5% and latency 3.7s → 5.5s. A genuine safety/recall
    tradeoff.
- **Decision:** gate **ON by default** (`strict_grounding=True` in API AskRequest)
  — refusing a real question is safer than fabricating in a clinical tool. No UI
  toggle (user chose not to expose it). Verified: family-history trap now refuses;
  real questions still answer with citations.
- Wrote the polished portfolio README (arch diagram, eval table, the tradeoff
  story, quickstart) and `docs/architecture.md`.

## Environment notes
- Python 3.9 venv at `.venv/`. Docker Desktop + Ollama required at runtime;
  neither auto-starts on reboot (`open -a Docker`, `open -a Ollama`,
  `docker compose up -d qdrant`, then re-ingest).
- See `KNOWN_ISSUES.md` for current limitations and deferred fixes.
