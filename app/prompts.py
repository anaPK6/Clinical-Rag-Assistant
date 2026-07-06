"""Prompt templates for the clinical RAG assistant."""
from __future__ import annotations

# ── QA with citations ──
# The model is told to answer ONLY from the numbered context and to cite the
# chunk numbers it used with [n] markers. We verify those markers afterward.
QA_SYSTEM = (
    "You are a careful clinical assistant. Answer the question using ONLY the "
    "numbered context passages provided. Do not use outside knowledge. "
    "If the answer is not in the context, say you cannot find it in the note. "
    "After each sentence or claim, cite the passage number(s) it comes from "
    "using square brackets, e.g. [1] or [2][3]. Be concise and factual."
)

QA_USER_TEMPLATE = (
    "Context passages:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer (with [n] citations):"
)


def format_context(chunks) -> str:
    """Render retrieved chunks as a numbered list for the prompt.
    Numbering is 1-based and is the citation key the model must use."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (note: {c.note_id}, section: {c.section})\n{c.text}")
    return "\n\n".join(lines)


# ── Summarization ──
SUMMARY_SYSTEM = (
    "You are a clinical documentation assistant. Produce a concise, structured "
    "summary of the clinical note using ONLY the provided context. Use short "
    "labeled sections. Do not invent information not present in the context."
)

SUMMARY_USER_TEMPLATE = (
    "Clinical note context:\n"
    "{context}\n\n"
    "Write a structured summary with these headers where information exists: "
    "Reason for Visit, Key History, Hospital Course / Findings, Diagnoses, "
    "Medications, Follow-up."
)


# ── Relevance gate (Week 5) ──
# A cheap pre-check: does the retrieved context actually CONTAIN the answer?
# This guards against over-answering when retrieval surfaces loosely-related
# text (e.g. "completed family" for a "family history" question). Judges
# semantic relevance, which a similarity-score threshold cannot (scores of
# answerable vs unanswerable questions overlap too much — measured on the gold
# set). Returns strict JSON so we can act on it deterministically.
RELEVANCE_GATE_SYSTEM = (
    "You are a strict relevance judge. Given a question and context passages, "
    "decide whether the context DIRECTLY contains the information needed to "
    "answer the question. A passage that merely mentions a related word is NOT "
    "enough — the specific answer must be present. Respond with JSON only: "
    '{"answerable": true|false}.'
)

RELEVANCE_GATE_USER = (
    "Question: {question}\n\n"
    "Context passages:\n{context}\n\n"
    "Does the context directly contain the answer? JSON only."
)


# ── Entity extraction (Week 3) ──
# The model returns JSON only. Every entity must include a `source_text` copied
# VERBATIM from the note so we can verify it exists (grounding). Empty lists are
# required when a category is absent — never invent entries.
EXTRACTION_SYSTEM = (
    "You are a clinical information extraction system. Extract structured "
    "entities from the clinical note. Return ONLY valid JSON matching the "
    "requested schema. For every entity, copy the `source_text` field VERBATIM "
    "from the note (an exact substring) — do not paraphrase it. If a category "
    "has no entries in the note, return an empty list for it. Never invent "
    "information that is not present in the note."
)

EXTRACTION_USER_TEMPLATE = (
    "Clinical note:\n"
    "\"\"\"\n{note_text}\n\"\"\"\n\n"
    "Extract the following and return a JSON object with EXACTLY these keys:\n"
    "  \"diagnoses\":   [ {{\"name\", \"status\", \"source_text\"}} ]\n"
    "  \"medications\": [ {{\"name\", \"dose\", \"route\", \"frequency\", \"source_text\"}} ]\n"
    "  \"allergies\":   [ {{\"substance\", \"reaction\", \"source_text\"}} ]\n"
    "  \"procedures\":  [ {{\"name\", \"date\", \"source_text\"}} ]\n"
    "  \"follow_ups\":  [ {{\"instruction\", \"timeframe\", \"source_text\"}} ]\n\n"
    "Use null for any field whose value is not stated. `source_text` must be an "
    "exact substring of the note above. Return JSON only, no commentary."
)
