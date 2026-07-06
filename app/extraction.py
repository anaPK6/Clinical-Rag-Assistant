"""Structured clinical entity extraction with source grounding (Week 3).

Flow:
  1. Load the full note text (from Qdrant chunks, in order).
  2. Ask the LLM for schema-constrained JSON (5 entity categories), each
     entity carrying a verbatim `source_text`.
  3. GROUND each entity: verify its source_text actually appears in the note
     and attach the char span. Entities whose source_text can't be located are
     kept but flagged grounded=False — so downstream/UI can show which
     extractions are verified vs. unverified. This is the extraction analog of
     Week 2's citation verification.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.llm import generate_json
from app.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE
from app.schemas import ExtractedEntity, ExtractionResult
from app.rag_pipeline import _load_note_chunks


_CATEGORY_KEYS = {
    "diagnoses": "diagnoses",
    "medications": "medications",
    "allergies": "allergies",
    "procedures": "procedures",
    "follow_ups": "follow_ups",
}
# The primary display field per category (used for logging / dedup).
_PRIMARY_FIELD = {
    "diagnoses": "name",
    "medications": "name",
    "allergies": "substance",
    "procedures": "name",
    "follow_ups": "instruction",
}


def _reconstruct_note(note_id: str) -> str:
    """Rebuild the full note text from its ordered chunks. Chunks were stored
    stripped, so we join with newlines — grounding uses normalized matching so
    exact whitespace doesn't matter."""
    chunks = _load_note_chunks(note_id)
    return "\n".join(c.text for c in chunks)


def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase for tolerant substring matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _ground(source_text: str, note_text: str, note_norm: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """Return (grounded, char_start, char_end). First try an exact substring
    match; fall back to a normalized-whitespace match to tolerate minor LLM
    reformatting."""
    if not source_text:
        return False, None, None

    # Exact match first (gives precise offsets).
    idx = note_text.find(source_text)
    if idx != -1:
        return True, idx, idx + len(source_text)

    # Strip a leading list marker the LLM may have added ("2. ", "- ", "* ")
    # that isn't part of the note's own text at that position, then retry exact.
    stripped = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", source_text)
    if stripped != source_text:
        idx = note_text.find(stripped)
        if idx != -1:
            return True, idx, idx + len(stripped)

    # Normalized match: locate the normalized source within the normalized note,
    # then map back approximately by searching for the first few words.
    src_norm = _normalize(source_text)
    if src_norm and src_norm in note_norm:
        # Approximate span: find the first distinctive word of the source.
        first_words = src_norm.split(" ")[:3]
        probe = first_words[0] if first_words else ""
        approx = note_text.lower().find(probe)
        if approx != -1:
            return True, approx, approx + len(source_text)
        return True, None, None  # grounded but span unknown

    return False, None, None


def _build_entities(category: str, items: list, note_text: str, note_norm: str) -> List[ExtractedEntity]:
    entities: List[ExtractedEntity] = []
    if not isinstance(items, list):
        return entities
    for item in items:
        if not isinstance(item, dict):
            continue
        source_text = str(item.get("source_text") or "")
        grounded, cs, ce = _ground(source_text, note_text, note_norm)
        data = {k: v for k, v in item.items() if k != "source_text"}
        entities.append(
            ExtractedEntity(
                type=category.rstrip("s") if category != "follow_ups" else "follow_up",
                data=data,
                source_text=source_text,
                grounded=grounded,
                char_start=cs,
                char_end=ce,
            )
        )
    return entities


def extract_entities(note_id: str) -> ExtractionResult:
    """Extract all 5 entity categories for one note, grounded to source spans."""
    note_text = _reconstruct_note(note_id)
    if not note_text.strip():
        return ExtractionResult(note_id=note_id)

    note_norm = _normalize(note_text)
    prompt = EXTRACTION_USER_TEMPLATE.format(note_text=note_text)
    raw = generate_json(prompt, system=EXTRACTION_SYSTEM)

    result = ExtractionResult(note_id=note_id)
    result.diagnoses = _build_entities("diagnoses", raw.get("diagnoses", []), note_text, note_norm)
    result.medications = _build_entities("medications", raw.get("medications", []), note_text, note_norm)
    result.allergies = _build_entities("allergies", raw.get("allergies", []), note_text, note_norm)
    result.procedures = _build_entities("procedures", raw.get("procedures", []), note_text, note_norm)
    result.follow_ups = _build_entities("follow_ups", raw.get("follow_ups", []), note_text, note_norm)
    return result
