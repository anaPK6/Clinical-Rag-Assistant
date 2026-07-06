"""Human-friendly display names for demo notes.

The internal note_id (used in Qdrant, citations, char-spans) never changes.
This is a DISPLAY layer only: a curated map of note_id -> friendly title, so a
user can type a short number/title instead of "mt_0018_vasectomy-4".

CURATED_NOTES lists the handful of well-structured demo notes we want front and
center (diverse specialties, clean chunking). Notes not in this list still work
by their raw id; they just aren't in the numbered menu.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# Ordered: (note_id, friendly title). Order = the number the user types.
CURATED_NOTES: List[Tuple[str, str]] = [
    ("discharge_001",                 "Heart Failure Discharge Summary"),
    ("radiology_002",                 "Chest CT — Lung Nodule"),
    ("mt_0000_allergic-rhinitis",     "Allergic Rhinitis Clinic Visit"),
    ("mt_0012_moyamoya-disease",      "Neurology — Moyamoya Disease"),
    ("mt_0018_vasectomy-4",           "Urology — Vasectomy Op Note"),
    ("mt_0027_umbilical-hernia-repair","Umbilical Hernia Repair Op Note"),
    ("mt_0019_airway-compromise-foreign-body", "Airway Obstruction — Foreign Body"),
    ("mt_0014_bony-impacted-teeth-removal",    "Oral Surgery — Impacted Teeth"),
]


def menu() -> str:
    """Numbered, human-readable menu of the curated demo notes."""
    lines = ["Available notes:"]
    for i, (_, title) in enumerate(CURATED_NOTES, start=1):
        lines.append(f"  {i}. {title}")
    return "\n".join(lines)


def resolve(ref: str) -> Optional[str]:
    """Resolve a user reference to an internal note_id.

    Accepts: a menu number ("1"), a friendly title (case-insensitive,
    substring ok), or a raw note_id passed straight through.
    Returns the note_id, or None if it can't be resolved.
    """
    ref = ref.strip()
    if not ref:
        return None

    # 1) menu number
    if ref.isdigit():
        idx = int(ref) - 1
        if 0 <= idx < len(CURATED_NOTES):
            return CURATED_NOTES[idx][0]
        return None

    # 2) exact raw note_id
    for nid, _ in CURATED_NOTES:
        if ref == nid:
            return nid

    # 3) title match (substring, case-insensitive)
    low = ref.lower()
    for nid, title in CURATED_NOTES:
        if low in title.lower():
            return nid

    # 4) fall through: assume it's a raw id we just don't have curated
    return ref


def title_for(note_id: str) -> str:
    """Friendly title for a note_id, or the raw id if uncurated."""
    for nid, title in CURATED_NOTES:
        if nid == note_id:
            return title
    return note_id
