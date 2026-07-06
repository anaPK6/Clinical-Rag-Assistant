"""Section-aware chunking for clinical notes.

Clinical notes (MTSamples, discharge summaries, radiology reports) are
semi-structured: they have section headers like CHIEF COMPLAINT,
HOSPITAL COURSE, DISCHARGE MEDICATIONS, IMPRESSION, etc.

Strategy:
  1. Detect section headers and split the note into sections.
  2. Within each oversized section, sub-split on size with overlap, on
     paragraph/sentence boundaries where possible.

Every emitted chunk carries:
  - note_id     : which note it came from
  - section     : the clinical section it belongs to
  - chunk_index : ordinal within the note
  - char_start / char_end : exact offsets into the ORIGINAL note text

The char span is what makes citations *verifiable* later (Week 2): we can
re-slice the original note and confirm the cited text actually exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List

from app.config import MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS

# A header line is a short, mostly-uppercase line, optionally ending in ":".
# Matches "CHIEF COMPLAINT:", "HOSPITAL COURSE", "DISCHARGE MEDICATIONS:",
# "PAST MEDICAL HISTORY", "IMPRESSION:", "FINDINGS:", etc.
#
# Two real-world formats must both work:
#   (a) standalone-line headers  ->  "CHIEF COMPLAINT:\n<content>"  (our synthetic notes)
#   (b) inline headers           ->  "SUBJECTIVE:,  <content>..."   (real MTSamples)
#
# We detect a header as an UPPERCASE run of >=3 chars ending in a colon, that
# either starts the text or follows a newline/period/space. The regex scans the
# whole note so it catches inline headers; char offsets are taken from match
# positions so spans stay exact.
# A header begins at one of these boundaries:
#   - start of the note
#   - after a newline                      (standalone-line headers, synthetic)
#   - after ".," or ". "                    (inline headers, real MTSamples uses ".,")
# followed by UPPERCASE header text and a colon.
_HEADER_RE = re.compile(
    r"(?:^|(?<=\n)|(?<=\.,)|(?<=\.\s))"     # allowed header boundaries
    r"[ \t]*"
    r"(?P<header>[A-Z][A-Z0-9 /&'\-()]{2,60}?)"  # uppercase header text
    r"[ \t]*:"                              # required trailing colon
)


@dataclass
class Chunk:
    note_id: str
    section: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    def to_payload(self) -> dict:
        """Qdrant point payload. text is included so retrieval returns the
        snippet without a second lookup."""
        return asdict(self)


# A header candidate is rejected if its text has too many lowercase letters
# (i.e. it's really a sentence like "Note: the patient..."), guarding against
# false positives.
def _is_real_header(header_text: str) -> bool:
    stripped = header_text.strip()
    if not (3 <= len(stripped) <= 60):
        return False
    lowercase = sum(c.islower() for c in stripped)
    return lowercase <= 2


def split_sections(text: str) -> List[tuple]:
    """Split note text into (section_name, section_text, start_offset) tuples.

    Scans for header markers ("HEADER:") anywhere they can legally begin — at
    the start of the note, or after a newline or period. This handles both
    standalone-line headers and inline "HEADER:, content" (real MTSamples).

    Content before the first detected header is assigned "PREAMBLE".
    Offsets index into the original `text`, so char spans stay exact.
    """
    # Collect valid header match positions.
    boundaries = []  # (match_start, header_name, content_start)
    for m in _HEADER_RE.finditer(text):
        name = m.group("header").strip()
        if _is_real_header(name):
            boundaries.append((m.start("header"), name, m.end()))

    sections: List[tuple] = []

    # PREAMBLE: anything before the first header.
    first = boundaries[0][0] if boundaries else len(text)
    if text[:first].strip():
        sections.append(("PREAMBLE", text[:first], 0))

    # Each header's section runs from its content start to the next header.
    for i, (_, name, content_start) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[content_start:end]
        if body.strip():
            sections.append((name, body, content_start))

    return sections


def _split_oversized(body: str, base_offset: int):
    """Yield (text, start, end) windows for a section body that exceeds
    MAX_CHUNK_CHARS. Splits on paragraph boundaries first, falling back to a
    hard character window with overlap. Offsets are absolute (into the note).
    """
    if len(body) <= MAX_CHUNK_CHARS:
        yield body, base_offset, base_offset + len(body)
        return

    # Prefer to break on blank lines (paragraphs); accumulate until the cap.
    paras = re.split(r"(\n\s*\n)", body)  # keep delimiters to preserve offsets
    window = ""
    window_start = base_offset
    cursor = base_offset

    for piece in paras:
        if window and len(window) + len(piece) > MAX_CHUNK_CHARS:
            yield window, window_start, window_start + len(window)
            # start next window with a tail overlap for context continuity
            overlap = window[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
            window = overlap + piece
            window_start = cursor - len(overlap)
        else:
            window += piece
        cursor += len(piece)

    if window.strip():
        yield window, window_start, window_start + len(window)


def chunk_note(note_id: str, text: str) -> List[Chunk]:
    """Chunk one note into section-aware chunks with verifiable char spans."""
    chunks: List[Chunk] = []
    idx = 0
    for section_name, body, start in split_sections(text):
        for sub_text, c_start, c_end in _split_oversized(body, start):
            if not sub_text.strip():
                continue
            chunks.append(
                Chunk(
                    note_id=note_id,
                    section=section_name,
                    chunk_index=idx,
                    text=sub_text.strip(),
                    char_start=c_start,
                    char_end=c_end,
                )
            )
            idx += 1
    return chunks
