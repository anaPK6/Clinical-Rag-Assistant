"""Ingestion runner: note(s) -> chunks -> BGE embeddings -> Qdrant.

Usage:
    # ingest every note in data/sample_notes/
    python -m app.ingest

    # ingest a specific file or directory
    python -m app.ingest path/to/note.txt
    python -m app.ingest data/sample_notes/

The note_id is derived from the filename stem (minus a trailing
".synthetic"), so discharge_001.synthetic.txt -> note_id "discharge_001".
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from app.chunking import chunk_note
from app.embeddings import embed_passages
from app.vector_store import (
    ensure_collection,
    upsert_chunks,
    delete_note,
    count,
    get_client,
)

DEFAULT_DIR = Path("data/sample_notes")


def note_id_from_path(path: Path) -> str:
    stem = path.stem  # drops .txt
    if stem.endswith(".synthetic"):
        stem = stem[: -len(".synthetic")]
    return stem


def ingest_file(path: Path, client=None) -> int:
    """Ingest one note file. Returns the chunk count stored."""
    text = path.read_text(encoding="utf-8", errors="replace")
    note_id = note_id_from_path(path)

    chunks = chunk_note(note_id, text)
    if not chunks:
        print(f"  [skip] {path.name}: no chunks produced")
        return 0

    vectors = embed_passages([c.text for c in chunks])

    # Replace any prior version of this note, then upsert fresh chunks.
    delete_note(note_id, client=client)
    n = upsert_chunks(chunks, vectors, client=client)

    sections = sorted({c.section for c in chunks})
    print(f"  [ok]   {path.name}: note_id={note_id!r}  chunks={n}")
    print(f"         sections: {', '.join(sections)}")
    return n


def collect_files(arg: str | None) -> List[Path]:
    target = Path(arg) if arg else DEFAULT_DIR
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.glob("*.txt"))
    raise SystemExit(f"Path not found: {target}")


def main(argv: List[str]) -> None:
    files = collect_files(argv[0] if argv else None)
    if not files:
        raise SystemExit("No .txt notes found to ingest.")

    client = get_client()
    ensure_collection(client)

    print(f"Ingesting {len(files)} note(s) into Qdrant...")
    total = 0
    for f in files:
        total += ingest_file(f, client=client)

    print(f"\nDone. {total} chunks ingested this run.")
    print(f"Collection now holds {count(client)} total chunks.")


if __name__ == "__main__":
    main(sys.argv[1:])
