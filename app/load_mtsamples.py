"""Load real MTSamples notes from the CSV and ingest them into Qdrant.

MTSamples CSV columns: description, medical_specialty, sample_name,
transcription, keywords. We use `transcription` as the note text.

Usage:
    # ingest a sample of N notes (default 30)
    python -m app.load_mtsamples

    # ingest N notes, optionally filtered to a specialty
    python -m app.load_mtsamples --n 50 --specialty Radiology

We ingest a SAMPLE, not all ~5000 notes: enough to make the demo realistic
without a long embed run. note_id is derived from the row index + a slug so
it's stable and human-readable.

Reminder: data/ is gitignored — MTSamples must not be committed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from app.chunking import chunk_note
from app.embeddings import embed_passages
from app.vector_store import ensure_collection, upsert_chunks, delete_note, count, get_client

CSV_PATH = Path("data/mtsamples/mtsamples.csv")


def _slug(text: str, maxlen: int = 30) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:maxlen] or "note"


def ingest_sample(n: int = 30, specialty: str | None = None) -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"MTSamples CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["transcription"].notna()]
    if specialty:
        df = df[df["medical_specialty"].str.contains(specialty, case=False, na=False)]
        if df.empty:
            raise SystemExit(f"No notes found for specialty {specialty!r}")

    df = df.head(n)
    client = get_client()
    ensure_collection(client)

    print(f"Ingesting {len(df)} MTSamples note(s)"
          f"{f' [{specialty}]' if specialty else ''}...")

    total_chunks = 0
    for row_idx, row in df.iterrows():
        note_id = f"mt_{row_idx:04d}_{_slug(row.get('sample_name', row_idx))}"
        text = str(row["transcription"])

        chunks = chunk_note(note_id, text)
        if not chunks:
            continue
        vectors = embed_passages([c.text for c in chunks])
        delete_note(note_id, client=client)
        upsert_chunks(chunks, vectors, client=client)
        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} chunks from {len(df)} notes ingested.")
    print(f"Collection now holds {count(client)} total chunks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of notes to ingest")
    ap.add_argument("--specialty", type=str, default=None, help="filter by specialty substring")
    args = ap.parse_args()
    ingest_sample(n=args.n, specialty=args.specialty)


if __name__ == "__main__":
    main()
