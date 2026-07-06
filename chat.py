"""Interactive CLI to try the clinical RAG assistant by hand.

Run:  python chat.py

Works one note (one patient chart) at a time by default — this prevents
answers from accidentally blending facts across different patients.

Commands:
  use <n>              select a note by its menu number (e.g. 'use 1')
  <question>           ask about the SELECTED note (with citations)
  summary [n]          structured summary of the selected (or given) note
  extract [n]          extract dx/meds/allergies/procedures/follow-ups
  all <question>       DELIBERATELY search across ALL notes (cross-patient!)
  notes                show the numbered note menu
  quit                 exit
"""
from __future__ import annotations

from app.rag_pipeline import answer_question, summarize_note
from app.extraction import extract_entities
from app.vector_store import get_client
from app.config import QDRANT_COLLECTION
from app.note_catalog import menu, resolve, title_for


def list_note_ids():
    client = get_client()
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION, limit=1000,
        with_payload=True, with_vectors=False,
    )
    return sorted({p.payload["note_id"] for p in points})


def print_answer(r):
    print("\n" + "-" * 68)
    print(r.answer.strip())
    print("-" * 68)
    if r.citations:
        print("SOURCES:")
        for c in r.citations:
            print(f"  [{c.marker}] {title_for(c.note_id)} / {c.section}  (chars {c.char_start}-{c.char_end})")
            print(f"       “{c.snippet[:90].strip()}”")
    else:
        print("SOURCES: (none — answer not grounded in the notes)")
    if r.dropped_markers:
        print(f"  ⚠ rejected fabricated citations: {r.dropped_markers}")
    print()


def print_extraction(r):
    print(f"\nEntities for {r.note_id}  {r.counts()}")
    labels = [
        ("diagnoses", "DIAGNOSES"), ("medications", "MEDICATIONS"),
        ("allergies", "ALLERGIES"), ("procedures", "PROCEDURES"),
        ("follow_ups", "FOLLOW-UPS"),
    ]
    for attr, title in labels:
        ents = getattr(r, attr)
        if not ents:
            continue
        print(f"\n{title}:")
        for e in ents:
            primary = next(iter(e.data.values()), "?") if e.data else "?"
            extra = ", ".join(f"{k}={v}" for k, v in e.data.items()
                               if k not in (list(e.data)[:1]) and v)
            mark = "✓" if e.grounded else "⚠ unverified"
            print(f"  • {primary}" + (f"  ({extra})" if extra else "") + f"   [{mark}]")


def main():
    selected = None  # the note (patient chart) currently in focus; None = none picked
    print("Clinical RAG assistant — interactive mode. Type 'help' for commands.")
    print("Pick a note with 'use <number>' before asking (e.g. 'use 1').\n")
    print(menu())
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if not line:
            continue

        if line in ("quit", "exit", "q"):
            print("bye"); return
        if line == "help":
            print(__doc__); continue
        if line == "notes":
            print(menu()); continue

        # ── select which note (patient chart) you're working on ──
        if line.startswith("use "):
            nid = resolve(line[4:].strip())
            if not nid or nid not in list_note_ids():
                print(f"Couldn't find that note. Type 'notes' to see the list.")
            else:
                selected = nid
                print(f"Now working on: {title_for(selected)}")
            continue

        # ── explicit cross-note search (labeled cross-patient) ──
        if line.startswith("all "):
            q = line[4:].strip()
            print("  ⚠ Searching ACROSS ALL NOTES — answers may span multiple patients.")
            r = answer_question(q, note_ids=None)
            print_answer(r)
            continue

        if line.startswith("summary"):
            arg = line[8:].strip() if len(line) > 8 else ""
            nid = resolve(arg) if arg else selected
            if not nid:
                print("No note selected. Type 'use <n>' first, or 'summary <n>'.")
            else:
                print(f"\n[{title_for(nid)}]\n" + summarize_note(nid))
            continue
        if line.startswith("extract"):
            arg = line[8:].strip() if len(line) > 8 else ""
            nid = resolve(arg) if arg else selected
            if not nid:
                print("No note selected. Type 'use <n>' first, or 'extract <n>'.")
            else:
                print_extraction(extract_entities(nid))
            continue

        # ── a bare line (or 'ask ...') is a question about the SELECTED note ──
        q = line[4:].strip() if line.startswith("ask ") else line
        if not selected:
            print("No note selected. Type 'use <note_id>' to pick a patient's note first,")
            print("or 'all <question>' to deliberately search across every note.")
            continue

        r = answer_question(q, note_ids=[selected])
        print_answer(r)


if __name__ == "__main__":
    main()
