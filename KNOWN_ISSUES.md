# Known Issues & Limitations

An honest, current list of what doesn't work perfectly, why, and what would fix
it. Documenting these is deliberate: being able to see and explain a model's
limits (thanks to citations/grounding) is a feature of this project, not an
oversight. Nothing here is hidden behind a happy-path demo.

Status key: 🔴 open · 🟡 mitigated/partial · 🟢 has a planned fix

---

## 1. 🟢 Loose semantic match → over-answering ("family history" case) — MITIGATED
**Symptom.** On the vasectomy note, asking *"what is the patient's family
history?"* returns *"The patient has a completed family. [1][2]"* — citing real
text ("Fertile male with completed family"). But "completed family" (family-
planning status) is **not** "family history" (relatives' medical history), which
this note doesn't contain. It should have refused, like it correctly does for
"any allergies?" on the same note.

**Cause.** Pure vector retrieval matched the surface word "family"; the 8B model
answered from the closest chunk instead of judging that the note doesn't actually
address the question. Presence of a related word ≠ relevance.

**Note:** the citations are still honest — the answer *is* grounded in real, cited
text. The failure is one of *intent/relevance*, not fabrication. The citation
layer is what lets a reader catch the mismatch.

**RESOLVED in Week 5 (with measured data):**
- Tried a **score threshold** first — rejected it: answerable (0.32–0.70) and
  unanswerable (0.43–0.60) similarity scores overlap too much to separate.
- Shipped an **LLM relevance gate** instead (ON by default). Refusal on
  unanswerable questions: 60% → **100%**. New cost: **false-refusal rate 10.5%**
  (see issue #6) and +~2s latency. For a clinical tool, refusing a real question
  beats fabricating an answer to a fake one, so the gate is the default.

---

## 6. 🟡 Relevance gate causes ~10% false refusals
**Symptom.** With the gate on (default), the 8B relevance judge occasionally
labels an answerable question "not answerable" and refuses it (measured 10.5% on
the gold set — e.g. "what were the discharge diagnoses?" wrongly refused).

**Cause.** Small model as a strict relevance judge is over-cautious.

**Trade-off (deliberate).** This is the accepted cost of eliminating fabrication
on unanswerable questions. A larger model or a two-vote gate would likely reduce
it. The gate can be disabled per-request (`strict_grounding=false`) to recover
the 0% false-refusal behavior at the cost of over-answering.

---

## 2. 🟡 8B model paraphrases `source_text` on some extractions
**Symptom.** On messier real notes, some extracted entities are flagged
`⚠ unverified` because the model returned a paraphrased `source_text` that isn't
a verbatim substring of the note (saw ~3/6 on one real note; 21/21 on the clean
synthetic note).

**Cause.** Expected limitation of a local 8B model — it doesn't always copy
source text exactly. Flagged at project start.

**Mitigation.** The grounding layer *surfaces* this honestly (✓ vs ⚠) rather than
hiding it. Unverified entities are still shown, just not vouched for. A stronger
model or stricter prompt would raise the grounded rate.

---

## 3. 🟡 Retrieval ranking weak on list-style sections
**Symptom.** Questions like "what medications is the patient on?" sometimes rank a
generic header or history chunk above the actual medication list (the list still
appears within top-k, so the LLM sees it).

**Cause.** Pure dense vector search on short, list-heavy clinical text.

**Planned improvement.** Hybrid search (dense + keyword) and/or a reranker.

---

## 4. 🟡 Header-format long tail → some notes chunk coarsely
**Symptom.** A few real notes (e.g. some echocardiograms, the tracheostomy note)
fall into one large "PREAMBLE" chunk because their formatting doesn't match the
header patterns. Citations on those point to a big span (still valid, just coarse).

**Cause.** MTSamples has more header-format variety in its long tail than the
patterns cover. The 8 curated demo notes are chosen to avoid this.

**Planned improvement.** Broaden header detection; add a size-based fallback split
for header-less notes.

---

## 5. 🟢 Services don't auto-start on reboot
**Symptom.** After a reboot, the app can't reach Qdrant/Ollama.
**Fix (documented):** `open -a Docker && open -a Ollama`, wait ~15s,
`docker compose up -d qdrant`, then re-ingest. Will be folded into a single
`make up` / compose target in Week 4.

---

## Scope / honesty notes
- This is a **portfolio project on public sample data (MTSamples)**, not a medical
  device. It summarizes/retrieves what is *written* in a note; it does not
  diagnose or advise.
- Answers are non-deterministic (LLM); facts should be stable, wording varies.
- Only a curated subset of notes is ingested by default (30–100), not all ~5000.
