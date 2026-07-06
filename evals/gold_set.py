"""Hand-labeled gold evaluation set for the Clinical RAG Assistant.

Each item is a question against ONE note, with:
  - expected_keywords: facts the answer MUST contain (correctness check)
  - expected_section:  the note section the citation SHOULD come from
                       (None for refusal cases)
  - should_refuse:     True if the answer is NOT in the note and the system
                       should decline rather than fabricate

Grounded in the real content of the curated demo notes (verified against the
ingested chunks). Includes deliberate negative/refusal cases — those are the
ones that test the anti-hallucination behavior, the project's core claim.

Kept in code (not CSV) so it's self-documenting and importable; it contains no
PHI (synthetic + public MTSamples snippets only).
"""
from __future__ import annotations

GOLD = [
    # ── discharge_001 (Heart Failure Discharge Summary) ──
    dict(note_id="discharge_001", question="What allergies does the patient have?",
         expected_keywords=["penicillin", "sulfa"], expected_section="ALLERGIES",
         should_refuse=False),
    dict(note_id="discharge_001", question="Why was the patient admitted?",
         expected_keywords=["heart failure"], expected_section="HOSPITAL COURSE",
         should_refuse=False),
    dict(note_id="discharge_001", question="What medications is the patient discharged on?",
         expected_keywords=["lisinopril", "furosemide"], expected_section="DISCHARGE MEDICATIONS",
         should_refuse=False),
    dict(note_id="discharge_001", question="What were the discharge diagnoses?",
         expected_keywords=["heart failure"], expected_section="DISCHARGE DIAGNOSES",
         should_refuse=False),
    dict(note_id="discharge_001", question="What is the chief complaint?",
         expected_keywords=["shortness of breath"], expected_section="CHIEF COMPLAINT",
         should_refuse=False),
    dict(note_id="discharge_001", question="What is the follow-up plan?",
         expected_keywords=["cardiology"], expected_section="FOLLOW-UP",
         should_refuse=False),
    # negative / refusal
    dict(note_id="discharge_001", question="What is the patient's COVID-19 vaccination status?",
         expected_keywords=[], expected_section=None, should_refuse=True),
    dict(note_id="discharge_001", question="What was the patient's blood pressure on admission?",
         expected_keywords=[], expected_section=None, should_refuse=True),

    # ── radiology_002 (Chest CT — Lung Nodule) ──
    dict(note_id="radiology_002", question="What did the CT chest show?",
         expected_keywords=["nodule"], expected_section="FINDINGS",
         should_refuse=False),
    dict(note_id="radiology_002", question="How big is the lung nodule?",
         expected_keywords=["2.3"], expected_section="FINDINGS",
         should_refuse=False),
    dict(note_id="radiology_002", question="What is the recommendation?",
         expected_keywords=["biopsy"], expected_section="IMPRESSION",
         should_refuse=False),
    # negative
    dict(note_id="radiology_002", question="What medications is the patient taking?",
         expected_keywords=[], expected_section=None, should_refuse=True),

    # ── mt_0000_allergic-rhinitis ──
    dict(note_id="mt_0000_allergic-rhinitis", question="What is the assessment?",
         expected_keywords=["allergic rhinitis"], expected_section="ASSESSMENT",
         should_refuse=False),
    dict(note_id="mt_0000_allergic-rhinitis", question="What medications is the patient currently on?",
         expected_keywords=["allegra"], expected_section="MEDICATIONS",
         should_refuse=False),
    dict(note_id="mt_0000_allergic-rhinitis", question="Does the patient have any known drug allergies?",
         expected_keywords=["no known"], expected_section="ALLERGIES",
         should_refuse=False),
    dict(note_id="mt_0000_allergic-rhinitis", question="What was the patient's blood pressure?",
         expected_keywords=["124/78"], expected_section="OBJECTIVE",
         should_refuse=False),

    # ── mt_0018_vasectomy-4 ──
    dict(note_id="mt_0018_vasectomy-4", question="What procedure was performed?",
         expected_keywords=["vasectomy"], expected_section="PROCEDURE",
         should_refuse=False),
    dict(note_id="mt_0018_vasectomy-4", question="Were there any complications?",
         expected_keywords=["none"], expected_section="COMPLICATIONS",
         should_refuse=False),
    dict(note_id="mt_0018_vasectomy-4", question="What was the estimated blood loss?",
         expected_keywords=["minimal"], expected_section="BLOOD LOSS",
         should_refuse=False),
    # negative — the "family history" trap (should refuse; note has no family Hx)
    dict(note_id="mt_0018_vasectomy-4", question="What is the patient's family medical history?",
         expected_keywords=[], expected_section=None, should_refuse=True),

    # ── mt_0027_umbilical-hernia-repair ──
    dict(note_id="mt_0027_umbilical-hernia-repair", question="What was the preoperative diagnosis?",
         expected_keywords=["umbilical hernia"], expected_section="PREOPERATIVE DIAGNOSIS",
         should_refuse=False),
    dict(note_id="mt_0027_umbilical-hernia-repair", question="What anesthesia was used?",
         expected_keywords=["general"], expected_section="ANESTHESIA",
         should_refuse=False),
    dict(note_id="mt_0027_umbilical-hernia-repair", question="What procedure was performed?",
         expected_keywords=["repair", "umbilical hernia"], expected_section="PROCEDURE PERFORMED",
         should_refuse=False),
    # negative
    dict(note_id="mt_0027_umbilical-hernia-repair", question="What are the patient's discharge medications?",
         expected_keywords=[], expected_section=None, should_refuse=True),
]
