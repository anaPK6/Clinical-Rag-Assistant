"""Pydantic schemas for structured clinical entity extraction (Week 3).

Design principle (same as citations): every extracted entity carries the
verbatim `source_text` it was drawn from. After the LLM returns JSON, we
verify that source_text actually appears in the note and attach its char
span — so extractions are grounded, not hallucinated. Entities whose
source_text can't be found in the note are flagged (not silently trusted).

These models also serve as the API response schemas in Week 4.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    name: str = Field(description="the diagnosis or condition")
    status: Optional[str] = Field(
        default=None, description="e.g. active, resolved, chronic, suspected"
    )
    source_text: str = Field(description="verbatim snippet from the note")


class Medication(BaseModel):
    name: str = Field(description="drug name")
    dose: Optional[str] = Field(default=None, description="e.g. 20 mg")
    route: Optional[str] = Field(default=None, description="e.g. PO, IV")
    frequency: Optional[str] = Field(default=None, description="e.g. daily, BID")
    source_text: str = Field(description="verbatim snippet from the note")


class Allergy(BaseModel):
    substance: str = Field(description="the allergen")
    reaction: Optional[str] = Field(default=None, description="e.g. rash, hives")
    source_text: str = Field(description="verbatim snippet from the note")


class Procedure(BaseModel):
    name: str = Field(description="the procedure performed or planned")
    date: Optional[str] = Field(default=None, description="date if stated")
    source_text: str = Field(description="verbatim snippet from the note")


class FollowUp(BaseModel):
    instruction: str = Field(description="the follow-up action or plan")
    timeframe: Optional[str] = Field(default=None, description="e.g. 1 week")
    source_text: str = Field(description="verbatim snippet from the note")


class ExtractedEntity(BaseModel):
    """One entity after grounding: the parsed fields plus provenance."""
    type: str  # diagnosis | medication | allergy | procedure | follow_up
    data: dict  # the entity's fields (name/dose/etc.)
    source_text: str
    grounded: bool  # True if source_text was found verbatim in the note
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class ExtractionResult(BaseModel):
    note_id: str
    diagnoses: List[ExtractedEntity] = []
    medications: List[ExtractedEntity] = []
    allergies: List[ExtractedEntity] = []
    procedures: List[ExtractedEntity] = []
    follow_ups: List[ExtractedEntity] = []

    def counts(self) -> dict:
        return {
            "diagnoses": len(self.diagnoses),
            "medications": len(self.medications),
            "allergies": len(self.allergies),
            "procedures": len(self.procedures),
            "follow_ups": len(self.follow_ups),
        }
