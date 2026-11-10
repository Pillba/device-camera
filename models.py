from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OcrCandidate:
    text: str
    confidence: Optional[float] = None
    source: str = ""


@dataclass(frozen=True)
class FieldConfidence:
    score: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class MedicationRecord:
    medicine_name: str = ""
    strength: str = ""
    dosage_form: str = ""
    instructions: str = ""
    quantity: str = ""
    refills: str = ""
    raw_text: str = ""
    confidence: Dict[str, FieldConfidence] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "medicine_name": self.medicine_name,
            "instructions": self.instructions,
            "quantity": self.quantity,
            "raw_text": self.raw_text,
        }
        optional = {
            "strength": self.strength,
            "dosage_form": self.dosage_form,
            "refills": self.refills,
        }
        data.update({key: value for key, value in optional.items() if value})
        if self.confidence:
            data["confidence"] = {
                key: {"score": value.score, "evidence": value.evidence}
                for key, value in self.confidence.items()
            }
        if self.warnings:
            data["warnings"] = self.warnings
        return data
