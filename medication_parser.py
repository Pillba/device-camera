import difflib
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from models import FieldConfidence, MedicationRecord
from text_normalizer import normalize_text, normalized_search_text


DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|g|iu|h)\b", re.IGNORECASE
)
COMPOUND_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?-\d+(?:\.\d+)?\s?(?:mg|mcg|ml|g|iu)\b",
    re.IGNORECASE,
)
QUANTITY_PATTERNS = (
    re.compile(r"\b(?:qty|quantity)\.?\s*:?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"#\s?(\d+)\b"),
    re.compile(r"\b(\d+)\s?(?:tablets?|tabs?|capsules?|caps?|pills?)\b", re.IGNORECASE),
)
INSTRUCTION_PATTERN = re.compile(
    r"\b(?:take|wake)\s+(\d+)\s+"
    r"(?:tablet|tablets|tab|tabs|capsule|capsules)\b"
    r"(?:\s+by\s*(?:mou\w*|m)?)?\s+"
    r"(?:every\s+(\d+)\s*(?:hours?|hrs?|hr)|"
    r"(daily|t?wice daily|ice daily|once daily))\b",
    re.IGNORECASE,
)


def load_drug_names(path: str = "drug_names.txt") -> List[str]:
    names_path = Path(path)
    if not names_path.exists():
        return []
    return [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class MedicationParser:
    def __init__(self, drug_names: Optional[Iterable[str]] = None):
        if isinstance(drug_names, str):
            names = load_drug_names(drug_names)
        else:
            names = list(drug_names) if drug_names is not None else load_drug_names()
        self._names = {name.casefold(): name for name in names}

    def match_drug_name(self, text: str) -> Tuple[str, float]:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.casefold())
        candidates = words + [
            f"{words[index]} {words[index + 1]}"
            for index in range(len(words) - 1)
        ]
        best_name, best_score = "", 0.0
        for candidate in candidates:
            matches = difflib.get_close_matches(
                candidate, list(self._names), n=1, cutoff=0.82
            )
            if matches:
                score = difflib.SequenceMatcher(
                    None, candidate, matches[0]
                ).ratio()
                if score > best_score:
                    best_name, best_score = self._names[matches[0]], score
        return best_name, best_score

    @staticmethod
    def _find_strength(text: str) -> str:
        match = COMPOUND_DOSAGE_PATTERN.search(text) or DOSAGE_PATTERN.search(text)
        if not match:
            return ""
        strength = match.group(0).upper().replace(" ", "")
        return strength[:-1] + "MG" if strength.endswith("H") else strength

    @staticmethod
    def _find_quantity(text: str) -> str:
        for pattern in QUANTITY_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _find_instructions(text: str) -> str:
        match = INSTRUCTION_PATTERN.search(text)
        if not match:
            return ""
        amount, hours, frequency = match.groups()
        if hours:
            return f"TAKE {amount} TABLET BY MOUTH EVERY {hours} HOURS"
        if frequency in {"ice daily", "wice daily"}:
            frequency = "twice daily"
        return f"TAKE {amount} TABLET BY MOUTH {frequency.upper()}"

    def parse(self, text: str) -> MedicationRecord:
        normalized = normalize_text(text)
        search_text = normalized_search_text(normalized)
        name, name_score = self.match_drug_name(search_text)
        strength = self._find_strength(search_text)
        instructions = self._find_instructions(search_text)
        quantity = self._find_quantity(search_text)
        warnings: List[str] = []

        if not name:
            warnings.append("Medication name was not matched to the local catalogue.")
        if not instructions:
            warnings.append("Directions were not recognized.")
        if not quantity:
            warnings.append("Quantity was not recognized.")

        confidence = {
            "medicine_name": FieldConfidence(
                score=round(name_score, 3),
                evidence=["local drug catalogue"] if name else [],
            ),
            "instructions": FieldConfidence(
                score=1.0 if instructions else 0.0,
                evidence=["structured directions pattern"] if instructions else [],
            ),
            "quantity": FieldConfidence(
                score=1.0 if quantity else 0.0,
                evidence=["quantity pattern"] if quantity else [],
            ),
        }
        return MedicationRecord(
            medicine_name=name,
            strength=strength,
            dosage_form="TABLET" if re.search(r"\bTABLETS?\b", search_text) else "",
            instructions=instructions,
            quantity=quantity,
            raw_text=normalized,
            confidence=confidence,
            warnings=warnings,
        )


def records_agree(records: Iterable[MedicationRecord], minimum: int = 2) -> bool:
    usable = [record for record in records if record.medicine_name]
    if len(usable) < minimum:
        return False
    first = usable[0]
    return sum(
        record.medicine_name == first.medicine_name
        and record.instructions == first.instructions
        and record.quantity == first.quantity
        for record in usable
    ) >= minimum
