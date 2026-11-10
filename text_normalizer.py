import re
from typing import List


SAFE_OCR_REPLACEMENTS = {
    "M G": "MG",
    "M L": "ML",
    "W AKE": "TAKE",
}


def normalize_text(text: str) -> str:
    """Normalize OCR spacing while preserving line boundaries."""
    lines: List[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    normalized = "\n".join(lines)
    for source, target in SAFE_OCR_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return normalized


def normalized_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).upper()
