from typing import Callable, List

from models import OcrCandidate


class OcrEngine:
    """Tesseract adapter; preprocessing and OCR are isolated from parsing."""

    def __init__(self, image_to_string: Callable):
        self._image_to_string = image_to_string

    def recognize(self, variants: List[object], psm_modes=(6, 11)) -> List[OcrCandidate]:
        candidates: List[OcrCandidate] = []
        for psm in psm_modes:
            for index, variant in enumerate(variants):
                text = self._image_to_string(variant, config=f"--psm {psm}")
                if text and text.strip():
                    candidates.append(
                        OcrCandidate(text=text, source=f"variant-{index}/psm-{psm}")
                    )
        return candidates
