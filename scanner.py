import os
from collections import Counter
from typing import Dict, List

import cv2
import pytesseract

from medication_parser import MedicationParser
from med7_validator import Med7Validator
from models import MedicationRecord
from ocr_engine import OcrEngine
from storage import save_record


def build_variants(image):
    """Build illumination-resistant variants aligned with the capture guide."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    label = gray[
        int(height * 0.18):int(height * 0.82),
        int(width * 0.08):int(width * 0.92),
    ]
    resized_label = cv2.resize(
        label, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )
    denoised = cv2.bilateralFilter(resized_label, 9, 75, 75)
    clahe = cv2.createCLAHE(
        clipLimit=2.5, tileGridSize=(8, 8)
    ).apply(denoised)
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(clahe, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    gaussian = cv2.GaussianBlur(clahe, (0, 0), 3.0)
    sharpened = cv2.addWeighted(clahe, 1.5, gaussian, -0.5, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(resized_label, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(resized_label, background, scale=255)
    _, normalized_threshold = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return [resized_label, clahe, sharpened, adaptive, normalized_threshold]


def scan_image(image_path: str, parser: MedicationParser) -> List[MedicationRecord]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    engine = OcrEngine(
        lambda variant, config: pytesseract.image_to_string(
            variant, config=f"{config} --oem 3 --dpi 300"
        )
    )
    med7 = Med7Validator()
    candidates = engine.recognize(build_variants(image), psm_modes=(6, 11))
    records = []
    for candidate in candidates:
        record = parser.parse(candidate.text)
        if not record.medicine_name and med7.available:
            medication_name = med7.extract_name(candidate.text, parser.drug_names)
            record = parser.parse(candidate.text, medication_name)
        records.append(record)
    return records


def choose_consensus(records: List[MedicationRecord]) -> MedicationRecord:
    required_fields = ("medicine_name", "strength")
    consensus = {}
    for field in required_fields:
        values = Counter(
            getattr(record, field)
            for record in records
            if getattr(record, field)
        )
        if not values:
            raise ValueError(f"Could not reliably extract {field}.")
        value, votes = values.most_common(1)[0]
        if votes < 2:
            raise ValueError(f"No matching {field} result across OCR variants.")
        consensus[field] = value

    quantity_values = Counter(
        record.quantity for record in records if record.quantity
    )
    quantity = quantity_values.most_common(1)[0][0] if quantity_values else ""
    best_record = max(
        records,
        key=lambda record: sum(
            getattr(record, field) == value
            for field, value in consensus.items()
        ),
    )
    return MedicationRecord(
        medicine_name=consensus["medicine_name"],
        strength=consensus["strength"],
        quantity=quantity,
        dosage_form=best_record.dosage_form,
        instructions=best_record.instructions,
        refills=best_record.refills,
        raw_text=best_record.raw_text,
        confidence=best_record.confidence,
        warnings=[],
    )


def scan_and_save(
    image_path: str,
    parser: MedicationParser,
    directory: str = "records",
) -> Dict[str, object]:
    result = choose_consensus(scan_image(image_path, parser))
    data = result.as_dict()
    data["path"] = save_record(data, directory)
    return data
