import os
from typing import Dict, List

import cv2
import pytesseract

from medication_parser import MedicationParser
from models import MedicationRecord
from ocr_engine import OcrEngine
from storage import save_record


def build_variants(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # A label-focused pass keeps the high-contrast text from competing with
    # background objects in a camera frame. The full-frame variants remain as
    # a fallback for images where the label is not centered.
    label = gray[
        int(height * 0.14):int(height * 0.96),
        int(width * 0.39):int(width * 0.76),
    ]
    sources = [gray, label]
    variants = []
    for source in sources:
        resized = cv2.resize(
            source, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC
        )
        variants.append(resized)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(resized)
        variants.append(clahe)
        blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
        _, otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(otsu)
        variants.append(
            cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
            )
        )
    label_clahe = variants[5]
    label_height = label_clahe.shape[0]
    text_lines = [
        label_clahe[int(label_height * 0.16):int(label_height * 0.38)],
        label_clahe[int(label_height * 0.37):int(label_height * 0.68)],
    ]
    variants.append(cv2.vconcat(text_lines))
    return variants


def scan_image(image_path: str, parser: MedicationParser) -> List[MedicationRecord]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    engine = OcrEngine(
        lambda variant, config: pytesseract.image_to_string(variant, config=config)
    )
    candidates = engine.recognize(build_variants(image))
    return [parser.parse(candidate.text) for candidate in candidates]


def choose_consensus(records: List[MedicationRecord]) -> MedicationRecord:
    groups = {}
    for record in records:
        key = (record.medicine_name, record.instructions, record.quantity)
        if all(key):
            groups.setdefault(key, []).append(record)
    if not groups:
        raise ValueError("No matching medication result across OCR variants.")
    matches = max(groups.values(), key=len)
    if len(matches) < 2:
        raise ValueError("No matching medication result across OCR variants.")
    return max(matches, key=lambda record: sum(
        field.score for field in record.confidence.values()
    ))


def scan_and_save(image_path: str, parser: MedicationParser, directory: str = "records") -> Dict[str, object]:
    result = choose_consensus(scan_image(image_path, parser))
    if result.warnings:
        raise ValueError("Scan requires user confirmation: " + "; ".join(result.warnings))
    data = result.as_dict()
    data["path"] = save_record(data, directory)
    return data
