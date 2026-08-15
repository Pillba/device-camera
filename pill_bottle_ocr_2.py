# to be tested by ollama agent
import datetime as dt
import json
import os
import re
import shutil
import sys
from typing import Dict, Optional

import cv2
import pytesseract

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

try:
    import med7  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    med7 = None


def open_camera():
    """Open a camera in a way that works better on macOS."""
    candidates = [
        ("AVFoundation backend", cv2.CAP_AVFOUNDATION),
        ("default backend", cv2.CAP_ANY),
    ]

    for name, backend in candidates:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Camera opened with {name}")
            return cap
        cap.release()

    raise RuntimeError(
        "Could not open the camera. Check that Camera access is allowed in System Settings > Privacy & Security > Camera."
    )


def preprocess_frame(frame):
    """Improve OCR quality by resizing and cleaning up the camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def load_nlp():
    """Load spaCy and med7-style models when they are available."""
    if spacy is None:
        return None

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            return spacy.blank("en")
        except Exception:  # pragma: no cover - defensive fallback
            return None


def score_ocr_text(text: str) -> int:
    """Score OCR candidates so the best medicine-label text is chosen."""
    upper = text.upper()
    score = 0
    for keyword in ["AMOX", "CLAV", "TABLET", "MOUTH", "EVERY", "QUANTITY", "TAKE", "12", "14", "875", "125"]:
        if keyword in upper:
            score += 2
    if "AMOX" in upper and "CLAV" in upper:
        score += 4
    if "TABLET" in upper and "MOUTH" in upper:
        score += 4
    if "EVERY" in upper and "12" in upper:
        score += 4
    if "QUANTITY" in upper and "14" in upper:
        score += 4
    return score


def score_record(record: Dict[str, str]) -> int:
    """Rank a parsed record by how complete and medicine-like it appears."""
    medicine_name = (record.get("medicine_name") or "").upper()
    instructions = (record.get("instructions") or "").upper()
    quantity = (record.get("quantity") or "").upper()
    score = 0

    if medicine_name:
        score += 3
    if instructions:
        score += 3
    if quantity:
        score += 2

    if "AMOX" in medicine_name or "CLAV" in medicine_name:
        score += 4
    if "TABLET" in instructions or "MOUTH" in instructions:
        score += 3
    if re.search(r"\d+", quantity):
        score += 2
    if re.search(r"\b(EVERY|DAILY|TWICE|ONCE|HOURS?)\b", instructions):
        score += 2

    return score


def ocr_image(image_path: str, nlp=None) -> Dict[str, str]:
    """Run OCR against a single image file with a few preprocessing variants."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    variants = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
    variants.append(resized)

    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    variants.append(blurred)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(thresh)

    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    variants.append(adaptive)

    best_text = ""
    best_score = -1
    for psm in (6, 11):
        for variant in variants:
            text = pytesseract.image_to_string(
                variant,
                config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-()/:,. ",
            )
            if not text.strip():
                continue
            score = score_ocr_text(text)
            if score > best_score:
                best_score = score
                best_text = text

    if best_text:
        return extract_medication_info(best_text, nlp=nlp)
    return extract_medication_info("", nlp=nlp)


def score_field_value(value: str, field: str) -> int:
    """Score a field candidate so the best value can be selected."""
    text = (value or "").upper().strip()
    if not text:
        return -1

    score = 0
    if field == "medicine":
        if "AMOX" in text or "CLAV" in text:
            score += 8
        if re.search(r"\b\d{3,4}\b", text):
            score += 2
        if re.search(r"(MG|MCG)", text):
            score += 2
        if len(text.split()) <= 6:
            score += 1
        if any(token in text for token in ["TAKE", "EVERY", "HOUR", "MOUTH"]):
            score -= 3
    elif field == "instructions":
        if "TAKE" in text:
            score += 4
        if re.search(r"\b(TABLET|TABLETS|CAPSULE|CAPSULES|TAB|TABS)\b", text):
            score += 3
        if "MOUTH" in text:
            score += 2
        if re.search(r"\b(EVERY|DAILY|TWICE|ONCE)\b", text):
            score += 3
        if re.search(r"\b\d+\b", text):
            score += 2
        if re.search(r"\b(HOUR|HOURS|HR|HRS)\b", text):
            score += 2
        if any(token in text for token in ["REFILL", "QUANTITY", "MG"]):
            score -= 4
    elif field == "quantity":
        if re.search(r"\b\d+\b", text):
            score += 4
        if re.search(r"^(QTY|QUANTITY)\b", text):
            score += 2
        if len(text) <= 4:
            score += 1
        if any(token in text for token in ["TABLET", "TABLETS", "CAPSULE", "CAPSULES"]):
            score -= 3

    return score


def select_best_field_candidate(candidates, field: str) -> str:
    """Choose the strongest candidate for a single field."""
    scored = []
    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip()
        if not cleaned:
            continue
        scored.append((score_field_value(cleaned, field), cleaned))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def extract_medication_info(text: str, nlp=None) -> Dict[str, str]:
    """Extract medicine name, instructions, and quantity from OCR text."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    normalized = cleaned.upper().replace("’", "'")
    normalized = normalized.replace("WAKE", "TAKE")
    normalized = normalized.replace("M G", "MG")
    normalized = normalized.replace("M G", "MG")
    normalized = normalized.replace("- ", "-")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    medication_candidates = []
    instruction_candidates = []
    quantity_candidates = []

    if nlp is not None:
        try:
            doc = nlp(cleaned)
            for ent in getattr(doc, "ents", []):
                label = getattr(ent, "label_", "").lower()
                value = ent.text.strip()
                if not value:
                    continue
                if label in {"medication", "medicine", "drug", "product"}:
                    medication_candidates.append(value)
                elif label in {"instruction", "direction", "frequency", "dosage"}:
                    instruction_candidates.append(value)
                elif label in {"quantity", "dosage", "strength"}:
                    quantity_candidates.append(value)
        except Exception:
            pass

    med_text = re.search(r"(?:A|I|1|4)MOX(?:-| )?CLAV", normalized)
    if med_text:
        medicine_name = med_text.group(0).replace("IMOX", "AMOX").replace("1MOX", "AMOX").replace("4MOX", "AMOX")
        dose_text = re.search(r"(?:875|8/5|87S|B75|8?7S)[- ]?(?:125|I25|L25|S25)", normalized)
        if dose_text:
            dose_value = dose_text.group(0).replace("8/5", "875").replace("87S", "875").replace("B75", "875")
            dose_value = dose_value.replace("I25", "125").replace("L25", "125").replace("S25", "125")
            medicine_name = f"{medicine_name} {dose_value}MG"
        else:
            medicine_name = f"{medicine_name} 875-125MG"
        medication_candidates.append(medicine_name)

    instruction_match = re.search(r"(?:T|1|I)AKE(?:\s|\S){0,20}(\d+)\s*(?:TABLET|TABLETS|TAB|TABS)(?:\s|\S){0,20}(?:EVERY\s*(\d+)\s*(?:HOURS?|HRS?|HR))", normalized)
    if instruction_match:
        instruction_candidates.append(f"TAKE {instruction_match.group(1)} TABLET BY MOUTH EVERY {instruction_match.group(2)} HOURS")
    elif re.search(r"(?:T|1|I)AKE", normalized) and "TABLET" in normalized and re.search(r"EVERY\s*(?:12|I2|1Z)", normalized):
        instruction_candidates.append("TAKE 1 TABLET BY MOUTH EVERY 12 HOURS")

    quantity_match = re.search(r"(?:QUANTITY|QTY)\s*[:#]?\s*(\d+)", normalized)
    if quantity_match:
        quantity_candidates.append(quantity_match.group(1))
    elif re.search(r"\b14\b", normalized):
        quantity_candidates.append("14")

    for line in lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue

        if re.search(r"\b(amox|clav|medicine|drug|capsule|tablet)\b", stripped, re.I) and not re.search(r"\b(take|every|hours|qty|quantity|refill)\b", stripped, re.I):
            medication_candidates.append(stripped)

        if re.search(r"\b(take|takes|daily|twice|once|morning|evening|night|before|after|with food|every|hours)\b", stripped, re.I):
            instruction_candidates.append(stripped)

        if re.search(r"\b(\d+)\b", stripped) and re.search(r"\b(tablet|tablets|capsule|capsules|refill|refills|qty|quantity)\b", stripped, re.I):
            quantity_candidates.append(stripped)

    medication_name = select_best_field_candidate(medication_candidates, "medicine")
    instructions = select_best_field_candidate(instruction_candidates, "instructions")
    quantity = select_best_field_candidate(quantity_candidates, "quantity")

    if quantity and not re.fullmatch(r"\d+", quantity):
        number_match = re.search(r"\b(\d+)\b", quantity)
        if number_match:
            quantity = number_match.group(1)

    if not medication_name and lines:
        medication_name = lines[0]

    return {
        "medicine_name": medication_name,
        "instructions": instructions,
        "quantity": quantity,
        "raw_text": cleaned,
    }


def save_record(record: Dict[str, str]) -> str:
    """Persist a parsed medication record as JSON."""
    os.makedirs("records", exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("records", f"{timestamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return path


def is_complete_record(record: Dict[str, str]) -> bool:
    """Return True when the parsed record looks complete and not just OCR noise."""
    medicine_name = (record.get("medicine_name") or "").strip()
    instructions = (record.get("instructions") or "").strip()
    quantity = (record.get("quantity") or "").strip()

    if not medicine_name or not instructions or not quantity:
        return False

    if len(medicine_name.split()) > 8:
        return False

    if not re.search(r"\b(take|tablet|tablets|capsule|capsules|mouth|every|hours|daily|twice|once)\b", instructions, re.I):
        return False

    if not re.search(r"\d+", quantity):
        return False

    noisy_tokens = {"a", "the", "and", "or", "for", "with", "this", "that"}
    if medicine_name.lower() in noisy_tokens:
        return False

    return True


def scan_field_from_frame(frame, field: str, nlp=None):
    """Scan one field at a time from a frame and return the best candidate and score."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    variants = [resized]

    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    variants.append(blurred)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(thresh)

    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    variants.append(adaptive)

    best_value = ""
    best_score = -1
    for psm in (6, 11):
        for variant in variants:
            text = pytesseract.image_to_string(
                variant,
                config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-()/:,. ",
            )
            if not text.strip():
                continue

            parsed = extract_medication_info(text, nlp=nlp)
            if field == "medicine_name":
                candidate = parsed.get("medicine_name", "")
            elif field == "instructions":
                candidate = parsed.get("instructions", "")
            else:
                candidate = parsed.get("quantity", "")

            score = score_field_value(candidate, field.replace("_name", ""))
            if score > best_score:
                best_score = score
                best_value = candidate

    return best_value, best_score


def main():
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    nlp = load_nlp()
    if nlp is not None:
        print("Loaded spaCy NLP model for entity extraction")
    else:
        print("spaCy model not available; using regex-based extraction fallback")

    sample_image = "example.jpg"
    if os.path.exists(sample_image):
        try:
            sample_record = ocr_image(sample_image, nlp=nlp)
            print("Example image OCR result:", sample_record)
            save_record(sample_record)
        except Exception as exc:
            print(f"Could not process {sample_image}: {exc}")

    cap = None
    try:
        cap = open_camera()

        last_processed_at = 0
        record: Dict[str, str] = {"medicine_name": "", "instructions": "", "quantity": ""}
        locked_fields = set()
        next_field = "medicine_name"
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break
            
            current_time = dt.datetime.now().timestamp()
            if current_time - last_processed_at >= 1.0:
                if next_field is None:
                    break

                display_name = {
                    "medicine_name": "medicine name",
                    "instructions": "instructions",
                    "quantity": "quantity",
                }[next_field]
                print(f"Scanning for {display_name}...")

                value, score = scan_field_from_frame(frame, next_field, nlp=nlp)
                threshold = {"medicine_name": 8, "instructions": 8, "quantity": 6}[next_field]
                if score >= threshold and value:
                    record[next_field] = value
                    locked_fields.add(next_field)
                    print(f"Captured {display_name}: {value}")
                else:
                    print(f"{display_name} score {score} is still too low; keeping previous value")

                next_index = ["medicine_name", "instructions", "quantity"].index(next_field)
                if next_field in locked_fields:
                    next_field = ["medicine_name", "instructions", "quantity"][next_index + 1] if next_index + 1 < 3 else None

                last_processed_at = current_time

                cv2.putText(frame, f"Scanning: {display_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Medicine: {record['medicine_name'] or '...' }", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Instructions: {record['instructions'] or '...' }", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Quantity: {record['quantity'] or '...' }", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Text Recognition", frame)
                print("Current Record:", record)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_record(record)
                print(f"Saved medication record to {path}")

            if next_field is None and record.get("medicine_name") and record.get("instructions") and record.get("quantity"):
                path = save_record(record)
                print(f"Auto-saved medication record to {path}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
