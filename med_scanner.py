import datetime as dt
import json
import os
import re
import shutil
import sys
import difflib
from typing import Dict, Optional

import cv2
import pytesseract

try:
    import spacy
except ImportError:  
    spacy = None

import med7  # spaCy-based named-entity-recognition model trained specifically for clinical/medical text


# put medication names in drug_names.txt (one per line) to improve reliability

DRUG_NAME_FILE = "drug_names.txt"
PLACEHOLDER_DRUG_NAMES = [
    "Metformin", "Lisinopril", "Atorvastatin", "Levothyroxine",
    "Amlodipine", "Metoprolol", "Omeprazole", "Losartan",
    "Gabapentin", "Hydrochlorothiazide", "Simvastatin", "Sertraline",
    "Amoxicillin",
]


def load_drug_names(path=DRUG_NAME_FILE):
    # read the drug list from a text file if it exists
    if os.path.exists(path):
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(names)} drug names from {path}")
        return names
    # no file yet, just use the placeholder list
    print(f"WARNING: {path} not found, using placeholder list.")
    return PLACEHOLDER_DRUG_NAMES


KNOWN_DRUG_NAMES = load_drug_names()

# matches a single dosage value like "500mg" or "0.5 mcg"
DOSAGE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|iu)\b", re.IGNORECASE)
# matches combo strengths like "875-125MG"
COMPOUND_DOSAGE_PATTERN = re.compile(r"\b\d+-\d+\s?(mg|mcg|ml|g|iu)\b", re.IGNORECASE)


def find_dosage(text):
    # check for combo strength first (like 875-125MG), then plain dosage
    match = COMPOUND_DOSAGE_PATTERN.search(text)
    if match:
        return match.group(0)
    match = DOSAGE_PATTERN.search(text)
    return match.group(0) if match else None


# different ways a quantity might show up on a label: "qty: 30", "quantity: 30", "#30", "30 tablets"
QUANTITY_PATTERNS = [
    re.compile(r"qty\.?\s*:?\s*(\d+)", re.IGNORECASE),
    re.compile(r"quantity\.?\s*:?\s*(\d+)", re.IGNORECASE),
    re.compile(r"#\s?(\d+)\b"),
    re.compile(r"\b(\d+)\s?(tablets?|tabs?|capsules?|caps?|pills?)\b", re.IGNORECASE),
]
# matches "TAKE 1 TABLET EVERY 8 HOURS" / "TAKE 2 CAPSULES DAILY" style instructions
INSTRUCTION_PATTERN = re.compile(
    # ("(?:T|1|I)AKE" instead of "TAKE" because OCR sometimes misreads the leading T as 1 or I)
    r"(?:T|1|I)AKE(?:\s|\S){0,20}(\d+)\s*(?:TABLET|TABLETS|TAB|TABS|CAPSULE|CAPSULES)"
    r"(?:\s|\S){0,25}(?:EVERY\s*(\d+)\s*(?:HOURS?|HRS?|HR)|(DAILY|TWICE DAILY|ONCE DAILY))",
    re.IGNORECASE,
)


def open_camera():
    """Open a camera."""
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
        "Could not open the camera. Check that Camera access is allowed."
    )


def load_nlp():
    """Load spaCy and med7-style models when they are available."""
    if spacy is None:
        return None  # spacy isn't installed, caller falls back to regex-only extraction

    try:
        # Loads spaCy's generic English model, not med7
        return spacy.load("en_core_web_sm")
    except OSError:
        # en_core_web_sm model files aren't downloaded; use an empty pipeline
        try:
            return spacy.blank("en")
        except Exception:  # defensive fallback
            return None


def match_drug_name(text):
    """Fuzzy-match OCR text against the known drug list. Checks single
    words and word-pairs, lowercased so ALL-CAPS OCR output still matches
    a title-case name. Returns (name, similarity_score) or (None, 0)."""
    # lowercase everything first so text still matches
    lower_to_original = {n.lower(): n for n in KNOWN_DRUG_NAMES}
    lower_names = list(lower_to_original.keys())

    # check single words and pairs of words (for drug names with 2 words)
    words = text.lower().split()
    # also try adjacent word pairs, e.g. "Lisinopril HCTZ"
    candidates = list(words) + [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    best_match, best_score = None, 0
    for candidate in candidates:
        # find the closest known drug name (if any are within 75% similarity)
        matches = difflib.get_close_matches(candidate, lower_names, n=1, cutoff=0.75)
        if matches:
            # get the exact similarity ratio so different candidates/options can be compared against each other
            score = difflib.SequenceMatcher(None, candidate, matches[0]).ratio()
            if score > best_score:
                # keep the original casing (e.g. "Metformin"), not the lowercased match
                best_match, best_score = lower_to_original[matches[0]], score
    return best_match, best_score

# Scores 1 OCR pass
def score_ocr_text(text: str) -> int:
    """Score an OCR candidate on how medication-label-like it looks.
    ocr_image() runs Tesseract multiple times per photo (different image
    preprocessing + page-segmentation modes) and keeps whichever pass scores
    highest here, since noisier images miss these.
    Max possible score is 18 (8 + 3 + 4 + 3); most real labels won't hit
    every point, so this is a relative ranking, not a pass/fail threshold.
    """
    name, name_score = match_drug_name(text)
    score = 0
    # drug name match is weighted most heavily (up to 8), scaled by how
    # close the fuzzy match was (name_score is a 0-1 similarity ratio)
    if name:
        score += int(name_score * 8)  # points for a drug name match and then recognizable dosage (e.g. "500mg") is full points
    if find_dosage(text):
        score += 3  # points for a dosage value
    # a full "take X tablet(s) every Y hours/daily" match is strong evidence for real instructions, so it scores highest
    if INSTRUCTION_PATTERN.search(text):
        score += 4  # points for a full instruction match
    # fall back to a looser check (just "TAKE" + "TABLET/CAPSULE" present somewhere) when the strict pattern doesn't match - worth less since it
    # doesn't confirm dose or frequency, just that instruction-like words exist
    elif re.search(r"\bTAKE\b", text, re.I) and re.search(r"\b(TABLET|TABLETS|CAPSULE|CAPSULES)\b", text, re.I):
        score += 2  # partial credit for a rough instruction match
    # any recognizable quantity phrasing ("qty: 30", "#30", "30 tablets")
    if any(p.search(text) for p in QUANTITY_PATTERNS):
        score += 3  # points for a quantity value
    return score

# Runs multiple OCR passes on a single image and returns the best result
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

# Scores one candidate value for a specific field (medicine name, instructions, or quantity)
def score_field_value(value: str, field: str) -> int:
    """Score a field candidate so the best value can be selected."""
    text = (value or "").upper().strip()
    if not text:
        return -1

    score = 0
    if field == "medicine":
        # score based on how close this is to a known drug name
        _, name_score = match_drug_name(text)
        score += int(name_score * 8)
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

# Select the best candidate for a specific field from a list of candidates
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

# Extract medicine name, instructions, and quantity from OCR text based on the score
def extract_medication_info(text: str, nlp=None) -> Dict[str, str]:
    """Extract medicine name, instructions, and quantity from OCR text."""
    # collapse whitespace/newlines from OCR into single spaces
    cleaned = re.sub(r"\s+", " ", text).strip()
    # uppercase + fix common OCR misreads to match more reliably
    normalized = cleaned.upper().replace("’", "'")
    normalized = normalized.replace("WAKE", "TAKE")  # OCR often misreads "TAKE" as "WAKE"
    normalized = normalized.replace("M G", "MG")  # OCR sometimes splits "MG" with a space
    normalized = normalized.replace("- ", "-")  # tidy up compound dosages like "875 -125MG"
    # keep the original line breaks 
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # select_best_field_candidate() scores each candidate with score_field_value() and keeps the highest scoring one per field
    medication_candidates = []
    instruction_candidates = []
    quantity_candidates = []

    # source 1: spaCy/med7 named-entity recognition, if a model was loaded
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

    # source 2: fuzzy match the drug name against the known drug list, adding on a dosage string (e.g. "500mg") if one is found nearby
    name_match, name_score = match_drug_name(cleaned)
    if name_match:
        dosage_text = find_dosage(cleaned)
        if dosage_text:
            medication_candidates.append(f"{name_match} {dosage_text.upper()}")
        else:
            medication_candidates.append(name_match)

    # source 3: look for a "take X every Y hours/daily" style instruction and reformat it into regular phrasing
    instruction_match = INSTRUCTION_PATTERN.search(normalized)
    if instruction_match:
        qty_word, hours, freq_word = instruction_match.groups()
        if hours:
            instruction_candidates.append(f"TAKE {qty_word} TABLET BY MOUTH EVERY {hours} HOURS")
        elif freq_word:
            instruction_candidates.append(f"TAKE {qty_word} TABLET BY MOUTH {freq_word}")

    # source 4: try each quantity pattern (qty:, quantity:, #30, "30 tablets") until one matches
    quantity_match = None
    for pattern in QUANTITY_PATTERNS:
        quantity_match = pattern.search(normalized)
        if quantity_match:
            quantity_candidates.append(quantity_match.group(1))
            break

    # source 5: fall back to scanning each line for keyword hints, in case none of the more targeted patterns above matched anything on that line
    for line in lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue

        # a line that mentions the medicine/drug/tablet but isn't an instruction or quantity, its probably the medicine name
        if re.search(r"\b(medicine|drug|capsule|tablet)\b", stripped, re.I) and not re.search(r"\b(take|every|hours|qty|quantity|refill)\b", stripped, re.I):
            medication_candidates.append(stripped)

        # a line with dosing-frequency language is probably the instructions
        if re.search(r"\b(take|takes|daily|twice|once|morning|evening|night|before|after|with food|every|hours)\b", stripped, re.I):
            instruction_candidates.append(stripped)

        # a line with both a number and a unit word is probably the quantity
        if re.search(r"\b(\d+)\b", stripped) and re.search(r"\b(tablet|tablets|capsule|capsules|refill|refills|qty|quantity)\b", stripped, re.I):
            quantity_candidates.append(stripped)

    # picks the single best candidate per field out of everything collected above
    medication_name = select_best_field_candidate(medication_candidates, "medicine")
    instructions = select_best_field_candidate(instruction_candidates, "instructions")
    quantity = select_best_field_candidate(quantity_candidates, "quantity")

    # the winning quantity candidate might be a whole phrase (e.g. "30
    # tablets") rather than a bare number - pull just the digits back out
    if quantity and not re.fullmatch(r"\d+", quantity):
        number_match = re.search(r"\b(\d+)\b", quantity)
        if number_match:
            quantity = number_match.group(1)

    # last resort: if nothing scored as a medicine name, just guess the first line
    if not medication_name and lines:
        medication_name = lines[0]

    return {
        "medicine_name": medication_name,
        "instructions": instructions,
        "quantity": quantity,
        "raw_text": cleaned,
    }


# Save the parsed medication record to a timestamped JSON file in records/
def save_record(record: Dict[str, str]) -> str:
    """Persist a parsed medication record as JSON."""
    os.makedirs("records", exist_ok=True)
    # filename is the timestamp, so each save gets its own file
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("records", f"{timestamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return path


# Check whether a record has all 3 fields filled in
def is_complete_record(record: Dict[str, str]) -> bool:
    """Return True when the parsed record looks complete and not just OCR noise."""
    medicine_name = (record.get("medicine_name") or "").strip()
    instructions = (record.get("instructions") or "").strip()
    quantity = (record.get("quantity") or "").strip()

    # all 3 fields have to have something in them for this 
    if not medicine_name or not instructions or not quantity:
        return False

    # a real drug name isn't a full sentence - this many words means OCR
    # noise probably got picked up as the "name"
    if len(medicine_name.split()) > 8:
        return False

    # instructions should actually contain dosing language, not just any text
    if not re.search(r"\b(take|tablet|tablets|capsule|capsules|mouth|every|hours|daily|twice|once)\b", instructions, re.I):
        return False

    # quantity should contain an actual number
    if not re.search(r"\d+", quantity):
        return False

    # reject common filler words that sometimes get picked up as the "name"
    noisy_tokens = {"a", "the", "and", "or", "for", "with", "this", "that"}
    if medicine_name.lower() in noisy_tokens:
        return False

    return True


def scan_field_from_frame(frame, field: str, nlp=None):
    """Scan one field at a time from a frame and return the best candidate and score.

    This is the live-camera counterpart to ocr_image() - same idea (try
    several preprocessing variants and PSM modes, keep the best-scoring
    result) but per-video-frame and scoped to just one field at a time,
    since main() scans for medicine name, then instructions, then quantity
    one at a time as the user points the camera at each part of the label.
    """
    # same preprocessing approach as ocr_image(): grayscale, upscale, blur,
    # then two different threshold methods to produce several OCR attempts
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
    # psm 6 = assume a uniform block of text, psm 11 = sparse text - try both
    # since a medication label doesn't always look like a neat paragraph
    for psm in (6, 11):
        for variant in variants:
            text = pytesseract.image_to_string(
                variant,
                config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-()/:,. ",
            )
            if not text.strip():
                continue

            # run the full extractor even though we only want one field back,
            # since it's the same candidate-gathering logic as extract_medication_info()
            parsed = extract_medication_info(text, nlp=nlp)
            if field == "medicine_name":
                candidate = parsed.get("medicine_name", "")
            elif field == "instructions":
                candidate = parsed.get("instructions", "")
            else:
                candidate = parsed.get("quantity", "")

            # "medicine_name" -> "medicine" to match score_field_value()'s field names
            score = score_field_value(candidate, field.replace("_name", ""))
            if score > best_score:
                best_score = score
                best_value = candidate

    return best_value, best_score


# process an optional sample image, then run the live camera scan loop
def main():
    # point pytesseract at the system tesseract binary if one is installed
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    nlp = load_nlp()
    if nlp is not None:
        print("Loaded spaCy NLP model for entity extraction")
    else:
        print("spaCy model not available; using regex-based extraction fallback")

    # optional test run against a static image, if its present
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
        # fields that have already been captured with a high enough score,
        # so the loop knows to move on and stop re-scanning them
        locked_fields = set()
        # which field the camera is currently trying to capture - cycles
        # medicine_name -> instructions -> quantity -> None (done)
        next_field = "medicine_name"
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break

            current_time = dt.datetime.now().timestamp()
            # only run OCR once per second rather than every frame
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
                # quantity has a lower bar since it's a short, simple field
                # that's harder to rack up points on than name/instructions
                threshold = {"medicine_name": 8, "instructions": 8, "quantity": 6}[next_field]
                if score >= threshold and value:
                    # good enough - lock it in and move on to the next field
                    record[next_field] = value
                    locked_fields.add(next_field)
                    print(f"Captured {display_name}: {value}")
                else:
                    # not confident enough yet - keep whatever was captured
                    # before (if anything) and try again next second
                    print(f"{display_name} score {score} is still too low; keeping previous value")

                # advance to the next field only once the current one is locked in
                next_index = ["medicine_name", "instructions", "quantity"].index(next_field)
                if next_field in locked_fields:
                    next_field = ["medicine_name", "instructions", "quantity"][next_index + 1] if next_index + 1 < 3 else None

                last_processed_at = current_time

                # draw the current scanning status and captured fields onto the preview window
                cv2.putText(frame, f"Scanning: {display_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Medicine: {record['medicine_name'] or '...' }", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Instructions: {record['instructions'] or '...' }", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Quantity: {record['quantity'] or '...' }", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Text Recognition", frame)
                print("Current Record:", record)

            # 'q' quits, 's' manually saves the record as-is at any point
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_record(record)
                print(f"Saved medication record to {path}")

            # once all 3 fields are captured, auto-save and stop
            if next_field is None and record.get("medicine_name") and record.get("instructions") and record.get("quantity"):
                path = save_record(record)
                print(f"Auto-saved medication record to {path}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except RuntimeError as exc:
        # raised by open_camera() when no camera backend could be opened
        print(exc)
        sys.exit(1)
    finally:
        # always release the camera and close the preview window, even on error
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
