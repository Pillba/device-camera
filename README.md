# Medication Label OCR

This project extracts structured medication information from printed
prescription bottle labels using local OCR. The current image pipeline is split
into OCR, normalization, parsing, consensus validation, and storage
responsibilities so the parser can be tested without a camera.

## Setup

Activate the Conda environment used by this project:

```sh
conda activate med7
python -m pip install -r requirements.txt
```

The application also requires the Tesseract executable. On macOS, install it
with Homebrew if it is not already available:

```sh
brew install tesseract
```

Verify the environment:

```sh
python -c "import cv2, pytesseract; print('Python OCR dependencies are available')"
tesseract --version
```

The new parser does not require spaCy or Med7 directly. The `med7` environment
is retained for compatibility with the earlier `med_scanner.py` prototype and
for future clinical NLP experiments.

## Raspberry Pi installation

These instructions target Raspberry Pi OS (64-bit, Bookworm or newer) with a
USB camera. The Pi uses a Python virtual environment instead of the macOS
Conda environment:

```sh
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-opencv tesseract-ocr
git clone <repository-url>
cd device-camera
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytesseract
```

The `python3-opencv` package avoids compiling OpenCV on the Pi. If you use a
regular isolated virtual environment instead, install all Python dependencies
with `python -m pip install -r requirements.txt`; this may download a
platform-compatible OpenCV wheel and can require more memory and storage.

Verify the installation:

```sh
python -c "import cv2, pytesseract; print(cv2.__version__)"
tesseract --version
```

Connect the USB camera, then capture a fixture without opening a preview
window:

```sh
python camera_capture.py --camera-index 0 --output captures/pi-capture.jpg --no-preview
```

Run the parser tests with:

```sh
python -m unittest -v test_medication_parser.py
```

The captured-image tests require the fixture images and can be slow on a
Raspberry Pi because they run multiple OCR passes. A desktop or laptop is
recommended for the full OCR fixture suite.

## Image scanning

The reusable image pipeline is exposed through `scanner.py` and processes
`example.jpg` with the local drug catalogue:

```sh
python -c "from medication_parser import MedicationParser; from scanner import scan_and_save; print(scan_and_save('example.jpg', MedicationParser('drug_names.txt')))"
```

The scanner requires at least two OCR variants to agree before saving a record.
If the result is incomplete or uncertain, it raises an error instead of silently
saving OCR noise.

## USB camera capture

Capture a still image from a USB camera without running OCR:

```sh
conda activate med7
python camera_capture.py --output captures/capture.jpg
```

The preview window displays the live camera feed. Press **Space** to save the
current frame, or **Q** or **Escape** to cancel. To save the first available
frame without opening a preview window:

```sh
python camera_capture.py --camera-index 0 --output captures/capture.jpg --no-preview
```

Use `--camera-index 1` or another index if the USB camera is not device 0.
Captured images are standalone files; this module does not invoke OCR or write
medication records.

## Tests

Run the parser and consensus tests with:

```sh
python -m unittest -v test_medication_parser.py
```

The captured-image tests use the clear Prednisone fixtures
`captures/capture2.jpg` and `captures/capture3.jpg` and exercise the complete
OpenCV/Tesseract-to-parser path:

```sh
python -m unittest -v test_captured_image.py
```

If the captures or OCR dependencies are unavailable, these tests are skipped.
Create or replace a capture fixture with:

```sh
python camera_capture.py --output captures/capture.jpg
```

## Features

- **OCR extraction**: Runs multiple OpenCV preprocessing variants and Tesseract
  modes.
- **Structured parsing**: Extracts medicine name, strength, directions, dosage
  form, quantity, and warnings.
- **Conservative matching**: Medication names must match the local
  `drug_names.txt` catalogue; OCR noise is not promoted to a drug name.
- **Consensus validation**: A result must agree across multiple OCR variants
  before the image pipeline saves it.
- **Atomic JSON storage**: Records are written to `records/` only after
  successful validation.

## Project structure

| File                   | Purpose                                               |
| ---------------------- | ----------------------------------------------------- |
| `scanner.py`           | End-to-end image scanning and consensus orchestration |
| `camera_capture.py`    | Standalone USB camera still-image capture             |
| `ocr_engine.py`        | Tesseract adapter for OCR candidates                  |
| `text_normalizer.py`   | Line-preserving OCR normalization                     |
| `medication_parser.py` | Drug matching and field extraction                    |
| `models.py`            | Typed OCR and medication result models                |
| `storage.py`           | Atomic JSON record persistence                        |
| `med_scanner.py`       | Earlier live-camera prototype                         |
| `pill_bottle_ocr.py`   | Older sample-specific OCR prototype                   |
| `drug_names.txt`       | Local medication-name catalogue                       |

## Scope and limitations

The initial architecture targets printed prescription bottle labels and uses
offline Tesseract. Low-confidence results must be reviewed by a user before
being treated as trusted medication data. Handwritten prescriptions, arbitrary
medical documents, and cloud OCR are out of scope for the initial version.
