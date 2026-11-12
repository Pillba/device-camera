import argparse
import json
from pathlib import Path
from typing import List, Optional

from camera_capture import capture_image
from medication_parser import MedicationParser
from scanner import scan_image, choose_consensus
from storage import save_record


PROJECT_ROOT = Path(__file__).resolve().parent


def scan_from_camera(
    camera_index: int = 0,
    image_path: str | None = None,
    records_directory: str | None = None,
) -> dict:
    """Capture one image, validate it, save the record, and return it."""
    result = scan_label(camera_index=camera_index, image_path=image_path)
    record_directory = (
        Path(records_directory)
        if records_directory is not None
        else PROJECT_ROOT / "records"
    )
    result["path"] = save_record(result, str(record_directory))
    return result


def scan_label(
    camera_index: int = 0,
    image_path: str | None = None,
) -> dict:
    """Capture and validate a label without persisting a medication record."""
    capture_path = (
        Path(image_path)
        if image_path is not None
        else PROJECT_ROOT / "captures" / "capture.jpg"
    )
    captured_path = capture_image(
        output_path=str(capture_path),
        camera_index=camera_index,
        preview=True,
    )
    parser = MedicationParser(str(PROJECT_ROOT / "drug_names.txt"))
    return choose_consensus(scan_image(captured_path, parser)).as_dict()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a medication label and print extracted information."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Captured image path (default: device-camera/captures/capture.jpg)",
    )
    parser.add_argument(
        "--records-directory",
        default=None,
        help="Directory for validated JSON records (default: device-camera/records)",
    )
    args = parser.parse_args(argv)

    try:
        result = scan_from_camera(
            camera_index=args.camera_index,
            image_path=args.output,
            records_directory=args.records_directory,
        )
    except KeyboardInterrupt:
        print("Capture cancelled.")
        return 1
    except (IOError, OSError, RuntimeError, ValueError) as error:
        print(f"Scan failed: {error}")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
