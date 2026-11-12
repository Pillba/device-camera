import argparse
import json
from typing import List, Optional

from camera_capture import capture_image
from medication_parser import MedicationParser
from scanner import scan_image, choose_consensus
from storage import save_record


def scan_from_camera(
    camera_index: int = 0,
    image_path: str = "captures/capture.jpg",
    records_directory: str = "records",
) -> dict:
    """Capture one image, validate it, save the record, and return it."""
    result = scan_label(camera_index=camera_index, image_path=image_path)
    result["path"] = save_record(result, records_directory)
    return result


def scan_label(
    camera_index: int = 0,
    image_path: str = "captures/capture.jpg",
) -> dict:
    """Capture and validate a label without persisting a medication record."""
    captured_path = capture_image(
        output_path=image_path,
        camera_index=camera_index,
        preview=True,
    )
    parser = MedicationParser("drug_names.txt")
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
        default="captures/capture.jpg",
        help="Captured image path (default: captures/capture.jpg)",
    )
    parser.add_argument(
        "--records-directory",
        default="records",
        help="Directory for validated JSON records (default: records)",
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
