import argparse
import os
import time
from typing import List, Optional

import cv2


def add_capture_guide(frame):
    """Return a preview copy with a centered medication-label hitbox."""
    preview = frame.copy()
    height, width = preview.shape[:2]
    box_width = int(width * 0.78)
    box_height = int(height * 0.55)
    left = (width - box_width) // 2
    top = (height - box_height) // 2
    right = left + box_width
    bottom = top + box_height
    color = (0, 220, 0)

    cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
    cv2.putText(
        preview,
        "Align medication label inside frame",
        (left, max(30, top - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "SPACE: capture  Q/ESC: cancel",
        (left, min(height - 15, bottom + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return preview


def open_usb_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Open a USB camera by device index and fail explicitly if unavailable."""
    for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        camera = cv2.VideoCapture(camera_index, backend)
        if camera.isOpened():
            return camera
        camera.release()
    raise RuntimeError(
        f"Could not open USB camera at index {camera_index}. "
        "Check the connection and camera permissions."
    )


def save_frame(frame, output_path: str) -> str:
    """Save a captured frame and verify that OpenCV wrote it successfully."""
    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    if not cv2.imwrite(output_path, frame):
        raise IOError(f"Could not save captured image to {output_path}")
    return output_path


def capture_image(
    output_path: str = "captures/capture.jpg",
    camera_index: int = 0,
    preview: bool = True,
    warmup_seconds: float = 2.0,
) -> str:
    """Capture one image from a USB camera and return its path.

    With preview enabled, press Space to capture or Q/Escape to cancel.
    With preview disabled, frames are discarded during the warm-up period so
    autofocus and exposure can settle before the image is saved.
    """
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be non-negative")

    camera = open_usb_camera(camera_index)
    try:
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        warmup_until = time.monotonic() + (warmup_seconds if not preview else 0)
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("USB camera opened, but no frame was received.")

            if not preview:
                if time.monotonic() >= warmup_until:
                    return save_frame(frame, output_path)
                continue

            cv2.imshow("USB Camera Capture", add_capture_guide(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                return save_frame(frame, output_path)
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Capture cancelled by user.")
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture one image from a USB camera.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--output",
        default="captures/capture.jpg",
        help="Output image path (default: captures/capture.jpg)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Warm up and save a frame without opening a preview window",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Seconds to warm up autofocus/exposure before capture (default: 2)",
    )
    args = parser.parse_args(argv)

    try:
        path = capture_image(
            output_path=args.output,
            camera_index=args.camera_index,
            preview=not args.no_preview,
            warmup_seconds=args.warmup_seconds,
        )
    except ValueError as error:
        print(f"Capture failed: {error}")
        return 1
    except KeyboardInterrupt:
        print("Capture cancelled.")
        return 1
    except (IOError, RuntimeError) as error:
        print(f"Capture failed: {error}")
        return 1

    print(f"Captured image: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
