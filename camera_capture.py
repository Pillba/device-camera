import argparse
import os
from typing import List, Optional

import cv2


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
) -> str:
    """Capture one image from a USB camera and return its path.

    With preview enabled, press Space to capture or Q/Escape to cancel.
    With preview disabled, the first available frame is saved immediately.
    """
    camera = open_usb_camera(camera_index)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("USB camera opened, but no frame was received.")

            if not preview:
                return save_frame(frame, output_path)

            cv2.imshow("USB Camera Capture", frame)
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
        help="Save the first available frame without opening a preview window",
    )
    args = parser.parse_args(argv)

    try:
        path = capture_image(
            output_path=args.output,
            camera_index=args.camera_index,
            preview=not args.no_preview,
        )
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
