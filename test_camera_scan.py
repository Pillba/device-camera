import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import camera_scan


class CameraScanTests(unittest.TestCase):
    @patch("camera_scan.scan_and_save")
    @patch("camera_scan.capture_image", return_value="captures/test.jpg")
    def test_captures_scans_and_prints_structured_result(
        self, capture_image, scan_and_save
    ):
        scan_and_save.return_value = {
            "medicine_name": "Amoxicillin",
            "strength": "500MG",
            "quantity": "10",
            "path": "records/result.json",
        }
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = camera_scan.main(
                [
                    "--camera-index",
                    "2",
                    "--output",
                    "captures/test.jpg",
                    "--records-directory",
                    "test-records",
                ]
            )

        self.assertEqual(exit_code, 0)
        capture_image.assert_called_once_with(
            output_path="captures/test.jpg",
            camera_index=2,
            preview=True,
        )
        scan_and_save.assert_called_once()
        self.assertEqual(json.loads(output.getvalue())["medicine_name"], "Amoxicillin")

    @patch("camera_scan.capture_image", side_effect=RuntimeError("camera unavailable"))
    def test_capture_failure_is_reported_without_success_output(self, capture_image):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = camera_scan.main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Scan failed: camera unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
