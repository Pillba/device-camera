import os
import unittest

try:
    import cv2  # noqa: F401
    import pytesseract  # noqa: F401
    OCR_DEPENDENCIES_AVAILABLE = True
except ImportError:
    OCR_DEPENDENCIES_AVAILABLE = False


CAPTURE_PATHS = (
    os.path.join("captures", "capture2.jpg"),
    os.path.join("captures", "capture3.jpg"),
)


@unittest.skipUnless(
    all(os.path.exists(path) for path in CAPTURE_PATHS) and OCR_DEPENDENCIES_AVAILABLE,
    "capture2.jpg, capture3.jpg, and OCR dependencies are required",
)
class CapturedImageTests(unittest.TestCase):
    def _records_for(self, path):
        from medication_parser import MedicationParser
        from scanner import scan_image

        return scan_image(path, MedicationParser("drug_names.txt"))

    def test_captured_labels_produce_medication_candidates(self):
        for path in CAPTURE_PATHS:
            records = self._records_for(path)
            self.assertGreater(len(records), 0, path)
            self.assertTrue(
                any(
                    record.medicine_name.casefold() == "prednisone"
                    for record in records
                ),
                path,
            )

    def test_clear_captured_label_extracts_name_and_strength(self):
        records = self._records_for(CAPTURE_PATHS[0])
        prednisone_records = [
            record
            for record in records
            if record.medicine_name.casefold() == "prednisone"
        ]

        self.assertTrue(prednisone_records)
        self.assertTrue(any(record.strength == "20MG" for record in prednisone_records))


if __name__ == "__main__":
    unittest.main()
