import unittest

from medication_parser import MedicationParser, records_agree


class MedicationParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = MedicationParser(
            ["Amoxicillin", "Metformin", "Lisinopril"]
        )

    def test_extracts_structured_label_fields(self):
        record = self.parser.parse(
            "AMOXICILLIN 500 MG\nTAKE 1 CAPSULE BY MOUTH EVERY 8 HOURS\n"
            "QTY: 21"
        )
        self.assertEqual(record.medicine_name, "Amoxicillin")
        self.assertEqual(record.strength, "500MG")
        self.assertEqual(record.quantity, "21")
        self.assertFalse(record.warnings)

    def test_does_not_promote_unmatched_ocr_noise_to_medicine(self):
        record = self.parser.parse("address pharmacy noise\n123 main street")
        self.assertEqual(record.medicine_name, "")
        self.assertIn("local catalogue", record.warnings[0])

    def test_requires_matching_records_for_consensus(self):
        first = self.parser.parse("METFORMIN 500MG\nQTY 30")
        second = self.parser.parse("METFORMIN 500MG\nTAKE 1 TABLET DAILY\nQTY 30")
        different = self.parser.parse("LISINOPRIL 500MG\nQTY 30")
        self.assertTrue(records_agree([first, second]))
        self.assertFalse(records_agree([first, different]))


if __name__ == "__main__":
    unittest.main()
