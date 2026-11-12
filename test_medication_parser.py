import unittest

from medication_parser import MedicationParser, records_agree
from models import MedicationRecord
from scanner import choose_consensus


class MedicationParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = MedicationParser(
            ["Amoxicillin", "Metformin", "Lisinopril", "Levocetirizine"]
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

    def test_medication_name_hint_can_validate_an_entity(self):
        record = self.parser.parse(
            "EVOCETIRIZINE 5MG\nQTY 30",
            medication_name_hint="Levocetirizine",
        )
        self.assertEqual(record.medicine_name, "Levocetirizine")
        self.assertEqual(record.strength, "5MG")
        self.assertEqual(record.quantity, "30")

    def test_handles_common_ocr_errors_on_label_fields(self):
        parser = MedicationParser(["Levocetirizine"])
        record = parser.parse("TEVOCETIRIZINE 5th\nTY 30")

        self.assertEqual(record.medicine_name, "Levocetirizine")
        self.assertEqual(record.strength, "5MG")
        self.assertEqual(record.quantity, "30")

    def test_recovers_strength_after_ocr_drops_the_unit(self):
        parser = MedicationParser(["Prednisone"])
        record = parser.parse("PREDNISONE 20\nTAKE 1 TABLET")

        self.assertEqual(record.medicine_name, "Prednisone")
        self.assertEqual(record.strength, "20MG")

    def test_requires_matching_records_for_consensus(self):
        first = self.parser.parse("METFORMIN 500MG\nQTY 30")
        second = self.parser.parse("METFORMIN 500MG\nTAKE 1 TABLET DAILY\nQTY 30")
        different = self.parser.parse("LISINOPRIL 500MG\nQTY 30")
        self.assertTrue(records_agree([first, second]))
        self.assertFalse(records_agree([first, different]))

    def test_consensus_merges_partial_records_without_stale_warnings(self):
        complete_name = MedicationRecord(
            medicine_name="Prednisone",
            strength="20MG",
            raw_text="PREDNISONE 20MG",
        )
        partial_quantity = MedicationRecord(
            medicine_name="Prednisone",
            strength="20MG",
            quantity="10",
            raw_text="QTY 10",
            warnings=["Quantity was not recognized."],
        )

        result = choose_consensus([complete_name, partial_quantity])

        self.assertEqual(result.medicine_name, "Prednisone")
        self.assertEqual(result.strength, "20MG")
        self.assertEqual(result.quantity, "10")
        self.assertFalse(result.warnings)

    def test_consensus_does_not_require_quantity(self):
        first = MedicationRecord(
            medicine_name="Prednisone",
            strength="20MG",
            raw_text="PREDNISONE 20MG",
        )
        second = MedicationRecord(
            medicine_name="Prednisone",
            strength="20MG",
            raw_text="PREDNISONE 20MG",
        )

        result = choose_consensus([first, second])

        self.assertEqual(result.medicine_name, "Prednisone")
        self.assertEqual(result.strength, "20MG")
        self.assertEqual(result.quantity, "")
        self.assertFalse(result.warnings)


if __name__ == "__main__":
    unittest.main()
