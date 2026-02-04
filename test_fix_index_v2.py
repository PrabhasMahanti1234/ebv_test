
import re
import unittest
from pdf_extraction import _is_index_entry, _is_extracted_data_from_index_page

class TestIndexDetectionV2(unittest.TestCase):
    def test_page_number_in_requirements(self):
        # Scenario: OCR puts page number in 'drug_requirements'
        item = {"drug_name": "AFLURIA", "drug_tier": None, "drug_requirements": "301"}
        # Currently this returns False (Fail). We want it to be True.
        self.assertTrue(_is_index_entry(item), "Should detect page number in requirements")

    def test_page_number_in_name_no_dots(self):
        # Scenario: OCR eats the dots
        item = {"drug_name": "AFLURIA 301", "drug_tier": None, "drug_requirements": None}
        self.assertTrue(_is_index_entry(item), "Should detect page number at end of name even without dots")

    def test_page_number_in_name_spaces(self):
        # Scenario: Dots are spaces
        item = {"drug_name": "AFLURIA                         301", "drug_tier": None}
        self.assertTrue(_is_index_entry(item), "Should detect page number with long spaces")

    def test_page_detection_mixed_bad_ocr(self):
        # Scenario: Page 310 from user screenshot
        # A mix of correct and incorrect parsings
        page_data = [
            {"drug_name": "AFLURIA", "drug_tier": "301", "drug_requirements": None}, # Detected by v1
            {"drug_name": "AJOVY", "drug_tier": None, "drug_requirements": "250"},   # Missed by v1?
            {"drug_name": "AKEEGA 102", "drug_tier": None, "drug_requirements": None}, # Missed by v1?
            {"drug_name": "AKTEN ................. 265", "drug_tier": None},           # Detected by v1
            {"drug_name": "albendazole", "drug_tier": "35", "drug_requirements": None}
        ]
        self.assertTrue(_is_extracted_data_from_index_page(page_data), "Should detect mixed OCR index page")

if __name__ == '__main__':
    unittest.main()
