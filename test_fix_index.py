
import re
import unittest
from pdf_extraction import _is_index_entry, _is_extracted_data_from_index_page

class TestIndexDetection(unittest.TestCase):
    def test_single_index_entry(self):
        # Case 1: High tier number (obvious page number)
        item1 = {"drug_name": "Aspirin", "drug_tier": "301"}
        self.assertTrue(_is_index_entry(item1), "Should detect high tier number as index")

        # Case 2: Tier number > 6 but < 100 (likely page number)
        item2 = {"drug_name": "Tylenol", "drug_tier": "15"}
        self.assertTrue(_is_index_entry(item2), "Should detect tier 15 as index")

        # Case 3: Valid Tier
        item3 = {"drug_name": "Amoxicillin", "drug_tier": "1"}
        self.assertFalse(_is_index_entry(item3), "Should NOT detect tier 1 as index")

        # Case 4: Valid Tier "Tier 3"
        item4 = {"drug_name": "Amoxicillin", "drug_tier": "Tier 3"}
        self.assertFalse(_is_index_entry(item4), "Should NOT detect 'Tier 3' as index")

        # Case 5: Dot leader in name
        item5 = {"drug_name": "Advil . . . . . . . 50", "drug_tier": "1"} 
        self.assertTrue(_is_index_entry(item5), "Should detect dot leaders in name as index")

    def test_page_detection_rule(self):
        # Create a fake page of extracted data that SHOULD be detected as an index page
        # Scenario: All "tiers" are actually page numbers
        index_page_data = []
        for i in range(20):
            index_page_data.append({
                "drug_name": f"Drug {i}",
                "drug_tier": str(10 + i), # Tiers 10-29 (Page numbers)
                "drug_requirements": None
            })
        
        self.assertTrue(_is_extracted_data_from_index_page(index_page_data), 
                        "Should detect page where tiers are actually page numbers")

        # Scenario: Real drug page (Uniform tiers 1-5)
        real_drug_page = []
        for i in range(20):
            real_drug_page.append({
                "drug_name": f"Real Drug {i} 500mg",
                "drug_tier": str((i % 5) + 1), # Tiers 1-5
                "drug_requirements": "QL" if i % 2 == 0 else None
            })
        
        self.assertFalse(_is_extracted_data_from_index_page(real_drug_page),
                         "Should NOT detect real drug page as index")
        
        # Scenario: Real drug page with one weird entry
        real_drug_page_glitch = real_drug_page.copy()
        real_drug_page_glitch.append({"drug_name": "Bad Parse", "drug_tier": "45"})
        self.assertFalse(_is_extracted_data_from_index_page(real_drug_page_glitch),
                         "Should NOT false positive on a single glitchy entry")

if __name__ == '__main__':
    unittest.main()
