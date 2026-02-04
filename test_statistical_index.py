
import unittest
from pdf_extraction import _is_index_entry, _is_extracted_data_from_index_page, _extract_number_from_any_field

class TestStatisticalIndexDetection(unittest.TestCase):
    """Test the new statistical variance-based index detection"""
    
    def test_extract_number_from_fields(self):
        """Test number extraction from various fields"""
        # Number in tier
        self.assertEqual(_extract_number_from_any_field({"drug_tier": "301"}), 301)
        # Number in requirements
        self.assertEqual(_extract_number_from_any_field({"drug_requirements": "270"}), 270)
        # Number at end of name
        self.assertEqual(_extract_number_from_any_field({"drug_name": "AFLURIA 301"}), 301)
        # No number
        self.assertIsNone(_extract_number_from_any_field({"drug_name": "ASPIRIN"}))
    
    def test_individual_index_entries(self):
        """Test individual entry detection"""
        # High tier number
        self.assertTrue(_is_index_entry({"drug_name": "AFLURIA", "drug_tier": "301"}))
        # High requirements number
        self.assertTrue(_is_index_entry({"drug_name": "AJOVY", "drug_requirements": "250"}))
        # Dot leaders
        self.assertTrue(_is_index_entry({"drug_name": "AKTEN .......... 265"}))
        # Number merged in name
        self.assertTrue(_is_index_entry({"drug_name": "AKEEGA 102"}))
        # Valid drug (should be False)
        self.assertFalse(_is_index_entry({"drug_name": "Amoxicillin 500mg", "drug_tier": "1"}))
    
    def test_statistical_variance_index_page(self):
        """Test page-level detection using variance"""
        # Simulating page 314 - index page with page numbers
        index_page = [
            {"drug_name": "AFLURIA", "drug_tier": "301"},
            {"drug_name": "AJOVY", "drug_tier": "250"},
            {"drug_name": "AKEEGA", "drug_tier": "102"},
            {"drug_name": "AKTEN", "drug_tier": "265"},
            {"drug_name": "ALBUTEROL", "drug_tier": "44"},
            {"drug_name": "ALCENSA", "drug_tier": "96"},
            {"drug_name": "ALINIA", "drug_tier": "36"},
            {"drug_name": "ALLOPURINOL", "drug_tier": "197"},
            {"drug_name": "ALPRAZOLAM", "drug_tier": "40"},
            {"drug_name": "ALTABAX", "drug_tier": "164"},
        ]
        # Mean ~159.5, Stdev ~89 → should be detected as index
        self.assertTrue(_is_extracted_data_from_index_page(index_page))
    
    def test_real_drug_page_not_detected(self):
        """Test that real drug pages are NOT marked as index"""
        real_drug_page = [
            {"drug_name": "Amoxicillin 500mg", "drug_tier": "1", "drug_requirements": "QL"},
            {"drug_name": "Aspirin 81mg", "drug_tier": "1"},
            {"drug_name": "Atorvastatin 20mg", "drug_tier": "2", "drug_requirements": "PA"},
            {"drug_name": "Lisinopril 10mg", "drug_tier": "1"},
            {"drug_name": "Metformin 1000mg", "drug_tier": "1", "drug_requirements": "ST"},
            {"drug_name": "Omeprazole 20mg", "drug_tier": "2"},
            {"drug_name": "Simvastatin 40mg", "drug_tier": "1"},
            {"drug_name": "Levothyroxine 50mcg", "drug_tier": "1"},
        ]
        # Mean ~1.25, Stdev ~0.46 → should NOT be detected as index
        self.assertFalse(_is_extracted_data_from_index_page(real_drug_page))
    
    def test_mixed_page_with_majority_index(self):
        """Test page with mix but majority are index entries"""
        mixed_page = [
            {"drug_name": "AFLURIA", "drug_tier": "301"},  # Index
            {"drug_name": "AJOVY", "drug_tier": "250"},    # Index
            {"drug_name": "Aspirin 81mg", "drug_tier": "1"},  # Real drug
            {"drug_name": "AKEEGA", "drug_tier": "102"},   # Index
            {"drug_name": "AKTEN", "drug_tier": "265"},    # Index
        ]
        # 4/5 are index (80%) → should be detected
        self.assertTrue(_is_extracted_data_from_index_page(mixed_page))

if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
