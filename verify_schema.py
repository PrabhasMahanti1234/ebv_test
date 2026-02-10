import logging
import json
from pdf_extraction import _extract_drug_from_item

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdl_extraction():
    """
    Test extraction logic with sample data simulating the 4-column PDL format.
    """
    print("🧪 Testing PDL Extraction Logic...")

    # Sample data mimicking OCR output for the 4-column format
    # Columns: Category | Preferred | Preferred with PA | Non-Preferred
    sample_items = [
        # Item from 'Preferred' column
        {
            "Drug Name": "tramadol 50 mg tablet",
            "category": "ANALGESICS",
            "preferred_agent": "yes",
            "non_preferred_agent": "no",
            "requirements": None
        },
        # Item from 'Preferred with PA' column
        {
            "Drug Name": "tramadol/APAP tablet",
            "category": "ANALGESICS",
            "preferred_agent": "yes",
            "non_preferred_agent": "no",
            "requirements": "PA"
        },
        # Item from 'Non-Preferred' column
        {
            "Drug Name": "hydromorphone suppository",
            "category": "ANALGESICS",
            "preferred_agent": "no",
            "non_preferred_agent": "yes",
            "requirements": None
        }
    ]

    print(f"\nProcessing {len(sample_items)} sample items...")
    
    results = []
    for item in sample_items:
        extracted = _extract_drug_from_item(item, page_number=1)
        results.append(extracted)
        print(f"\nInput: {item['Drug Name']}")
        print(f"Extracted: {json.dumps(extracted, indent=2)}")

    # Verification checks
    
    # Check 1: Preferred drug
    assert results[0]['preferred_agent'] == 'yes', "Failed: Preferred agent should be 'yes'"
    assert results[0]['non_preferred_agent'] == 'no', "Failed: Non-preferred agent should be 'no'"
    assert results[0]['drug_requirements'] is None, "Failed: Requirements should be None"

    # Check 2: Preferred with PA
    assert results[1]['preferred_agent'] == 'yes', "Failed: Preferred agent should be 'yes'"
    assert results[1]['drug_requirements'] == 'PA', "Failed: Requirements should be 'PA'"

    # Check 3: Non-Preferred
    assert results[2]['preferred_agent'] == 'no', "Failed: Preferred agent should be 'no'"
    assert results[2]['non_preferred_agent'] == 'yes', "Failed: Non-preferred agent should be 'yes'"

    print("\n✅ All logic tests passed!")

if __name__ == "__main__":
    test_pdl_extraction()
