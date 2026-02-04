
import re
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _is_header_row(drug_name: str) -> bool:
    """
    Detects if a row is likely a therapeutic category header rather than a drug.
    """
    if not drug_name:
        return False
        
    name_lower = drug_name.lower()
    
    # Pattern 1: Explicit "Drugs to Treat" phrase (very strong signal)
    if "drugs to treat" in name_lower:
        return True
        
    # Pattern 2: Category-like structure (Hyphens) + NO Dosage Info
    # Headers often have " - " but NO strength/form info (mg, ml, tab, cap)
    
    # Regex for dosage/form - reused from RE_DOSAGE_FORM but broader
    has_dosage = re.search(r'\d+\s*(mg|ml|mcg|unit|%|tab|cap|sol|cream|gel|patch|spray)', name_lower)
    
    if not has_dosage:
        # If it has " - " and no dosage, it's suspicious.
        if " - " in name_lower:
            return True
            
        # Check for all caps words at start (common in headers)
        # Only if the name is reasonably long (avoid filtering "ASPIRIN")
        parts = drug_name.split(' ')
        if len(parts) > 0 and parts[0].isupper() and len(parts[0]) > 4 and len(drug_name) > 20:
             return True
             
    return False

# Test cases
test_cases = [
    "Urinary Antispasmodics - Direct Muscle Relaxants",
    "Urinary Antispasmodics-Direct Muscle Relaxants", # No spaces around hyphen
    "Urinary Antispasmodics – Direct Muscle Relaxants", # En-dash or Em-dash
    "Urinary Antispasmodics — Direct Muscle Relaxants",
    "AMOXICILLIN 500 MG",
    "LANTUS SOLOSTAR 100 UNIT/ML",
    "ANTI-INFECTIVE AGENTS - MISC.",
    "DERMATOLOGICALS",
    "ANALGESICS - OPIOID",
    "VACCINES"
]

print("Running tests for _is_header_row:")
for test in test_cases:
    result = _is_header_row(test)
    print(f"'{test}': {result}")
