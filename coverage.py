import re
import pandas as pd
import logging
from typing import Optional, Tuple
from database import get_db_connection
# from clasify import ml_predict_coverage_status, ml_predict_coverage_status_batch
# Setup logging
logger = logging.getLogger(__name__)

# # Setup ML Output Logger
# ml_logger = logging.getLogger('ml_coverage_logger')
# ml_logger.setLevel(logging.INFO)
# # Prevent propagation to root logger to avoid duplicate logs in main log
# ml_logger.propagate = False 

# # Create file handler
# fh = logging.FileHandler('ml_output.log', encoding='utf-8')
# fh.setLevel(logging.INFO)

# # Create formatter
# formatter = logging.Formatter('%(asctime)s - %(message)s')
# fh.setFormatter(formatter)

# # Add handler to logger
# if not ml_logger.handlers:
#     ml_logger.addHandler(fh)

# --- PRE-COMPILED REGEX PATTERNS (OPTIMIZATION) ---
NC_INDICATORS = ["NC", "NF", "NOT COVERED", "EXCLUDED", "NON-FORMULARY", "NOT-COVERED"]
RE_NC = re.compile(r'\b(?:' + '|'.join(map(re.escape, NC_INDICATORS)) + r')\b', re.IGNORECASE)

# --- COVERAGE PRIORITY MAPPING (MOST RESTRICTIVE -> LEAST) ---
COVERAGE_PRIORITY = {
    "Not Covered": 1,
    "Covered with PA": 2,
    "Covered with ST": 3,
    "Covered with Conditions": 4,
    "Covered": 5
}

PA_KEYWORDS = [
    'prior authorization', 'prior auth', 'pa required', 'pa needed',
    'pa', 
    'authorization required',
    'requires prior authorization',
    'must be approved', 'approval needed', 'prior approval needed'
]
RE_PA = re.compile(r'(?:^|[^a-zA-Z0-9])(?:' + '|'.join(map(re.escape, PA_KEYWORDS)) + r')(?:\d*)(?:$|[^a-zA-Z0-9])', re.IGNORECASE)

ST_KEYWORDS = [
    'step therapy', 'step-therapy', 'st'
]
RE_ST = re.compile(r'(?:^|[^a-zA-Z0-9])(?:' + '|'.join(map(re.escape, ST_KEYWORDS)) + r')(?:\d*)(?:$|[^a-zA-Z0-9])', re.IGNORECASE)

OTHER_CONDITIONS = ["QL", "AL", "LA", "BVD"]
RE_OTHER_CONDITIONS = re.compile(r'(?:^|[^a-zA-Z0-9])(?:' + '|'.join(map(re.escape, OTHER_CONDITIONS)) + r')(?:$|[^a-zA-Z0-9])', re.IGNORECASE)

# def determine_ml_coverage_status(coverage_statuses):
#     """
#     Determine final coverage status when multiple statuses are found.
#     Order of precedence: Not Covered > Covered with Conditions > Covered
#     """
#     normalized = {s.strip().title() for s in coverage_statuses if s}

#     if "Not Covered" in normalized:
#         return "Not Covered"
#     elif "Covered With Pa" in normalized or "Covered With Prior Authorization" in normalized: 
#         return "Covered with PA"
#     elif "Covered With St" in normalized or "Covered With Step Therapy" in normalized:  
#         return "Covered with ST"
#     elif "Covered With Conditions" in normalized or "Covered With Condition" in normalized:
#         return "Covered with Condition"
#     elif "Covered" in normalized:
#         return "Covered"
#     else:
#         return "Covered"


def detect_prior_authorization(requirements_text):
    """
    Detect if Prior Authorization is required based on requirements text
    Returns True if PA is required, False otherwise
    """
    if not requirements_text or pd.isna(requirements_text):
        return False
    
    text = str(requirements_text).strip()
    
    if not text or text.lower() in ['', 'none', 'null', 'nan']:
        return False
    
    return bool(RE_PA.search(text))


def detect_step_therapy(requirements_text):
    """
    Detect if Step Therapy is required based on requirements text  
    Returns True if ST is required, False otherwise
    """
    if not requirements_text or pd.isna(requirements_text):
        return False
    
    text = str(requirements_text).strip()
    
    if not text or text.lower() in ['', 'none', 'null', 'nan']:
        return False
    
    return bool(RE_ST.search(text))

def det_coverage_status(
    acronym=None,
    expansion=None,
    explanation=None,
    requirements_text=None,
    tier_text=None,
    conn=None,
    state_name=None,
    payer_name=None,
    # ml_predict_fn=None,
    drug_name=None,
    acronym_cache=None
):
    """
    Main orchestrator for coverage status determination.
    Consolidated logic: Tier NC > Python Logic (PA/ST/QL) > DB Lookup > Default
    """
    # Identifier for logging
    log_id = drug_name or acronym or "Unknown Drug"

    # Use the imported ML prediction function if not provided
    # if ml_predict_fn is None:
    #     ml_predict_fn = ml_predict_coverage_status

    tier_text_clean = str(tier_text or "").upper().strip()
    req_text_clean = str(requirements_text or "").upper().strip()

    # --- NEW: DETERMINISTIC PRIORITY RESOLUTION ---
    # Collect all potential coverage results and resolve by priority (Most Restrictive -> Least)
    # Each entry is a tuple: (status, confidence, source)
    detected_results = []
    
    # 1. Check Tier NC Priority
    if tier_text_clean and RE_NC.search(tier_text_clean):
        detected_results.append(("Not Covered", 100.0, "Python Logic (NC Tier)"))
        logger.info(f"[{log_id}] Detected 'Not Covered' (100.0) from Tier NC")

    # 2. Check Requirement NC Priority
    is_nc_req = bool(RE_NC.search(req_text_clean))
    if is_nc_req:
        detected_results.append(("Not Covered", 90.0, "Python Logic (NC Req)"))
        logger.info(f"[{log_id}] Detected 'Not Covered' (90.0) from Req NC")

    # 3. Check Python Rule Logic: PA, ST, QL, AL, LA, BVD, etc.
    is_pa = detect_prior_authorization(req_text_clean)
    is_st = detect_step_therapy(req_text_clean)
    is_other_cond = bool(RE_OTHER_CONDITIONS.search(req_text_clean))
    is_specialty = "SP" in tier_text_clean or "SPECIALTY" in tier_text_clean

    if is_st:
        detected_results.append(("Covered with ST", 85.0, "Python Logic"))
        logger.info(f"[{log_id}] Detected 'Covered with ST' (85.0) from Python Logic")
    if is_pa:
        detected_results.append(("Covered with PA", 85.0, "Python Logic"))
        logger.info(f"[{log_id}] Detected 'Covered with PA' (85.0) from Python Logic")
    if is_other_cond or is_specialty:
        detected_results.append(("Covered with Conditions", 85.0, "Python Logic"))
        logger.info(f"[{log_id}] Detected 'Covered with Conditions' (85.0) from Python Logic")

    # 4. Check DB Lookup (Full and Sub-acronyms)
    db_cvg_status = None
    db_source = None
    db_conf = 0.0

    if acronym:
        acronym_str = str(acronym).strip()
        
        # 1. Tier & Requirement Extraction Logic
        raw_parts = []
        
        # a. Extract Tier tokens (1B, 5^, Tier 1-6)
        # Note: We keep them intact as requested. Use whitespace/separator boundaries.
        tier_regex = r'(?:^|[,\s;])(Tier\s*[1-6]|[1-6][A-Z\^]?|[1-6])(?=$|[,\s;])'
        found_tiers = re.findall(tier_regex, acronym_str, flags=re.IGNORECASE)
        for t in found_tiers:
            t_clean = t.strip().upper()
            # Handle "5^" -> "Tier 5" and "^"
            if re.match(r'^[1-6]\^$', t_clean):
                raw_parts.append(f"Tier {t_clean[0]}")
                raw_parts.append("^")
            # Handle "1B" -> "1B"
            elif re.match(r'^[1-6][A-Z]$', t_clean):
                raw_parts.append(t_clean)
            # Handle "Tier 1" -> "Tier 1"
            elif t_clean.startswith("TIER"):
                digit_match = re.search(r'\d+', t_clean)
                if digit_match:
                    raw_parts.append(f"Tier {digit_match.group()}")
            # Handle bare digit "2" -> "Tier 2"
            elif re.match(r'^[1-6]$', t_clean):
                raw_parts.append(f"Tier {t_clean}")
            else:
                raw_parts.append(t_clean)

        # b. Requirement Parsing: Remove everything inside parentheses
        # QL(20 EA per fill retail; 20 per fill mail) -> QL
        cleaned_reqs = re.sub(r'\(.*?\)', ' ', acronym_str)
        
        # c. Token Cleaning: Remove commas, split by ; or whitespace
        split_parts = re.split(r'[,\s;]+', cleaned_reqs)
        
        # d. Valid Acronym Filter
        # Allowed patterns: 
        # 1. Tier tokens: ^[0-9][A-Z]$ (e.g. 1B)
        # 2. Requirement acronyms: PA, QL, ST, SP, AL, NM, B/D, ^
        # 3. Tier names: Tier 1-6
        # Reject: spaces, lowercase, >4 chars, English words, standalone digits (unless Tier prefix), parentheses content
        valid_req_acronyms = {"PA", "QL", "ST", "SP", "AL", "NM", "B/D", "^"}
        
        for part in split_parts:
            part = part.strip()
            # Reject if empty, contains spaces, or contains lowercase
            if not part or " " in part or any(c.islower() for c in part):
                continue
            
            # Special case: QL= parts
            if part.startswith("QL="):
                if "QL" not in raw_parts:
                    raw_parts.append("QL")
                continue
                
            # Pattern 1: Tier tokens (e.g., 1B). Note: Standalone digits handled by tier_regex above
            is_tier_token = bool(re.match(r'^[0-9][A-Z]$', part))
            
            # Pattern 2: Requirement acronyms
            is_req_acronym = part in valid_req_acronyms
            
            # Pattern 3: Tier names (Note: already handled by regex split if Tier 1)
            is_tier_name = bool(re.match(r'^Tier[0-9]$', part, re.IGNORECASE))
            
            if (is_tier_token or is_req_acronym or is_tier_name) and len(part) <= 4:
                if part not in raw_parts:
                    raw_parts.append(part)

        sub_acronyms = list(dict.fromkeys(raw_parts)) # Deduplicate preserving order

        from utils import lookup_expansion, determine_db_coverage
        
        # Try full lookup first if no parentheses
        if (conn or acronym_cache) and '(' not in acronym_str:
            _, _, cvg = lookup_expansion(acronym, state_name, payer_name, conn, acronym_cache=acronym_cache)
            if cvg:
                db_cvg_status = cvg
                db_source = f"DB Lookup - {acronym}"
                db_conf = 85.0

        # Try sub-acronyms if no full match
        if not db_cvg_status:
            res = determine_db_coverage(sub_acronyms, conn, state_name, payer_name, acronym_cache=acronym_cache)
            if res:
                db_cvg_status, db_conf, db_source = res

        if db_cvg_status:
            # Map DB status to priority status if necessary
            # Note: determine_db_coverage already returns standard statuses
            detected_results.append((db_cvg_status, db_conf, db_source))
            logger.info(f"[{log_id}] Detected '{db_cvg_status}' ({db_conf}) from {db_source}")

    # Resolve Final Status by Priority
    if detected_results:
        # Filter for valid statuses that exist in our priority map
        valid_results = [r for r in detected_results if r[0] in COVERAGE_PRIORITY]
        
        if valid_results:
            # Select the most restrictive result (lowest priority number)
            # This ensures status, confidence, and source stay together
            best_result = min(valid_results, key=lambda r: COVERAGE_PRIORITY.get(r[0], 999))
            
            final_status, conf, source = best_result
            manual_review = bool(conf < 80.0)
            
            logger.info(f"Coverage for '{log_id}': {final_status} (Source: {source}, Confidence: {conf}, Priority Resolution of {[r[0] for r in detected_results]})")
            return final_status, conf, source, manual_review

    # 5. Final Fallback - Default to Covered if nothing else detected
    logger.info(f"Coverage for '{log_id}': Covered (Source: Final Default Fallback)")
    return "Covered", 50.0, "Default", True


def normalize_drug_tier(raw_tier):
    """
    Clean raw tier text and normalize to standard tier names.
    Handles both 'Tier X' format and 'Generic/Brand' format.
    """
    if not raw_tier:
        return None
    
    # Simple cleaning for special characters and formatting artifacts
    cleaned = re.sub(r'\\[a-zA-Z]+\s*\{([^}]*)\}', r'\1', str(raw_tier))
    cleaned = cleaned.replace('$', '').replace('mathrm', '')
    cleaned = re.sub(r'^[,\s\$\{\}\\]+|[,\s\$\{\}\\]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if not cleaned:
        return None
    
    cleaned_lower = cleaned.lower().strip()
    
    # Map Generic/Brand format to tier equivalents
    tier_mapping = {
        'generic': 'Generic',
        'brand': 'Brand', 
        'specialty': 'Specialty',
        'generic/brand': 'Generic/Brand',
        'G': 'Generic',
        'B': 'Brand',
    }
    
    # Check if it matches a known mapping
    if cleaned_lower in tier_mapping:
        return tier_mapping[cleaned_lower]
    
    # Check for Tier X pattern or bare digit
    tier_match = re.search(r'tier\s*(\d+)', cleaned_lower, re.IGNORECASE)
    if tier_match:
        return f"Tier {tier_match.group(1)}"
    
    # Check for bare digit (e.g., "3" -> "Tier 3")
    digit_match = re.match(r'^(\d+)$', cleaned_lower)
    if digit_match:
        return f"Tier {digit_match.group(1)}"
    
    # Return the cleaned value as-is if it looks valid
    return cleaned

def infer_drug_tier_from_text(text: Optional[str]) -> Optional[str]:
    """
    Tries to find explicit tier mentions (e.g., "Tier 1", "Tier 2") inside a longer text blob.
    This is a conservative function to avoid incorrectly identifying dosages or other numbers as tiers.
    """
    if not text or pd.isna(text):
        return None

    # Search for the pattern "Tier" followed by 1 or 2 digits.
    match = re.search(r'(Tier\s*\d{1,2})', str(text), re.IGNORECASE)

    if match:
        tier_str = match.group(1).strip()
        # Normalize "Tier1" or "tier 1" to "Tier 1"
        return normalize_drug_tier(tier_str)

    return None
