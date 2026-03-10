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
    Consolidated logic: Tier NC > Python Logic (PA/ST/QL) > ML Model > Default
    """
    # Identifier for logging
    log_id = drug_name or acronym or "Unknown Drug"

    # Use the imported ML prediction function if not provided
    # if ml_predict_fn is None:
    #     ml_predict_fn = ml_predict_coverage_status

    tier_text_clean = str(tier_text or "").upper().strip()
    req_text_clean = str(requirements_text or "").upper().strip()

    # nc_indicators removed - using RE_NC

    # 1. Both Null Fallback: If both are null, return Covered/Unknown immediately
    if not tier_text_clean and not req_text_clean:
        logger.info(f"Coverage for '{log_id}': Covered/Unknown (Source: Default Fallback - Both Tier and Req null)")
        return "Covered with Conditions", 50.0, "Default", True

    # 2. Tier NC Priority: if tier has nc indicators it is not covered
    # Use regex word boundaries to avoid false positives (e.g., "NC" in "CONC")
    if tier_text_clean and RE_NC.search(tier_text_clean):
        logger.info(f"Coverage for '{log_id}': Not Covered (Source: Python Logic - NC Tier '{tier_text_clean}')")
        return "Not Covered", 100.0, "Python Logic (NC Tier)", False

    # 3. Python Rule Logic: Check for PA, ST, QL, AL, LA, BVD, etc.
    is_pa = detect_prior_authorization(req_text_clean)
    is_st = detect_step_therapy(req_text_clean)
    is_other_cond = bool(RE_OTHER_CONDITIONS.search(req_text_clean))
    is_specialty = "SP" in tier_text_clean or "SPECIALTY" in tier_text_clean
    is_nc_req = bool(RE_NC.search(req_text_clean))

    if is_nc_req:
        logger.info(f"Coverage for '{log_id}': Not Covered (Source: Python Logic - NC Requirement '{req_text_clean}')")
        return "Not Covered", 90.0, "Python Logic (NC Req)", False
    
    if is_pa:
        reason = ["PA"]
        if is_st: reason.append("ST")
        if is_other_cond: reason.append("Other Conditions")
        if is_specialty: reason.append("Specialty Tier")
        logger.info(f"Coverage for '{log_id}': Covered with PA (Source: Python Logic - {', '.join(reason)})")
        return "Covered with PA", 85.0, "Python Logic", False
    
    if is_st:
        reason = ["ST"]
        if is_other_cond: reason.append("Other Conditions")
        if is_specialty: reason.append("Specialty Tier")
        logger.info(f"Coverage for '{log_id}': Covered with ST (Source: Python Logic - {', '.join(reason)})")
        return "Covered with ST", 85.0, "Python Logic", False

    if is_other_cond or is_specialty:
        reason = []
        if is_other_cond: reason.append("Other Conditions")
        if is_specialty: reason.append("Specialty Tier")
        logger.info(f"Coverage for '{log_id}': Covered with Conditions (Source: Python Logic - {', '.join(reason)})")
        return "Covered with Conditions", 85.0, "Python Logic", False
    # 4. ML Model Logic (Requirement and Tier)
    # if acronym:
    #     # Split acronyms by common separators (comma, hyphen, slash, whitespace)
    #     raw_parts = [a for a in re.split(r'[,\s\-/]+', str(acronym)) if a]
    #     sub_acronyms = []
    #     for part in raw_parts:
    #         token = part.strip()
            
    #         # Special case: If QL is followed by "=", treat the acronym as just "QL"
    #         if token.upper().startswith("QL="):
    #             sub_acronyms.append("QL")
    #             continue

    #         # Handle digit + symbol format (e.g., 5^, 3*)
    #         # Only split if no letters are present (avoids splitting 5ST, 4PA)
    #         if not re.search(r'[A-Z]', token, re.IGNORECASE):
    #             match = re.match(r'^(\d+)([^\w\s]+)$', token)
    #             if match:
    #                 digit, symbol = match.groups()
    #                 sub_acronyms.append(f"Tier {digit}")
    #                 sub_acronyms.append(symbol)
    #                 continue
            
    #         sub_acronyms.append(token)

        
    #     ml_predictions = []
        
    #     # Ensure we can look up expansion
    #     from utils import lookup_expansion

    #     for sub_acronym in sub_acronyms:
    #         # Determine initial expansion/explanation for this part
    #         # If there's only one acronym, use the provided arguments as base
    #         sub_expansion = expansion if len(sub_acronyms) == 1 else None
    #         sub_explanation = explanation if len(sub_acronyms) == 1 else None
            
    #         # DB Lookup to enrich or fill missing info
    #         if conn:
    #             db_exp, db_expl, _ = lookup_expansion(sub_acronym, state_name, payer_name, conn)
    #             sub_expansion = sub_expansion or db_exp
    #             sub_explanation = sub_explanation or db_expl
            
    #         # Predict
    #         pred_label, pred_conf = ml_predict_fn(
    #             payer_name, 
    #             state_name, 
    #             sub_acronym, 
    #             sub_expansion, 
    #             sub_explanation
    #         )
            
    #         # Log individual prediction to ml_output.log
    #         ml_logger.info(f"Drug: {log_id} | Input: {sub_acronym} | Expansion: {sub_expansion} | Explanation: {sub_explanation} | Prediction: {pred_label} | Confidence: {pred_conf} | Payer: {payer_name} | State: {state_name}")

    #         if pred_label:
    #             ml_predictions.append((pred_label, pred_conf))
        
    #     if ml_predictions:
    #         # ml_predictions is a list of (prediction_label, prediction_confidence) tuples
    #         statuses = [prediction_label for prediction_label, _ in ml_predictions]
    #         final_status = determine_ml_coverage_status(statuses)
    #         manual_verification_required = False
            
    #         # Determine confidence based on the winning status
    #         final_conf = 0.0
    #         normalized_target = final_status.lower().replace("conditions", "condition")
            
    #         for label, conf in ml_predictions:
    #              normalized_label = label.lower().replace("conditions", "condition")
    #              if normalized_label == normalized_target:
    #                  final_conf = max(final_conf, conf)
            
    #         # Log final decision to ml_output.log
    #         ml_logger.info(f"Drug: {log_id} | Final ML Status: {final_status} | Final Confidence: {final_conf}")

    #         # Convert to percentage with 2 decimal places without rounding
    #         final_conf_pct = float(int(final_conf * 10000) / 100.0)

    #         logger.info(f"Coverage for '{log_id}': {final_status} (Source: ML Model - {acronym})")
    #         if final_conf_pct < 80.0:
    #             manual_verification_required = True
    #         return final_status, final_conf_pct, "ML Model", manual_verification_required

    #4. Cvg status look up using lookup_expansion()
    if acronym:
        # Split acronyms by common separators (comma, hyphen, slash, whitespace)
        raw_parts = [a for a in re.split(r'[,\s\-/]+', str(acronym)) if a]
        sub_acronyms = []
        for part in raw_parts:
            token = part.strip()
            
            # Special case: If QL is followed by "=", treat the acronym as just "QL"
            if token.upper().startswith("QL="):
                sub_acronyms.append("QL")
                continue

            # Handle variants like "Tier 5^", "5^", "Tier-1", "1"
            # Regex to handle optional prefix (Tier, tier-, etc.), then digit, then trailing symbols
            match = re.match(r'^(?:TIER\s*|tier\s*|tier\-)?(\d+)\s*([^\w\s]*)$', token, re.IGNORECASE)
            if match:
                digit, symbol = match.groups()
                # Normalize digit to "Tier X"
                sub_acronyms.append(f"Tier {digit}")
                if symbol:
                    # Extract individual symbols if multiple are present (e.g., "^*")
                    for char in symbol:
                        if char.strip():
                            sub_acronyms.append(char)
                continue
            
            sub_acronyms.append(token)
                
        from utils import lookup_expansion, determine_db_coverage

        # 1. Try lookup with the full original acronym string first (most specific)
        if conn or acronym_cache:
            _, _, cvg_status = lookup_expansion(acronym, state_name, payer_name, conn, acronym_cache=acronym_cache)
            if cvg_status:
                conf = 85.0
                manual_review = bool(conf < 80.0)
                logger.info(f"Coverage for '{log_id}': {cvg_status} (Source: DB Lookup - {acronym})")
                return cvg_status, conf, "DB Lookup", manual_review
        
        # 2. Check sub-acronyms individually if no full-string match was found
        res = determine_db_coverage(sub_acronyms, conn, state_name, payer_name, acronym_cache=acronym_cache)
        if res:
            db_cvg_status, db_conf, db_status_msg = res
            manual_review = bool(db_conf < 80.0)
            logger.info(f"Coverage for '{log_id}': {db_cvg_status} (Source: {db_status_msg} - {acronym})")
            return db_cvg_status, db_conf, "DB Lookup", manual_review

 

    # 5. Final Fallback (Should be rare)
    logger.info(f"Coverage for '{log_id}': Covered with Conditions/Unknown parameters (Source: Final Default Fallback)")
    return "Covered with Conditions", 50.0, "Default", True


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
