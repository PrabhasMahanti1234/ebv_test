import re
import pandas as pd
import logging
from typing import Optional, Tuple
from database import get_db_connection
from clasify import ml_predict_coverage_status, ml_predict_coverage_status_batch
# Setup logging
logger = logging.getLogger(__name__)

# Setup ML Output Logger
ml_logger = logging.getLogger('ml_coverage_logger')
ml_logger.setLevel(logging.INFO)
# Prevent propagation to root logger to avoid duplicate logs in main log
ml_logger.propagate = False 

# Create file handler
fh = logging.FileHandler('ml_output.log')
fh.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)

# Add handler to logger
if not ml_logger.handlers:
    ml_logger.addHandler(fh)

def determine_ml_coverage_status(coverage_statuses):
    """
    Determine final coverage status when multiple statuses are found.
    Order of precedence: Not Covered > Covered with Conditions > Covered
    """
    normalized = {s.strip().title() for s in coverage_statuses if s}

    if "Not Covered" in normalized:
        return "Not Covered"
    elif "Covered With Pa" in normalized or "Covered With Prior Authorization" in normalized: 
        return "Covered with PA"
    elif "Covered With St" in normalized or "Covered With Step Therapy" in normalized:  
        return "Covered with ST"
    elif "Covered With Conditions" in normalized or "Covered With Condition" in normalized:
        return "Covered with Condition"
    elif "Covered" in normalized:
        return "Covered"
    else:
        return "Covered/Unknown"

def detect_prior_authorization(requirements_text):
    """
    Detect if Prior Authorization is required based on requirements text
    Returns True if PA is required, False otherwise
    """
    if not requirements_text or pd.isna(requirements_text):
        return False
    
    requirements_lower = str(requirements_text).lower().strip()
    
    if not requirements_lower or requirements_lower in ['', 'none', 'null', 'nan']:
        return False
    
    # Common PA indicators
    pa_keywords = [
        'prior authorization', 'prior auth', 'pa required', 'pa needed',
        'pa;', 'PA', 'pa', 
        'pa,', 'authorization required', 'PA;',
        'PA,', 'requires prior authorization',
        'must be approved', 'approval needed', 'prior approval needed'
    ]
    
    return any(keyword in requirements_lower for keyword in pa_keywords)

def detect_step_therapy(requirements_text):
    """
    Detect if Step Therapy is required based on requirements text  
    Returns True if ST is required, False otherwise
    """
    if not requirements_text or pd.isna(requirements_text):
        return False
    
    requirements_lower = str(requirements_text).lower().strip()
    
    if not requirements_lower or requirements_lower in ['', 'none', 'null', 'nan']:
        return False
    
    # Common ST indicators
    st_keywords = [
        'step therapy', 'step-therapy', 'st', 'ST',
        'ST;', 'st;', 'st,',
        'ST,'
    ]
    
    return any(keyword in requirements_lower for keyword in st_keywords)

def det_coverage_status(
    acronym=None,
    expansion=None,
    explanation=None,
    requirements_text=None,
    tier_text=None,
    conn=None,
    state_name=None,
    payer_name=None,
    ml_predict_fn=None,
    drug_name=None
):
    """
    Main orchestrator for coverage status determination.
    Consolidated logic: Tier NC > Python Logic (PA/ST/QL) > ML Model > Default
    """
    # Identifier for logging
    log_id = drug_name or acronym or "Unknown Drug"

    # Use the imported ML prediction function if not provided
    if ml_predict_fn is None:
        ml_predict_fn = ml_predict_coverage_status

    tier_text_clean = str(tier_text or "").upper().strip()
    req_text_clean = str(requirements_text or "").upper().strip()
    nc_indicators = ["NC", "NF", "NOT COVERED", "EXCLUDED", "NON-FORMULARY", "NOT-COVERED"]

    # 1. Both Null Fallback: If both are null AND no acronym context, return Covered/Unknown
    #    BUT if we have an acronym, use the acronym itself as the requirements indicator
    #    e.g. acronym="PA" → Covered with PA, "ST" → Covered with ST, "QL" → Covered with Conditions
    if not tier_text_clean and not req_text_clean:
        if not acronym:
            logger.info(f"Coverage for '{log_id}': Covered/Unknown (Source: Default Fallback - Both Tier and Req null)")
            return "Covered/Unknown", 0.5, "Default"
        
        # ✅ Derive requirements signal from the acronym itself
        acronym_upper = str(acronym).upper().strip()
        acronym_tokens = set(t.strip() for t in acronym_upper.replace("/", " ").replace(",", " ").split())
        
        # NC/NF → Not Covered
        nc_acronyms = {"NC", "NF", "NOT COVERED", "NON-FORMULARY", "EXCLUDED"}
        if acronym_tokens & nc_acronyms:
            logger.info(f"Coverage for '{log_id}': Not Covered (Source: Acronym NC Indicator '{acronym}')")
            return "Not Covered", 0.9, "Acronym (NC)"
        
        # PA → Covered with PA
        pa_acronyms = {"PA", "PRIOR AUTH", "PRIOR AUTHORIZATION"}
        if acronym_tokens & pa_acronyms:
            # Also check if ST present in same acronym e.g. "PA ST"
            st_acronyms = {"ST", "STEP THERAPY"}
            if acronym_tokens & st_acronyms:
                logger.info(f"Coverage for '{log_id}': Covered with PA+ST (Source: Acronym '{acronym}')")
                return "Covered with PA", 0.85, "Acronym (PA+ST)"
            logger.info(f"Coverage for '{log_id}': Covered with PA (Source: Acronym '{acronym}')")
            return "Covered with PA", 0.85, "Acronym (PA)"
        
        # ST → Covered with ST
        st_acronyms = {"ST", "STEP THERAPY"}
        if acronym_tokens & st_acronyms:
            logger.info(f"Coverage for '{log_id}': Covered with ST (Source: Acronym '{acronym}')")
            return "Covered with ST", 0.85, "Acronym (ST)"
        
        # QL/LA/AL/BVD/MME/SP/DL/7D/EC → Covered with Conditions
        conditions_acronyms = {"QL", "LA", "AL", "BVD", "MME", "SP", "DL", "7D", "EC", "QUANTITY LIMIT",
                                "QUANTITY LIMITS", "LIMITED ACCESS", "SPECIALTY", "DISPENSING LIMIT"}
        if acronym_tokens & conditions_acronyms or "QL" in acronym_upper or "LIMIT" in acronym_upper:
            logger.info(f"Coverage for '{log_id}': Covered with Conditions (Source: Acronym '{acronym}')")
            return "Covered with Conditions", 0.8, "Acronym (Conditions)"
        
        # If acronym is known but doesn't match above, fall through to ML model below
        logger.info(f"Coverage for '{log_id}': No acronym rule matched for '{acronym}', proceeding to ML")


    # 2. Tier NC Priority: if tier has nc indicators it is not covered
    if tier_text_clean and any(ind in tier_text_clean for ind in nc_indicators):
        logger.info(f"Coverage for '{log_id}': Not Covered (Source: Python Logic - NC Tier '{tier_text_clean}')")
        return "Not Covered", 1.0, "Python Logic (NC Tier)"

    # 3. Python Rule Logic: Check for PA, ST, QL, AL, LA, BVD, etc.
    is_pa = detect_prior_authorization(req_text_clean)
    is_st = detect_step_therapy(req_text_clean)
    is_other_cond = any(code in req_text_clean for code in ["QL", "AL", "LA", "BVD"])
    is_specialty = "SP" in tier_text_clean or "SPECIALTY" in tier_text_clean
    is_nc_req = any(ind in req_text_clean for ind in nc_indicators)

    if is_nc_req:
        logger.info(f"Coverage for '{log_id}': Not Covered (Source: Python Logic - NC Requirement '{req_text_clean}')")
        return "Not Covered", 0.9, "Python Logic (NC Req)"
    
    if is_pa:
        reason = ["PA"]
        if is_st: reason.append("ST")
        if is_other_cond: reason.append("Other Conditions")
        if is_specialty: reason.append("Specialty Tier")
        logger.info(f"Coverage for '{log_id}': Covered with PA (Source: Python Logic - {', '.join(reason)})")
        return "Covered with PA", 0.85, "Python Logic"
    
    if is_st:
        reason = ["ST"]
        if is_other_cond: reason.append("Other Conditions")
        if is_specialty: reason.append("Specialty Tier")
        logger.info(f"Coverage for '{log_id}': Covered with ST (Source: Python Logic - {', '.join(reason)})")
        return "Covered with ST", 0.85, "Python Logic"

    if is_other_cond or is_specialty:
        reason = []
        if is_other_cond: reason.append("Other Conditions")
        if is_specialty: reason.append("Specialty Tier")
        logger.info(f"Coverage for '{log_id}': Covered with Conditions (Source: Python Logic - {', '.join(reason)})")
        return "Covered with Conditions", 0.85, "Python Logic"

#    # 4. ML Model Logic (Requirement and Tier)
    if acronym:
        # Split acronyms by comma if present
        sub_acronyms = [a.strip() for a in str(acronym).split(',') if a.strip()]
        
        ml_predictions = []
        
        # Ensure we can look up expansion
        from utils import lookup_expansion

        for sub_acronym in sub_acronyms:
            # Determine initial expansion/explanation for this part
            # If there's only one acronym, use the provided arguments as base
            sub_expansion = expansion if len(sub_acronyms) == 1 else None
            sub_explanation = explanation if len(sub_acronyms) == 1 else None
            
            # DB Lookup to enrich or fill missing info
            if conn:
                db_exp, db_expl, _ = lookup_expansion(sub_acronym, state_name, payer_name, conn)
                sub_expansion = sub_expansion or db_exp
                sub_explanation = sub_explanation or db_expl
            
            # Predict
            pred_label, pred_conf = ml_predict_fn(
                payer_name, 
                state_name, 
                sub_acronym, 
                sub_expansion, 
                sub_explanation
            )
            
            # Log individual prediction to ml_output.log
            ml_logger.info(f"Drug: {log_id} | Input: {sub_acronym} | Expansion: {sub_expansion} | Explanation: {sub_explanation} | Prediction: {pred_label} | Confidence: {pred_conf} | Payer: {payer_name} | State: {state_name}")

            if pred_label:
                ml_predictions.append((pred_label, pred_conf))
        
        if ml_predictions:
            statuses = [p[0] for p in ml_predictions]
            final_status = determine_ml_coverage_status(statuses)
            
            # Determine confidence based on the winning status
            final_conf = 0.0
            normalized_target = final_status.lower().replace("conditions", "condition")
            
            for label, conf in ml_predictions:
                 normalized_label = label.lower().replace("conditions", "condition")
                 if normalized_label == normalized_target:
                     final_conf = max(final_conf, conf)
            
            # Log final decision to ml_output.log
            ml_logger.info(f"Drug: {log_id} | Final ML Status: {final_status} | Final Confidence: {final_conf}")

            logger.info(f"Coverage for '{log_id}': {final_status} (Source: ML Model - {acronym})")
            return final_status, final_conf, "ML Model"
    # ========================================
    # TIER 4: DEFAULT FALLBACK
    # ========================================
    # If no specific logic matched, default to "Covered"
    # This applies to standard tier drugs without special requirements
    logger.info(f"No specific coverage logic matched. Defaulting to 'Covered' with confidence 0.5")
    return "Covered", 0.5, "Default"

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
