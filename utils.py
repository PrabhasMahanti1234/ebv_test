import re
import os
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher
import logging
import hashlib
from typing import Optional
from config import EXCEL_FILE_PATH, PDF_FOLDER
import time
import random
import json
from functools import wraps
from config import MAX_RETRIES, BACKOFF_MULTIPLIER, RATE_LIMIT_DELAY, COST_TRACKER, BEDROCK_COST_PER_1K_TOKENS, MISTRAL_OCR_COST_PER_1K_PAGES, bedrock

logger = logging.getLogger(__name__)

def rate_limited_api_call(func):
    """Decorator to add rate limiting and retry logic to API calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_call_time = getattr(wrapper, 'last_call_time', 0)
        
        for attempt in range(MAX_RETRIES):
            try:
                # Ensure minimum delay between calls
                time_since_last_call = time.time() - last_call_time
                if time_since_last_call < RATE_LIMIT_DELAY:
                    sleep_time = RATE_LIMIT_DELAY - time_since_last_call
                    logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                result = func(*args, **kwargs)
                wrapper.last_call_time = time.time()
                return result
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if attempt < MAX_RETRIES - 1:
                        # Exponential backoff with jitter
                        delay = RATE_LIMIT_DELAY * (BACKOFF_MULTIPLIER ** attempt)
                        jitter = random.uniform(0.1, 0.5)
                        total_delay = delay + jitter
                        
                        logger.warning(f"Rate limit hit, attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {MAX_RETRIES} attempts")
                        return {}  # Return empty dict for header mapping
                else:
                    logger.error(f"API error: {e}")
                    return {}
        
        return {}
    
    wrapper.last_call_time = 0
    return wrapper

def estimate_tokens(text):
    """Estimate token count - roughly 4 characters per token"""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)

def track_bedrock_cost(payer_name, prompt_text, response_text):
    """Track Bedrock API costs for a specific payer"""
    global COST_TRACKER
    
    # Estimate tokens
    prompt_tokens = estimate_tokens(prompt_text)
    response_tokens = estimate_tokens(response_text)
    total_tokens = prompt_tokens + response_tokens
    
    # Calculate cost
    cost = (total_tokens / 1000.0) * BEDROCK_COST_PER_1K_TOKENS
    
    # Update payer-specific tracking
    COST_TRACKER['payer_costs'][payer_name]['bedrock_tokens'] += total_tokens
    COST_TRACKER['payer_costs'][payer_name]['bedrock_cost'] += cost
    COST_TRACKER['payer_costs'][payer_name]['llm_calls'] += 1
    COST_TRACKER['payer_costs'][payer_name]['total_cost'] += cost
    
    # Update global tracking
    COST_TRACKER['total_tokens'] += total_tokens
    COST_TRACKER['total_llm_calls'] += 1
    COST_TRACKER['total_cost'] += cost
    
    logger.debug(f"Bedrock cost for {payer_name}: ${cost:.8f} ({total_tokens} tokens)")

def track_bedrock_cost_precalculated(payer_name, tokens, cost, calls):
    """Tracks Bedrock API costs that have already been calculated."""
    global COST_TRACKER
    
    # Update payer-specific tracking
    COST_TRACKER['payer_costs'][payer_name]['bedrock_tokens'] += tokens
    COST_TRACKER['payer_costs'][payer_name]['bedrock_cost'] += cost
    COST_TRACKER['payer_costs'][payer_name]['llm_calls'] += calls
    COST_TRACKER['payer_costs'][payer_name]['total_cost'] += cost
    
    # Update global tracking
    COST_TRACKER['total_tokens'] += tokens
    COST_TRACKER['total_llm_calls'] += calls
    COST_TRACKER['total_cost'] += cost
    
    logger.debug(f"Tracked Bedrock cost for {payer_name}: ${cost:.8f} ({tokens} tokens)")

def track_mistral_cost(payer_name, page_count):
    """Track Mistral OCR costs for a specific payer"""
    global COST_TRACKER
    
    # Calculate cost
    cost = (page_count / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES
    
    # Update payer-specific tracking
    COST_TRACKER['payer_costs'][payer_name]['mistral_ocr_pages'] += page_count
    COST_TRACKER['payer_costs'][payer_name]['mistral_cost'] += cost
    COST_TRACKER['payer_costs'][payer_name]['pdfs_processed'] += 1
    COST_TRACKER['payer_costs'][payer_name]['total_cost'] += cost
    
    # Update global tracking
    COST_TRACKER['total_pages'] += page_count
    COST_TRACKER['total_pdfs_processed'] += 1
    COST_TRACKER['total_cost'] += cost
    
    logger.info(f"Mistral OCR cost for {payer_name}: ${cost:.4f} ({page_count} pages)")

def normalize_text(text):
    """Normalize text for comparison"""
    if not text:
        return ""
    normalized = re.sub(r'[^a-zA-Z0-9]', '', str(text).lower())
    return normalized

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def clean_drug_name(text: str) -> str:
    """Cleans drug names, removing LaTeX, HTML, markdown, and extra spaces/punctuation."""
    
    if text is None or pd.isna(text):
        return "" 
    
    
    if not isinstance(text, str):
        cleaned_text = str(text)
    
    
    # Step 1: Extract and remove requirements
    cleaned_text, requirements = extract_requirements_from_drug_name(text)
    
    # --- FIX: Add a specific replacement for the 'mathrm' artifact ---
    # This is a common OCR error for units like 'ml'.
    cleaned_text = cleaned_text.replace('mathrm', '')
    
    # Step 2: Remove all LaTeX commands with braces (e.g., \mathrm{mg}, \text{ml}, etc.)
    cleaned_text = re.sub(r'\\[a-zA-Z]+\s*\{([^}]*)\}', r'\1', cleaned_text)
    
    # Step 3: Remove $ signs (math mode)
    cleaned_text = cleaned_text.replace('$', '')
    
    # Step 4: Remove leftover LaTeX symbols like ${ }^{DL}
    cleaned_text = re.sub(r'\{\s*\}', '', cleaned_text)  # Remove empty {}
    cleaned_text = re.sub(r'\^\{[^}]*\}', '', cleaned_text)  # Remove ^{DL}
    cleaned_text = re.sub(r'\^.', '', cleaned_text)  # Remove ^D style
    cleaned_text = re.sub(r'\\', ' ', cleaned_text)  # Remove stray backslashes
    
    # Step 5: Scientific notation normalization
    cleaned_text = re.sub(r'(\d+)\s*X\s*10\s*EXP\s*(\d+)', r'\1×10^\2', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(\d+)\s*EXP\s*(\d+)', r'\1×10^\2', cleaned_text)
    cleaned_text = re.sub(r'(\d+)\s*\*\s*10\s*\^\s*(\d+)', r'\1×10^\2', cleaned_text)
    
    # Step 6: Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    
    # Step 7: Remove markdown formatting
    cleaned_text = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\*(.+?)\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'_(.+?)_', r'\1', cleaned_text)
    
    # Step 8: Remove leading/trailing punctuation and normalize spaces
    cleaned_text = re.sub(r'^[,\s\$\{\}\\]+|[,\s\$\{\}\\]+$', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

def generate_filename(state_name, payer_name, plan_name):
    """Generate filename based on naming convention"""
    state_clean = re.sub(r'[^a-zA-Z0-9]', '', str(state_name)).strip() # Remove all non-alphanumeric
    payer_clean = re.sub(r'[^a-zA-Z0-9 ]', '', str(payer_name)).replace(' ', '-').strip() # Replace spaces with hyphens, remove other special chars
    plan_clean = re.sub(r'[^a-zA-Z0-9 ]', '', str(plan_name)).replace(' ', '-').strip() # Replace spaces with hyphens, remove other special chars
    return f"{state_clean}_{payer_clean}_{plan_clean}.pdf"

def parse_date_string(date_str):
    """Safely parse date string"""
    if pd.isna(date_str) or str(date_str).strip() == '':
        return None
    try:
        return datetime.strptime(str(date_str), '%d-%b-%y').date()
    except ValueError:
        logger.warning(f"Could not parse date '{date_str}'")
        return None

def validate_required_files():
    """Validate that required files and directories exist"""
    if not os.path.exists(EXCEL_FILE_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE_PATH}")
    
    # Remove or comment out the PDF folder checks below:
    # if not os.path.exists(PDF_FOLDER):
    #     raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")
    # pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    # if not pdf_files:
    #     raise FileNotFoundError(f"No PDF files found in: {PDF_FOLDER}")
    # logger.info(f"Validation passed: Found {len(pdf_files)} PDF files to process")


def extract_requirements_from_drug_name(drug_name_cell):
    """
    Extract requirements that appear after the drug name in the same cell
    Returns: (cleaned_drug_name, extracted_requirements)
    """
    if drug_name_cell is None or pd.isna(drug_name_cell):
        return "", ""

    drug_text = str(drug_name_cell).strip()
    if not drug_text:
        return "", ""

    # Common requirement patterns at the end of drug names
    requirement_patterns = [
        r'\s+(DL,LA|LA,DL|DL|LA|AV,DL|DL,AV|MO|AV|CI|PDS|CI,DL$)',
        r'\s+(PA|ST|QL|BvD|SP|TI|NM)(?:\s|$)',  # Other common requirement codes
        r'\s+([A-Z]{2,3}(?:,[A-Z]{2,3})*)$',
        r'\s*\^?\s*\{([^}]+)\}\s*\$?\s*$'  # Generic pattern for 2-3 letter codes
    ]

    extracted_requirements = []
    cleaned_name = drug_text

    for pattern in requirement_patterns:
        matches = re.findall(pattern, drug_text, re.IGNORECASE)
        if matches:
            for match in matches:
                extracted_requirements.append(match.strip())
            # Remove the matched requirement from the drug name
            cleaned_name = re.sub(pattern, '', cleaned_name, flags=re.IGNORECASE).strip()

    # Remove LaTeX \text{...} artifacts anywhere
    cleaned_name = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', cleaned_name)

    # Remove unwanted chars from both start and end (commas, spaces, $, {, }, \\)
    cleaned_name = re.sub(r'^[,\s\$\{\\}\\]+|[,\s\$\{\\}\\]+$', '', cleaned_name).strip()

    # Join extracted requirements
    final_requirements = ', '.join(extracted_requirements) if extracted_requirements else ""

    return cleaned_name, final_requirements

def lookup_expansion(acronym, state_name, payer_name, conn):
    """
    Look up expansion and coverage status from tier_requirement_expansion SQL table
    using acronym + state_name + payer_name.
    If acronym contains '$', uses ILIKE pattern matching with surrounding spaces.
    Returns (expansion, explanation, coverage_status) tuple if found, else (None, None, None).
    """
    if not acronym:
        return None, None, None

    acronym = str(acronym).strip().upper()
    state_name = str(state_name).strip() if state_name else None
    payer_name = str(payer_name).strip() if payer_name else None

    print(f"Searching for acronym: '{acronym}', state: '{state_name}', payer: '{payer_name}'")  # Debug

    with conn.cursor() as cur:
        # Check if acronym contains '$' - use pattern matching
        if '$' in acronym:
            
            pattern = f'%{acronym}%'
            
            cur.execute(
                """
                SELECT expansion, explanation, coverage_status 
                FROM tier_requirement_expansion 
                WHERE acronym ILIKE %s
                  AND (%s IS NULL OR UPPER(state_name) = UPPER(%s))
                  AND (%s IS NULL OR UPPER(payer_name) = UPPER(%s))
                LIMIT 1
                """,
                (pattern, state_name, state_name, payer_name, payer_name)
            )
        else:
            # Original exact match logic
            cur.execute(
                """
                SELECT expansion, explanation, coverage_status 
                FROM tier_requirement_expansion 
                WHERE UPPER(acronym) = %s 
                  AND (%s IS NULL OR UPPER(state_name) = UPPER(%s))
                  AND (%s IS NULL OR UPPER(payer_name) = UPPER(%s))
                LIMIT 1
                """,
                (acronym, state_name, state_name, payer_name, payer_name)
            )
        
        result = cur.fetchone()
        print("result: ", result)

    return (result[0], result[1], result[2]) if result else (None, None, None)

def parse_requirements(text):
    """
    Parse requirements text into individual components handling multiple formats.
    Examples:
        "ST,QL(180 per 30 days)" -> [("ST", None), ("QL", "180 per 30 days")]
        "ST;QL(60 per 30 days)" -> [("ST", None), ("QL", "60 per 30 days")]
    """
    if not text or pd.isna(text):
        return []
    
    requirements = []
    current = ''
    paren_count = 0
    
    # Iterate through characters to properly handle nested parentheses
    for char in str(text):
        if char == '(' and paren_count == 0:
            paren_count += 1
            current += char
        elif char == ')' and paren_count > 0:
            paren_count -= 1
            current += char
        elif char in ',;|:' and paren_count == 0:
            if current.strip():
                requirements.append(current.strip())
            current = ''
        else:
            current += char
    
    if current.strip():
        requirements.append(current.strip())
    
    # Parse each requirement into code and parameters
    parsed = []
    for req in requirements:
        # Allow / and - in code
        match = re.match(r'^([A-Za-z0-9/\-]+)(?:\s*\((.*?)\))?', req.strip())
        if match:
            code = match.group(1).strip().upper()
            params = match.group(2).strip() if match.group(2) else None
            parsed.append((code, params))
        else:
            # Handle case where requirement is just a code
            parsed.append((req.strip().upper(), None))
            
    return parsed

def normalize_requirement_code(code: str) -> str:
    """
    Normalize a requirement code for lookup.
    - Cleans LaTeX, stray $, spaces, and artifacts so that "$\$ 0$" becomes "$0"
    - Keeps $ if present at start, removes spaces and artifacts
    """
    if not code:
        return ""
    code = code.strip()
    # Special handling: normalize any variant of "$\$ 0$" to "$0"
    if re.fullmatch(r'[\$\\\s]*0[\$\\\s]*', code):
        return "$0"
    # If code starts with $, preserve it, clean the rest
    if code.startswith("$"):
        # Remove all backslashes, extra spaces, but keep the $ and numbers
        cleaned = re.sub(r'[\\s]+', '', code[1:])  # Remove backslashes and spaces after $
        code = "$" + cleaned
    else:
        code = clean_special_chars(code)
    return code.upper()

def determine_coverage_status(requirements_text, tier_text, conn, state_name, payer_name):
    """
    Determine coverage status by checking requirements in expansion table.
    - If ANY requirement is 'Covered with Conditions' → return 'Covered with Conditions'
    - If ALL requirements are 'Covered' → return 'Covered'
    - Default to 'Covered' if no requirements
    - No hard-coding of requirement codes or statuses; all logic is table-driven.
    """
    import re

    if not requirements_text and not tier_text:
        return "Covered"

    # Combine and split all requirements using common delimiters
    all_text = f"{requirements_text or ''} {tier_text or ''}"
    # Updated regex to better capture $0 variants
    requirements = re.findall(r'(\$[\\\s]*0[\\\s]*\$?|\$\/\$|\$[0-9]+|[A-Za-z0-9/\-]{2,})', all_text)

    if not requirements:
        return "Covered"

    statuses = []
    for code in requirements:
        norm_code = normalize_requirement_code(code)
        print(f"Looking up normalized code: '{norm_code}' from original: '{code}'")  # Debug print
        _, _, status = lookup_expansion(norm_code, state_name, payer_name, conn)
        if status:
            statuses.append(status.strip())

    if not statuses:
        return "Covered"

    if "Covered with Condition" in statuses or "Covered with Conditions" in statuses:
        return "Covered with Conditions"
    if all(status == "Covered" for status in statuses):
        return "Covered"
    # If there are mixed or unknown statuses, default to "Covered with Conditions"
    return "Covered with Conditions"

def parse_requirement(req_text):
    """
    Parse a single requirement into code and parameters.
    Examples:
        "ST" -> ("ST", None)
        "QL(180 per 30 days)" -> ("QL", "180 per 30 days")
    """
    if not req_text or pd.isna(req_text):
        return None, None
        
    # Match requirement code and optional parameters
    match = re.match(r'^([A-Za-z]+)(?:\s*\(([^)]+)\))?', req_text.strip())
    if match:
        code = match.group(1).strip().upper()
        params = match.group(2).strip() if match.group(2) else None
        return code, params
    return req_text.strip().upper(), None

def determine_final_coverage_status(coverage_statuses):
    """
    Determine final coverage status when multiple statuses are found.
    Order of precedence: Not Covered > Covered with Conditions > Covered
    """
    unique_statuses = set(coverage_statuses)
    
    if "Not Covered" in unique_statuses:
        return "Not Covered"
    elif "Covered with Conditions" in unique_statuses:
        return "Covered with Conditions"
    elif "Covered" in unique_statuses:
        return "Covered"
    else:
        return "Covered with Conditions"  # Safe default

# Optional: Keep Claude as fallback for cases where lookup table doesn't have coverage_status
def determine_coverage_status_with_claude_fallback(requirements_text, tier_text, conn, state_name, payer_name):
    """
    Determine coverage status using direct lookup first, Claude as fallback.
    """
    
    # Collect raw codes from requirements and tier columns
    raw_inputs = []
    for raw_text in [requirements_text, tier_text]:
        if raw_text and not pd.isna(raw_text):
            parts = re.split(r'[;,|:]', str(raw_text))
            raw_inputs.extend([p.strip() for p in parts if p.strip()])

    # Look up from SQL table
    coverage_statuses = []
    expansions_without_status = []
    
    for code in raw_inputs:
        expansion, explanation, coverage_status = lookup_expansion(code, state_name, payer_name, conn)
        
        if coverage_status:  # Direct lookup found coverage status
            coverage_statuses.append(coverage_status)
        elif expansion:  # Found expansion but no coverage status - use Claude
            if explanation:
                combined_text = f"{expansion} ({explanation})"
            else:
                combined_text = expansion
            expansions_without_status.append(combined_text)

    # If we have direct coverage statuses, use those
    if coverage_statuses:
        direct_status = determine_final_coverage_status(coverage_statuses)
        
        # If we also have expansions without status, check with Claude
        if expansions_without_status:
            claude_status = call_claude_for_coverage("; ".join(expansions_without_status))
            
            # Combine results - if either indicates conditions, return conditions
            if direct_status == "Covered with Conditions" or claude_status == "Covered with Conditions":
                return "Covered with Conditions"
            elif direct_status == "Not Covered" or claude_status == "Not Covered":
                return "Not Covered"
            else:
                return "Covered"
        else:
            return direct_status
    
    # No direct coverage statuses found
    elif expansions_without_status:
        return call_claude_for_coverage("; ".join(expansions_without_status))
    
    # Fallback logic
    if not requirements_text or pd.isna(requirements_text):
        return "Covered"
    return "Covered with Conditions"


def call_claude_for_coverage(expansion_text):
    """
    Use Claude 3 Haiku (AWS Bedrock) to classify coverage status.
    (Keep this function unchanged for fallback scenarios)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
You are an expert in interpreting health plan formulary rules.
Given the following expansion text: "{expansion_text}", decide if the drug is:
- "Covered with Conditions" (if it requires prior authorization, step therapy, quantity limits, specialty restrictions, or non-formulary)
- "Covered" (if no restrictions are indicated)

Task: Decide if the drug is:
- "Covered with Conditions" → if it requires prior authorization, step therapy, quantity limits, specialty restrictions, or is non-formulary
- "Covered" → if no restrictions are indicated

⚠️ Important: Respond with ONLY one of these exact phrases:
- Covered
- Covered with Conditions
No explanation, no extra text.
"""
                }
            ],
        }
    ]

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "temperature": 0,
                "messages": messages
            })
        )

        result = json.loads(response["body"].read())

        if "content" in result and len(result["content"]) > 0:
            return result["content"][0]["text"].strip()

        raise ValueError(f"Unexpected Claude response format: {result}")
    
    except Exception as e:
        print(f"[ERROR] Claude API call failed: {e}")
        return "Covered with Conditions"

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

def calculate_bytes_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of a file's content provided in bytes."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content)
    return sha256_hash.hexdigest()


def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file on disk."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks of 4K for memory efficiency
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


import re
from typing import Optional

def clean_special_chars(cleaned_text):
    """Simple cleaning for special characters and formatting artifacts"""
    if not cleaned_text:
        return ""
    
    if cleaned_text is None or pd.isna(cleaned_text):
        return "" 
    
    if not isinstance(cleaned_text, str):
        cleaned_text = str(cleaned_text)
    # This is a common OCR error for units like 'ml'.
    cleaned_text = cleaned_text.replace('mathrm', '')
    
    # Step 2: Remove all LaTeX commands with braces (e.g., \mathrm{mg}, \text{ml}, etc.)
    cleaned_text = re.sub(r'\\[a-zA-Z]+\s*\{([^}]*)\}', r'\1', cleaned_text)
    
    # Step 3: Remove $ signs (math mode)
    cleaned_text = cleaned_text.replace('$', '')
    
    # Step 4: Remove leftover LaTeX symbols like ${ }^{DL}
    cleaned_text = re.sub(r'\{\s*\}', '', cleaned_text)  # Remove empty {}
    cleaned_text = re.sub(r'\^\{[^}]*\}', '', cleaned_text)  # Remove ^{DL}
    cleaned_text = re.sub(r'\^.', '', cleaned_text)  # Remove ^D style
    cleaned_text = re.sub(r'\\', ' ', cleaned_text)  # Remove stray backslashes
    
    # Step 5: Scientific notation normalization
    cleaned_text = re.sub(r'(\d+)\s*X\s*10\s*EXP\s*(\d+)', r'\1×10^\2', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(\d+)\s*EXP\s*(\d+)', r'\1×10^\2', cleaned_text)
    cleaned_text = re.sub(r'(\d+)\s*\*\s*10\s*\^\s*(\d+)', r'\1×10^\2', cleaned_text)
    
    # Step 6: Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    
    # Step 7: Remove markdown formatting
    cleaned_text = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\*(.+?)\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'_(.+?)_', r'\1', cleaned_text)
    
    # Step 8: Remove leading/trailing punctuation and normalize spaces
    cleaned_text = re.sub(r'^[,\s\$\{\}\\]+|[,\s\$\{\}\\]+$', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def normalize_drug_tier(raw_tier):
    """Clean raw tier text and return the cleaned value or None"""
    if not raw_tier:
        return None
    
    cleaned = clean_special_chars(raw_tier)
    return cleaned if cleaned else None

def infer_drug_tier_from_text(text: Optional[str]) -> Optional[str]:
    """
    Tries to find explicit tier mentions (e.g., "Tier 1", "Tier 2") inside a longer text blob.
    This is a conservative function to avoid incorrectly identifying dosages or other numbers as tiers.
    """
    if not text or pd.isna(text):
        return None

    # Search for the pattern "Tier" followed by 1 or 2 digits.
    # This is much safer than just cleaning the text and returning it.
    match = re.search(r'(Tier\s*\d{1,2})', str(text), re.IGNORECASE)

    if match:
        # Return the found tier, e.g., "Tier 1"
        return match.group(1).strip()

    # If no "Tier X" pattern is found, return None to prevent incorrect inference.
    return None