import os
import re
import json
import pandas as pd
import logging
import traceback
import requests
import httpx
import uuid
import time
import concurrent.futures
import PyPDF2
from typing import Tuple, List
from pathlib import Path
from mistralai.models.sdkerror import SDKError
from mistralai import Mistral
from mistralai.models import DocumentURLChunk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from config import (
    PDF_FOLDER, PROCESS_COUNT, MISTRAL_API_KEY, BEDROCK_MODEL_ID, BEDROCK_COST_PER_1K_TOKENS,bedrock,
    MISTRAL_OCR_COST_PER_1K_PAGES, BEDROCK_COST_PER_1K_TOKENS, LLM_PAGE_WORKERS,
    MAX_RETRIES, BACKOFF_MULTIPLIER, CLIENT_TIMEOUT, CONNECT_TIMEOUT
)
from database import get_db_connection, batch_determine_coverage_status, get_cached_result, cache_result, update_plan_file_hash, insert_acronyms_to_ref_table, insert_drug_formulary_data, delete_drug_formulary_records_for_plan
from utils import (
    similarity, clean_drug_name, detect_prior_authorization,
    detect_step_therapy, calculate_file_hash, rate_limited_api_call,
    track_bedrock_cost_precalculated, track_mistral_cost, determine_coverage_status,
    normalize_drug_tier, infer_drug_tier_from_text, calculate_bytes_hash,
    parse_complex_drug_name, similarity, normalize_requirement_code
)

logger = logging.getLogger(__name__)
 
MAX_PDF_PAGES = 2000
PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "default.txt"
PROMPT_MAPPINGS_FILE = PROMPTS_DIR / "prompt_mappings.json"

# Cache prompts in memory to avoid repeated file reads
_PROMPT_CACHE = {}
_AVAILABLE_PROMPTS = {}
_PROMPT_MAPPINGS = {}


# json5 is not a standard library, so we handle its absence gracefully.
try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    JSON5_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available. Page count check will be skipped.")


def _sanitize_escape_sequences(json_string: str) -> str:
    r"""
    Sanitizes invalid escape sequences in JSON string values.
    Fixes issues like \e, \x, invalid \u sequences that break JSON parsing.
    """
    # Pattern to match string values in JSON (content between quotes)
    # This regex handles escaped quotes and finds string boundaries
    def fix_escapes_in_string(match):
        string_content = match.group(1)  # Content inside quotes (without the quotes)
        # Replace invalid escape sequences
        # Valid escapes are: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        # Invalid ones like \e, \x (not followed by hex), etc. need to be escaped
        
        # Fix invalid single-character escapes (not in list of valid ones)
        # Pattern matches \ followed by a character that's not a valid escape
        fixed = re.sub(r'\\(?![nrtbfu"\\/0-9x])', r'\\\\', string_content)
        
        # Fix invalid \u sequences (must be followed by 4 hex digits)
        # Replace \u not followed by 4 hex digits with \\u
        fixed = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', fixed)
        
        # Fix incomplete \x sequences (must be followed by 2 hex digits)
        # Replace \x not followed by 2 hex digits with \\x
        fixed = re.sub(r'\\x(?![0-9a-fA-F]{2})', r'\\\\x', fixed)
        
        return f'"{fixed}"'
    
    # Match strings: "content" but handle escaped quotes
    # This is tricky - we'll use a state machine approach
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(json_string):
        char = json_string[i]
        
        if escape_next:
            # Current char is escaped - check if it's valid
            valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u', 'x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
            
            if char == 'u':
                # Check if next 4 chars are hex digits
                if i + 4 < len(json_string) and all(c in '0123456789abcdefABCDEF' for c in json_string[i+1:i+5]):
                    result.append(f'\\u{json_string[i+1:i+5]}')
                    i += 5
                    escape_next = False
                    continue
                else:
                    # Invalid \u - escape the backslash
                    result.append('\\\\u')
                    escape_next = False
                    # Don't increment i - process 'u' normally
                    continue
            elif char == 'x':
                # Check if next 2 chars are hex digits
                if i + 2 < len(json_string) and all(c in '0123456789abcdefABCDEF' for c in json_string[i+1:i+3]):
                    result.append(f'\\x{json_string[i+1:i+3]}')
                    i += 3
                    escape_next = False
                    continue
                else:
                    # Invalid \x - escape the backslash
                    result.append('\\\\x')
                    escape_next = False
                    continue
            elif char in valid_escapes:
                result.append(f'\\{char}')
                escape_next = False
            else:
                # Invalid escape - escape the backslash, keep the char
                result.append(f'\\\\{char}')
                escape_next = False
            
            i += 1
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            i += 1
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def _extract_partial_json_arrays(json_string: str) -> dict:
    """
    Attempts to extract JSON arrays even when the full JSON object is corrupted.
    Uses regex to find and parse individual array sections.
    """
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}
    
    # Pattern to match array sections: "key": [...]
    array_pattern = r'"(\w+)":\s*(\[[\s\S]*?\])'
    
    extracted = {}
    
    for match in re.finditer(array_pattern, json_string):
        key = match.group(1)
        array_str = match.group(2)
        
        if key not in ['drug_table', 'acronyms', 'tiers']:
            continue
        
        try:
            # Try to parse just this array
            # First fix trailing commas in the array
            array_str = re.sub(r',\s*(\])', r'\1', array_str)
            # Try to extract valid JSON objects/strings from the array
            parsed_array = json.loads(array_str)
            extracted[key] = parsed_array
        except:
            # If parsing fails, try to extract objects using regex
            if key == 'drug_table':
                # Extract drug objects: {"drug_name": "...", "drug_tier": "...", ...}
                drug_objects = []
                object_pattern = r'\{[^}]*"drug_name"[^}]*\}'
                for obj_match in re.finditer(object_pattern, array_str):
                    try:
                        obj_str = obj_match.group(0)
                        obj_str = re.sub(r',\s*(\})', r'\1', obj_str)
                        obj = json.loads(obj_str)
                        drug_objects.append(obj)
                    except:
                        continue
                extracted[key] = drug_objects
            else:
                # For acronyms and tiers, try simpler extraction
                extracted[key] = []
    
    # Fill missing keys with empty lists
    for key in ['drug_table', 'acronyms', 'tiers']:
        if key not in extracted:
            extracted[key] = []
    
    return extracted


def robust_json_repair(json_string: str):
    """
    Attempts to repair common JSON errors from LLMs with multiple strategies:
    1. Fix trailing commas
    2. Sanitize invalid escape sequences
    3. Use lenient parsers (json5)
    4. Extract partial JSON when full parse fails
    """
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}

    if not isinstance(json_string, str) or not json_string.strip():
        return default_output

    # 1. Find the start and end of the main JSON object.
    start_index = json_string.find('{')
    end_index = json_string.rfind('}')
    
    if start_index == -1 or end_index == -1:
        logger.warning("Could not find a JSON object in the LLM response.")
        return default_output
    
    # Extract the JSON part of the string.
    json_string = json_string[start_index : end_index + 1]

    # 2. Fix the most common LLM error: trailing commas before '}' or ']'.
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)

    # 3. NEW: Sanitize invalid escape sequences
    try:
        json_string = _sanitize_escape_sequences(json_string)
    except Exception as e:
        logger.debug(f"Escape sequence sanitization encountered an issue: {e}")

    # 4. Attempt to parse the cleaned JSON.
    try:
        # Use json5 for more lenient parsing if available.
        if JSON5_AVAILABLE:
            try:
                parsed = json5.loads(json_string)
                return _sanitize_output(parsed, default_output)
            except Exception as e:
                logger.debug(f"json5 parsing failed, falling back to standard json: {e}")
        
        # Fallback to the standard json library.
        parsed = json.loads(json_string)
        return _sanitize_output(parsed, default_output)

    except json.JSONDecodeError as e:
        logger.debug(f"Standard JSON parsing failed: {e}. Attempting partial extraction...")
        
        # 5. NEW: Try partial extraction as last resort
        try:
            partial_result = _extract_partial_json_arrays(json_string)
            if any(partial_result.values()):  # If we extracted anything
                logger.info(f"Successfully extracted partial JSON: {sum(len(v) for v in partial_result.values())} total items")
                return _sanitize_output(partial_result, default_output)
        except Exception as partial_error:
            logger.debug(f"Partial extraction also failed: {partial_error}")
        
        # Log the failed JSON to a file for analysis.
        logger.error(f"JSON parsing failed definitively after all repair attempts: {e}")
        try:
            with open("failed_llm_json.log", "a", encoding="utf-8") as f:
                f.write(f"=== JSON Parse Error ===\n")
                f.write(f"Error: {e}\n")
                f.write(f"Original String (first 1000 chars): {json_string[:1000]}\n")
                f.write(f"{'='*50}\n\n")
        except Exception as log_error:
            logger.warning(f"Failed to write to debug log: {log_error}")
            
        return default_output


def _sanitize_output(parsed_data, default_output):
    """
    Ensures the parsed output conforms to the expected dictionary structure
    with the correct keys, returning empty lists for any missing keys.
    """
    if not isinstance(parsed_data, dict):
        return default_output

    # Ensure all three primary keys exist in the final output.
    sanitized = {
        "drug_table": parsed_data.get("drug_table", []),
        "acronyms": parsed_data.get("acronyms", []),
        "tiers": parsed_data.get("tiers", []),
    }
    return sanitized


def extract_metadata_from_filename(filename):
    """Extract state, payer, and plan name from filename"""
    base = os.path.splitext(filename)[0]
    parts = base.split("_", 2)
    if len(parts) != 3:
        logger.error(f"Filename format incorrect: {filename}. Expected State_Payer_Plan.")
        raise ValueError(f"Filename format incorrect: {filename}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def is_index_page(markdown: str) -> bool:
    """
    Detect if a page is an index/table of contents.
    Returns True if index, False otherwise.
    """
    lines = markdown.splitlines()
    
    # Count table cells that look like index entries
    table_index_count = 0
    table_cell_count = 0
    
    # Pattern for index entries: text followed by dots and/or spaces, then a number
    index_entry_pattern = re.compile(r'[A-Za-z0-9\s\(\)\-]+\.{2,}\s*\d+')
    
    for line in lines:
        # Check if this is a table row with content
        if '|' in line and not line.strip().startswith(':'):
            # Split by pipe and process each cell
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            
            for cell in cells:
                if cell:  # Non-empty cell
                    table_cell_count += 1
                    # Check if cell contains an index-like entry
                    if index_entry_pattern.search(cell):
                        table_index_count += 1
    
    # If we found table-based index entries, check if threshold is met
    if table_cell_count > 0:
        index_ratio = table_index_count / table_cell_count
        # If at least 40% of table cells are index entries, it's an index page
        if index_ratio >= 0.4:
            logger.info("Detected index page based on table content.")
            return True
    
    # Also check for non-table format
    content_lines = [
        line.strip() for line in lines 
        if line.strip() and not line.strip().startswith('|') and not line.strip().startswith(':')
    ]
    
    if not content_lines:
        return False
    
    # Match lines like "DRUGNAME.....5" or "DRUGNAME 5" or "DRUGNAME (DOSE).....5"
    index_pattern = re.compile(r'^[A-Za-z0-9\s\(\)\-\.,]+\.{2,}\s*\d+\s*$')
    alt_pattern = re.compile(r'^[A-Za-z0-9\s\(\)\-\.,]+\s+\d+\s*$')
    
    index_lines = sum(
        1 for line in content_lines
        if index_pattern.search(line) or alt_pattern.search(line)
    )
    
    # If at least 10% of content lines match the index pattern, it's likely an index page
    if len(content_lines) > 0 and (index_lines / len(content_lines)) >= 0.1:
        logger.info("Detected index page based on line patterns.")
        return True
        
    return False

def is_aca_drug_list_page(markdown: str) -> bool:
    """
    Detects if a page is part of an 'ACA Drug List' or 'Preventative Medications' section
    using a heuristic scoring system. This is more robust than simple keyword matching.

    Returns True if the page's score exceeds a confidence threshold, False otherwise.
    """
    score = 0
    # A score of 10 or more gives high confidence that this page should be skipped.
    CONFIDENCE_THRESHOLD = 10
    
    lower_markdown = markdown.lower()

    # --- Feature 1: The Strongest Signal - The BRAND/GENERIC Table Header ---
    # This structure is unique to these lists and absent from the main formulary.
    # We use regex to be precise about the table format.
    if re.search(r'\|\s*brand\s*\|\s*generic\s*\|', lower_markdown):
        logger.debug("ACA page score +8 for BRAND/GENERIC header.")
        score += 8

    # --- Feature 2: High-Confidence Titles ---
    # These titles are very unlikely to appear on a standard formulary page.
    # We check if they appear as standalone lines (typical for a title).
    high_confidence_titles = [
        r'^\s*aca drug list\s*$',
        r'^\s*preventative medications and preferred contraceptives\s*$',
        r'^\s*breast cancer prevention\s*$',
        r'^\s*tobacco cessation\s*$',
        r'^\s*bowel preparation\s*$',
        r'^\s*pre-exposure prophylaxis \(prep\)\*\*\s*$'
    ]
    for title_pattern in high_confidence_titles:
        if re.search(title_pattern, lower_markdown, re.MULTILINE):
            logger.debug(f"ACA page score +5 for title: {title_pattern}")
            score += 5

    # --- Feature 3: Supporting Keywords ---
    # These words add confidence but aren't strong enough on their own.
    supporting_keywords = [
        'affordable care act',
        'preventive services',
        'contraceptives',
        'statins*',
        'fluoride products',
        'iron products'
    ]
    for keyword in supporting_keywords:
        if keyword in lower_markdown:
            logger.debug(f"ACA page score +2 for keyword: {keyword}")
            score += 2
            
    # --- Final Decision ---
    if score >= CONFIDENCE_THRESHOLD:
        logger.info(f"Detected ACA/Preventative drug list page with a confidence score of {score}. Skipping.")
        return True

    return False

 
@rate_limited_api_call
def extract_structured_data_with_llm(page_markdown: str, payer_name: str = None):
    """
    Uses Claude 3 Haiku to parse markdown and extract structured drug data.
    """
    costs = {'tokens': 0, 'cost': 0.0, 'calls': 1}
    if not bedrock:
        logger.error("Bedrock client is not initialized. Cannot extract structured data.")
        return {"drug_table": [], "acronyms": [], "tiers": []}, costs

    if is_index_page(page_markdown):
        logger.info("Skipping LLM call for index/table of contents page.")
        return {"drug_table": [], "acronyms": [], "tiers": []}, {'tokens': 0, 'cost': 0.0, 'calls': 0}
    
    if is_aca_drug_list_page(page_markdown):
        logger.info("Skipping LLM call for ACA Drug List/Preventative Medications page.")
        return {"drug_table": [], "acronyms": [], "tiers": []}, {'tokens': 0, 'cost': 0.0, 'calls': 0}

    system_prompt = get_payer_prompt(payer_name)
    user_message = f"<INPUT_MARKDOWN>\n{page_markdown}\n</INPUT_MARKDOWN>"

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096,
            "system": system_prompt, "messages": [{"role": "user", "content": user_message}]
        })
        response = bedrock.invoke_model(body=body, modelId=BEDROCK_MODEL_ID)
        response_body = json.loads(response.get('body').read())
        response_text = response_body['content'][0]['text']
        
        usage = response_body.get('usage', {})
        total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        costs['tokens'] = total_tokens
        costs['cost'] = (total_tokens / 1000.0) * BEDROCK_COST_PER_1K_TOKENS

        logger.debug(f"Raw LLM response (first 500 chars): {response_text[:500]}...")
        
        # Use the robust repair and parsing function.
        structured_data = robust_json_repair(response_text)

        logger.info(f"Successfully processed page. Extracted: {len(structured_data.get('drug_table', []))} drugs, {len(structured_data.get('acronyms', []))} acronyms, {len(structured_data.get('tiers', []))} tiers.")
        
        # Defensive filtering to remove non-English terms the LLM might have missed.
        blocklist = {'nivel'}
        for key in ['acronyms', 'tiers']:
            if key in structured_data and isinstance(structured_data[key], list):
                filtered_list = []
                for item in structured_data[key]:
                    if isinstance(item, dict):
                        acronym = str(item.get('acronym') or '').lower()
                        expansion = str(item.get('expansion') or '').lower()
                        # Only keep the item if neither field contains a blocked word
                        if acronym not in blocklist and expansion not in blocklist:
                            filtered_list.append(item)
                    elif isinstance(item, str):
                        # Also filter out simple strings that are in the blocklist
                        if item.lower() not in blocklist:
                            logger.warning(f"LLM returned a string '{item}' in list '{key}'. Converting to dict.")
                            filtered_list.append({'acronym': item, 'expansion': None, 'explanation': None})
                structured_data[key] = filtered_list

        return structured_data, costs

    except Exception as e:
        logger.error(f"Error in Claude 3 Haiku LLM data extraction: {e}")
        response_text_for_log = locals().get('response_text', 'No response text captured')
        try:
            with open("llm_errors.log", "a", encoding="utf-8") as f:
                f.write(f"=== LLM Error ===\n")
                f.write(f"Error: {e}\n")
                f.write(f"Response text: {response_text_for_log}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write(f"{'='*50}\n\n")
        except Exception as log_error:
            logger.warning(f"Failed to write to error log: {log_error}")
        return {"drug_table": [], "acronyms": [], "tiers": []}, costs

def _load_prompts_config():
    """Scans the prompts directory and loads all prompts and mappings into memory."""
    global _AVAILABLE_PROMPTS, _PROMPT_MAPPINGS, _PROMPT_CACHE

    if _AVAILABLE_PROMPTS: # Already loaded
        return

    logger.info("[PROMPT SELECTION] Initializing and caching prompts from disk...")
    if not PROMPTS_DIR.is_dir():
        logger.warning(f"Prompts directory not found: {PROMPTS_DIR}")
        return

    # Scan for all .txt files and create a mapping of normalized_name -> file_path
    for file_path in PROMPTS_DIR.glob("*.txt"):
        normalized_name = file_path.stem.lower().replace(" ", "_").replace("-", "_")
        _AVAILABLE_PROMPTS[normalized_name] = file_path

    # Load the fuzzy matching configuration from the new JSON file
    if PROMPT_MAPPINGS_FILE.exists():
        try:
            with open(PROMPT_MAPPINGS_FILE, 'r', encoding='utf-8') as f:
                _PROMPT_MAPPINGS = json.load(f)
            logger.info(f"Loaded {len(_PROMPT_MAPPINGS)} fuzzy prompt mappings from {PROMPT_MAPPINGS_FILE.name}")
        except Exception as e:
            logger.error(f"Failed to load prompt mappings from {PROMPT_MAPPINGS_FILE.name}: {e}")


def get_payer_prompt(payer_name: str = None) -> str:
    """
    Loads a payer-specific prompt, falling back to a default.
    This version is optimized to scan the directory once and uses a config file for fuzzy matching.
    """
    # Ensure prompts and mappings are loaded into memory
    _load_prompts_config()

    # Helper to load a prompt from file and cache it
    def _load_and_cache_prompt(file_path: Path, prompt_key: str) -> str:
        if prompt_key in _PROMPT_CACHE:
            return _PROMPT_CACHE[prompt_key]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            _PROMPT_CACHE[prompt_key] = prompt
            return prompt
        except Exception as e:
            logger.error(f"Failed to read prompt file {file_path.name}: {e}")
            return None # Return None to trigger fallback

    # 1. Handle default case
    if not payer_name or not payer_name.strip():
        logger.info("[PROMPT SELECTION] No payer name provided. Using DEFAULT prompt.")
        return _load_and_cache_prompt(DEFAULT_PROMPT_FILE, 'default') or _load_default_prompt()

    # 2. Try for an exact match using the normalized name
    payer_normalized = payer_name.strip().lower().replace(" ", "_").replace("-", "_")
    if payer_normalized in _AVAILABLE_PROMPTS:
        prompt_file = _AVAILABLE_PROMPTS[payer_normalized]
        logger.info(f"[PROMPT SELECTION] ✓ Found direct match for '{payer_name}' -> {prompt_file.name}")
        return _load_and_cache_prompt(prompt_file, payer_normalized) or _load_default_prompt()

    # 3. Try for a fuzzy match using the mappings configuration
    for key, prompt_filename_stem in _PROMPT_MAPPINGS.items():
        if key.lower() in payer_name.strip().lower():
            if prompt_filename_stem in _AVAILABLE_PROMPTS:
                prompt_file = _AVAILABLE_PROMPTS[prompt_filename_stem]
                logger.info(f"[PROMPT SELECTION] ✓ Found fuzzy match for '{payer_name}' via key '{key}' -> {prompt_file.name}")
                return _load_and_cache_prompt(prompt_file, prompt_filename_stem) or _load_default_prompt()
            else:
                 logger.warning(f"Fuzzy mapping for '{key}' points to a non-existent prompt file: '{prompt_filename_stem}.txt'")


    # 4. Fallback to default
    logger.info(f"[PROMPT SELECTION] ✗ No specific prompt found for '{payer_name}'. Using DEFAULT.")
    return _load_and_cache_prompt(DEFAULT_PROMPT_FILE, 'default') or _load_default_prompt()


def _load_default_prompt() -> str:
    """Loads the default prompt from file."""
    try:
        if DEFAULT_PROMPT_FILE.exists():
            with open(DEFAULT_PROMPT_FILE, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            logger.info(f"[PROMPT SELECTION] ✓ Loaded DEFAULT prompt file: {DEFAULT_PROMPT_FILE.name}")
            return prompt
        else:
            logger.warning(f"[PROMPT SELECTION] Default prompt file not found: {DEFAULT_PROMPT_FILE}. Using hardcoded fallback prompt.")
            # Fallback to a basic prompt if file doesn't exist
            return """You are a data extraction expert for pharmaceutical formularies. Extract drug information and return as JSON with keys: "drug_table", "acronyms", "tiers"."""
    except Exception as e:
        logger.error(f"[PROMPT SELECTION] Error loading default prompt from {DEFAULT_PROMPT_FILE.name}: {e}. Using hardcoded fallback prompt.")
        return """You are a data extraction expert for pharmaceutical formularies. Extract drug information and return as JSON with keys: "drug_table", "acronyms", "tiers"."""

        

def create_resilient_mistral_client():
    """
    Creates a Mistral client with robust timeouts and retry logic to prevent
    'Server disconnected' errors during large file uploads.
    """
    timeout = httpx.Timeout(CLIENT_TIMEOUT, connect=CONNECT_TIMEOUT)
    # The transport adapter handles the retry logic for specific HTTP errors
    transport = httpx.HTTPTransport(retries=MAX_RETRIES)
    client = httpx.Client(timeout=timeout, transport=transport)
    return Mistral(api_key=MISTRAL_API_KEY, client=client)


def process_pdf_with_mistral_ocr(pdf_input, payer_name=None):
    """
    Processes a PDF (from a file path or a BytesIO object) using Mistral OCR 
    and a parallelized LLM pipeline for data extraction, with robust retry logic.
    """
    log_name = getattr(pdf_input, 'name', pdf_input) if not isinstance(pdf_input, str) else pdf_input
    logger.info(f"Analyzing PDF with parallel LLM pipeline: {log_name}")

    if PYPDF2_AVAILABLE and isinstance(pdf_input, BytesIO):
        try:
            reader = PyPDF2.PdfReader(pdf_input)
            num_pages = len(reader)
            if num_pages > MAX_PDF_PAGES:
                logger.error(f"PDF has {num_pages} pages, exceeding limit of {MAX_PDF_PAGES}.")
                return {"drug_table": [], "acronyms": [], "tiers": []}, "", {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
            pdf_input.seek(0)
        except Exception as e:
            logger.warning(f"Failed to check PDF page count: {e}")

    total_costs = {'mistral_pages': 0, 'mistral_cost': 0.0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    
    # Use the resilient client for all API interactions
    mistral_client = create_resilient_mistral_client()

    try:
        if isinstance(pdf_input, BytesIO):
            file_bytes = pdf_input.getvalue()
            file_name = "temp_in_memory.pdf"
        else:
            pdf_file = Path(pdf_input)
            file_bytes = pdf_file.read_bytes()
            file_name = pdf_file.name

        uploaded_file = None
        # Manual retry loop for the initial upload for extra safety
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} to upload '{file_name}' to Mistral...")
                uploaded_file = mistral_client.files.upload(
                    file={"file_name": file_name, "content": file_bytes},
                    purpose="ocr",
                )
                logger.info("File uploaded successfully to Mistral.")
                break
            except (SDKError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_MULTIPLIER ** attempt
                    logger.warning(f"Network or Server error during upload: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to upload file to Mistral after {MAX_RETRIES} attempts due to a persistent network/server error.")
                    raise # Re-raise the exception to fail the worker

        if not uploaded_file:
            # Return the correct structure on failure
            return {"drug_table": [], "acronyms": [], "tiers": []}, "", total_costs

        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=120)
        ocr_response = mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=False
        )

        page_count = len(ocr_response.pages)
        total_costs['mistral_pages'] = page_count
        total_costs['mistral_cost'] = (page_count / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES

        all_structured_data, all_acronyms, all_tiers, all_raw_pages = [], [], [], []

        logger.info(f"Processing {page_count} pages in parallel with up to {LLM_PAGE_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=LLM_PAGE_WORKERS) as executor:
            future_to_page = {executor.submit(extract_structured_data_with_llm, page.markdown, payer_name): page_idx + 1 for page_idx, page in enumerate(ocr_response.pages)}
            # Collect raw markdown content separately to avoid race conditions
            for page in ocr_response.pages:
                 all_raw_pages.append(page.markdown)

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    structured_records, llm_costs = future.result()
                    logger.info(f"--- Completed processing for Page {page_num}/{page_count} ---")
                    total_costs['bedrock_tokens'] += llm_costs.get('tokens', 0)
                    total_costs['bedrock_cost'] += llm_costs.get('cost', 0)
                    total_costs['bedrock_calls'] += llm_costs.get('calls', 0)
                    if structured_records:
                        for drug in structured_records.get('drug_table', []):
                            drug['page_number'] = page_num
                            all_structured_data.append(drug)
                        all_acronyms.extend(structured_records.get('acronyms', []))
                        all_tiers.extend(structured_records.get('tiers', []))
                except Exception as exc:
                    logger.error(f"Page {page_num} generated an exception during result processing: {exc}")

        full_raw_content = "\n\n--- PAGE BREAK ---\n\n".join(all_raw_pages)
        
        # --- CONSOLIDATE RESULTS ---
        # Instead of returning multiple lists, return a single dictionary.
        full_structured_data = {
            "drug_table": all_structured_data,
            "acronyms": all_acronyms,
            "tiers": all_tiers
        }
        
        logger.info(f"Final results: {len(all_structured_data)} structured records extracted from PDF.")

        try:
            mistral_client.files.delete(file_id=uploaded_file.id)
            logger.info(f"Deleted uploaded file from Mistral: {uploaded_file.id}")
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file {uploaded_file.id}: {e}")

        return full_structured_data, full_raw_content, total_costs

    except Exception as e:
        logger.error(f"A critical error occurred in the main PDF processing pipeline for {log_name}: {e}")
        traceback.print_exc()
        # Return empty structures to prevent downstream errors
        return {"drug_table": [], "acronyms": [], "tiers": []}, "", total_costs


def get_plan_and_payer_info(state_name, payer, plan_name):
    """Get plan_id and payer_id from database with exact and fuzzy matching."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            logger.info(f"Looking for: State='{state_name}', Payer='{payer}', Plan='{plan_name}'")
            exact_query = """
                SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.formulary_url
                FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
                WHERE LOWER(TRIM(pd.state_name)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(py.payer_name)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(pd.plan_name)) ILIKE LOWER(TRIM(%s));
            """
            cursor.execute(exact_query, (state_name, payer, f'%{plan_name}%')) # Use ILIKE for plan name
            result = cursor.fetchone()
            if result:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = result
                logger.info(f"Found match in DB: Plan='{db_plan_name}', Payer='{db_payer_name}'")
                return plan_id, payer_id, db_payer_name, db_plan_name, formulary_url

            logger.warning(f"No exact match for '{plan_name}'. Falling back to fuzzy matching...")
            cursor.execute("""
                SELECT pd.plan_id, pd.payer_id, py.payer_name, pd.plan_name, pd.formulary_url
                FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
                WHERE LOWER(TRIM(pd.state_name)) = LOWER(TRIM(%s))
            """, (state_name,))
            all_records_in_state = cursor.fetchall()
            if not all_records_in_state:
                 logger.error(f"Fuzzy match failed: No plans found for state '{state_name}'")
                 return None, None, None, None, None

            best_match, best_score = None, 0.70 # Increased threshold
            for record in all_records_in_state:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = record
                payer_score = similarity(payer, db_payer_name)
                plan_score = similarity(plan_name, db_plan_name)
                total_score = (payer_score * 0.4) + (plan_score * 0.6)
                if total_score > best_score:
                    best_score = total_score
                    best_match = record

            if best_match:
                plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = best_match
                logger.info(f"Found fuzzy match (score: {best_score:.2f}): Plan='{db_plan_name}', Payer='{db_payer_name}'")
                return plan_id, payer_id, db_payer_name, db_plan_name, formulary_url

            logger.error(f"Fuzzy match failed for plan '{plan_name}' in state '{state_name}'.")
            return None, None, None, None, None

        except Exception as e:
            logger.error(f"Error in get_plan_and_payer_info: {e}")
            return None, None, None, None, None

def deduplicate_dicts(dicts, primary_key='acronym'):
    """Deduplicates a list of dictionaries, merging to keep the most complete info."""
    if not dicts:
        return []
    merged_entries = {}
    for item in dicts:
        key_value = item.get(primary_key)
        if not key_value:
            continue
        key = str(key_value).strip().lower()
        if key not in merged_entries:
            merged_entries[key] = item.copy()
        else:
            current_best = merged_entries[key]
            for field in ['expansion', 'explanation']:
                new_value = item.get(field)
                if new_value and len(str(new_value)) > len(str(current_best.get(field) or '')):
                    current_best[field] = new_value
    return list(merged_entries.values())

# --- WORKER AND ORCHESTRATOR FOR LOCAL PDFS ---

# def process_pdfs_in_parallel():
#     """Processes all PDFs in a local folder in parallel using a ProcessPoolExecutor."""
#     logger.info("STEP 2: Processing Local PDF Files in Parallel")
#     all_processed_data = []
#     pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
#     if not pdf_files:
#         logger.warning(f"No PDF files found in '{PDF_FOLDER}'.")
#         return [], {}

#     # Define a generous timeout for each PDF file in seconds (e.g., 20 minutes)
#     PDF_PROCESSING_TIMEOUT = 1200 

#     logger.info(f"Found {len(pdf_files)} PDFs. Starting parallel processing with up to {PROCESS_COUNT} workers.")
#     success_count, error_count, skipped_count = 0, 0, 0
#     with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
#         future_to_filename = {executor.submit(process_single_pdf_worker, filename, PDF_FOLDER): filename for filename in pdf_files}
#         for future in as_completed(future_to_filename):
#             filename = future_to_filename[future]
#             try:
#                 # Wait for the result, but no longer than the timeout
#                 status, _, result_data, costs = future.result(timeout=PDF_PROCESSING_TIMEOUT)
                
#                 if status == 'SUCCESS':
#                     success_count += 1
#                     payer_name = result_data['db_payer_name']
#                     if costs['mistral_pages'] > 0:
#                         track_mistral_cost(payer_name, costs['mistral_pages'])
#                     if costs['bedrock_tokens'] > 0:
#                         track_bedrock_cost_precalculated(payer_name, costs['bedrock_tokens'], costs['bedrock_cost'], costs['bedrock_calls'])
#                     all_processed_data.extend(result_data["processed_records"])
#                 elif status == 'SKIPPED':
#                     skipped_count += 1
#                     logger.warning(f"Skipped file: {filename}. Reason: {result_data}")
#                 elif status == 'ERROR':
#                     error_count += 1
#                     logger.error(f"Error processing file: {filename}. Reason: {result_data}")

#             except concurrent.futures.TimeoutError:
#                 error_count += 1
#                 logger.error(f"CRITICAL: Processing timed out for file: {filename} after {PDF_PROCESSING_TIMEOUT} seconds. The worker is likely stuck. Moving on.")
#             except Exception as e:
#                 error_count += 1
#                 logger.error(f"Critical error processing result for {filename}: {e}", exc_info=True)

#     logger.info("--- Local PDF Processing Complete ---")
#     logger.info(f"Summary: {success_count} successful, {error_count} failed, {skipped_count} skipped")
#     logger.info(f"Total structured records aggregated: {len(all_processed_data)}")
#     return all_processed_data, {}


# --- WORKER AND ORCHESTRATOR FOR URLS ---

def get_all_plans_with_formulary_url():
    """Fetch all plans marked 'processing' with a non-null formulary_url."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pd.state_name, py.payer_name, pd.plan_name, pd.plan_id, py.payer_id, pd.formulary_url, pd.file_hash
            FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
            WHERE pd.formulary_url IS NOT NULL AND pd.formulary_url != '' AND pd.status = 'processing'
        """)
        return cursor.fetchall()


# def process_single_pdf_worker(filename: str, pdf_folder_path: str):
#     """
#     Worker function for processing a single local PDF file.
#     Includes caching, data extraction, normalization, and record creation.
#     """
#     log_prefix = f"[Worker for {filename}]"
#     zero_costs = {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
 
#     try:
#         full_path = os.path.join(pdf_folder_path, filename)
#         if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:
#             return 'ERROR', filename, "File not found or is empty.", zero_costs
 
#         state_name, payer, plan_name = extract_metadata_from_filename(filename)
#         plan_id, payer_id, db_payer_name, db_plan_name, formulary_url = get_plan_and_payer_info(state_name, payer, plan_name)
#         if not plan_id:
#             return 'SKIPPED', filename, f"Plan not found in DB for: {state_name}, {payer}, {plan_name}", zero_costs
 
#         file_hash = calculate_file_hash(full_path)
#         update_plan_file_hash(plan_id, file_hash)
 
#         # --- CORRECTED CACHING LOGIC ---
#         cached_data, raw_content = get_cached_result(file_hash)
#         costs = zero_costs
#         full_structured_data = None  # Initialize
 
#         if cached_data is None: # Cache MISS
#             logger.info(f"{log_prefix} Cache MISS. Starting full processing...")
#             full_structured_data, raw_content, costs = process_pdf_with_mistral_ocr(full_path, db_payer_name)
#             cache_result(file_hash, full_structured_data, raw_content)
#         else: # Cache HIT
#             logger.info(f"{log_prefix} Cache HIT. Using pre-processed data.")
#             full_structured_data = cached_data
 
#         # --- UNPACK DATA FOR POST-PROCESSING ---
#         if not isinstance(full_structured_data, dict):
#             logger.error(f"{log_prefix} Corrupted cache or processing error. Expected a dictionary, got {type(full_structured_data)}")
#             full_structured_data = {"drug_table": [], "acronyms": [], "tiers": []}
 
#         drug_table_data = full_structured_data.get('drug_table', [])
#         all_acronyms = full_structured_data.get('acronyms', [])
#         all_tiers = full_structured_data.get('tiers', [])
       
#         # **CRITICAL STEP**: Create the DataFrame from the unpacked list.
#         structured_df = pd.DataFrame(drug_table_data)
 
#         # **NOW THIS CHECK IS SAFE**:
#         if structured_df.empty and not all_acronyms and not all_tiers:
#             return 'SKIPPED', filename, "No structured data could be extracted.", costs
 
#         # --- Acronym and Tier processing ---
#         all_acronyms, all_tiers = _reclassify_definitions(all_acronyms, all_tiers)
#         all_tiers = _parse_and_split_tier_definitions(all_tiers)
 
#         for tier_dict in all_tiers:
#             acronym = tier_dict.get('acronym')
#             if acronym and str(acronym).strip().isdigit():
#                 tier_dict['acronym'] = f"Tier {str(acronym).strip()}"
 
#         dedup_acronyms = deduplicate_dicts(all_acronyms)
#         dedup_tiers = deduplicate_dicts(all_tiers)
 
#         logger.info("Filtering out non-formulary definitions before insertion.")
       
#         # List of keywords that are not true acronyms or tiers
#         blocklist_keywords = ['prenatal', 'aspirin', 'statin', 'fluoride', 'tobacco', 'nicotine']
 
#         def is_valid_formulary_definition(item):
#             acronym = str(item.get('acronym', '')).lower().strip()
#             if not acronym:
#                 return False
#             # Rule 1: Check if the acronym starts with any blocked keyword.
#             if any(acronym.startswith(keyword) for keyword in blocklist_keywords):
#                 return False
#             # Rule 2: Filter out items that are clearly drug names (long text without numbers/special chars).
#             if len(acronym.replace(' ', '')) > 20 and acronym.isalpha():
#                  return False
#             return True
 
#         filtered_acronyms = [item for item in dedup_acronyms if is_valid_formulary_definition(item)]
#         filtered_tiers = [item for item in dedup_tiers if is_valid_formulary_definition(item)]
 
#         acronyms_removed_count = len(dedup_acronyms) - len(filtered_acronyms)
#         tiers_removed_count = len(dedup_tiers) - len(filtered_tiers)
 
#         if acronyms_removed_count > 0 or tiers_removed_count > 0:
#             logger.warning(
#                 f"Filtered out {acronyms_removed_count} invalid acronyms and "
#                 f"{tiers_removed_count} invalid tiers based on keyword blocklist."
#             )
 
#         all_definitions = filtered_acronyms + filtered_tiers
 
#         all_definitions = dedup_acronyms + dedup_tiers
#         if all_definitions:
#             # This step is crucial for handling shared formulary documents (cache hits).
#             # The 'all_definitions' list comes from the cached result, but we associate
#             # it with the current plan's specific state, payer, and plan name.
#             # This ensures that if Plan A and Plan B share a PDF, the definitions
#             # are correctly mapped to *both* plans in the reference table,
#             # mirroring how drug data is mapped in the drug_formulary_details table.
#             insert_acronyms_to_ref_table(all_definitions, state_name, payer, plan_name, "pp_formulary_names")
 
#         if structured_df.empty:
#             logger.info(f"{log_prefix} Acronyms/Tiers processed, but no drug records found.")
#             return 'SUCCESS', filename, {"processed_records": [], "db_payer_name": db_payer_name}, costs
 
#         processed_records = []
#         for _, row in structured_df.iterrows():
#             try:
#                 raw_drug_name = str(row.get('drug_name', '') or '')
#                 requirements_text = str(row.get('drug_requirements', '') or '').strip()
#                 cleaned_drug_name = clean_drug_name(raw_drug_name)
#                 if not cleaned_drug_name: continue
 
#                 raw_tier = row.get('drug_tier', None)
#                 drug_tier_normalized = normalize_drug_tier(raw_tier) or infer_drug_tier_from_text(requirements_text) or infer_drug_tier_from_text(raw_drug_name)
 
#                 with get_db_connection() as conn:
#                     coverage_status = determine_coverage_status(requirements_text, drug_tier_normalized, conn, state_name, db_payer_name)
 
#                 record = {
#                     "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
#                     "drug_name": cleaned_drug_name, "state_name": state_name,
#                     "coverage_status": coverage_status, "drug_tier": drug_tier_normalized,
#                     "drug_requirements": requirements_text or None,
#                     "is_prior_authorization_required": "Yes" if detect_prior_authorization(requirements_text) else "No",
#                     "is_step_therapy_required": "Yes" if detect_step_therapy(requirements_text) else "No",
#                     "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
#                     "confidence_score": 0.95, "source_url": formulary_url,
#                     "plan_name": db_plan_name, "payer_name": db_payer_name, "file_name": filename,
#                     "ndc_code": None, "jcode": None, "coverage_details": None,
#                 }
#                 processed_records.append(record)
#             except Exception as e:
#                 logger.warning(f"{log_prefix} Error processing extracted row: {row}. Error: {e}")
#                 continue
#             pass
 
#         if processed_records:
#             return 'SUCCESS', filename, {"processed_records": processed_records, "db_payer_name": db_payer_name}, costs
#         else:
#             return 'SKIPPED', filename, "Data extracted, but no valid drug records could be processed.", costs
 
#     except Exception as e:
#         return 'ERROR', filename, f"An unexpected error occurred in worker: {e}\n{traceback.format_exc()}", zero_costs
import datetime
 
 
def process_single_pdf_url_worker(plan_info):
    """Worker: Download PDF from URL and process it entirely in-memory."""
    state_name, payer_name, plan_name, plan_id, payer_id, formulary_url, old_file_hash = plan_info
    log_prefix = f"[URL Worker for {plan_name}]"
    zero_costs = {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    start_time = time.time()

    try:
        # --- NEW: Proactive URL Validation ---
        # Check for empty URLs or strings that look like phone numbers before making a network request.
        if not formulary_url or re.match(r'^[\d\s\(\)-]{7,}$', str(formulary_url).strip()):
            error_message = f"Invalid Formulary URL detected (is blank or resembles a phone number): '{formulary_url}'"
            logger.error(f"{log_prefix} {error_message}")
            return 'ERROR', plan_name, error_message, zero_costs

        if not formulary_url.startswith(('http://', 'https://')):
            formulary_url = 'https://' + formulary_url
            logger.info(f"{log_prefix} URL scheme was missing. Corrected to: {formulary_url}")


        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        # First, try the secure way. If it fails, log a warning and retry insecurely.
        try:
            with requests.get(formulary_url, timeout=90, headers=headers, stream=True, verify=True) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                    error_details = f"Invalid content type: {content_type}"
                    logger.error(f"{log_prefix} {error_details}")
                    return 'ERROR', plan_name, error_details, zero_costs
                pdf_content_bytes = resp.content
        except requests.exceptions.SSLError as e:
            logger.warning(f"{log_prefix} SSL verification failed: {e}. This often means the URL is wrong or the server is misconfigured.")
            logger.warning(f"{log_prefix} Retrying with SSL verification DISABLED. This is insecure and should only be a last resort.")
            with requests.get(formulary_url, timeout=90, headers=headers, stream=True, verify=False) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                     error_details = f"Invalid content type on retry: {content_type}"
                     logger.error(f"{log_prefix} {error_details}")
                     return 'ERROR', plan_name, error_details, zero_costs
                pdf_content_bytes = resp.content

        pdf_bytes_io = BytesIO(pdf_content_bytes)
 
        new_file_hash = calculate_bytes_hash(pdf_content_bytes)
 
        if old_file_hash and old_file_hash != new_file_hash:
            logger.info(f"New formulary version detected for plan '{plan_name}' (hash changed from {old_file_hash[:8]}... to {new_file_hash[:8]}...).")
            delete_drug_formulary_records_for_plan(plan_id)
         
        update_plan_file_hash(plan_id, new_file_hash)

        cached_data, raw_content = get_cached_result(new_file_hash)
        costs = zero_costs
        full_structured_data = None


        if cached_data is None:
            logger.info(f"{log_prefix} Cache MISS for new hash. Starting full processing...")
            full_structured_data, raw_content, costs = process_pdf_with_mistral_ocr(pdf_bytes_io, payer_name)
            cache_result(new_file_hash, full_structured_data, raw_content)
        else:
            logger.info(f"{log_prefix} Cache HIT for new hash. Using pre-processed data.")
            full_structured_data = cached_data

        if not isinstance(full_structured_data, dict):
            logger.error(f"{log_prefix} Corrupted cache or processing error. Expected a dictionary, got {type(full_structured_data)}")
            full_structured_data = {"drug_table": [], "acronyms": [], "tiers": []}

        drug_table_data = full_structured_data.get('drug_table', [])
        all_acronyms = full_structured_data.get('acronyms', [])
        all_tiers = full_structured_data.get('tiers', [])

        structured_df = pd.DataFrame(drug_table_data)

        if not structured_df.empty:
            expanded_rows = []
            for _, row in structured_df.iterrows():
                # Get all original data from the row first
                drug_name_full = str(row.get('drug_name', ''))
                drug_tier = row.get('drug_tier')
                drug_requirements = row.get('drug_requirements')
                page_number = row.get('page_number')  

                parsed_drugs = parse_complex_drug_name(drug_name_full)
                
                if not parsed_drugs: # If parsing fails, keep the original row
                    expanded_rows.append(row.to_dict())
                    continue

                for parsed_drug in parsed_drugs:
                    for strength in parsed_drug['strengths']:
                        new_drug_name = f"{parsed_drug['base_name']} {strength}".strip()
                        if parsed_drug['brand_name']:
                            new_drug_name += f" ({parsed_drug['brand_name']})"
                        
                        # Create the new row, making sure to include the page number
                        expanded_rows.append({
                            "drug_name": new_drug_name,
                            "drug_tier": drug_tier,
                            "drug_requirements": drug_requirements,
                            "page_number": page_number   
                        })
            structured_df = pd.DataFrame(expanded_rows)

        if structured_df.empty and not all_acronyms and not all_tiers:
            return 'SKIPPED', plan_name, "No structured data extracted from URL PDF.", costs

        all_acronyms, all_tiers = _reclassify_definitions(all_acronyms, all_tiers)
        all_tiers = _parse_and_split_tier_definitions(all_tiers)
        
        for tier_dict in all_tiers:
            acronym = tier_dict.get('acronym')
            if acronym and str(acronym).strip().isdigit():
                tier_dict['acronym'] = f"Tier {str(acronym).strip()}"

        dedup_acronyms = deduplicate_dicts(all_acronyms)
        dedup_tiers = deduplicate_dicts(all_tiers)

        all_definitions = dedup_acronyms + dedup_tiers
        if all_definitions:
            insert_acronyms_to_ref_table(all_definitions, state_name, payer_name, plan_name, "pp_formulary_names")
        
        if structured_df.empty:
            logger.info(f"{log_prefix} Acronyms/Tiers processed, but no drug records found.")
            return 'SUCCESS', plan_name, {"processed_records": [], "db_payer_name": payer_name}, costs
 
        if not structured_df.empty:
            requirement_tier_pairs = set()
            for _, row in structured_df.iterrows():
                req_code = str(row.get('drug_requirements', '') or '').strip()
                req_code = normalize_requirement_code(req_code)
                tier = normalize_drug_tier(row.get('drug_tier', None)) or infer_drug_tier_from_text(req_code)
                requirement_tier_pairs.add((req_code, tier))
            with get_db_connection() as conn:
                coverage_map = batch_determine_coverage_status(requirement_tier_pairs, conn, state_name, payer_name)
 
        processed_records = []
        for _, row in structured_df.iterrows():
            cleaned_drug_name = clean_drug_name(str(row.get('drug_name', '') or ''))
            if not cleaned_drug_name: continue
            requirements_text = str(row.get('drug_requirements', '') or '').strip()
            requirements_text = normalize_requirement_code(requirements_text)
            drug_tier_normalized = normalize_drug_tier(row.get('drug_tier', None)) or infer_drug_tier_from_text(requirements_text) or infer_drug_tier_from_text(cleaned_drug_name)
 
            coverage_status = coverage_map.get((requirements_text, drug_tier_normalized))
            
            if (
                coverage_status and coverage_status.lower() == "covered"
                and "pa" in requirements_text.lower()
            ):
                coverage_status = "Covered with Conditions"
            record = {
                "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
                "drug_name": cleaned_drug_name, "state_name": state_name, "coverage_status": coverage_status,
                "drug_tier": drug_tier_normalized, "drug_requirements": requirements_text or None,
                "page_number": row.get('page_number', None),
                "is_prior_authorization_required": "Yes" if detect_prior_authorization(requirements_text) else "No",
                "is_step_therapy_required": "Yes" if detect_step_therapy(requirements_text) else "No",
                "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
                "confidence_score": 0.95, "source_url": formulary_url,
                "plan_name": plan_name, "payer_name": payer_name,
                "file_name": f"{state_name}_{payer_name}_{plan_name}.pdf",
                "ndc_code": None, "jcode": None, "coverage_details": None,
                "ndc_code": None, "jcode": None, "coverage_details": None,
            }
            processed_records.append(record)

        if processed_records:
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"{log_prefix} Total processing time (worker start to DB push): {total_time:.2f} seconds.")
            with open("worker_timing.log", "a", encoding="utf-8") as f:
                f.write(f"{log_prefix} | Total time: {total_time:.2f} seconds\n")
            return 'SUCCESS', plan_name, {"processed_records": processed_records, "db_payer_name": payer_name}, costs
        else:
            return 'SKIPPED', plan_name, "Data extracted, but no valid drug records were processed.", costs

    except requests.exceptions.ProxyError as e:
        logger.error(f"{log_prefix} Proxy Error: Could not connect to the proxy. {e}", exc_info=True)
        return 'ERROR', plan_name, f"Proxy Error: {e}", zero_costs
    except Exception as e:
        logger.error(f"{log_prefix} Error: {e}", exc_info=True)
        return 'ERROR', plan_name, str(e), zero_costs
 

def process_pdfs_from_urls_in_parallel():
    """Process PDFs by downloading from URLs in plan_details, in parallel."""
    logger.info("STEP 2: Processing PDF Files from URLs in plan_details")
    successfully_processed_plan_ids = []

    plans = get_all_plans_with_formulary_url()
    if not plans:
        logger.warning("No plans with formulary URLs found to process.")
        return [], {}

    URL_PROCESSING_TIMEOUT = 1200

    logger.info(f"Found {len(plans)} plans with URLs to process.")
    success_count, error_count, skipped_count = 0, 0, 0
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
        future_to_plan = {executor.submit(process_single_pdf_url_worker, plan): plan for plan in plans}

        for future in as_completed(future_to_plan):
            plan_info = future_to_plan[future]
            plan_name_log = plan_info[2]
            try:
                status, _, result_data, costs = future.result(timeout=URL_PROCESSING_TIMEOUT)
                
                if status == 'SUCCESS':
                    logger.info(f"Aggregating results for SUCCESSFUL plan: {plan_name_log}")
                    success_count += 1
                    
                    processed_records = result_data.get("processed_records", [])
                    if processed_records:
                        logger.info(f"Deduplicating {len(processed_records)} records before insertion for plan '{plan_name_log}'.")
                        unique_records = {}
                        for record in processed_records:
                            # Create a key based on the database's UNIQUE constraint
                            key = (
                                record.get('plan_id'),
                                record.get('drug_name'),
                                record.get('drug_tier'),
                                record.get('drug_requirements')
                            )
                            if key not in unique_records:
                                unique_records[key] = record
                        
                        deduplicated_data = list(unique_records.values())
                        records_removed = len(processed_records) - len(deduplicated_data)
                        if records_removed > 0:
                            logger.warning(f"Removed {records_removed} duplicate records from the batch for '{plan_name_log}'.")

                        if deduplicated_data:
                            logger.info(f"Inserting {len(deduplicated_data)} unique records for plan '{plan_name_log}' into the database.")
                            insert_drug_formulary_data(deduplicated_data)
                            
                            plan_id = deduplicated_data[0].get('plan_id')
                            if plan_id and plan_id not in successfully_processed_plan_ids:
                                successfully_processed_plan_ids.append(plan_id)

                    payer_name = result_data['db_payer_name']
                    if costs['mistral_pages'] > 0:
                        track_mistral_cost(payer_name, costs['mistral_pages'])
                    if costs['bedrock_tokens'] > 0:
                        track_bedrock_cost_precalculated(payer_name, costs['bedrock_tokens'], costs['bedrock_cost'], costs['bedrock_calls'])
                
                elif status == 'SKIPPED':
                    logger.warning(f"Skipped plan: {plan_name_log}. Reason: {result_data}")
                    skipped_count += 1
                elif status == 'ERROR':
                    logger.error(f"Error processing plan: {plan_name_log}. Reason: {result_data}")
                    error_count += 1
            
            except concurrent.futures.TimeoutError:
                error_count += 1
                logger.error(f"CRITICAL: Processing timed out for plan: {plan_name_log} after {URL_PROCESSING_TIMEOUT} seconds. The worker is likely stuck. Moving on.")
            except Exception as e:
                logger.error(f"A critical error occurred while processing result for {plan_name_log}: {e}", exc_info=True)
                error_count += 1

    logger.info("--- URL PDF Processing Complete ---")
    logger.info(f"Summary: {success_count} successful, {error_count} failed, {skipped_count} skipped")
    return successfully_processed_plan_ids, {}

def _parse_and_split_tier_definitions(tier_list: list) -> list:
    """
    Parses tier definitions where the acronym and expansion might be combined in one field.
    This corrects LLM outputs like {"acronym": "Tier 1 - Generic", "expansion": None}
    into {"acronym": "Tier 1", "expansion": "Generic"}.
    """
    if not tier_list:
        return []

    processed_tiers = []
    for tier_dict in tier_list:
        if not isinstance(tier_dict, dict):
            continue

        acronym_raw = tier_dict.get('acronym')
        expansion_raw = tier_dict.get('expansion')

        if isinstance(acronym_raw, str) and ' - ' in acronym_raw:
            parts = acronym_raw.split(' - ', 1)
            new_acronym = parts[0].strip()
            new_expansion = parts[1].strip()
            
            tier_dict['acronym'] = new_acronym
            
            if not expansion_raw:
                tier_dict['expansion'] = new_expansion
        
        processed_tiers.append(tier_dict)
        
    return processed_tiers

def _reclassify_definitions(acronyms_list: list, tiers_list: list) -> Tuple[list, list]:
    """
    Sorts definitions into acronyms or tiers based on heuristics to correct LLM misclassifications.
    """
    if not tiers_list and not acronyms_list:
        return [], []

    corrected_acronyms = []
    corrected_tiers = []
    
    TIER_KEYWORDS = {'aca', 'preventive', 'specialty', 'preferred', 'generic', 'brand'}

    for item in tiers_list:
        if not isinstance(item, dict): continue
        acronym = str(item.get('acronym') or '').strip().lower()

        if acronym.startswith('tier') or acronym in TIER_KEYWORDS:
            corrected_tiers.append(item)
        else:
            corrected_acronyms.append(item)
            
    for item in acronyms_list:
        if not isinstance(item, dict): continue
        acronym = str(item.get('acronym') or '').strip().lower()

        if acronym.startswith('tier') or acronym in TIER_KEYWORDS:
            corrected_tiers.append(item)
        else:
            corrected_acronyms.append(item)

    return corrected_acronyms, corrected_tiers


def is_valid_formulary_definition(item: dict) -> bool:
    """
    Automatically detects if an extracted item is a valid formulary definition.
    """
    acronym = str(item.get('acronym', '')).strip()
    expansion = str(item.get('expansion', '')).strip()

    if not acronym or not expansion:
        return False

    tier_description_words = {'preferred', 'non-preferred', 'generic', 'brand', 'specialty'}
    if len(acronym) <= 4 and any(word in expansion.lower() for word in tier_description_words):
        return False

    if len(acronym.split()) > 3:
        return False
        
    sim_score = similarity(acronym, expansion)
    if sim_score > 0.75:
        return False
        
    if acronym.isdigit():
        return False

    return True





                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            