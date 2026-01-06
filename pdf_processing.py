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
import random
from typing import Tuple, List
from pathlib import Path
from mistralai.models.sdkerror import SDKError
from typing import List, Dict, Optional, Union
from mistralai import Mistral
from mistralai.models import DocumentURLChunk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from utils import is_english

import threading

from config import (
    PDF_FOLDER, PROCESS_COUNT, MISTRAL_API_KEY, BEDROCK_MODEL_ID, BEDROCK_COST_PER_1K_TOKENS,bedrock,
    MISTRAL_OCR_COST_PER_1K_PAGES, BEDROCK_COST_PER_1K_TOKENS, LLM_PAGE_WORKERS,
    MAX_RETRIES, BACKOFF_MULTIPLIER, CLIENT_TIMEOUT, CONNECT_TIMEOUT, PDF_PAGE_PROCESSING_CONFIG,
    OCR_CHUNK_WORKERS, MISTRAL_OCR_RATE_LIMIT, ENABLE_PAGE_PREFILTER, MIN_PAGE_TEXT_LENGTH, SKIP_INDEX_PAGES
)
from database import get_db_connection, batch_determine_coverage_status, get_cached_result, cache_result, update_plan_file_hash, insert_acronyms_to_ref_table, insert_drug_formulary_data, delete_drug_formulary_records_for_plan
from utils import (
    similarity, clean_drug_name, detect_prior_authorization,
    detect_step_therapy, calculate_file_hash, rate_limited_api_call,
    track_bedrock_cost_precalculated, track_mistral_cost, determine_coverage_status,
    normalize_drug_tier, infer_drug_tier_from_text, calculate_bytes_hash,
    parse_complex_drug_name, similarity, normalize_requirement_code, transform_viewer_url
)

logger = logging.getLogger(__name__)

# PDF Processing Constants
MAX_PDF_PAGES = 2000
ENHANCED_PDF_DPI = 200
USE_ENHANCED_PDF = False


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

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def initialize_worker():
    """
    An initializer function for each worker in the ProcessPoolExecutor.
    """
    pass # logging config removed
    pass # _load_prompts_config() removed


# -----------------------------
# OPTIMIZATION HELPERS
# -----------------------------

# Thread-safe rate limiter for Mistral OCR API (Optimization 7)
_mistral_last_call_time = 0
_mistral_rate_lock = threading.Lock()

def mistral_rate_limited_call(func):
    """
    A fast rate limiter specifically for Mistral OCR calls.
    Uses MISTRAL_OCR_RATE_LIMIT instead of the slower RATE_LIMIT_DELAY.
    """
    def wrapper(*args, **kwargs):
        global _mistral_last_call_time
        with _mistral_rate_lock:
            elapsed = time.time() - _mistral_last_call_time
            if elapsed < MISTRAL_OCR_RATE_LIMIT:
                time.sleep(MISTRAL_OCR_RATE_LIMIT - elapsed)
            _mistral_last_call_time = time.time()
        return func(*args, **kwargs)
    return wrapper


def prefilter_pages_with_pymupdf(pdf_input: BytesIO, page_indices: List[int]) -> List[int]:
    """
    OPTIMIZATION 5: Smart page pre-filtering using PyMuPDF text extraction.
    Quickly scans pages and removes those that are likely empty, index pages, or TOC.
    This runs BEFORE OCR to avoid wasting API calls on useless pages.
    
    Args:
        pdf_input: BytesIO object containing the PDF
        page_indices: List of 1-based page numbers to consider
        
    Returns:
        Filtered list of 1-based page numbers worth processing
    """
    if not ENABLE_PAGE_PREFILTER or not PYMUPDF_AVAILABLE:
        logger.info("📄 Page pre-filtering disabled or PyMuPDF not available")
        return page_indices
    
    filtered_pages = []
    skipped_pages = []
    
    try:
        pdf_input.seek(0)
        src_doc = fitz.open(stream=pdf_input.getvalue(), filetype="pdf")
        
        for page_num in page_indices:
            if page_num < 1 or page_num > len(src_doc):
                continue
                
            page = src_doc[page_num - 1]
            text = page.get_text().strip()
            text_lower = text.lower()
            
            # Skip pages with very little text (blank or image-only pages)
            if len(text) < MIN_PAGE_TEXT_LENGTH:
                skipped_pages.append((page_num, "too little text"))
                continue
            
            # Skip pages that look like index/TOC (Optimization 5)
            if SKIP_INDEX_PAGES:
                # Check first 500 chars for index indicators
                header_text = text_lower[:500]
                index_indicators = [
                    "table of contents",
                    "alphabetical index",
                    "drug index",
                    "index of drugs",
                    "formulary index",
                    "............",  # Common in TOC
                    "...........",
                ]
                
                is_index = False
                for indicator in index_indicators:
                    if indicator in header_text:
                        is_index = True
                        break
                
                # Also check for page number patterns typical in TOC (e.g., "drug name ... 42")
                if not is_index:
                    # Count lines that end with just a number (common TOC pattern)
                    lines = text.split('\n')
                    toc_pattern_count = 0
                    for line in lines[:30]:  # Check first 30 lines
                        line = line.strip()
                        if re.match(r'^.{10,}\.{3,}\s*\d+$', line) or re.match(r'^.{10,}\s+\d+$', line):
                            toc_pattern_count += 1
                    
                    if toc_pattern_count > 10:  # If many lines match TOC pattern
                        is_index = True
                
                if is_index:
                    skipped_pages.append((page_num, "index/TOC page"))
                    continue
            
            # Page passed all filters
            filtered_pages.append(page_num)
        
        src_doc.close()
        
        if skipped_pages:
            logger.info(f"📄 [PRE-FILTER] Skipped {len(skipped_pages)} pages: {skipped_pages[:5]}{'...' if len(skipped_pages) > 5 else ''}")
        
        logger.info(f"📄 [PRE-FILTER] Kept {len(filtered_pages)} of {len(page_indices)} pages after filtering")
        
    except Exception as e:
        logger.warning(f"⚠️ Page pre-filtering failed: {e}. Using original page list.")
        return page_indices
    
    return filtered_pages if filtered_pages else page_indices


def process_single_chunk_parallel(chunk_info: dict) -> dict:
    """
    OPTIMIZATION 1: Process a single chunk of pages for parallel execution.
    This function is designed to be called by ThreadPoolExecutor for concurrent chunk processing.
    
    Args:
        chunk_info: Dictionary containing:
            - chunk_idx: Index of this chunk
            - chunk_pages: List of original page numbers in this chunk
            - pdf_bytes: The original PDF as bytes
            - mistral_client: Mistral client instance
            - ocr_schema: The OCR annotation schema
            
    Returns:
        Dictionary with extracted drugs, acronyms, and page count
    """
    chunk_idx = chunk_info['chunk_idx']
    chunk_pages = chunk_info['chunk_pages']
    pdf_bytes = chunk_info['pdf_bytes']
    ocr_schema = chunk_info['ocr_schema']
    
    result = {
        'chunk_idx': chunk_idx,
        'drugs': [],
        'acronyms': [],
        'pages_processed': 0,
        'error': None
    }
    
    try:
        # Create a fresh Mistral client for this thread (thread-safe)
        timeout = httpx.Timeout(CLIENT_TIMEOUT, connect=CONNECT_TIMEOUT)
        transport = httpx.HTTPTransport(retries=MAX_RETRIES)
        client = httpx.Client(timeout=timeout, transport=transport)
        mistral_client = Mistral(api_key=MISTRAL_API_KEY, client=client)
        
        # Extract pages for this chunk
        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunk_doc = fitz.open()
        
        for page_num in chunk_pages:
            if 1 <= page_num <= len(src_doc):
                chunk_doc.insert_pdf(src_doc, from_page=page_num-1, to_page=page_num-1)
        
        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()
        src_doc.close()
        
        # Apply Mistral rate limiting (Optimization 7)
        @mistral_rate_limited_call
        def upload_chunk():
            return mistral_client.files.upload(
                file={"file_name": f"chunk_{chunk_idx}.pdf", "content": chunk_bytes},
                purpose="ocr"
            )
        
        chunk_uploaded = upload_chunk()
        chunk_signed_url = mistral_client.files.get_signed_url(file_id=chunk_uploaded.id, expiry=300)
        
        # Process chunk with OCR - with retry logic for 500/502 errors
        max_retries = 3
        retry_delay = 2  # seconds
        ocr_response = None
        
        for attempt in range(max_retries):
            try:
                ocr_response = mistral_client.ocr.process(
                    model="mistral-ocr-latest",
                    document=DocumentURLChunk(document_url=chunk_signed_url.url),
                    document_annotation_format=ocr_schema,
                    include_image_base64=False
                )
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                if "500" in error_str or "502" in error_str or "503" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Chunk {chunk_idx + 1}: OCR API error (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise on final attempt
                else:
                    raise  # Non-retryable error
        
        if ocr_response is None:
            result['error'] = "OCR API failed after retries"
            return result
        
        result['pages_processed'] = len(ocr_response.pages)
        
        # Log chunk info for debugging page mapping
        logger.info(f"Chunk {chunk_idx + 1}: Processing {len(chunk_pages)} pages. Original PDF page numbers: {chunk_pages}")
        
        # Extract from document-level annotation
        if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
            chunk_json = ocr_response.document_annotation
            if isinstance(chunk_json, str):
                # Try to parse JSON, with repair for malformed responses
                try:
                    chunk_json = json.loads(chunk_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Chunk {chunk_idx + 1}: JSON parse error: {e}. Attempting repair...")
                    # Try to repair common JSON issues
                    repaired = chunk_json
                    # Fix unterminated strings by truncating at the error point
                    if "Unterminated string" in str(e):
                        # Find the last complete JSON object
                        try:
                            # Try to find a valid JSON subset
                            for end_pos in range(len(repaired) - 1, 0, -1):
                                test_str = repaired[:end_pos] + '"}]}'
                                try:
                                    chunk_json = json.loads(test_str)
                                    logger.info(f"✅ Chunk {chunk_idx + 1}: JSON repaired successfully")
                                    break
                                except:
                                    continue
                            else:
                                chunk_json = {"DrugInformation": [], "FormularyAbbreviations": []}
                        except:
                            chunk_json = {"DrugInformation": [], "FormularyAbbreviations": []}
                    else:
                        chunk_json = {"DrugInformation": [], "FormularyAbbreviations": []}
            
            drug_info_list = chunk_json.get("DrugInformation", [])
            
            for drug_idx, item in enumerate(drug_info_list):
                if isinstance(item, dict):
                    drug_name = item.get("Drug Name", "")
                    if not drug_name or len(drug_name) < 2:
                        continue
                    
                    # =====================================================================
                    # DRUG DATA MAPPING FOR PDL FORMAT
                    # =====================================================================
                    # Therapeutic Category → drug_name (already extracted as "Drug Name")
                    # Comment (number) → drug_tier
                    # B,G,O + P,N,R,NR combined with ";" → drug_requirements
                    # =====================================================================
                    
                    # Get the raw values from OCR
                    bgo = item.get("BGO", "").strip() if item.get("BGO") else ""
                    pnrnr = item.get("PNRNR", "").strip() if item.get("PNRNR") else ""
                    
                    # Combine BGO and PNRNR into drug_requirements
                    # If OCR already provided combined "requirements", use that as fallback
                    ocr_requirements = item.get("requirements", "")
                    
                    if bgo or pnrnr:
                        # Combine with semicolon separator: "G; P", "B; N", etc.
                        parts = [p for p in [bgo, pnrnr] if p]
                        drug_requirements = "; ".join(parts) if parts else None
                    elif ocr_requirements:
                        drug_requirements = ocr_requirements
                    else:
                        drug_requirements = None
                    
                    # =====================================================================
                    # PAGE NUMBER MAPPING - Store ACTUAL PDF page numbers
                    # =====================================================================
                    ocr_page_num = item.get("page_number")
                    
                    # Case 1: OCR returned a chunk-relative page number (1 to len(chunk_pages))
                    if ocr_page_num and isinstance(ocr_page_num, int) and 1 <= ocr_page_num <= len(chunk_pages):
                        actual_pdf_page = chunk_pages[ocr_page_num - 1]
                    # Case 2: OCR returned a page number that matches one of our original PDF pages
                    elif ocr_page_num and isinstance(ocr_page_num, int) and ocr_page_num in chunk_pages:
                        actual_pdf_page = ocr_page_num
                    # Case 3: Fallback - distribute evenly based on position in list
                    else:
                        if len(drug_info_list) > 0 and len(chunk_pages) > 0:
                            position_ratio = drug_idx / len(drug_info_list)
                            page_index = min(int(position_ratio * len(chunk_pages)), len(chunk_pages) - 1)
                            actual_pdf_page = chunk_pages[page_index]
                        else:
                            actual_pdf_page = chunk_pages[0] if chunk_pages else 1
                    
                    result['drugs'].append({
                        "drug_name": drug_name,
                        "drug_tier": item.get("drug tier"),
                        "drug_requirements": drug_requirements,
                        "category": item.get("category"),
                        "page_number": actual_pdf_page
                    })
            
            for item in chunk_json.get("FormularyAbbreviations", []):
                if isinstance(item, dict):
                    result['acronyms'].append({
                        "acronym": item.get("Acronym"),
                        "expansion": item.get("Expansion"),
                        "explanation": item.get("Explanation")
                    })
        
        # Also check page-level annotations
        # PAGE MAPPING: page_idx is 0-based index within the chunk (0, 1, 2, 3)
        # chunk_pages[page_idx] gives the ACTUAL PDF page number (e.g., 5, 6, 7, 8)
        for page_idx, page in enumerate(ocr_response.pages):
            # Map chunk page index to ACTUAL PDF page number
            if page_idx < len(chunk_pages):
                actual_pdf_page = chunk_pages[page_idx]
            else:
                logger.warning(f"Chunk {chunk_idx + 1}: page_idx {page_idx} out of range for chunk_pages (len={len(chunk_pages)})")
                actual_pdf_page = chunk_pages[-1] if chunk_pages else page_idx + 1  # Fallback
            
            if hasattr(page, 'document_annotation') and page.document_annotation:
                page_json = page.document_annotation
                if isinstance(page_json, str):
                    page_json = json.loads(page_json)
                if isinstance(page_json, dict):
                    drugs_on_page = page_json.get("DrugInformation", [])
                    if drugs_on_page:
                        logger.debug(f"Chunk {chunk_idx + 1}: Page-level found {len(drugs_on_page)} drugs on PDF page {actual_pdf_page}")
                    for item in drugs_on_page:
                        if isinstance(item, dict):
                            # Combine BGO and PNRNR for PDL format
                            bgo = item.get("BGO", "").strip() if item.get("BGO") else ""
                            pnrnr = item.get("PNRNR", "").strip() if item.get("PNRNR") else ""
                            ocr_requirements = item.get("requirements", "")
                            
                            if bgo or pnrnr:
                                parts = [p for p in [bgo, pnrnr] if p]
                                drug_requirements = "; ".join(parts) if parts else None
                            elif ocr_requirements:
                                drug_requirements = ocr_requirements
                            else:
                                drug_requirements = None
                            
                            result['drugs'].append({
                                "drug_name": item.get("Drug Name"),
                                "drug_tier": item.get("drug tier"),
                                "drug_requirements": drug_requirements,
                                "category": item.get("category"),
                                "page_number": actual_pdf_page
                            })
                    for item in page_json.get("FormularyAbbreviations", []):
                        if isinstance(item, dict):
                            result['acronyms'].append({
                                "acronym": item.get("Acronym"),
                                "expansion": item.get("Expansion"),
                                "explanation": item.get("Explanation")
                            })
        
        # Cleanup
        try:
            mistral_client.files.delete(file_id=chunk_uploaded.id)
        except:
            pass
        
        # Log completion with page distribution info
        if result['drugs']:
            page_counts = {}
            for d in result['drugs']:
                p = d.get('page_number')
                page_counts[p] = page_counts.get(p, 0) + 1
            logger.info(f"✅ Chunk {chunk_idx + 1} complete: {len(result['drugs'])} drugs from original pages {chunk_pages}. Distribution: {page_counts}")
        else:
            logger.info(f"✅ Chunk {chunk_idx + 1} complete: 0 drugs from original pages {chunk_pages}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.warning(f"⚠️ Chunk {chunk_idx + 1} failed: {e}")
    
    return result


def enhance_pdf(pdf_input, dpi=ENHANCED_PDF_DPI):
    """
    Enhance a PDF by converting it to high-resolution images and 
    reconstructing it as a new high-quality PDF.
    
    This improves OCR accuracy, especially for tables, by:
    1. Rendering each page at high DPI (300)
    2. Creating a new PDF with these high-quality renders
    
    Args:
        pdf_input: File path (str/Path) or BytesIO object
        dpi: Resolution for enhancement (default 300)
        
    Returns:
        BytesIO object containing the enhanced PDF, or None if enhancement fails
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF (fitz) not available. Cannot enhance PDF.")
        return None
    
    logger.info(f"Enhancing PDF at {dpi} DPI for better OCR quality...")
    
    try:
        # Open the source PDF
        if isinstance(pdf_input, (str, Path)):
            src_doc = fitz.open(str(pdf_input))
        elif isinstance(pdf_input, BytesIO):
            pdf_input.seek(0)
            src_doc = fitz.open(stream=pdf_input.getvalue(), filetype="pdf")
        else:
            logger.error("pdf_input must be a file path or BytesIO object")
            return None
        
        page_count = len(src_doc)
        logger.info(f"Source PDF has {page_count} pages. Starting enhancement...")
        
        # Create a new PDF document for the enhanced version
        enhanced_doc = fitz.open()
        
        zoom = dpi / 72  # 72 is default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(page_count):
            # Get the source page
            src_page = src_doc[page_num]
            
            # Render page to high-resolution pixmap
            pix = src_page.get_pixmap(matrix=matrix, alpha=False)
            
            # Create a new page in enhanced doc with same dimensions as original
            # but insert the high-res image
            page_rect = src_page.rect
            new_page = enhanced_doc.new_page(width=page_rect.width, height=page_rect.height)
            
            # Insert the high-res pixmap as an image on the new page
            # Scale it back down to fit the original page dimensions
            new_page.insert_image(page_rect, pixmap=pix)
            
            if (page_num + 1) % 10 == 0 or page_num == page_count - 1:
                logger.info(f"Enhanced page {page_num + 1}/{page_count}")
        
        src_doc.close()
        
        # Save enhanced PDF to BytesIO
        enhanced_bytes = BytesIO()
        enhanced_doc.save(enhanced_bytes, garbage=4, deflate=True)
        enhanced_doc.close()
        
        enhanced_bytes.seek(0)
        enhanced_size_mb = len(enhanced_bytes.getvalue()) / (1024 * 1024)
        logger.info(f"Successfully created enhanced PDF: {page_count} pages, {enhanced_size_mb:.2f} MB")
        
        return enhanced_bytes
        
    except Exception as e:
        logger.error(f"Failed to enhance PDF: {e}")
        traceback.print_exc()
        return None
    
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

    # Pattern to match array sections: "key": [...] (with proper brace matching)
    # This handles nested arrays and objects better
    extracted = {}

    for key in ['drug_table', 'acronyms', 'tiers']:
        try:
            # Find the key and its array
            key_pattern = rf'"{key}"\s*:\s*\['
            match = re.search(key_pattern, json_string)
            if not match:
                extracted[key] = []
                continue
            
            # Find the matching closing bracket
            start_pos = match.end() - 1  # Position of opening [
            bracket_count = 0
            end_pos = -1
            
            for i in range(start_pos, len(json_string)):
                if json_string[i] == '[':
                    bracket_count += 1
                elif json_string[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i
                        break
            
            if end_pos == -1:
                extracted[key] = []
                continue
            
            # Extract the array string including brackets
            array_str = json_string[start_pos:end_pos + 1]
            
            # Clean and parse
            array_str = re.sub(r',\s*(\])', r'\1', array_str)  # Remove trailing commas
            array_str = re.sub(r',\s*(\})', r'\1', array_str)  # Remove trailing commas before }
            
            try:
                parsed_array = json.loads(array_str)
                extracted[key] = parsed_array if isinstance(parsed_array, list) else []
            except json.JSONDecodeError:
                # If still failing, try object-by-object extraction for drug_table
                if key == 'drug_table':
                    drug_objects = []
                    # More flexible pattern that handles nested braces
                    brace_depth = 0
                    current_obj = ""
                    in_object = False
                    
                    for char in array_str[1:-1]:  # Skip the outer [ ]
                        if char == '{':
                            brace_depth += 1
                            in_object = True
                            current_obj += char
                        elif char == '}':
                            current_obj += char
                            brace_depth -= 1
                            if brace_depth == 0 and in_object:
                                # Try to parse this object
                                try:
                                    clean_obj = re.sub(r',\s*\}', '}', current_obj)
                                    obj = json.loads(clean_obj)
                                    if 'drug_name' in obj:
                                        drug_objects.append(obj)
                                except:
                                    pass
                                current_obj = ""
                                in_object = False
                        elif in_object:
                            current_obj += char
                    
                    extracted[key] = drug_objects
                else:
                    # For acronyms and tiers, try simple string extraction
                    if key == 'acronyms':
                        # Try to extract dictionary objects
                        try:
                            items = re.findall(r'\{[^}]+\}', array_str)
                            acronym_list = []
                            for item in items:
                                try:
                                    acronym_list.append(json.loads(item))
                                except:
                                    pass
                            extracted[key] = acronym_list
                        except:
                            extracted[key] = []
                    else:
                        extracted[key] = []
        except Exception as e:
            logger.debug(f"Failed to extract {key}: {e}")
            extracted[key] = []

    return extracted


# def robust_json_repair(json_string: str):
#     """
#     Attempts to repair common JSON errors from LLMs, primarily focusing on trailing commas,
#     escaping backslashes, and stripping non-JSON content before parsing.
#     """
#     default_output = {"drug_table": [], "acronyms": [], "tiers": []}

#     if not isinstance(json_string, str) or not json_string.strip():
#         return default_output

#     # 1. Find the start and end of the main JSON object.
#     start_index = json_string.find('{')
#     end_index = json_string.rfind('}')

#     if start_index == -1 or end_index == -1:
#         logger.warning("Could not find a JSON object in the LLM response.")
#         return default_output

#     # Extract the JSON part of the string.
#     json_string = json_string[start_index : end_index + 1]

#     # 2. Escape every backslash by doubling it.
#     json_string = json_string.replace("\\", "\\\\")

#     # 3. Fix the most common LLM error: trailing commas before '}' or ']'.
#     json_string = re.sub(r',\s*([}\]])', r'\1', json_string)

#     # 4. Attempt to parse the cleaned JSON.
#     try:
#         # Use json5 for more lenient parsing if available.
#         if JSON5_AVAILABLE:
#             try:
#                 parsed = json5.loads(json_string)
#                 return _sanitize_output(parsed, default_output)
#             except Exception as e:
#                 logger.debug(f"json5 parsing failed, falling back to standard json: {e}")

#         # Fallback to the standard json library.
#         parsed = json.loads(json_string)
#         return _sanitize_output(parsed, default_output)

#     except json.JSONDecodeError as e:
#         logger.error(f"JSON parsing failed definitively after repair attempts: {e}")
#         logger.debug(f"Problematic JSON string for debugging: {json_string}")

#         # Log the failed JSON to a file for analysis.
#         try:
#             with open("failed_llm_json.log", "a", encoding="utf-8") as f:
#                 f.write(f"=== JSON Parse Error ===\n")
#                 f.write(f"Original String: {json_string}\n")
#                 f.write(f"{'='*50}\n\n")
#         except Exception as log_error:
#             logging.warning(f"Failed to write to debug log: {log_error}")

#         return default_output



def robust_json_repair(json_string: str):
    """
    Attempts to repair common JSON errors from LLMs, primarily focusing on trailing commas,
    escaping backslashes, and stripping non-JSON content before parsing.
    """
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}

    if not isinstance(json_string, str) or not json_string.strip():
        return default_output

    original_string = json_string
    
    # 1. Remove markdown code fences FIRST - LLMs often ignore our instructions
    lines = json_string.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are ONLY markdown fences
        if stripped in ['```', '```json', '```JSON']:
            continue
        # Remove inline markdown fences from the start/end of lines
        if stripped.startswith('```'):
            line = line.replace('```json', '').replace('```JSON', '').replace('```', '', 1)
        if stripped.endswith('```'):
            line = line[:line.rfind('```')]
        if line.strip():  # Only keep non-empty lines
            cleaned_lines.append(line)
    json_string = '\n'.join(cleaned_lines)

    # 2. Find the FIRST complete JSON object (not first { to last })
    # This handles "Extra data" errors from multiple JSON objects
    start_index = json_string.find('{')
    if start_index == -1:
        logger.warning("Could not find a JSON object in the LLM response.")
        return default_output
    
    # Find the matching closing brace for the first opening brace
    brace_count = 0
    end_index = -1
    for i in range(start_index, len(json_string)):
        if json_string[i] == '{':
            brace_count += 1
        elif json_string[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = i
                break
    
    if end_index == -1:
        logger.warning("Could not find matching closing brace for JSON object.")
        return default_output
    
    # Extract ONLY the first complete JSON object
    json_string = json_string[start_index : end_index + 1]
    
    # 3. Escape every backslash by doubling it - BUT be careful with already escaped quotes
    # First, protect already-escaped quotes
    json_string = json_string.replace('\\"', '<<<ESCAPED_QUOTE>>>')
    json_string = json_string.replace("\\", "\\\\")
    # Restore the escaped quotes
    json_string = json_string.replace('<<<ESCAPED_QUOTE>>>', '\\"')

    # 4. Fix the most common LLM error: trailing commas before '}' or ']'.
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
    # 5. Fix missing commas between objects in arrays (common LLM error)
    # Pattern: }{  should be },{
    json_string = re.sub(r'\}\s*\{', '},{', json_string)

    # 6. Attempt to parse the cleaned JSON.
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
        logger.error(f"â Œ JSON parsing failed definitively after repair attempts: {e}")
        logger.error(f"   Error location: Line {e.lineno}, Column {e.colno}")
        
        # TRY ONE MORE TIME: Maybe our brace matching was wrong, try simpler approach
        # Look for drug_table, acronyms, tiers arrays in the original string
        logger.warning("âš ï¸  Trying alternative extraction from original response...")
        try:
            # Try to find and parse just the outermost JSON in original string
            alt_start = original_string.find('{')
            if alt_start != -1:
                # Find content that looks like our expected structure
                if '"drug_table"' in original_string or '"acronyms"' in original_string:
                    # Try to extract up to the first occurrence of closing the main object
                    # This is a heuristic approach
                    test_string = original_string[alt_start:]
                    # Try parsing progressively larger chunks
                    for attempt_end in range(len(test_string) - 1, max(0, len(test_string) - 500), -10):
                        if test_string[attempt_end] == '}':
                            try:
                                test_json = test_string[:attempt_end + 1]
                                # Quick cleanup
                                test_json = re.sub(r',\s*([}\]])', r'\1', test_json)
                                parsed_alt = json.loads(test_json)
                                if isinstance(parsed_alt, dict):
                                    result = _sanitize_output(parsed_alt, default_output)
                                    if result.get('drug_table') or result.get('acronyms'):
                                        logger.info(f"âœ… Alternative parsing succeeded! Found {len(result.get('drug_table', []))} drugs")
                                        return result
                            except:
                                continue
        except Exception as alt_error:
            logger.debug(f"Alternative extraction also failed: {alt_error}")

        # Log the failed JSON to a file for analysis.
        try:
            with open("failed_llm_json.log", "a", encoding="utf-8") as f:
                f.write(f"=== JSON Parse Error at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Error: {e}\n")
                f.write(f"Error Line {e.lineno}, Column {e.colno}\n")
                f.write(f"Problematic segment around error:\n")
                # Show lines around the error
                lines = json_string.split('\n')
                if e.lineno and 0 < e.lineno <= len(lines):
                    start_line = max(0, e.lineno - 3)
                    end_line = min(len(lines), e.lineno + 2)
                    for i in range(start_line, end_line):
                        marker = " >>> " if i == e.lineno - 1 else "     "
                        f.write(f"{marker}Line {i+1}: {lines[i]}\n")
                f.write(f"\nCleaned JSON String:\n{json_string}\n")
                f.write(f"\nOriginal Response (first 2000 chars):\n{original_string[:2000]}\n")
                f.write(f"{'='*80}\n\n")
        except Exception as log_error:
            logger.warning(f"Failed to write to debug log: {log_error}")
        
        # TRY LAST-RESORT EXTRACTION using regex to find individual drug records
        logger.warning("âš ï¸  Attempting last-resort partial JSON extraction...")
        partial_data = _extract_partial_json_arrays(original_string)
        if partial_data and partial_data.get('drug_table'):
            logger.info(f"âœ… Partial extraction recovered {len(partial_data['drug_table'])} drugs!")
            return partial_data

        return default_output


def _is_extracted_data_from_index_page(drug_table: List[dict]) -> bool:
    """
    Detect if extracted drug data appears to come from an index/table of contents page.
    Returns True if the data looks like an index, False otherwise.

    Index page indicators:
    - High proportion of entries with page numbers in drug name
    - Drug names that look like section headers or categories
    - Numeric patterns suggesting page references

    NOTE: Missing tier/requirements alone is NOT enough to mark as index page,
    since some formulary formats (like VDP) legitimately have empty tier fields.
    """
    if not drug_table or len(drug_table) == 0:
        return False

    total_records = len(drug_table)

    # NEW: Added more specific counters for different index patterns.
    page_number_pattern_count = 0
    missing_both_count = 0
    category_header_count = 0

    # NEW: List of common section headers found in index pages to filter out.
    category_keywords = [
        'index', 'contents', 'table of contents', 'formulary index',
        'drug index', 'medication index', 'drug class', 'medication list',
        'section', 'chapter', 'overview', 'introduction', 'summary', 'appendix'
    ]

    for entry in drug_table:
        drug_name = str(entry.get('drug_name', '')).strip()
        drug_tier = str(entry.get('drug_tier', '')).strip()
        drug_requirements = str(entry.get('drug_requirements', '')).strip()

        # NEW: Strongest indicator - check for page number patterns in the extracted drug name.
        if re.search(r'\.{2,}\s*\d{1,3}\s*$', drug_name) or re.search(r'\s{3,}\d{1,3}\s*$', drug_name):
            page_number_pattern_count += 1

        # MODIFIED: Stricter check for missing data.
        has_tier = drug_tier and drug_tier not in ['', 'none', 'null', 'n/a']
        has_requirements = drug_requirements and drug_requirements not in ['', 'none', 'null', 'n/a']
        if not has_tier and not has_requirements:
            missing_both_count += 1

        # NEW: Check for category/section header keywords.
        drug_name_lower = drug_name.lower()
        if any(keyword in drug_name_lower for keyword in category_keywords) and len(drug_name) < 50:
            category_header_count += 1

    # NEW: More robust decision logic with stricter thresholds.
    page_number_ratio = page_number_pattern_count / total_records
    missing_both_ratio = missing_both_count / total_records
    category_ratio = category_header_count / total_records
    
    # DEBUG: Log the ratios for diagnosis
    logger.info(f"🔍 Index detection stats: total={total_records}, page_num_pattern={page_number_pattern_count} ({page_number_ratio:.1%}), missing_both={missing_both_count} ({missing_both_ratio:.1%}), category_headers={category_header_count} ({category_ratio:.1%})")

    # If 40% or more entries have a clear page number pattern, it's an index.
    if page_number_ratio >= 0.4:
        logger.info(f"⚠️ Index page detected: {page_number_ratio:.1%} of entries have page number patterns.")
        return True

    # DISABLED: This check was too aggressive - disabled for now
    # if missing_both_ratio >= 0.98:
    #     logger.info(f"Index page detected: {missing_both_ratio:.1%} of entries missing both tier and requirements.")
    #     return True

    # If a high percentage of entries are category headers.
    if category_ratio >= 0.6:
        logger.info(f"⚠️ Index page detected: {category_ratio:.1%} of entries are category headers.")
        return True

    # A combination of page numbers and missing data is also a strong signal.
    # FIXED: Increased missing_both_ratio threshold from 0.7 to 0.85 to be less aggressive
    if page_number_ratio >= 0.2 and missing_both_ratio >= 0.85:
        logger.info(f"⚠️ Index page detected: Page numbers ({page_number_ratio:.1%}) + missing data ({missing_both_ratio:.1%}).")
        return True
    
    logger.info(f"✅ NOT an index page - passing through")
    return False

def _consolidate_and_clean_drug_table(drug_table: List[dict]) -> List[dict]:
    """
    A definitive, multi-stage function to fix fragmented and incorrect drug extractions.
    It performs three critical operations in the correct order:
    1. CONSOLIDATE: Merges fragmented lines into a single drug name.
    2. PROPAGATE: Fills down the correct tier and requirements within drug groups.
    3. FILTER: Removes any remaining invalid or junk records.
    """
    if not drug_table:
        return []

    # --- Stage 1: CONSOLIDATE FRAGMENTS ---
    consolidated_list = []
    if not drug_table:
        return []

    # This buffer holds parts of a drug name that span multiple lines
    current_drug_parts = []
    last_drug = None

    for i, drug in enumerate(drug_table):
        drug_name = str(drug.get("drug_name") or "").strip()
        drug_tier = str(drug.get("drug_tier") or "").strip()

        # A line is a fragment if it lacks a tier AND starts with a non-letter
        # or a common dosage form word. This is much safer.
        is_fragment = (not drug_tier and
                       (not re.match(r'^[a-zA-Z]', drug_name) or
                        re.match(r'^(oral|tablet|capsule|injection|solution|suspension|cream|ointment|lotion|mg|mcg|ml)\b', drug_name, re.IGNORECASE)))

        # Override: If it's a known medical supply, it's NEVER a fragment.
        supply_keywords = ['needle', 'syringe', 'pad', 'strip', 'lancet', 'sensor', 'cap', 'duo']
        if any(keyword in drug_name.lower() for keyword in supply_keywords):
            is_fragment = False

        if not is_fragment:
            # This is a new drug entry. First, save the previously buffered drug.
            if last_drug:
                last_drug['drug_name'] = ' '.join(current_drug_parts)
                consolidated_list.append(last_drug)

            # Start a new buffer for the current drug
            current_drug_parts = [drug_name]
            last_drug = drug
        else:
            # This is a fragment, add it to the current buffer.
            if drug_name:
                current_drug_parts.append(drug_name)

    # Add the very last drug that was being processed
    if last_drug:
        last_drug['drug_name'] = ' '.join(current_drug_parts)
        consolidated_list.append(last_drug)

    # --- Stage 2 & 3: PROPAGATE & FINAL FILTER ---
    final_list = []
    # (The propagation from _clean_and_propagate_drug_groups already handles context)

    # Common section header patterns to filter out
    section_header_patterns = [
        r'^\s*drug\s+name\s*$', r'^\s*drug\s+tier\s*$', r'^\s*requirements\s*$',
        r'^Therapeutic\s+Category', r'^Drug\s+Class', r'^Category:',
        r'^\[.*Category.*\]$', r'^Section\s+\d+', r'^Chapter\s+\d+',
        r'^\s*notes\s*&\s*restrictions\s*$', r'\.{3,}\s*\d+\s*$', r'\s{3,}\d+\s*$'
    ]

    for drug in consolidated_list:
        name = str(drug.get('drug_name') or '').strip()

        is_section_header = any(re.search(pattern, name, re.IGNORECASE) for pattern in section_header_patterns)
        if is_section_header:
            logger.info(f"Filtering out potential junk/header record: '{name}'")
            continue

        # A record is valid if it has a name with at least one real word.
        if name and re.search(r'[a-zA-Z]{3,}', name):
            final_list.append(drug)
        else:
            logger.warning(f"Filtering out invalid junk record (missing valid name): {drug}")

    return final_list

def _clean_and_propagate_drug_groups(drug_table: List[dict]) -> List[dict]:
    """
    Corrected function that fills in missing context (tier/requirements) for
    fragmented drug entries without incorrectly overwriting valid, extracted data.
    """
    if not drug_table:
        return []

    # --- Stage 1: Build a context map from rows that have data ---
    context_map = {}
    for i, drug in enumerate(drug_table):
        # Use a simple base name for grouping (e.g., first word)
        drug_name = str(drug.get('drug_name', '')).strip()
        if not drug_name:
            continue

        base_name = drug_name.split()[0].lower()
        has_tier = drug.get('drug_tier') and str(drug.get('drug_tier')).strip()
        has_reqs = drug.get('drug_requirements') and str(drug.get('drug_requirements')).strip()

        # If this row has good data, store it as the context for this group
        if has_tier or has_reqs:
            context_map[base_name] = {
                'tier': drug.get('drug_tier'),
                'reqs': drug.get('drug_requirements')
            }

    # --- Stage 2: Apply context ONLY to rows that are missing it ---
    corrected_table = []
    for drug in drug_table:
        drug_name = str(drug.get('drug_name', '')).strip()
        if not drug_name:
            continue

        base_name = drug_name.split()[0].lower()

        # Check if the current drug is missing data and if a context exists for it
        is_missing_tier = not (drug.get('drug_tier') and str(drug.get('drug_tier')).strip())
        is_missing_reqs = not (drug.get('drug_requirements') and str(drug.get('drug_requirements')).strip())

        if (is_missing_tier or is_missing_reqs) and base_name in context_map:
            correct_context = context_map[base_name]

            # **THE FIX**: Only fill if the field is empty. Do NOT overwrite.
            if is_missing_tier and correct_context.get('tier'):
                drug['drug_tier'] = correct_context['tier']
                logger.debug(f"Propagated tier '{correct_context['tier']}' to '{drug_name}'")

            if is_missing_reqs and correct_context.get('reqs'):
                drug['drug_requirements'] = correct_context['reqs']
                logger.debug(f"Propagated reqs '{correct_context['reqs']}' to '{drug_name}'")

        corrected_table.append(drug)

    return corrected_table


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

def extract_printed_page_number_from_markdown(markdown: Optional[str]) -> Optional[int]:
    """
    Try to extract a printed page number from OCR markdown/text for a page.

    Heuristics tried (in order):
      1. A lone number on one of the last 6 non-empty lines: "1" or "12"
      2. "Page 1", "Page 1 of 10", "Pg. 1", "p. 1", possibly with punctuation
      3. A number appearing near the end of the page text (last 200 chars)
      4. If nothing found -> return None

    Returns int when found, else None.
    """
    if not markdown:
        return None

    # normalize line endings & split
    lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
    # 1) Check last few lines for a single number
    for ln in reversed(lines[-6:]):  # footers are usually within the last few lines
        if re.fullmatch(r'\d{1,4}', ln):
            try:
                return int(ln)
            except ValueError:
                pass

    # Prepare a short tail of the text (footers typically near the end)
    tail = "\n".join(lines[-12:]) if lines else markdown
    tail_search = tail[-400:] if len(tail) > 400 else tail  # limit search area

    # 2) Common "Page X" patterns
    page_patterns = [
        r'page[\s:\.-]*?(\d{1,4})\b',     # "Page 1", "page-1"
        r'pg[\s\.]*?(\d{1,4})\b',         # "Pg. 1"
        r'\bp[\s\.]*?(\d{1,4})\b',        # "p. 1"
        r'(\d{1,4})\s+of\s+\d{1,4}\b',    # "1 of 10"
    ]
    for pat in page_patterns:
        m = re.search(pat, tail_search, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except (IndexError, ValueError):
                continue

    # 3) As a last resort, look for any lone number near the very end
    m = re.search(r'(\d{1,3})\s*$', tail_search)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    return None


def apply_effective_page_numbers(
    ocr_pages: List[Union[str, Dict]],
    structured_records_per_page: List[List[Dict]],
) -> List[Dict]:
    """
    Given OCR pages and the structured records produced per OCR page,
    return a single flattened list of structured records where each record will have:
      - 'page_number'        : printed page number when found, otherwise the OCR page index (1-based)
      - 'ocr_page_index'     : the OCR page sequence index (1-based) (for auditing)

    Parameters
    - ocr_pages: list where each item is either:
        * a markdown string for the page, or
        * a dict having a 'markdown' or 'text' key
      The list order represents the OCR page sequence.
    - structured_records_per_page: list with same length as ocr_pages; each element is a list
      of structured records (dicts) extracted from that OCR page.

    Returns:
      - flattened list of dict records with page_number & ocr_page_index injected.
    """
    if len(ocr_pages) != len(structured_records_per_page):
        logger.warning(
            "apply_effective_page_numbers: ocr_pages length != structured_records_per_page length "
            f"({len(ocr_pages)} vs {len(structured_records_per_page)})"
        )

    flattened: List[Dict] = []
    n_pages = min(len(ocr_pages), len(structured_records_per_page))

    for idx in range(n_pages):
        ocr_page = ocr_pages[idx]
        # robustly obtain page markdown text
        if isinstance(ocr_page, str):
            markdown = ocr_page
        elif isinstance(ocr_page, dict):
            markdown = (
                ocr_page.get("markdown")
                or ocr_page.get("text")
                or ocr_page.get("page_text")
                or ""
            )
        else:
            markdown = ""

        ocr_page_index = idx + 1  # 1-based
        printed_page = extract_printed_page_number_from_markdown(markdown)
        effective_page = printed_page if printed_page is not None else ocr_page_index

        logger.debug(
            "Page mapping: ocr_index=%d printed_footer=%s effective=%d",
            ocr_page_index,
            str(printed_page),
            effective_page,
        )

        records = structured_records_per_page[idx] or []
        for rec in records:
            # preserve existing page_number if present? we overwrite intentionally
            rec["page_number"] = effective_page
            # add raw OCR index for auditing
            rec["ocr_page_index"] = ocr_page_index
            flattened.append(rec)

    # If lengths differ, handle any remaining pages (defensive)
    if len(ocr_pages) > n_pages:
        logger.warning("apply_effective_page_numbers: extra OCR pages without structured records.")
    if len(structured_records_per_page) > n_pages:
        logger.warning("apply_effective_page_numbers: extra structured pages without OCR pages; "
                       "assigning ocr_page_index=None")
        for idx in range(n_pages, len(structured_records_per_page)):
            for rec in structured_records_per_page[idx]:
                rec["page_number"] = None
                rec["ocr_page_index"] = None
                flattened.append(rec)

    return flattened


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
    Detect if a page is an index/table of contents with enhanced detection logic.
    Returns True if index, False otherwise.
    """

    lower_markdown = markdown.lower()
    
    if 'drug name' in lower_markdown and ('tier' in lower_markdown or 'requirements' in lower_markdown):
        return False
    lines = markdown.splitlines()
    upper_markdown = markdown.upper()
    
    # === CRITICAL: Check for "Alphabetical Index" with actual drug data FIRST ===
    # Many formularies title their main drug list "Alphabetical Index" even though it contains actual drugs
    # This check MUST run before generic "index" keyword matching
    if 'alphabetical index' in lower_markdown:
        # Check if it has actual drug data (tier values, dosage forms, requirements)
        has_tier_column = 'tier' in lower_markdown and '|' in lower_markdown
        has_category_column = 'category' in lower_markdown and '|' in lower_markdown  # Hennepin Health specific
        has_special_code_column = 'special code' in lower_markdown and '|' in lower_markdown  # Hennepin Health specific
        has_dosage_forms = any(form in lower_markdown for form in ['tab', 'cap', 'soln', 'inj', 'mg', 'mcg', 'susp', 'cream'])
        has_requirements = any(req in lower_markdown for req in ['ql', 'pa', 'st', 'ol', 'inf', 'vac', 'otc', '90ds'])
        
        # If it has a tier column AND either dosage forms OR requirements OR special columns, it's a DRUG PAGE
        if has_tier_column and (has_dosage_forms or has_requirements or has_category_column or has_special_code_column):
            logger.info(f"âœ… 'Alphabetical Index' page has actual drug data. This is a DRUG PAGE, not an index!")
            return False
    
    # === Check for specific payer table structures (Hennepin Health, etc.) ===
    # If we detect specific column patterns that indicate drug data pages, it's NOT an index
    hennepin_health_pattern = ('drug name' in lower_markdown and 'special code' in lower_markdown and 
                                'tier' in lower_markdown and 'category' in lower_markdown and '|' in lower_markdown)
    
    if hennepin_health_pattern:
        logger.info("âœ… Detected Hennepin Health table structure (Drug Name | Special Code | Tier | Category). This is a DRUG PAGE, not an index!")
        return False
    
    # === Check for drug page indicators BEFORE index keywords ===
    negative_keywords = ['drug tier', 'coverage details', 'requirements', 'limitations', 'dosage form', 'special code']
    if any(keyword in lower_markdown for keyword in negative_keywords):
        # A high number of table headers for requirements is a very strong signal of a drug page.
        if lower_markdown.count('|') > 20 and 'drug name' in lower_markdown:
             logger.info("Detected drug page characteristics (negative keywords, table structure). Not an index.")
             return False

    # === Check for specific index keywords (but NOT "alphabetical index" which was already handled) ===
    # FIXED: Made more specific to avoid false positives on "Alphabetical Index" drug pages
    index_keywords = ['table of contents', 'formulary index']  # Removed generic 'index' and 'contents'
    
    for keyword in index_keywords:
        if keyword.upper() in upper_markdown:
            for line in lines[:10]:  # Check first 10 lines
                if keyword.upper() in line.upper() and (line.strip().startswith('#') or len(line.strip()) < 50):
                    logger.info(f"Detected index page based on keyword '{keyword}' in header.")
                    return True

    page_number_pattern = re.compile(r'\.{3,}\s*\d+\s*$')

    page_number_lines = 0
    total_meaningful_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('|') or stripped.startswith(':'):
            continue

        total_meaningful_lines += 1

        # Check if line ends with dots and page number (classic index format)
        if page_number_pattern.search(stripped):
            page_number_lines += 1

    if total_meaningful_lines > 0:
        page_number_ratio = page_number_lines / total_meaningful_lines
        # If 30% or more lines have the "...page_number" pattern, it's an index
        if page_number_ratio >= 0.30:
            logger.info(f"Detected index page: {page_number_ratio:.0%} of lines have '...page_number' pattern.")
            return True

    suspected_page_numbers = re.findall(r'\b(\d{2,3})\b', markdown)  # Find 2-3 digit numbers

    if len(suspected_page_numbers) > 20:  # More than 20 standalone numbers
        # Check if they're sequential or clustered (typical of page numbers)
        unique_numbers = set(int(n) for n in suspected_page_numbers)
        if len(unique_numbers) < len(suspected_page_numbers) * 0.5:  # Many duplicates
            logger.info(f"Detected index page: Found {len(suspected_page_numbers)} page-like numbers with limited variety.")
            return True

    # === ORIGINAL: Table-based index detection ===
    table_index_count = 0
    table_cell_count = 0
    index_entry_pattern = re.compile(r'[A-Za-z0-9\s\(\)\-]+\.{2,}\s*\d+')

    for line in lines:
        if '|' in line and not line.strip().startswith(':'):
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]

            for cell in cells:
                if cell:
                    table_cell_count += 1
                    if index_entry_pattern.search(cell):
                        table_index_count += 1

    if table_cell_count > 0:
        index_ratio = table_index_count / table_cell_count
        if index_ratio >= 0.4:
            logger.info("Detected index page based on table content.")
            return True

    content_lines = [
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith('|') and not line.strip().startswith(':')
    ]

    if not content_lines:
        return False

    # Improved patterns for index lines
    index_pattern = re.compile(r'^[A-Za-z0-9\s\(\)\-\.,]+\.{2,}\s*\d+\s*$')
    alt_pattern = re.compile(r'^[A-Za-z0-9\s\(\)\-\.,]+\s+\d{2,3}\s*$')  # Name followed by 2-3 digit number

    index_lines = sum(
        1 for line in content_lines
        if index_pattern.search(line) or alt_pattern.search(line)
    )

    if len(content_lines) > 0:
        index_line_ratio = index_lines / len(content_lines)
        # FIXED: Increased threshold from 30% to 50% to avoid false positives on drug pages
        # If at least 50% of content lines match the index pattern, it's likely an index page
        if index_line_ratio >= 0.5:
            logger.info(f"Detected index page based on line patterns: {index_line_ratio:.0%} match.")
            return True

    # === NEW: Detect if the page is mostly very short text entries (typical of index) ===
    if len(content_lines) > 10:  # Only check if we have enough lines
        avg_line_length = sum(len(line) for line in content_lines) / len(content_lines)
        short_lines = sum(1 for line in content_lines if len(line) < 50)
        short_line_ratio = short_lines / len(content_lines)

        if short_line_ratio >= 0.7 and avg_line_length < 40 and len(suspected_page_numbers) > 15:
            logger.info("Detected index page: High ratio of short lines with many numbers.")
            return True

    # === NEW: Detect table-based index with mostly empty cells (except drug name and page number) ===
    # This catches index pages that are formatted as tables but lack tier/requirement data
    # FIXED: Made less aggressive for formularies like Hennepin Health where some columns may be empty
    table_rows = [line for line in lines if '|' in line and not line.strip().startswith(':')]
    if len(table_rows) > 5:  # Need at least a few rows to analyze
        empty_cell_pattern = re.compile(r'\|\s*\|')  # Adjacent pipes with only whitespace
        rows_with_empty_cells = sum(1 for row in table_rows if empty_cell_pattern.search(row))

        # Also check for tables with very few columns filled (e.g., only 2 out of 5)
        # FIXED: Increased threshold from "non_empty_cells <= 2" to check for truly sparse tables
        rows_with_mostly_empty = 0
        for row in table_rows:
            cells = [cell.strip() for cell in row.split('|')]
            non_empty_cells = sum(1 for cell in cells if cell)
            # FIXED: Only flag if less than 2 cells are filled (was <= 2, now < 2)
            # This allows tables with Drug Name + Tier (2 columns) to pass through
            if len(cells) > 4 and non_empty_cells < 2:  # Most cells empty (changed from <= 2)
                rows_with_mostly_empty += 1

        empty_ratio = (rows_with_empty_cells + rows_with_mostly_empty) / len(table_rows)
        # FIXED: Increased threshold from 0.6 (60%) to 0.8 (80%) to be less aggressive
        if empty_ratio >= 0.8:  # 80% or more rows have mostly empty cells (was 60%)
            logger.info(f"Detected index page: Table with {empty_ratio:.0%} of rows having mostly empty cells.")
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
def extract_structured_data_with_llm(page_markdown: str, mistral_client: Mistral, payer_name: str = None):
    """
    Uses Mistral Chat with structured output to parse markdown and extract drug data.
    """
    costs = {'tokens': 0, 'cost': 0.0, 'calls': 1}
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}

    if not mistral_client:
        logger.error("Mistral client is not provided. Cannot extract structured data.")
        return default_output, costs

    if is_index_page(page_markdown):
        logger.info("Skipping LLM call for index/table of contents page.")
        return default_output, {'tokens': 0, 'cost': 0.0, 'calls': 0}

    if is_aca_drug_list_page(page_markdown):
        logger.info("Skipping LLM call for ACA Drug List/Preventative Medications page.")
        return default_output, {'tokens': 0, 'cost': 0.0, 'calls': 0}

    user_message = f"Extract drug formulary information from the following text:\n\n{page_markdown}"


    for attempt in range(MAX_RETRIES):
        try:
            response = mistral_client.chat.complete(
                model="mistral-large-latest",
    messages=[
    {"role": "system", "content": (
        "You are a professional medical data extractor. You will receive a page from a drug formulary. "
        "IMPORTANT: Drug Tiers are often found in SECTION HEADERS above the drug list (e.g., 'Tier 1 - Generic', 'Tier 2'). "
        "If a drug row does not have a specific tier column, you MUST use the most recent Section Header as the Tier for that drug. "
        "CRITICAL EXCLUSION: Do NOT mistreat 'Therapeutic Categories' (e.g., 'Pain Relief', 'Cardiovascular', 'Inflammatory Disease') as the Drug Tier. "
        "If the immediate header is a medical condition or class, LOOK HIGHER up the page for the true Tier Header (e.g. 'Tier 1', 'Generic', 'Preferred Brand'). "
        "   - Map 'Generic' or 'Generics' -> 'Tier 1' (or T1)"
        "   - Map 'Preferred Brand' -> 'Tier 2' (or T2)"
        "   - Map 'Non-Preferred' -> 'Tier 3' (or T3)"
        "   - Map 'Specialty' -> 'Tier 4' (or T4)"
        "   - CRITICAL: Only apply these mappings if they appear in SECTION HEADERS."
        "LAYOUT SPECIFIC: If the page has columns labeled 'Preferred' and 'Non-Preferred', treat these as TIERS. "
        "   - Drugs under 'Preferred' -> Tier = 'Preferred' "
        "   - Drugs under 'Non-Preferred' -> Tier = 'Non-Preferred' "
        "If you absolutely cannot find a Tier Header (and it's not the Preferred/Non-Preferred layout), return null (empty) for the Tier field. "
        "REQUIREMENTS extraction: "
        "   - Check for codes in PARENTHESES (e.g. (QL), (PA)) or SQUARE BRACKETS (e.g. [NP], [SP], [DL]). "
        "   - EXTRACT these codes into the 'Drug Edit' field. "
        "   - Do NOT map [NP] or [SP] text to the Drug Tier field UNLESS it is clearly a column header. Keep them in Drug Edit. "
        "   - Check for 'GEN 5' or 'Gen 5'. "
        "Do NOT use the document title (e.g., '3-Tier Drug List') as the tier. "
        "IMPORTANT: Some pages may have a 'Legend' or 'Introduction' at the top. Ignore the intro for drug rows, "
        "but YOU MUST extract any 'Tier Definitions' or 'Keys' (like 'T1 = Tier 1', 'QL = Quantity Limit') "
        "but YOU MUST extract any 'Tier Definitions' or 'Keys' (like 'T1 = Tier 1', 'QL = Quantity Limit') "
        "and put them into the 'FormularyAbbreviations' list. "
        "Extract every single drug listed in the tables below it. "
        f"Output JSON matching this schema: {json.dumps(DRUG_EXTRACTION_SCHEMA)}"
    )},
    {"role": "user", "content": user_message}
],
                response_format={"type": "json_object", "schema": DRUG_EXTRACTION_SCHEMA},
                temperature=0
            )

            response_content = response.choices[0].message.content

            usage = response.usage
            total_tokens = usage.total_tokens
            costs['tokens'] = total_tokens
            costs['cost'] = (total_tokens / 1000.0) * 0.002 # Placeholder cost

            try:
                structured_response = json.loads(response_content)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response from Mistral.")
                continue

            # Map to internal structure
            drug_table = []
            for item in structured_response.get("DrugInformation", []):
                drug_table.append({
                    "drug_name": item.get("Drug Name"),
                    "drug_tier": item.get("drug tier"),
                    "drug_requirements": item.get("requirements")
                })

            acronyms = []
            for item in structured_response.get("FormularyAbbreviations", []):
                acronyms.append({
                    "acronym": item.get("Acronym"),
                    "expansion": item.get("Expansion"),
                    "explanation": item.get("Explanation")
                })
            
            structured_data = {
                "drug_table": drug_table,
                "acronyms": acronyms,
                "tiers": [] 
            }

            logger.info(
                f"Successfully processed page. Extracted: "
                f"{len(structured_data['drug_table'])} drugs, "
                f"{len(structured_data['acronyms'])} acronyms."
            )

            return structured_data, costs

        except Exception as e:
            logger.error(f"Error in Mistral LLM data extraction (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_MULTIPLIER ** attempt)
            else:
                return default_output, costs
    
    return default_output, costs


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


def _parse_page_ranges(page_config_value: Union[str, list, None]) -> List[int]:
    """
    Parses a flexible page range configuration into a flat list of page numbers.
    Handles "all", lists of numbers, and lists of strings with ranges (e.g., "10-20").
    """
    if not page_config_value:
        return []

    pages = set()
    # Ensure the config value is a list to simplify processing
    if isinstance(page_config_value, str) and page_config_value.lower() != 'all':
        config_list = [item.strip() for item in page_config_value.split(',')]
    elif not isinstance(page_config_value, list):
        config_list = [page_config_value]
    else:
        config_list = page_config_value

    for item in config_list:
        item_str = str(item).strip()
        if '-' in item_str:
            try:
                start, end = map(int, item_str.split('-'))
                if start <= end:
                    pages.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"Ignoring malformed page range: '{item_str}'")
        else:
            try:
                pages.add(int(item_str))
            except ValueError:
                logger.warning(f"Ignoring invalid page number entry: '{item_str}'")
    return sorted(list(pages))


def _get_pages_to_process(filename: Optional[str], total_pages: int) -> List[int]:
    """
    Determines which page indices to process based on the configuration in config.py.
    Returns a list of 0-based page indices.
    """
    config = PDF_PAGE_PROCESSING_CONFIG

    # Default behavior is to process all pages
    selected_rule = "all"
    rule_source = "system default"

    # Find a specific rule for the filename
    if filename:
        for key, pages_rule in config.items():
            if key != "default" and key.lower() in filename.lower():
                selected_rule = pages_rule
                rule_source = f"specific rule for key '{key}'"
                break

    # If no specific rule was found, use the default
    if rule_source == "system default" and "default" in config:
        selected_rule = config["default"]
        rule_source = "configuration default"

    logger.info(f"Applying page processing rule for '{filename}' from {rule_source}: {selected_rule}")

    # Process the selected rule
    if isinstance(selected_rule, str) and selected_rule.lower() == "all":
        logger.info(f"Processing all {total_pages} pages for '{filename}'.")
        return list(range(total_pages))

    # Parse the rule, which can be a list of numbers and/or string ranges
    page_numbers_1_based = _parse_page_ranges(selected_rule)

    if not page_numbers_1_based:
        logger.warning(f"No valid pages specified by rule '{selected_rule}' for '{filename}'. No pages will be processed.")
        return []

    # Filter this numeric list to remove out-of-range pages and get valid 1-based pages
    valid_pages_1_based = [p for p in page_numbers_1_based if 1 <= p <= total_pages]

    # Log which pages were specified but invalid
    invalid_pages = [p for p in page_numbers_1_based if p not in valid_pages_1_based]
    if invalid_pages:
        logger.warning(f"Ignoring invalid/out-of-range pages for '{filename}': {invalid_pages}. Total pages in document: {total_pages}.")

    # Convert the final valid list to 0-based indices for processing
    page_indices_0_based = [p - 1 for p in valid_pages_1_based]

    logger.info(f"Final list of pages to process for '{filename}': {[p + 1 for p in page_indices_0_based]}")
    return sorted(list(set(page_indices_0_based)))


def process_pdf_with_mistral_ocr(pdf_input, payer_name=None, filename: Optional[str] = None):
    """
    Processes a PDF using Mistral OCR and a parallelized LLM pipeline.
    Includes an optional enhancement step to improve OCR quality for complex documents.
    """
    if not filename:
        filename = getattr(pdf_input, 'name', 'in_memory_file.pdf')

    logger.info(f"Analyzing PDF with parallel LLM pipeline: {filename}")

    if PYPDF2_AVAILABLE and isinstance(pdf_input, BytesIO):
        try:
            reader = PyPDF2.PdfReader(pdf_input)
            num_pages = len(reader.pages)
            if num_pages > MAX_PDF_PAGES:
                logger.error(f"PDF has {num_pages} pages, exceeding limit of {MAX_PDF_PAGES}.")
                return {"drug_table": [], "acronyms": [], "tiers": []}, "", {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
            pdf_input.seek(0)
        except Exception as e:
            logger.warning(f"Failed to check PDF page count: {e}")

    total_costs = {'mistral_pages': 0, 'mistral_cost': 0.0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    
    mistral_client = create_resilient_mistral_client()

    try:
        # --- OPTIMIZATION: Extract only the pages we need BEFORE uploading ---
        # This dramatically reduces upload size and processing time
        
        # First, determine which pages to process
        temp_page_count = num_pages if 'num_pages' in locals() else 0
        if temp_page_count == 0 and PYPDF2_AVAILABLE and isinstance(pdf_input, BytesIO):
            try:
                pdf_input.seek(0)
                reader = PyPDF2.PdfReader(pdf_input)
                temp_page_count = len(reader.pages)
                pdf_input.seek(0)
            except:
                temp_page_count = 100  # Fallback
        
        page_indices = _get_pages_to_process(filename, temp_page_count or 100)
        pages_to_extract = [p + 1 for p in page_indices]  # 1-based page numbers
        
        logger.info(f"📄 [OPTIMIZATION] Extracting only pages {pages_to_extract} from {temp_page_count}-page PDF before upload...")
        
        # Extract only needed pages using PyMuPDF
        pdf_to_process = pdf_input
        if PYMUPDF_AVAILABLE and pages_to_extract:
            try:
                if isinstance(pdf_input, BytesIO):
                    pdf_input.seek(0)
                    src_doc = fitz.open(stream=pdf_input.getvalue(), filetype="pdf")
                else:
                    src_doc = fitz.open(str(pdf_input))
                
                # Create a new PDF with only the needed pages
                extracted_doc = fitz.open()
                for page_num in pages_to_extract:
                    if 1 <= page_num <= len(src_doc):
                        extracted_doc.insert_pdf(src_doc, from_page=page_num-1, to_page=page_num-1)
                
                extracted_bytes = extracted_doc.tobytes()
                extracted_doc.close()
                src_doc.close()
                
                pdf_to_process = BytesIO(extracted_bytes)
                logger.info(f"✅ [OPTIMIZATION] Extracted {len(pages_to_extract)} pages ({len(extracted_bytes)/1024:.1f} KB) from original PDF.")
                
            except Exception as e:
                logger.warning(f"⚠️ Page extraction failed: {e}. Using full PDF.")
                if isinstance(pdf_input, BytesIO):
                    pdf_input.seek(0)
                pdf_to_process = pdf_input
        
        # --- PDF Enhancement Step (now only enhances the extracted pages) ---
        if USE_ENHANCED_PDF:
            enhanced_pdf_bytes = enhance_pdf(pdf_to_process)
            if enhanced_pdf_bytes:
                pdf_to_process = enhanced_pdf_bytes
                logger.info("Proceeding with enhanced PDF for OCR.")
            else:
                logger.warning("PDF enhancement failed. Falling back to original PDF.")
                if isinstance(pdf_to_process, BytesIO):
                    pdf_to_process.seek(0)
        
        # --- Document Upload and OCR ---
        if isinstance(pdf_to_process, BytesIO):
            file_bytes = pdf_to_process.getvalue()
            file_name_for_upload = "enhanced_temp.pdf" if pdf_to_process is not pdf_input else "temp_in_memory.pdf"
        else:
            pdf_file = Path(pdf_to_process)
            file_bytes = pdf_file.read_bytes()
            file_name_for_upload = pdf_file.name

        uploaded_file = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} to upload '{file_name_for_upload}' to Mistral...")
                uploaded_file = mistral_client.files.upload(
                    file={"file_name": file_name_for_upload, "content": file_bytes},
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
                    logger.error(f"Failed to upload file to Mistral after {MAX_RETRIES} attempts.")
                    raise

        if not uploaded_file:
            return {"drug_table": [], "acronyms": [], "tiers": []}, "", total_costs

        # --------------------------------------------------------------------------
        # PRIMARY FLOW: Native Mistral OCR with Structured JSON Output
        # Using ocr.process() with document_annotation_format - FAST like playground!
        # NOTE: document_annotation_format has a LIMIT of 8 pages per request
        # For larger documents, we process in chunks of 8 pages
        # --------------------------------------------------------------------------
        MAX_PAGES_PER_OCR_REQUEST = 4  # Mistral API limit for document_annotation_format
        
        logger.info(f"--- [PRIMARY FLOW] Native OCR Extraction for {filename} ---")
        try:
            num_pages_in_upload = len(pages_to_extract) if 'pages_to_extract' in locals() else 1
            
            # Check if we need to process in chunks
            if num_pages_in_upload > MAX_PAGES_PER_OCR_REQUEST:
                logger.info(f"📄 Document has {num_pages_in_upload} pages, exceeds {MAX_PAGES_PER_OCR_REQUEST} page limit. Processing in chunks...")
                
                # We need to process in chunks - delete the current uploaded file and process chunks
                try:
                    mistral_client.files.delete(file_id=uploaded_file.id)
                except: pass
                
                # =====================================================================
                # OPTIMIZATION 5: Smart page pre-filtering BEFORE chunking
                # =====================================================================
                if isinstance(pdf_input, BytesIO):
                    pdf_input.seek(0)
                    filtered_pages = prefilter_pages_with_pymupdf(pdf_input, pages_to_extract)
                    pdf_input.seek(0)
                else:
                    filtered_pages = pages_to_extract
                
                if len(filtered_pages) < len(pages_to_extract):
                    logger.info(f"🚀 [OPTIMIZATION 5] Pre-filtering reduced pages from {len(pages_to_extract)} to {len(filtered_pages)}")
                
                # Split filtered pages into chunks
                all_structured_data = []
                all_acronyms = []
                total_pages_processed = 0
                
                page_chunks = [filtered_pages[i:i + MAX_PAGES_PER_OCR_REQUEST] 
                              for i in range(0, len(filtered_pages), MAX_PAGES_PER_OCR_REQUEST)]
                
                logger.info(f"📦 Split into {len(page_chunks)} chunks of up to {MAX_PAGES_PER_OCR_REQUEST} pages each")
                
                # =====================================================================
                # OPTIMIZATION 1: PARALLEL CHUNK PROCESSING
                # Process multiple chunks concurrently using ThreadPoolExecutor
                # =====================================================================
                
                # Define the OCR annotation schema once for all chunks
                # IMPORTANT: This schema supports MULTIPLE PDF formats:
                # 
                # FORMAT 1 (Traditional): Drug Name | Drug Tier | Requirements/Limits columns
                # FORMAT 2 (PDL): B,G,O | Comment | P,N,R,NR | Therapeutic Category | PA Form Link
                #
                # The extraction code handles both formats dynamically.
                #
                ocr_annotation_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "drug_extraction_schema",
                        "schema": {
                            "type": "object",
                            "title": "StructuredData",
                            "properties": {
                                "DrugInformation": {
                                    "type": "array",
                                    "description": "Extract drugs from formulary/PDL drug listing tables. Supports multiple formats: 1) Traditional tables with 'Drug Name', 'Drug Tier', 'Requirements' columns, 2) PDL tables with 'B,G,O', 'Comment', 'P,N,R,NR', 'Therapeutic Category' columns. SKIP index pages, TOC, and category header rows.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Drug Name": {
                                                "type": "string",
                                                "description": "The drug name. In traditional format: from 'Drug Name' column. In PDL format: from 'Therapeutic Category' column. Examples: 'ACCURETIC', 'Lisinopril', 'APTIVUS ORAL CAPSULE 250 MG'."
                                            },
                                            "drug tier": {
                                                "type": ["string", "null"],
                                                "description": "The tier/comment value. In traditional format: from 'Drug Tier' column (1, 2, 3, 4). In PDL format: from 'Comment' column (numbers like 6, 11, 12). Null if not present."
                                            },
                                            "requirements": {
                                                "type": ["string", "null"],
                                                "description": "Requirements/restrictions. In traditional format: from 'Requirements/Limits' column (e.g., 'PA', 'QL (30/day)', 'ST'). In PDL format: the combined value is built from BGO and PNRNR columns. Null if none."
                                            },
                                            "BGO": {
                                                "type": ["string", "null"],
                                                "description": "PDL format only: Value from 'B, G or O' column. B=Brand, G=Generic, O=OTC. Null if not in PDL format."
                                            },
                                            "PNRNR": {
                                                "type": ["string", "null"],
                                                "description": "PDL format only: Value from 'P, N, R or NR' column. P=Preferred, N=Non-Preferred, R=Recommended, NR=Non-Recommended. Null if not in PDL format."
                                            },
                                            "category": {
                                                "type": ["string", "null"],
                                                "description": "Therapeutic category or section header: 'ACE Inhibitors', 'HIV/AIDS', 'Infections', etc. Null if none."
                                            },
                                            "page_number": {
                                                "type": ["integer", "null"],
                                                "description": "The page number where this drug is found in the document."
                                            },
                                            "pa_form_link": {
                                                "type": ["string", "null"],
                                                "description": "PA Form Link URL if present. Null if none."
                                            }
                                        }
                                    }
                                },
                                "FormularyAbbreviations": {
                                    "type": "array",
                                    "description": "Extract ALL abbreviation/legend definitions: 1) Traditional codes (PA=Prior Authorization, QL=Quantity Limit, ST=Step Therapy), 2) PDL codes (B=Brand, G=Generic, P=Preferred, N=Non-Preferred), 3) Comment reference table (numbers with descriptions), 4) Any tier legends.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Acronym": {"type": "string", "description": "The abbreviation code (B, G, O, P, N, R, NR, PA, QL, ST, or Comment numbers like 6, 11, 12, etc.)"},
                                            "Expansion": {"type": "string", "description": "What it stands for (Brand, Generic, Preferred, or for comment numbers: the full comment text)"},
                                            "Explanation": {"type": ["string", "null"], "description": "Additional explanation if any"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                # Prepare PDF bytes for parallel processing
                if isinstance(pdf_input, BytesIO):
                    pdf_input.seek(0)
                    pdf_bytes_for_parallel = pdf_input.getvalue()
                else:
                    with open(str(pdf_input), 'rb') as f:
                        pdf_bytes_for_parallel = f.read()
                
                # Prepare chunk info for parallel processing
                chunk_infos = [
                    {
                        'chunk_idx': chunk_idx,
                        'chunk_pages': chunk_pages,
                        'pdf_bytes': pdf_bytes_for_parallel,
                        'ocr_schema': ocr_annotation_schema
                    }
                    for chunk_idx, chunk_pages in enumerate(page_chunks)
                ]
                
                logger.info(f"🚀 [OPTIMIZATION 1] Processing {len(page_chunks)} chunks in PARALLEL with {OCR_CHUNK_WORKERS} workers...")
                start_parallel_time = time.time()
                
                # Process chunks in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=OCR_CHUNK_WORKERS) as executor:
                    future_to_chunk = {
                        executor.submit(process_single_chunk_parallel, chunk_info): chunk_info['chunk_idx']
                        for chunk_info in chunk_infos
                    }
                    
                    completed_chunks = 0
                    for future in as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            result = future.result()
                            completed_chunks += 1
                            
                            if result['error']:
                                logger.warning(f"⚠️ Chunk {chunk_idx + 1} returned with error: {result['error']}")
                            else:
                                all_structured_data.extend(result['drugs'])
                                all_acronyms.extend(result['acronyms'])
                                total_pages_processed += result['pages_processed']
                                
                                logger.info(f"📊 Chunk {chunk_idx + 1}/{len(page_chunks)} complete. "
                                           f"Total drugs so far: {len(all_structured_data)} "
                                           f"({completed_chunks}/{len(page_chunks)} chunks done)")
                        
                        except Exception as exc:
                            logger.error(f"⚠️ Chunk {chunk_idx + 1} generated exception: {exc}")
                
                parallel_duration = time.time() - start_parallel_time
                logger.info(f"⏱️ [OPTIMIZATION 1] Parallel processing completed in {parallel_duration:.1f}s "
                           f"(avg {parallel_duration/len(page_chunks):.2f}s per chunk)")
                
                # Return combined results from all chunks
                full_structured_data = {
                    "drug_table": all_structured_data,
                    "acronyms": all_acronyms,
                    "tiers": []
                }
                
                total_costs['mistral_pages'] = total_pages_processed
                total_costs['mistral_cost'] = (total_pages_processed / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES
                
                logger.info(f"✅ [PRIMARY FLOW SUCCESS] Parallel chunked OCR extraction completed for {filename}.")
                logger.info(f"📊 Extracted {len(all_structured_data)} drugs, {len(all_acronyms)} acronyms from {total_pages_processed} page(s) in {len(page_chunks)} chunks.")
                
                # Log page distribution for verification
                if all_structured_data:
                    page_counts = {}
                    for drug in all_structured_data:
                        p = drug.get('page_number')
                        page_counts[p] = page_counts.get(p, 0) + 1
                    logger.info(f"📄 Page distribution: {dict(sorted(page_counts.items()))}")
                    # Log sample drugs for debugging
                    logger.info(f"🔍 Sample drugs from extraction:")
                    for i, drug in enumerate(all_structured_data[:5]):
                        logger.info(f"   Drug {i+1}: '{drug.get('drug_name', '')[:50]}...' on page {drug.get('page_number')}")
                else:
                    logger.warning(f"⚠️ NO DRUGS EXTRACTED from parallel chunked processing!")
                
                logger.info(f"🚀 RETURNING from parallel chunked processing: {len(all_structured_data)} drugs in drug_table")
                return full_structured_data, "[PARALLEL CHUNKED OCR EXTRACTION]", total_costs
            
            # For documents <= 8 pages, process normally (single request)
            signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=300)
            logger.info(f"--- [PRIMARY FLOW] Processing {num_pages_in_upload} page(s) with mistral-ocr-latest ---")

            # Define the structured output schema for native OCR
            # Handles format like "DRUG NAME (PA, QL)" - extracts drug name and parenthesis codes separately
            ocr_annotation_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "drug_extraction_schema",
                    "schema": {
                        "type": "object",
                        "title": "StructuredData",
                        "properties": {
                            "DrugInformation": {
                                "type": "array",
                                "description": "Extract drugs ONLY from actual drug listing pages. SKIP Index pages, Table of Contents, Introduction, Appendix, and pages that only list drug names alphabetically without tier/requirements columns. Only extract from pages with FULL drug details (name + tier OR name + requirements).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Drug Name": {"type": "string", "description": "COMPLETE drug name with form and dosage. Examples: 'APTIVUS ORAL CAPSULE 250 MG', 'atazanavir oral capsule 150 mg, 300 mg'. Include ALL dosages/strengths listed."},
                                        "drug tier": {"type": ["string", "null"], "description": "From 'Drug Tier' column (1, 2, 3, 4) OR from page header. Null if not shown."},
                                        "requirements": {"type": ["string", "null"], "description": "From 'Requirements/Limits' column. Include FULL text: 'QL (120 per 30 days)', 'PA', 'PA; LA'. Null if none."},
                                        "category": {"type": ["string", "null"], "description": "Section header: 'Infections', 'HIV/AIDS', etc. Null if none."},
                                        "page_number": {"type": ["integer", "null"], "description": "The actual page number where this drug is found in the document."}
                                    }
                                }
                            },
                            "FormularyAbbreviations": {
                                "type": "array",
                                "description": "Abbreviation definitions found in legends/keys",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Acronym": {"type": "string", "description": "The abbreviation code (PA, QL, ST, etc.)"},
                                        "Expansion": {"type": "string", "description": "What it stands for (Prior Authorization, Quantity Limit, etc.)"},
                                        "Explanation": {"type": ["string", "null"], "description": "Additional explanation if any"}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            # Use native OCR endpoint with structured output - THIS IS FAST!
            ocr_response = mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document=DocumentURLChunk(document_url=signed_url.url),
                document_annotation_format=ocr_annotation_schema,
                include_image_base64=False
            )

            # Process OCR response - extract structured data
            original_pages = pages_to_extract if 'pages_to_extract' in locals() else []
            all_structured_data = []
            all_acronyms = []
            
            logger.info(f"OCR response has {len(ocr_response.pages)} pages")
            
            # Debug: Log the structure of the OCR response to understand where data is
            logger.info(f"OCR response attributes: {dir(ocr_response)}")
            if hasattr(ocr_response, 'document_annotation'):
                logger.info(f"document_annotation found: {ocr_response.document_annotation}")
            
            # Try to find structured data in multiple possible locations
            structured_json = None
            
            # Option 1: Check for document-level annotation
            if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
                try:
                    if isinstance(ocr_response.document_annotation, str):
                        structured_json = json.loads(ocr_response.document_annotation)
                    else:
                        structured_json = ocr_response.document_annotation
                    logger.info(f"Found structured data in document_annotation")
                except:
                    pass
            
            # Option 2: Check pages for markdown content and parse JSON from it
            if not structured_json:
                for page_idx, page in enumerate(ocr_response.pages):
                    # FIXED PAGE MAPPING: Map extracted page index to original page number
                    page_num = original_pages[page_idx] if page_idx < len(original_pages) else page_idx + 1
                    logger.debug(f"Non-chunked page_idx {page_idx} -> mapped to original page {page_num}")
                    logger.info(f"Page {page_num} attributes: {dir(page)}")
                    
                    # Check page.document_annotation
                    if hasattr(page, 'document_annotation') and page.document_annotation:
                        try:
                            if isinstance(page.document_annotation, str):
                                page_json = json.loads(page.document_annotation)
                            else:
                                page_json = page.document_annotation
                            logger.info(f"Found structured data in page.document_annotation")
                            
                            # Extract from page-level structured data
                            if isinstance(page_json, dict):
                                drugs_on_page = page_json.get("DrugInformation", [])
                                if drugs_on_page:
                                    logger.info(f"Page-level annotation: Found {len(drugs_on_page)} drugs on original page {page_num}")
                                for item in drugs_on_page:
                                    if isinstance(item, dict):
                                        all_structured_data.append({
                                            "drug_name": item.get("Drug Name"),
                                            "drug_tier": item.get("drug tier"),
                                            "drug_requirements": item.get("requirements"),
                                            "category": item.get("category"),
                                            "page_number": page_num
                                        })
                                for item in page_json.get("FormularyAbbreviations", []):
                                    if isinstance(item, dict):
                                        all_acronyms.append({
                                            "acronym": item.get("Acronym"),
                                            "expansion": item.get("Expansion"),
                                            "explanation": item.get("Explanation")
                                        })
                        except Exception as e:
                            logger.debug(f"Could not parse page.document_annotation: {e}")
                    
                    # Check page.annotations
                    if hasattr(page, 'annotations') and page.annotations:
                        logger.info(f"Page {page_num} has {len(page.annotations)} annotations")
                        for annotation in page.annotations:
                            logger.info(f"Annotation type: {type(annotation)}, attrs: {dir(annotation)}")
            
            # If we found document-level structured_json, process it
            if structured_json and isinstance(structured_json, dict):
                logger.info(f"Processing document-level structured JSON with keys: {structured_json.keys()}")
                for item in structured_json.get("DrugInformation", []):
                    if isinstance(item, dict):
                        # FIXED PAGE MAPPING: OCR's page_number is within the extracted document (1, 2, 3...)
                        # We need to map it back to original page numbers using original_pages
                        ocr_page_num = item.get("page_number")
                        if ocr_page_num and original_pages and 1 <= ocr_page_num <= len(original_pages):
                            # Map OCR page (1-based index within extracted doc) to original page number
                            actual_page = original_pages[ocr_page_num - 1]
                        elif original_pages:
                            # Fallback to first page of extracted doc
                            actual_page = original_pages[0]
                        else:
                            actual_page = ocr_page_num  # Last resort: use as-is
                        
                        logger.debug(f"Document-level: OCR page_num={ocr_page_num} -> mapped to original page {actual_page}")
                        
                        all_structured_data.append({
                            "drug_name": item.get("Drug Name"),
                            "drug_tier": item.get("drug tier"),
                            "drug_requirements": item.get("requirements"),
                            "category": item.get("category"),
                            "page_number": actual_page
                        })
                for item in structured_json.get("FormularyAbbreviations", []):
                    if isinstance(item, dict):
                        all_acronyms.append({
                            "acronym": item.get("Acronym"),
                            "expansion": item.get("Expansion"),
                            "explanation": item.get("Explanation")
                        })

            full_structured_data = {
                "drug_table": all_structured_data,
                "acronyms": all_acronyms,
                "tiers": []
            }

            # Track costs for native OCR
            total_costs['mistral_pages'] = len(ocr_response.pages)
            total_costs['mistral_cost'] = (total_costs['mistral_pages'] / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES

            logger.info(f"✅ [PRIMARY FLOW SUCCESS] Native OCR extraction completed for {filename}.")
            logger.info(f"📊 Extracted {len(all_structured_data)} drugs, {len(all_acronyms)} acronyms from {total_costs['mistral_pages']} page(s).")
            
            # Log page distribution for verification
            if all_structured_data:
                page_counts = {}
                for drug in all_structured_data:
                    p = drug.get('page_number')
                    page_counts[p] = page_counts.get(p, 0) + 1
                logger.info(f"📄 Page distribution: {dict(sorted(page_counts.items()))}")

            # Delete file and return
            try:
                mistral_client.files.delete(file_id=uploaded_file.id)
            except: pass
            
            return full_structured_data, "[NATIVE OCR EXTRACTION]", total_costs

        except Exception as primary_error:
            logger.warning(f"âš ï¸ [PRIMARY FLOW FAILED] Unified extraction failed for {filename}: {primary_error}")
            logger.info("--- [BACKUP FLOW] Falling back to traditional OCR + Per-Page LLM processing ---")

        # --------------------------------------------------------------------------
        # BACKUP FLOW: Original OCR + Per-Page LLM Loop
        # --------------------------------------------------------------------------
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=120)
        
        ocr_response = mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-2512",
            include_image_base64=False
        )

        page_count = len(ocr_response.pages)
        total_costs['mistral_pages'] = page_count
        total_costs['mistral_cost'] = (page_count / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES

        all_structured_data, all_acronyms, all_tiers = [], [], []

        # Since we already extracted only the needed pages, process ALL pages in this (smaller) document
        # Map extracted page index to original page number for tracking
        original_page_numbers = pages_to_extract if 'pages_to_extract' in locals() else list(range(1, page_count + 1))
        page_indices_to_process = list(range(page_count))  # Process all pages in the extracted doc

        if not page_indices_to_process:
            logger.warning(f"No valid pages selected for processing for file '{filename}'. Skipping LLM stage.")
        else:
            logger.info(f"[BACKUP FLOW] Processing {len(page_indices_to_process)} extracted pages in parallel with up to {LLM_PAGE_WORKERS} workers...")
            logger.info(f"Original page numbers being processed: {original_page_numbers}")
            
            with ThreadPoolExecutor(max_workers=LLM_PAGE_WORKERS) as executor:
                # Map extracted index to original page number
                future_to_page = {
                    executor.submit(extract_structured_data_with_llm, ocr_response.pages[idx].markdown, mistral_client, payer_name): (idx, original_page_numbers[idx] if idx < len(original_page_numbers) else idx + 1)
                    for idx in page_indices_to_process
                }

                for future in as_completed(future_to_page):
                    extracted_idx, original_page_num = future_to_page[future]
                    try:
                        structured_records, llm_costs = future.result()
                        logger.info(f"--- Completed processing for Original Page {original_page_num} (Extracted index {extracted_idx}) ---")
                        total_costs['bedrock_tokens'] += llm_costs.get('tokens', 0)
                        total_costs['bedrock_cost'] += llm_costs.get('cost', 0)
                        total_costs['bedrock_calls'] += llm_costs.get('calls', 0)
                        if structured_records:
                            extracted_count = len(structured_records.get('drug_table', []))
                            for drug in structured_records.get('drug_table', []):
                                if isinstance(drug, dict):
                                    drug['page_number'] = original_page_num  # Use original page number
                                else:
                                    logger.warning(f"Skipping non-dict item in drug_table: {drug}")
                                    continue
                                all_structured_data.append(drug)
                            if extracted_count > 0:
                                logger.debug(f"Assigned page_number={original_page_num} to {extracted_count} drugs from this page")
                            all_acronyms.extend(structured_records.get('acronyms', []))
                            all_tiers.extend(structured_records.get('tiers', []))
                    except Exception as exc:
                        logger.error(f"Page {original_page_num} generated an exception during result processing: {exc}")

        full_raw_content = "\n\n--- PAGE BREAK ---\n\n".join([p.markdown for p in ocr_response.pages])
        
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
        logger.error(f"A critical error occurred in the main PDF processing pipeline for {filename}: {e}")
        traceback.print_exc()
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



def process_single_pdf_url_worker(plan_info):
    """Worker: Download PDF from URL and process it entirely in-memory."""
    state_name, payer_name, plan_name, plan_id, payer_id, formulary_url, old_file_hash = plan_info
    log_prefix = f"[URL Worker for {plan_name}]"
    zero_costs = {'mistral_pages': 0, 'bedrock_tokens': 0, 'bedrock_cost': 0.0, 'bedrock_calls': 0}
    start_time = time.time()

    try:
        formulary_url = str(formulary_url).strip().replace('\u2026', '').replace('...', '')

        clean_url = str(formulary_url).strip().split(' ')[0]
        clean_url = clean_url.replace('\u2026', '').replace('...', '')

        formulary_url = transform_viewer_url(formulary_url)

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

        # proxy_user = os.getenv("PROXY_USER")
        # proxy_pass = os.getenv("PROXY_PASS")
        # proxy_host = os.getenv("PROXY_HOST")
        # proxy_port = os.getenv("PROXY_PORT")

        # proxies = None
        # if all([proxy_user, proxy_pass, proxy_host, proxy_port]):
        #     proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        #     proxies = {
        #         "http": proxy_url,
        #         "https": proxy_url,
        #     }
        #     logger.info(f"{log_prefix} Using authenticated proxy.")
        # else:
        #     logger.info(f"{log_prefix} Proxy environment variables not set. Attempting direct connection.")

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
            logger.warning(f"{log_prefix} SSL verification failed: {e}. Retrying with SSL verification DISABLED.")
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

        filename_for_config = f"{state_name}_{payer_name}_{plan_name}.pdf"

        if cached_data is None:
            logger.info(f"🔄 {log_prefix} Cache MISS for new hash. Starting full processing...")
            full_structured_data, raw_content, costs = process_pdf_with_mistral_ocr(
                pdf_bytes_io,
                payer_name,
                filename=filename_for_config
            )
            logger.info(f"🔍 {log_prefix} process_pdf_with_mistral_ocr returned: {len(full_structured_data.get('drug_table', []))} drugs")
            cache_result(new_file_hash, full_structured_data, raw_content)
        else:
            logger.info(f"💾 {log_prefix} Cache HIT for new hash. Using pre-processed data.")
            full_structured_data = cached_data
            # Log cached data details
            cached_drug_count = len(full_structured_data.get('drug_table', [])) if isinstance(full_structured_data, dict) else 0
            logger.info(f"🔍 {log_prefix} Cached data has {cached_drug_count} drugs")
            if cached_drug_count == 0:
                logger.warning(f"⚠️ {log_prefix} Cache has ZERO drugs! Consider clearing cache for this hash: {new_file_hash}")

        if not isinstance(full_structured_data, dict):
            logger.error(f"{log_prefix} Corrupted cache or processing error. Expected a dictionary, got {type(full_structured_data)}")
            full_structured_data = {"drug_table": [], "acronyms": [], "tiers": []}

        drug_table_data = full_structured_data.get('drug_table', [])
        
        logger.info(f"🔍 {log_prefix} RECEIVED from process_pdf_with_mistral_ocr: {len(drug_table_data)} records")

        if drug_table_data:
            page_numbers = [r.get('page_number') for r in drug_table_data if r.get('page_number')]
            if page_numbers:
                logger.info(f"{log_prefix} Initial extraction: {len(drug_table_data)} records from pages {min(page_numbers)}-{max(page_numbers)}")
                logger.info(f"{log_prefix} Page distribution: {dict(pd.Series(page_numbers).value_counts().sort_index())}")
            else:
                logger.warning(f"🔍 {log_prefix} NO PAGE NUMBERS in any of the {len(drug_table_data)} records!")
            
            # Log sample drugs
            logger.info(f"🔍 {log_prefix} Sample drugs received:")
            for i, drug in enumerate(drug_table_data[:5]):
                logger.info(f"   Drug {i+1}: '{drug.get('drug_name', '')[:50]}' | page={drug.get('page_number')} | tier={drug.get('drug_tier')}")

        # Check for index page but with detailed logging
        if drug_table_data:
            is_index = _is_extracted_data_from_index_page(drug_table_data)
            logger.info(f"🔍 {log_prefix} _is_extracted_data_from_index_page returned: {is_index}")
            if is_index:
                logger.warning(f"⚠️ {log_prefix} Detected index/TOC page based on extracted data patterns. Clearing {len(drug_table_data)} records!")
                drug_table_data = []

        if drug_table_data:
            logger.info(f"{log_prefix} Step 1: Running group propagation on {len(drug_table_data)} raw records.")
            drug_table_data = _clean_and_propagate_drug_groups(drug_table_data)
            logger.info(f"{log_prefix} After group propagation: {len(drug_table_data)} records.")

        if drug_table_data:
            logger.info(f"{log_prefix} Step 2: Consolidating fragments for {len(drug_table_data)} records.")
            drug_table_data = _consolidate_and_clean_drug_table(drug_table_data)
            logger.info(f"{log_prefix} After consolidation: {len(drug_table_data)} records.")

        all_acronyms = full_structured_data.get('acronyms', [])
        all_tiers = full_structured_data.get('tiers', [])

        structured_df = pd.DataFrame(drug_table_data)

        if not structured_df.empty:
            expanded_rows = []
            for _, row in structured_df.iterrows():
                drug_name_full = str(row.get('drug_name', ''))
                other_data = {k: v for k, v in row.items() if k != 'drug_name'}

                is_single_entry = False
                single_entry_keywords = ['kit', 'pak', 'pack', 'titration']
                if any(keyword in drug_name_full.lower() for keyword in single_entry_keywords):
                    is_single_entry = True

                if is_single_entry:
                    expanded_rows.append(row.to_dict())
                    continue

                parsed_drugs = parse_complex_drug_name(drug_name_full)

                if not parsed_drugs or (len(parsed_drugs) == 1 and not parsed_drugs[0]['strengths']):
                    expanded_rows.append(row.to_dict())
                    continue

                for parsed_drug in parsed_drugs:
                    base_name = parsed_drug['base_name']
                    brand_name_part = f" ({parsed_drug['brand_name']})" if parsed_drug['brand_name'] else ""

                    if parsed_drug['strengths']:
                        for strength in parsed_drug['strengths']:
                            new_drug_name = f"{base_name} {strength}{brand_name_part}".strip()
                            new_row = other_data.copy()
                            new_row["drug_name"] = new_drug_name
                            expanded_rows.append(new_row)
                    elif base_name:
                        new_drug_name = f"{base_name}{brand_name_part}".strip()
                        new_row = other_data.copy()
                        new_row["drug_name"] = new_drug_name
                        expanded_rows.append(new_row)

            structured_df = pd.DataFrame(expanded_rows)
            if expanded_rows:
                 logger.info(f"{log_prefix} After complex parsing, DataFrame has {len(structured_df)} rows.")

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

        def is_fully_english(item: dict) -> bool:
            """An inner helper function to check if all text fields in a dict are English."""
            acronym = item.get('acronym', '')
            expansion = item.get('expansion', '')
            explanation = item.get('explanation', '')
            
            # A record is valid ONLY IF the acronym, expansion, AND explanation are all English.
            return is_english(acronym) and is_english(expansion) and is_english(explanation)

        # Apply the new, stricter filter
        english_acronyms = [item for item in dedup_acronyms if is_fully_english(item)]
        english_tiers = [item for item in dedup_tiers if is_fully_english(item)]

        acronyms_removed = len(dedup_acronyms) - len(english_acronyms)
        tiers_removed = len(dedup_tiers) - len(english_tiers)

        if acronyms_removed > 0:
            logger.warning(f"Filtered out {acronyms_removed} non-English or mixed-language acronyms.")
        if tiers_removed > 0:
            logger.warning(f"Filtered out {tiers_removed} non-English or mixed-language tiers.")

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
        page_number_tracking = {}  # Track page numbers being assigned
        
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
            
            page_num = row.get('page_number', None)
            record = {
                "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
                "drug_name": cleaned_drug_name, "state_name": state_name, "coverage_status": coverage_status,
                "drug_tier": drug_tier_normalized, "drug_requirements": requirements_text or None,
                "page_number": page_num,
                "is_prior_authorization_required": "Yes" if detect_prior_authorization(requirements_text) else "No",
                "is_step_therapy_required": "Yes" if detect_step_therapy(requirements_text) else "No",
                "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
                "confidence_score": 0.95, "source_url": formulary_url,
                "plan_name": plan_name, "payer_name": payer_name,
                "file_name": f"{state_name}_{payer_name}_{plan_name}.pdf",
                "ndc_code": None, "jcode": None, "coverage_details": None,
            }
            processed_records.append(record)
            
            # Track page numbers for verification
            if page_num:
                page_number_tracking[page_num] = page_number_tracking.get(page_num, 0) + 1
        
        if page_number_tracking:
            logger.info(f"{log_prefix} Final records by page (will be saved to DB): {dict(sorted(page_number_tracking.items())[:10])}")

        if processed_records:
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"{log_prefix} Total processing time (worker start to DB push): {total_time:.2f} seconds.")
            return 'SUCCESS', plan_name, {"processed_records": processed_records, "db_payer_name": payer_name}, costs
        else:
            return 'SKIPPED', plan_name, "Data extracted, but no valid drug records were processed.", costs

    except requests.exceptions.ConnectionError as e:
        logger.error(f"{log_prefix} A network connection error occurred: {e}", exc_info=True)
        logger.error(f"{log_prefix} This is often caused by a firewall, a missing/incorrect proxy configuration, or the server being down.")
        logger.error(f"{log_prefix} Please check your network connection and ensure that if you are behind a proxy, the HTTP_PROXY and HTTPS_PROXY environment variables are set correctly.")
        return 'ERROR', plan_name, f"Connection Error: {e}", zero_costs
    except Exception as e:
        logger.error(f"{log_prefix} An unexpected error occurred in worker: {e}", exc_info=True)
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
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT, initializer=initialize_worker) as executor:
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

                        # FINAL FRAGMENT FILTER: Block any remaining orphan dosage rows before DB insertion
                        if deduplicated_data:
                            fragment_pattern = re.compile(r'^[/\d\(\)\.]+.*?(mg|ml|mcg|hr|%|tab|cap|gm|gram|unit)', re.IGNORECASE)
                            # Keywords often found in headers or non-drug text that should be blocked.
                            junk_keywords = {'index', 'contents', 'introduction', 'appendix', 'formulary', 'drug', 'tier', 'notes'}

                            validated_data = []
                            blocked_records = 0

                            for record in deduplicated_data:
                                drug_name = str(record.get('drug_name', '')).strip()
                                page_number = record.get('page_number')
                                drug_requirements = record.get('drug_requirements')

                                if page_number is not None and page_number <= 2 and not drug_requirements:
                                    if len(drug_name) < 5 or not re.search(r'[a-zA-Z]{3,}', drug_name):
                                        logger.warning(f"ðŸš« BLOCKED Page {page_number} record (missing requirements + suspicious name): '{drug_name}'")
                                        blocked_records += 1
                                        continue

                                if fragment_pattern.match(drug_name):
                                    logger.warning(f"ðŸš« BLOCKED FRAGMENT before DB insertion: '{drug_name}'")
                                    blocked_records += 1
                                    continue

                                name_parts = drug_name.split()
                                if not re.search(r'[a-zA-Z]{3,}', drug_name) or (len(name_parts) == 1 and name_parts[0].lower() in junk_keywords):
                                    logger.warning(f"ðŸš« BLOCKED INVALID NAME before DB insertion: '{drug_name}'")
                                    blocked_records += 1
                                    continue

                                validated_data.append(record)

                            if blocked_records > 0:
                                logger.warning(f"Blocked {blocked_records} invalid/junk records before DB insertion for '{plan_name_log}'.")

                            if validated_data:
                                logger.info(f"Inserting {len(validated_data)} validated records for plan '{plan_name_log}' into the database.")
                                insert_drug_formulary_data(validated_data)
                                plan_id = validated_data[0].get('plan_id')
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
                logger.error(f"CRITICAL: Processing timed out for plan: {plan_name_log} after {URL_PROCESSING_TIMEOUT} seconds.")
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