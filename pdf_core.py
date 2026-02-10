"""
pdf_core.py - Core PDF Processing and OCR Functions

This module contains stable core functions that rarely need modification:
- PDF file handling (enhancement, page extraction)
- Mistral OCR client management
- Rate limiting and retry logic
- Page filtering and pre-processing

Functions in this file:
- initialize_worker() - Worker initialization
- mistral_rate_limited_call() - Rate limiting decorator
- prefilter_pages_with_pymupdf() - Smart page pre-filtering
- process_single_chunk_parallel() - Parallel chunk processing
- enhance_pdf() - PDF quality enhancement
- create_resilient_mistral_client() - Mistral client creation
- _upload_pdf_to_mistral() - PDF upload with retry
- _extract_pages_from_pdf() - Page extraction
- _process_ocr_response() - OCR response processing
- _parse_page_ranges() - Page range parsing
- _get_pages_to_process() - Page selection logic
"""

import os
import re
import json
import logging
import time
import threading
import httpx
from io import BytesIO
from typing import List, Optional, Union
from pathlib import Path

from mistralai import Mistral
from mistralai.models import DocumentURLChunk

from config import (
    MISTRAL_API_KEY, MISTRAL_OCR_RATE_LIMIT, MAX_RETRIES, BACKOFF_MULTIPLIER,
    CLIENT_TIMEOUT, CONNECT_TIMEOUT, PDF_PAGE_PROCESSING_CONFIG,
    ENABLE_PAGE_PREFILTER, MIN_PAGE_TEXT_LENGTH, SKIP_INDEX_PAGES
)

# Import extraction functions
from pdf_extraction import (
    OCR_ANNOTATION_SCHEMA, _extract_drug_from_item, _extract_acronym_from_item,
    _build_requirements_from_item, filter_index_entries_from_mistral_response
)

logger = logging.getLogger(__name__)

# PDF Processing Constants
MAX_PDF_PAGES = 2000
ENHANCED_PDF_DPI = 200
USE_ENHANCED_PDF = False

# Pre-compiled regex patterns for index page detection (PERFORMANCE OPTIMIZATION)
# Pattern for dot leaders: matches ". . . ." or "......" or ". ." patterns
RE_DOT_LEADER = re.compile(r'(?:\.[\s\.]*){4,}')  # Requires 4+ dots to avoid false positives
# Pattern for page numbers at end of line
RE_PAGE_NUMBER_AT_END = re.compile(r'\s{3,}\d{1,3}\s*$')  # Requires 3+ spaces before page number
# Pattern for index entries: DRUG NAME followed by dots/spaces and page number
RE_INDEX_PATTERN = re.compile(r'^[A-Z][A-Za-z\s\-/,\(\)]+\.{3,}\s*\d{1,3}\s*$')  # Must have 3+ dots
# Pattern to detect if line has dosage info (real drugs have this, index entries don't)
RE_HAS_DOSAGE = re.compile(r'\d+\s*(mg|ml|mcg|g|%|unit|tablet|capsule|cap|tab|sol|cream)', re.IGNORECASE)

# Optional imports
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def initialize_worker():
    """An initializer function for each worker in the ProcessPoolExecutor."""
    pass


# Thread-safe rate limiter for Mistral OCR API (simplified - less lock contention)
_mistral_last_call_time = 0


def mistral_rate_limited_call(func):
    """
    A lightweight rate limiter for Mistral OCR calls.
    Mistral handles their own rate limiting, so we just add minimal delay.
    """
    def wrapper(*args, **kwargs):
        global _mistral_last_call_time
        elapsed = time.time() - _mistral_last_call_time
        if elapsed < MISTRAL_OCR_RATE_LIMIT:
            time.sleep(MISTRAL_OCR_RATE_LIMIT - elapsed)
        _mistral_last_call_time = time.time()
        return func(*args, **kwargs)
    return wrapper


# Shared Mistral client (PERFORMANCE OPTIMIZATION - reuse TCP connections)
_shared_mistral_client = None
_mistral_client_lock = threading.Lock()


def create_resilient_mistral_client():
    """
    Creates or returns a shared Mistral client with robust timeouts.
    Reuses the same client to avoid TCP connection overhead.
    """
    global _shared_mistral_client
    
    if _shared_mistral_client is None:
        with _mistral_client_lock:
            if _shared_mistral_client is None:  # Double-check locking
                timeout = httpx.Timeout(CLIENT_TIMEOUT, connect=CONNECT_TIMEOUT)
                transport = httpx.HTTPTransport(retries=MAX_RETRIES)
                client = httpx.Client(timeout=timeout, transport=transport)
                _shared_mistral_client = Mistral(api_key=MISTRAL_API_KEY, client=client)
                logger.info("✅ Created shared Mistral client (will be reused)")
    
    return _shared_mistral_client


def _upload_pdf_to_mistral(mistral_client, file_bytes: bytes, filename: str,
                           max_retries: int = 3, backoff_multiplier: float = 2.0):
    """
    Upload a PDF file to Mistral with retry logic.

    Args:
        mistral_client: Mistral client instance
        file_bytes: PDF content as bytes
        filename: Name for the uploaded file
        max_retries: Maximum number of retry attempts
        backoff_multiplier: Multiplier for exponential backoff

    Returns:
        Uploaded file object or None if failed
    """
    from mistralai.models.sdkerror import SDKError

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to upload '{filename}' to Mistral...")
            uploaded_file = mistral_client.files.upload(
                file={"file_name": filename, "content": file_bytes},
                purpose="ocr",
            )
            logger.info("File uploaded successfully to Mistral.")
            return uploaded_file
        except (SDKError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                delay = backoff_multiplier ** attempt
                logger.warning(f"Network or Server error during upload: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to upload file to Mistral after {max_retries} attempts.")
                raise
    return None


def _extract_pages_from_pdf(pdf_input, pages_to_extract: list):
    """
    Extract specific pages from a PDF using PyMuPDF.

    Args:
        pdf_input: PDF file path or BytesIO object
        pages_to_extract: List of 1-based page numbers to extract

    Returns:
        BytesIO containing only the extracted pages, or None if extraction fails
    """
    if not PYMUPDF_AVAILABLE or not pages_to_extract:
        return None

    try:
        if isinstance(pdf_input, BytesIO):
            pdf_input.seek(0)
            src_doc = fitz.open(stream=pdf_input.getvalue(), filetype="pdf")
        else:
            src_doc = fitz.open(str(pdf_input))

        extracted_doc = fitz.open()
        for page_num in pages_to_extract:
            if 1 <= page_num <= len(src_doc):
                extracted_doc.insert_pdf(src_doc, from_page=page_num-1, to_page=page_num-1)

        extracted_bytes = extracted_doc.tobytes()
        extracted_doc.close()
        src_doc.close()

        logger.info(f"✅ Extracted {len(pages_to_extract)} pages ({len(extracted_bytes)/1024:.1f} KB)")
        return BytesIO(extracted_bytes)

    except Exception as e:
        logger.warning(f"⚠️ Page extraction failed: {e}")
        return None


def _process_ocr_response(ocr_response, original_pages: list) -> tuple:
    """
    Process OCR response and extract structured data.

    Args:
        ocr_response: Mistral OCR response object
        original_pages: List of original PDF page numbers

    Returns:
        Tuple of (all_structured_data, all_acronyms, pages_processed)
    """
    all_structured_data = []
    all_acronyms = []

    # Try document-level annotation first
    if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
        try:
            doc_json = ocr_response.document_annotation
            if isinstance(doc_json, str):
                doc_json = json.loads(doc_json)

            if isinstance(doc_json, dict):
                # Extract PageHeaders to check if this is a valid drug page
                # If headers exist, we assume it's a valid page and bypass index filtering
                # Extract PageHeaders to check if this is a valid drug page
                # If headers exist, we check their content to distinguish valid drug lists from index pages
                page_headers = doc_json.get("PageHeaders", [])
                
                # Check Header Content
                header_str = " ".join([str(h).lower() for h in page_headers])
                is_index_header = "page" in header_str or "index" in header_str
                is_drug_list_header = "limit" in header_str or "requir" in header_str or "tier" in header_str or "note" in header_str or "drug" in header_str
                
                # Bypass ONLY if it looks like a drug list AND NOT an index page
                # This ensures we blindly add valid 2-column formats (Drug Name + Requirements)
                # But still filter index pages (Drug Name + Page Number)
                bypass_index_check = (is_drug_list_header and not is_index_header) or (page_headers and not is_index_header)
                
                if bypass_index_check:
                    logger.info(f"✅ Valid headers detected: {page_headers}. Bypassing index check for this page.")
                
                # FILTER OUT INDEX ENTRIES before processing
                drug_info_list = doc_json.get("DrugInformation", [])
                
                # Inject bypass flag if applicable
                if bypass_index_check:
                    for item in drug_info_list:
                        item["_bypass_index_check"] = True
                
                # ✅ BACKFILL TIER FROM BrandOrGeneric (Moved BEFORE filter)
                # Some templates put "GENERIC" (or page nums) in BrandOrGeneric column but leave Tier empty.
                # Must do this BEFORE filtering so we can catch "66, 77" page numbers.
                for item in drug_info_list:
                    if not item.get("drug tier") and item.get("BrandOrGeneric"):
                        item["drug tier"] = item.get("BrandOrGeneric")

                # Still call filter (it will respect the flag if set, or proceed normally if not)
                drug_info_list = filter_index_entries_from_mistral_response(drug_info_list)
                
                for item in drug_info_list:
                    if isinstance(item, dict):
                        ocr_page = item.get("page_number")
                        if ocr_page and original_pages and 1 <= ocr_page <= len(original_pages):
                            actual_page = original_pages[ocr_page - 1]
                        elif original_pages:
                            actual_page = original_pages[0]
                        else:
                            actual_page = ocr_page or 1
                        all_structured_data.append(_extract_drug_from_item(item, actual_page))

                for item in doc_json.get("FormularyAbbreviations", []):
                    if isinstance(item, dict):
                        all_acronyms.append(_extract_acronym_from_item(item))
        except Exception as e:
            logger.debug(f"Could not parse document_annotation: {e}")

    # Also check page-level annotations
    for page_idx, page in enumerate(ocr_response.pages):
        page_num = original_pages[page_idx] if page_idx < len(original_pages) else page_idx + 1

        if hasattr(page, 'document_annotation') and page.document_annotation:
            try:
                page_json = page.document_annotation
                if isinstance(page_json, str):
                    page_json = json.loads(page_json)

                if isinstance(page_json, dict):
                    # Extract PageHeaders to check if this is a valid drug page
                    page_headers = page_json.get("PageHeaders", [])
                    
                    # Check Header Content
                    header_str = " ".join([str(h).lower() for h in page_headers])
                    is_index_header = "page" in header_str or "index" in header_str
                    is_drug_list_header = "limit" in header_str or "requir" in header_str or "tier" in header_str or "note" in header_str or "drug" in header_str
                    
                    # Bypass ONLY if it looks like a drug list AND NOT an index page
                    bypass_index_check = (is_drug_list_header and not is_index_header) or (page_headers and not is_index_header)
                    
                    if bypass_index_check:
                        logger.info(f"✅ Valid page headers detected: {page_headers}. Bypassing index check.")

                    # FILTER OUT INDEX ENTRIES before processing
                    drug_info_list = page_json.get("DrugInformation", [])

                    # Inject bypass flag if applicable
                    if bypass_index_check:
                        for item in drug_info_list:
                            item["_bypass_index_check"] = True
                    drug_info_list = filter_index_entries_from_mistral_response(drug_info_list)
                    
                    for item in drug_info_list:
                        if isinstance(item, dict):
                            all_structured_data.append(_extract_drug_from_item(item, page_num))

                    for item in page_json.get("FormularyAbbreviations", []):
                        if isinstance(item, dict):
                            all_acronyms.append(_extract_acronym_from_item(item))
            except Exception as e:
                logger.debug(f"Could not parse page annotation: {e}")

    # POST-PROCESSING: Auto-generate LD (Limited Distribution) acronym if detected
    # Check if any drugs have Limited Distribution = true
    has_ld_flag = any(
        item.get("Limited Distribution") == True
        for item in all_structured_data if isinstance(item, dict)
    )
    
    if has_ld_flag:
        # Check if LD acronym already exists
        has_ld_acronym = any(
            (str(acr.get("acronym", "")).upper() in ["LD", "LIMITED DISTRIBUTION"])
            for acr in all_acronyms if isinstance(acr, dict)
        )
        
        if not has_ld_acronym:
            # Auto-add LD acronym entry
            all_acronyms.append({
                "acronym": "LD",
                "full_text": "Limited Distribution",
                "explanation": "Limited Distribution"
            })
            logger.info("✅ Auto-added LD (Limited Distribution) acronym")

    logger.info(f"📋 OCR extracted {len(all_structured_data)} drugs and {len(all_acronyms)} acronyms (raw, before cleaning)")
    return all_structured_data, all_acronyms, len(ocr_response.pages)


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

            # Skip pages with very little text
            if len(text) < MIN_PAGE_TEXT_LENGTH:
                skipped_pages.append((page_num, "too little text"))
                continue

            # Enhanced index/TOC page detection
            if SKIP_INDEX_PAGES:
                is_index = False
                index_reason = ""
                
                # Check 1: Explicit index indicators in header
                header_text = text_lower[:800]  # Check more of the header
                footer_text = text_lower[-300:] if len(text_lower) > 300 else text_lower
                
                # NEW: Skip index detection if page contains tier definitions
                # These pages have valuable content that should be processed
                tier_definition_indicators = [
                    "tier 1", "tier 2", "tier 3", "tier 4", "tier 5", "tier 6",
                    "preferred generic", "preferred brand", "non-preferred",
                    "generic drugs", "brand drugs", "specialty tier",
                    "this tier includes", "drugs in tier", "cost-sharing"
                ]
                has_tier_definitions = any(ind in text_lower for ind in tier_definition_indicators)
                
                if has_tier_definitions:
                    # This page has tier definitions - DO NOT skip it
                    logger.info(f"   ✅ Page {page_num}: Contains tier definitions - keeping for OCR")
                    filtered_pages.append(page_num)
                    continue
                
                index_indicators = [
                    "table of contents", "alphabetical index", "drug index",
                    "index of drugs", "formulary index", "index to drugs",
                    "medication index", "generic index", "brand index",
                    "attention fhk providers",
                    "alphabetical listing of drugs",  # NEW: From uploaded image
                    "listing of drugs", "drug listing",
                    "formulary listing", "medication listing"
                ]
                
                # Also check footer for "last updated date" pattern (common in index pages)
                footer_indicators = []
                for indicator in index_indicators:
                    if indicator in header_text:
                        is_index = True
                        index_reason = f"header contains '{indicator}'"
                        break
                
                # Check footer for index page indicators
                if not is_index:
                    for indicator in footer_indicators:
                        if indicator in footer_text:
                            is_index = True
                            index_reason = f"footer contains '{indicator}'"
                            break
                
                if not is_index:
                    lines = text.split('\n')
                    total_content_lines = 0
                    dot_leader_lines = 0
                    page_number_at_end_lines = 0
                    no_dosage_lines = 0
                    
                    for line in lines:
                        stripped = line.strip()
                        if len(stripped) < 5:  # Skip very short lines
                            continue
                        
                        total_content_lines += 1
                        
                        # Check 2: Dot leaders (. . . . or ......) pattern
                        if RE_DOT_LEADER.search(stripped):
                            dot_leader_lines += 1
                        
                        # Check 3: Page number at end pattern
                        if RE_PAGE_NUMBER_AT_END.search(stripped):
                            page_number_at_end_lines += 1
                        
                        # Check 4: No dosage info (index entries are just drug names)
                        if not RE_HAS_DOSAGE.search(stripped) and len(stripped) > 10:
                            no_dosage_lines += 1
                    
                    if total_content_lines > 10:
                        # Rule 1: If >30% of lines have dot leaders, it's definitely an index
                        # (increased from 20% to reduce false positives)
                        if dot_leader_lines / total_content_lines >= 0.30:
                            is_index = True
                            index_reason = f"{dot_leader_lines}/{total_content_lines} lines have dot leaders"
                        
                        # Rule 2: If >50% of lines end with page numbers, likely index
                        # (increased from 40% to reduce false positives)
                        elif page_number_at_end_lines / total_content_lines >= 0.50:
                            is_index = True
                            index_reason = f"{page_number_at_end_lines}/{total_content_lines} lines end with page numbers"
                        
                        # Rule 3: If >98% have no dosage info AND >40% end with page numbers
                        # (made stricter to avoid false positives on drug pages)
                        elif no_dosage_lines / total_content_lines >= 0.98 and page_number_at_end_lines / total_content_lines >= 0.40:
                            is_index = True
                            index_reason = f"No dosage info ({no_dosage_lines}/{total_content_lines}) + page numbers ({page_number_at_end_lines})"
                        
                        # Rule 4: Check for index pattern directly (stricter threshold)
                        elif not is_index:
                            toc_pattern_count = sum(
                                1 for line in lines
                                if RE_INDEX_PATTERN.match(line.strip())
                            )
                            if toc_pattern_count / total_content_lines >= 0.35:
                                is_index = True
                                index_reason = f"{toc_pattern_count}/{total_content_lines} lines match index pattern"

                if is_index:
                    skipped_pages.append((page_num, f"index/TOC page: {index_reason}"))
                    logger.debug(f"🚫 Skipping page {page_num}: {index_reason}")
                    continue

            filtered_pages.append(page_num)

        src_doc.close()

        if skipped_pages:
            logger.info(f"📄 [PRE-FILTER] Skipped {len(skipped_pages)} index/empty pages:")
            for pg, reason in skipped_pages[:5]:  # Show first 5
                logger.info(f"   📄 Page {pg}: {reason}")
            if len(skipped_pages) > 5:
                logger.info(f"   ... and {len(skipped_pages) - 5} more")
        logger.info(f"📄 [PRE-FILTER] Kept {len(filtered_pages)} of {len(page_indices)} pages for OCR")

    except Exception as e:
        logger.warning(f"⚠️ Page pre-filtering failed: {e}. Using original page list.")
        return page_indices

    return filtered_pages if filtered_pages else page_indices


def enhance_pdf(pdf_input, dpi=ENHANCED_PDF_DPI):
    """
    Enhance a PDF by converting it to high-resolution images and
    reconstructing it as a new high-quality PDF.

    This improves OCR accuracy, especially for tables, by:
    1. Rendering each page at high DPI (300)
    2. Creating a new PDF with these high-quality renders

    Args:
        pdf_input: File path (str/Path) or BytesIO object
        dpi: Resolution for enhancement (default 200)

    Returns:
        BytesIO object containing the enhanced PDF, or None if enhancement fails
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF (fitz) not available. Cannot enhance PDF.")
        return None

    logger.info(f"Enhancing PDF at {dpi} DPI for better OCR quality...")

    try:
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

        enhanced_doc = fitz.open()
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(page_count):
            src_page = src_doc[page_num]
            pix = src_page.get_pixmap(matrix=matrix, alpha=False)
            page_rect = src_page.rect
            new_page = enhanced_doc.new_page(width=page_rect.width, height=page_rect.height)
            new_page.insert_image(page_rect, pixmap=pix)

            if (page_num + 1) % 10 == 0 or page_num == page_count - 1:
                logger.info(f"Enhanced page {page_num + 1}/{page_count}")

        src_doc.close()

        enhanced_bytes = BytesIO()
        enhanced_doc.save(enhanced_bytes, garbage=4, deflate=True)
        enhanced_doc.close()

        enhanced_bytes.seek(0)
        enhanced_size_mb = len(enhanced_bytes.getvalue()) / (1024 * 1024)
        logger.info(f"Successfully created enhanced PDF: {page_count} pages, {enhanced_size_mb:.2f} MB")

        return enhanced_bytes

    except Exception as e:
        logger.error(f"Failed to enhance PDF: {e}")
        return None


def _parse_page_ranges(page_config_value: Union[str, list, None]) -> List[int]:
    """
    Parses a flexible page range configuration into a flat list of page numbers.
    Handles "all", lists of numbers, and lists of strings with ranges (e.g., "10-20").
    """
    if not page_config_value:
        return []

    pages = set()
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
    selected_rule = "all"
    rule_source = "system default"

    if filename:
        for key, pages_rule in config.items():
            if key != "default" and key.lower() in filename.lower():
                selected_rule = pages_rule
                rule_source = f"specific rule for key '{key}'"
                break

    if rule_source == "system default" and "default" in config:
        selected_rule = config["default"]
        rule_source = "configuration default"

    logger.info(f"Applying page processing rule for '{filename}' from {rule_source}: {selected_rule}")

    if isinstance(selected_rule, str) and selected_rule.lower() == "all":
        logger.info(f"Processing all {total_pages} pages for '{filename}'.")
        return list(range(total_pages))

    page_numbers_1_based = _parse_page_ranges(selected_rule)

    if not page_numbers_1_based:
        logger.warning(f"No valid pages specified by rule '{selected_rule}' for '{filename}'.")
        return []

    valid_pages_1_based = [p for p in page_numbers_1_based if 1 <= p <= total_pages]
    invalid_pages = [p for p in page_numbers_1_based if p not in valid_pages_1_based]

    if invalid_pages:
        logger.warning(f"Ignoring invalid/out-of-range pages for '{filename}': {invalid_pages}")

    page_indices_0_based = [p - 1 for p in valid_pages_1_based]
    logger.info(f"Final list of pages to process for '{filename}': {[p + 1 for p in page_indices_0_based]}")

    return sorted(list(set(page_indices_0_based)))
