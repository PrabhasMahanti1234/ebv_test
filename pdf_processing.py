"""
pdf_processing.py - Main PDF Processing Pipeline

This is the main entry point module that orchestrates the PDF processing pipeline.
"""

import os
import re
import json
import logging
import time
import traceback
import requests
import httpx
import concurrent.futures
from io import BytesIO
from typing import Optional, List, Tuple

from mistralai import Mistral
from mistralai.models import DocumentURLChunk

from config import (
    MISTRAL_API_KEY, MAX_PAGES_PER_OCR_REQUEST, MAX_OCR_WORKERS,
    ENABLE_PAGE_PREFILTER, SKIP_INDEX_PAGES
)

# Import from extraction module (frequently changed functions)
from pdf_extraction import (
    OCR_ANNOTATION_SCHEMA,
    _build_requirements_from_item,
    _extract_drug_from_item,
    _extract_acronym_from_item,
    filter_index_entries_from_mistral_response,
    is_index_content,
    robust_json_repair,
    _is_extracted_data_from_index_page,
    _consolidate_and_clean_drug_table,
    _clean_and_propagate_drug_groups,
    _sanitize_output,
    extract_metadata_from_filename,
    is_index_page,
    _parse_and_split_tier_definitions,
    _reclassify_definitions,
    is_valid_formulary_definition
)

# Import from core module (stable functions)
from pdf_core import (
    MAX_PDF_PAGES,
    ENHANCED_PDF_DPI,
    USE_ENHANCED_PDF,
    PYMUPDF_AVAILABLE,
    create_resilient_mistral_client,
    _upload_pdf_to_mistral,
    _extract_pages_from_pdf,
    _process_ocr_response,
    prefilter_pages_with_pymupdf,
    enhance_pdf,
    mistral_rate_limited_call,
    _parse_page_ranges,
    _get_pages_to_process
)

# Import database and utility functions
from database import (
    get_db_connection, batch_determine_coverage_status,
    get_cached_result, cache_result, update_plan_file_hash,
    insert_acronyms_to_ref_table, insert_drug_formulary_data,
    delete_drug_formulary_records_for_plan
)

from utils import (
    similarity, clean_drug_name, detect_prior_authorization,
    detect_step_therapy, calculate_file_hash, track_mistral_cost,
    determine_coverage_status, normalize_drug_tier, infer_drug_tier_from_text,
    calculate_bytes_hash, parse_complex_drug_name, normalize_requirement_code,
    transform_viewer_url
)

logger = logging.getLogger(__name__)

# Optional imports
try:
    import fitz
except ImportError:
    fitz = None

try:
    from langdetect import detect as detect_language
    LANGDETECT_AVAILABLE = False
except ImportError:
    LANGDETECT_AVAILABLE = False
# LANGDETECT_AVAILABLE = False


def process_single_chunk_parallel(chunk_info: dict) -> dict:
    """
    OPTIMIZATION 1: Process a single chunk of pages for parallel execution.
    """
    chunk_idx = chunk_info['chunk_idx']
    chunk_pages = chunk_info['chunk_pages']  # Sequential pages in extracted PDF (1,2,3,4)
    original_pages = chunk_info.get('original_pages', chunk_pages)  # Original PDF page numbers (270,271,272,273)
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
        mistral_client = create_resilient_mistral_client()

        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunk_doc = fitz.open()

        for page_num in chunk_pages:
            if 1 <= page_num <= len(src_doc):
                chunk_doc.insert_pdf(src_doc, from_page=page_num-1, to_page=page_num-1)

        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()
        src_doc.close()

        @mistral_rate_limited_call
        def upload_chunk():
            return mistral_client.files.upload(
                file={"file_name": f"chunk_{chunk_idx}.pdf", "content": chunk_bytes},
                purpose="ocr"
            )

        chunk_uploaded = upload_chunk()
        chunk_signed_url = mistral_client.files.get_signed_url(file_id=chunk_uploaded.id, expiry=300)

        max_retries = 2  # Reduced from 3
        retry_delay = 1  # Start at 1s instead of 2s
        ocr_response = None

        for attempt in range(max_retries):
            try:
                ocr_response = mistral_client.ocr.process(
                    model="mistral-ocr-latest",
                    document=DocumentURLChunk(document_url=chunk_signed_url.url),
                    document_annotation_format=ocr_schema,
                    include_image_base64=False
                )
                break
            except Exception as e:
                error_str = str(e)
                # Only retry on server errors (5xx), not client errors (4xx)
                if any(code in error_str for code in ["500", "502", "503", "504"]):
                    if attempt < max_retries - 1:
                        logger.warning(f"Chunk {chunk_idx + 1}: Server error, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
                else:
                    raise

        if ocr_response is None:
            result['error'] = "OCR API failed after retries"
            return result

        result['pages_processed'] = len(ocr_response.pages)

        if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
            chunk_json = ocr_response.document_annotation
            if isinstance(chunk_json, str):
                try:
                    chunk_json = json.loads(chunk_json)
                except json.JSONDecodeError as e:
                    logger.debug(f"Chunk {chunk_idx + 1}: JSON decode error, attempting repair...")
                    # Use robust_json_repair to fix malformed JSON
                    repaired = robust_json_repair(chunk_json)
                    if repaired.get("drug_table") or repaired.get("acronyms"):
                        chunk_json = {
                            "DrugInformation": repaired.get("drug_table", []),
                            "FormularyAbbreviations": repaired.get("acronyms", [])
                        }
                        logger.debug(f"Chunk {chunk_idx + 1}: JSON repaired")
                    else:
                        logger.error(f"Chunk {chunk_idx + 1}: JSON repair failed")
                        chunk_json = {"DrugInformation": [], "FormularyAbbreviations": []}

            drug_info_list = chunk_json.get("DrugInformation", [])
            
            # FILTER OUT INDEX ENTRIES (entries with page_number but no tier/requirements)
            drug_info_list = filter_index_entries_from_mistral_response(drug_info_list)

            for drug_idx, item in enumerate(drug_info_list):
                if isinstance(item, dict):
                    drug_name = item.get("Drug Name", "")
                    dosage_form = item.get("Dosage Form/Strength", "")
                    
                    # Combine drug name and dosage form/strength
                    if drug_name and dosage_form:
                        drug_name = f"{drug_name.strip()} {dosage_form.strip()}"
                    
                    if not drug_name or len(drug_name) < 2:
                        continue

                    # Extract tier - check multiple possible field names
                    drug_tier = (item.get("Tier") or 
                                item.get("drug tier") or 
                                item.get("drug_tier") or 
                                item.get("Tier Designation"))
                    
                    # Extract requirements - check multiple possible field names
                    drug_requirements = (item.get("Requirements") or 
                                        item.get("requirements") or 
                                        item.get("drug_requirements") or
                                        _build_requirements_from_item(item))

                    ocr_page_num = item.get("page_number")
                    # Map OCR page number (1,2,3,4 in chunk) to original PDF page (270,271,272,273)
                    if ocr_page_num and isinstance(ocr_page_num, int) and 1 <= ocr_page_num <= len(original_pages):
                        actual_pdf_page = original_pages[ocr_page_num - 1]
                    elif ocr_page_num and isinstance(ocr_page_num, int) and ocr_page_num in original_pages:
                        actual_pdf_page = ocr_page_num
                    else:
                        # Estimate page based on position in drug list
                        if len(drug_info_list) > 0 and len(original_pages) > 0:
                            position_ratio = drug_idx / len(drug_info_list)
                            page_index = min(int(position_ratio * len(original_pages)), len(original_pages) - 1)
                            actual_pdf_page = original_pages[page_index]
                        else:
                            actual_pdf_page = original_pages[0] if original_pages else 1

                    result['drugs'].append({
                        "drug_name": drug_name,
                        "drug_tier": drug_tier,
                        "drug_requirements": drug_requirements,
                        "category": item.get("category"),
                        "page_number": actual_pdf_page
                    })

            for item in chunk_json.get("FormularyAbbreviations", []):
                if isinstance(item, dict):
                    result['acronyms'].append(_extract_acronym_from_item(item))
        else:
            logger.warning(f"Chunk {chunk_idx + 1}: No document_annotation found")

        try:
            mistral_client.files.delete(file_id=chunk_uploaded.id)
        except:
            pass

        # NOTE: Chunk-level index detection DISABLED - causes valid drugs to be lost
        # when chunks contain mixed index + drug pages. Individual index entries
        # are filtered out later in _consolidate_and_clean_drug_table() instead.
        # if result['drugs'] and _is_extracted_data_from_index_page(result['drugs']):
        #     logger.warning(f"⚠️ Chunk {chunk_idx + 1}: Detected INDEX PAGE data, discarding {len(result['drugs'])} entries")
        #     result['drugs'] = []
        #     result['acronyms'] = []
        if result['drugs']:
            logger.info(f"Chunk {chunk_idx + 1} complete: {len(result['drugs'])} drugs")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Chunk {chunk_idx + 1} failed: {e}")

    return result


def process_pdf_with_mistral_ocr(pdf_input, payer_name=None, filename: Optional[str] = None):
    """Processes a PDF using Mistral OCR and a parallelized LLM pipeline."""
    try:
        if isinstance(pdf_input, BytesIO):
            pdf_input.seek(0)
            pdf_bytes = pdf_input.getvalue()
        else:
            with open(pdf_input, 'rb') as f:
                pdf_bytes = f.read()

        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(src_doc)
        src_doc.close()

        if total_pages > MAX_PDF_PAGES:
            logger.warning(f"PDF has {total_pages} pages, exceeds limit")
            return {"drug_table": [], "acronyms": [], "tiers": []}, "PDF_TOO_LARGE", {}

        page_indices_0_based = _get_pages_to_process(filename, total_pages)
        if not page_indices_0_based:
            page_indices_0_based = list(range(total_pages))

        pages_to_process = [p + 1 for p in page_indices_0_based]
        original_page_numbers = pages_to_process.copy()  # Keep original page numbers for metadata

        if ENABLE_PAGE_PREFILTER:
            pages_to_process = prefilter_pages_with_pymupdf(BytesIO(pdf_bytes), pages_to_process)
            original_page_numbers = pages_to_process.copy()
            
            # CRITICAL FIX: If pre-filter removed ALL pages, stop processing
            if not pages_to_process:
                logger.warning(f"⚠️ All pages were filtered out during pre-processing. Returning empty results.")
                return {
                    "drug_table": [],
                    "acronyms": [],
                    "tiers": []
                }, "[PRE-FILTER: ALL PAGES SKIPPED]", {}

        # Store original PDF bytes for chunk processing
        original_pdf_bytes = pdf_bytes
        
        if len(pages_to_process) < total_pages:
            extracted_pdf = _extract_pages_from_pdf(BytesIO(pdf_bytes), pages_to_process)
            if extracted_pdf:
                pdf_bytes = extracted_pdf.getvalue()
                src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages_in_upload = len(src_doc)
                src_doc.close()
                # CRITICAL FIX: After extraction, the new PDF has pages 1 to num_pages_in_upload
                # We need to use sequential page numbers for the extracted PDF
                logger.info(f"📄 Extracted {num_pages_in_upload} pages from original PDF (pages {pages_to_process[0]}-{pages_to_process[-1]})")
            else:
                num_pages_in_upload = total_pages
        else:
            num_pages_in_upload = total_pages

        mistral_client = create_resilient_mistral_client()
        total_costs = track_mistral_cost(payer_name, num_pages_in_upload)

        # PERFORMANCE FIX: Only upload full PDF if NOT using chunked processing
        # When chunking, each chunk uploads separately, so initial upload is wasted
        if num_pages_in_upload > MAX_PAGES_PER_OCR_REQUEST:
            # Build chunks using SEQUENTIAL page numbers (1 to num_pages_in_upload)
            # because we're working with the extracted PDF, not the original
            sequential_pages = list(range(1, num_pages_in_upload + 1))
            
            chunks = []
            for i in range(0, len(sequential_pages), MAX_PAGES_PER_OCR_REQUEST):
                chunk_pages = sequential_pages[i:i + MAX_PAGES_PER_OCR_REQUEST]
                # Also store the original page numbers for metadata mapping
                original_chunk_pages = original_page_numbers[i:i + MAX_PAGES_PER_OCR_REQUEST] if i < len(original_page_numbers) else chunk_pages
                chunks.append({
                    'chunk_idx': len(chunks),
                    'chunk_pages': chunk_pages,  # Sequential pages in extracted PDF
                    'original_pages': original_chunk_pages,  # Original PDF page numbers
                    'pdf_bytes': pdf_bytes,  # The extracted PDF
                    'ocr_schema': OCR_ANNOTATION_SCHEMA
                })

            total_chunks = len(chunks)
            logger.info("=" * 70)
            logger.info(f"🚀 PARALLEL CHUNKED PROCESSING STARTED")
            logger.info(f"   📄 Total pages to process: {num_pages_in_upload}")
            logger.info(f"   📦 Total chunks created: {total_chunks}")
            logger.info(f"   📊 Pages per chunk: {MAX_PAGES_PER_OCR_REQUEST}")
            logger.info(f"   👷 Max parallel workers: {MAX_OCR_WORKERS}")
            logger.info("=" * 70)

            all_drugs = []
            all_acronyms = []
            
            # Track chunk results
            chunks_completed = 0
            chunks_failed = 0
            chunks_with_data = 0
            total_drugs_extracted = 0
            failed_chunk_ids = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_single_chunk_parallel, chunk): chunk for chunk in chunks}

                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    chunk_idx = chunk['chunk_idx']
                    chunk_pages = chunk['chunk_pages']
                    original_pages = chunk.get('original_pages', chunk_pages)
                    
                    try:
                        result = future.result()
                        chunks_completed += 1
                        
                        if result.get('error'):
                            chunks_failed += 1
                            failed_chunk_ids.append(chunk_idx + 1)
                            logger.warning(f"❌ Chunk {chunk_idx + 1}/{total_chunks} FAILED: {result['error']}")
                            logger.warning(f"   Chunk pages: {chunk_pages[0]}-{chunk_pages[-1]} (original: {original_pages[0]}-{original_pages[-1]})")
                        else:
                            drugs_in_chunk = len(result.get('drugs', []))
                            acronyms_in_chunk = len(result.get('acronyms', []))
                            total_drugs_extracted += drugs_in_chunk
                            
                            if drugs_in_chunk > 0:
                                chunks_with_data += 1
                            
                            all_drugs.extend(result.get('drugs', []))
                            all_acronyms.extend(result.get('acronyms', []))
                            
                            logger.info(f"✅ Chunk {chunk_idx + 1}/{total_chunks} completed: "
                                       f"{drugs_in_chunk} drugs, {acronyms_in_chunk} acronyms | "
                                       f"Pages: {original_pages[0]}-{original_pages[-1]}")
                        
                        # Progress update every few chunks
                        if chunks_completed % 5 == 0 or chunks_completed == total_chunks:
                            pct = (chunks_completed / total_chunks) * 100
                            logger.info(f"📊 Progress: {chunks_completed}/{total_chunks} chunks ({pct:.1f}%) | "
                                       f"Drugs so far: {total_drugs_extracted}")
                            
                    except Exception as e:
                        chunks_completed += 1
                        chunks_failed += 1
                        failed_chunk_ids.append(chunk_idx + 1)
                        logger.error(f"❌ Chunk {chunk_idx + 1}/{total_chunks} EXCEPTION: {e}")

            # Final summary
            logger.info("=" * 70)
            logger.info(f"🏁 CHUNKED PROCESSING COMPLETE - SUMMARY")
            logger.info(f"   📦 Total chunks: {total_chunks}")
            logger.info(f"   ✅ Successful chunks: {chunks_completed - chunks_failed}")
            logger.info(f"   ❌ Failed chunks: {chunks_failed}")
            logger.info(f"   📊 Chunks with data: {chunks_with_data}")
            logger.info(f"   💊 Total drugs extracted (before cleaning): {total_drugs_extracted}")
            
            if failed_chunk_ids:
                logger.warning(f"   ⚠️ Failed chunk IDs: {failed_chunk_ids}")
            
            # Clean and consolidate
            all_drugs = _consolidate_and_clean_drug_table(all_drugs)
            
            logger.info(f"   💊 Total drugs after cleaning: {len(all_drugs)}")
            logger.info(f"   📝 Total acronyms: {len(all_acronyms)}")
            logger.info("=" * 70)

            return {
                "drug_table": all_drugs,
                "acronyms": all_acronyms,
                "tiers": []
            }, "[PARALLEL CHUNKED OCR EXTRACTION]", total_costs

        else:
            # Non-chunked path: Upload full PDF and process in single request
            logger.info(f"📄 Processing {num_pages_in_upload} pages in single OCR request (no chunking needed)")
            
            uploaded_file = _upload_pdf_to_mistral(mistral_client, pdf_bytes, filename or "formulary.pdf")
            if not uploaded_file:
                return {"drug_table": [], "acronyms": [], "tiers": []}, "UPLOAD_FAILED", {}

            signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=300)

            ocr_response = mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document=DocumentURLChunk(document_url=signed_url.url),
                document_annotation_format=OCR_ANNOTATION_SCHEMA,
                include_image_base64=False
            )

            # CRITICAL: Improved debug response writing
            try:
                debug_data = {}
                if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
                    debug_data["document_annotation"] = ocr_response.document_annotation
                
                # Also capture page-level annotations if any
                if hasattr(ocr_response, 'pages'):
                    debug_data["pages"] = []
                    for p in ocr_response.pages:
                        if hasattr(p, 'document_annotation') and p.document_annotation:
                            debug_data["pages"].append(p.document_annotation)
                
                with open("debug_response.json", "w", encoding="utf-8") as f:
                    json.dump(debug_data, f, indent=2, default=str)
                logger.info("✅ Written detailed OCR response to debug_response.json")
            except Exception as e:
                logger.error(f"Failed to write debug response: {e}")
            
            all_structured_data, all_acronyms, pages_processed = _process_ocr_response(ocr_response, original_page_numbers)
            logger.info(f"🐛 DEBUG: extracted {len(all_structured_data)} raw items from _process_ocr_response")
            if all_structured_data:
                logger.info(f"🐛 DEBUG First Item Bypass Flag: {all_structured_data[0].get('_bypass_index_check')}")

            all_structured_data = _consolidate_and_clean_drug_table(all_structured_data)
            logger.info(f"🐛 DEBUG: {len(all_structured_data)} items after _consolidate_and_clean_drug_table")

            try:
                mistral_client.files.delete(file_id=uploaded_file.id)
            except:
                pass

            return {
                "drug_table": all_structured_data,
                "acronyms": all_acronyms,
                "tiers": []
            }, "[NATIVE OCR EXTRACTION]", total_costs

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        traceback.print_exc()
        return {"drug_table": [], "acronyms": [], "tiers": []}, f"ERROR: {str(e)}", {}


def get_plan_and_payer_info(state_name, payer, plan_name):
    """Get plan_id and payer_id from database with exact and fuzzy matching."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT p.plan_id, p.payer_id
                FROM plan_details p
                JOIN payer_details pa ON p.payer_id = pa.payer_id
                WHERE LOWER(p.plan_name) = LOWER(%s) AND LOWER(pa.payer_name) = LOWER(%s)
                LIMIT 1
            """, (plan_name, payer))

            result = cursor.fetchone()
            if result:
                return result[0], result[1]

            cursor.execute("""
                SELECT payer_id, payer_name FROM payer_details
                WHERE LOWER(payer_name) LIKE LOWER(%s) LIMIT 1
            """, (f"%{payer}%",))

            payer_result = cursor.fetchone()
            if payer_result:
                payer_id = payer_result[0]
                cursor.execute("""
                    SELECT plan_id FROM plan_details
                    WHERE payer_id = %s AND LOWER(plan_name) LIKE LOWER(%s) LIMIT 1
                """, (payer_id, f"%{plan_name}%"))

                plan_result = cursor.fetchone()
                if plan_result:
                    return plan_result[0], payer_id

            return None, None
        finally:
            cursor.close()


def deduplicate_dicts(dicts, primary_key='acronym'):
    """Deduplicates a list of dictionaries, merging to keep the most complete info."""
    seen = {}
    for d in dicts:
        key = d.get(primary_key)
        if not key:
            continue
        if key not in seen:
            seen[key] = d.copy()
        else:
            for k, v in d.items():
                if v and not seen[key].get(k):
                    seen[key][k] = v
    return list(seen.values())


def get_all_plans_with_formulary_url():
    """Fetch all plans marked 'processing' with a non-null formulary_url."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.plan_id, p.plan_name, p.formulary_url, p.payer_id, p.state_name, pa.payer_name
            FROM plan_details p
            LEFT JOIN payer_details pa ON p.payer_id = pa.payer_id
            WHERE p.status = 'processing' AND p.formulary_url IS NOT NULL
        """)

        plans = cursor.fetchall()
        cursor.close()

    return [{"plan_id": p[0], "plan_name": p[1], "formulary_url": p[2], "payer_id": p[3], 
             "state_name": p[4], "payer_name": p[5]} for p in plans]


def process_single_pdf_url_worker(plan_info):
    """Worker: Download PDF from URL and process it entirely in-memory."""
    import uuid
    
    plan_id = plan_info['plan_id']
    plan_name = plan_info['plan_name']
    formulary_url = plan_info['formulary_url']
    payer_id = plan_info.get('payer_id')
    state_name = plan_info.get('state_name', 'Unknown')
    payer_name = plan_info.get('payer_name', plan_name)

    logger.info(f"Processing plan: {plan_name} (ID: {plan_id})")

    try:
        pdf_url = transform_viewer_url(formulary_url)
        log_prefix = f"Plan {plan_id}:"

        # Use browser-like headers to avoid 406 errors from websites blocking bots
        download_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
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
            with requests.get(pdf_url, timeout=120, headers=download_headers, stream=True, verify=True) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                    logger.warning(f"{log_prefix} Unexpected content type: {content_type}. Proceeding anyway.")
                pdf_content_bytes = resp.content
        except requests.exceptions.SSLError as e:
            logger.warning(f"{log_prefix} SSL verification failed: {e}. Retrying with SSL verification DISABLED.")
            with requests.get(pdf_url, timeout=120, headers=download_headers, stream=True, verify=False) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                    logger.warning(f"{log_prefix} Unexpected content type on retry: {content_type}. Proceeding anyway.")
                pdf_content_bytes = resp.content

        pdf_bytes = BytesIO(pdf_content_bytes)
        file_hash = calculate_bytes_hash(pdf_bytes.getvalue())

        cached_data, cached_content = get_cached_result(file_hash)
        if cached_data:
            logger.info(f"Using cached result for plan {plan_id}")
            return plan_id, cached_data

        structured_data, method, costs = process_pdf_with_mistral_ocr(
            pdf_bytes,
            payer_name=plan_name,
            filename=f"{plan_name}.pdf"
        )

        drug_table = structured_data.get("drug_table", [])
        acronyms = structured_data.get("acronyms", [])
        
        # ✅ HARD FILTER: Remove any drug with Tier > 6 (Per User Request)
        # This catches index entries like "66, 77" that slipped through extraction.
        # Standard tiers are 1, 2, 3, 4, 5, 6, Generic, Brand.
        valid_drugs = []
        for d in drug_table:
            tier_val = str(d.get("drug_tier", "") or "").strip()
            # Normalize list format "66, 77" -> check all numbers
            tier_nums = re.findall(r'\d+', tier_val)
            if tier_nums:
                # If ANY number is > 6, assume it's an index page number and drop it
                # Exception: "Tier 1", "Tier 2" -> these have numbers <= 6.
                # Only drop if number > 6.
                if any(int(n) > 6 for n in tier_nums):
                    logger.warning(f"🚫 Dropping invalid drug with High Tier (Index data): {d.get('drug_name')} - Tier: {tier_val}")
                    continue
            valid_drugs.append(d)
        
        if len(drug_table) != len(valid_drugs):
            logger.info(f"🧹 Hard Filter removed {len(drug_table) - len(valid_drugs)} invalid index entries.")
            drug_table = valid_drugs

        # FINAL VALIDATION: Check if the entire drug_table looks like index page data
        # This is a last line of defense before database insertion
        if drug_table and _is_extracted_data_from_index_page(drug_table):
            logger.warning(f"🚫 Plan {plan_id}: Detected INDEX PAGE data after extraction. Discarding {len(drug_table)} entries.")
            drug_table = []
            acronyms = []

        # NOTE: Plan-level index detection DISABLED - individual index entries 
        # are filtered in _consolidate_and_clean_drug_table() instead.
        # if _is_extracted_data_from_index_page(drug_table):
        #     logger.warning(f"Plan {plan_id}: Detected index page, skipping")
        #     return plan_id, {"drug_table": [], "acronyms": [], "status": "index_page"}

        if LANGDETECT_AVAILABLE:
            def is_fully_english(item: dict) -> bool:
                for value in item.values():
                    # Only check long strings - short drug names may be misdetected
                    if isinstance(value, str) and len(value) > 50:  # Increased from 20 to 50
                        try:
                            detected_lang = detect_language(value)
                            if detected_lang not in ['en', 'la']:  # Allow Latin (medical terms)
                                return False
                        except:
                            pass  # If detection fails, keep the drug
                return True

            before_lang_filter = len(drug_table)
            drug_table = [d for d in drug_table if is_fully_english(d)]
            filtered_by_lang = before_lang_filter - len(drug_table)
            if filtered_by_lang > 0:
                logger.warning(f"⚠️ Language filter removed {filtered_by_lang} drugs (kept {len(drug_table)})")

        # Clean and normalize drug data
        for drug in drug_table:
            if drug.get("drug_name"):
                cleaned_name, extracted_reqs = clean_drug_name(drug["drug_name"])
                drug["drug_name"] = cleaned_name
                
                # Merge extracted requirements
                if extracted_reqs:
                    existing_reqs = drug.get("drug_requirements")
                    if existing_reqs:
                        drug["drug_requirements"] = f"{existing_reqs}, {extracted_reqs}"
                    else:
                        drug["drug_requirements"] = extracted_reqs
            if drug.get("drug_tier"):
                drug["drug_tier"] = normalize_drug_tier(drug["drug_tier"])

        acronyms = deduplicate_dicts(acronyms, 'acronym')

        # Delete existing records for this plan
        delete_drug_formulary_records_for_plan(plan_id)
        
        # Enrich drug records with plan metadata for database insertion
        enriched_drug_records = []

        # Batch lookup for coverage status
        coverage_map = {}
        if drug_table:
            requirement_tier_pairs = set()
            for drug in drug_table:
                req_code = str(drug.get('drug_requirements', '') or '').strip()
                req_code_norm = normalize_requirement_code(req_code)
                # Tier is already normalized in previous loop
                tier = drug.get('drug_tier') or infer_drug_tier_from_text(req_code_norm)
                requirement_tier_pairs.add((req_code_norm, tier))
            
            with get_db_connection() as conn:
                coverage_map = batch_determine_coverage_status(requirement_tier_pairs, conn, state_name, payer_name)
        
        for drug in drug_table:
            requirements_text = str(drug.get('drug_requirements', '') or '').strip()
            requirements_text_norm = normalize_requirement_code(requirements_text)
            # Tier is already normalized
            drug_tier = drug.get('drug_tier')
            drug_tier_normalized = drug_tier or infer_drug_tier_from_text(requirements_text_norm) or infer_drug_tier_from_text(drug.get("drug_name"))
            
            coverage_status = coverage_map.get((requirements_text_norm, drug_tier_normalized), "Covered")

            # Refined fallback logic:
            # If status is "Covered" (either from DB or default), check requirements.
            # If requirements contain anything OTHER than "MO", it should be "Covered with Conditions".
            # "MO" alone (or empty) keeps it as "Covered".
            if coverage_status and coverage_status.lower() == "covered":
                reqs_lower = requirements_text.lower()
                # Split by common separators to check individual tokens if needed, 
                # or just check if there are other characters. 
                # A simple robust way: remove "mo", "m.o.", spaces, punctuation. If anything significant remains, it's a condition.
                # However, simpler logic might be: if any token is not "mo".
                
                # Normalize for check: remove punctuation/spaces
                # But we need to be careful not to flag "month" as "mo". 
                # Let's use the normalized requirement code which usually handles some cleaning, 
                # but here we want to check the raw-ish text for specific acronyms.
                
                # Let's tokenize by non-alphanumeric to be safe
                tokens = [t for t in re.split(r'[^a-zA-Z0-9]', reqs_lower) if t]
                
                has_other_conditions = False
                for token in tokens:
                    if token != 'mo':
                        has_other_conditions = True
                        break
                
                if has_other_conditions:
                    coverage_status = "Covered with Conditions"

            enriched_record = {
                "id": str(uuid.uuid4()),
                "plan_id": plan_id,
                "payer_id": payer_id,
                "plan_name": plan_name,
                "payer_name": payer_name,
                "drug_name": drug.get("drug_name"),
                "drug_tier": drug_tier_normalized,
                "drug_requirements": requirements_text or None,
                "page_number": drug.get("page_number"),
                "badge_colors": drug.get("badge_colors"),
                "preferred_agent": drug.get("preferred_agent"),
                "non_preferred_agent": drug.get("non_preferred_agent"),
                "state_name": state_name,
                "coverage_status": coverage_status,
                "ndc_code": None,
                "jcode": None,
                "is_prior_authorization_required": detect_prior_authorization(requirements_text),
                "is_step_therapy_required": detect_step_therapy(requirements_text),
                "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
                "coverage_details": None,
                "confidence_score": None,
                "source_url": formulary_url,
                "file_name": f"{plan_name}.pdf"
            }
            enriched_drug_records.append(enriched_record)
        
        # Insert enriched records into database
        if enriched_drug_records:
            # DEBUG LOG
            tiers_debug = [r.get("drug_tier") for r in enriched_drug_records[:5]]
            logger.info(f"🐛 DEBUG: Enriched Records Sample Tiers: {tiers_debug}")
            
            insert_drug_formulary_data(enriched_drug_records)
            logger.info(f"Inserted {len(enriched_drug_records)} drug records for plan {plan_id}")
        
        # Insert acronyms into pp_formulary_names table
        if acronyms:
            insert_acronyms_to_ref_table(acronyms, state_name, payer_name, plan_name, "pp_formulary_names")
            logger.info(f"Inserted {len(acronyms)} acronyms into pp_formulary_names for plan {plan_id}")

        result = {
            "drug_table": drug_table,
            "acronyms": acronyms,
            "method": method,
            "drug_count": len(drug_table)
        }
        cache_result(file_hash, result, None)  # Cache the structured data
        update_plan_file_hash(plan_id, file_hash)

        logger.info(f"Plan {plan_id}: Extracted {len(drug_table)} drugs")
        return plan_id, result

    except Exception as e:
        logger.error(f"Plan {plan_id} failed: {e}")
        traceback.print_exc()
        return plan_id, {"error": str(e), "drug_table": [], "acronyms": []}


def process_pdfs_from_urls_in_parallel():
    """Process PDFs by downloading from URLs in plan_details, in parallel."""
    plans = get_all_plans_with_formulary_url()

    if not plans:
        logger.warning("No plans with formulary URLs found")
        return [], []

    logger.info(f"Found {len(plans)} plans to process")

    processed_plan_ids = []
    all_results = []

    max_workers = min(4, len(plans))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plan = {executor.submit(process_single_pdf_url_worker, plan): plan for plan in plans}

        for future in concurrent.futures.as_completed(future_to_plan):
            plan = future_to_plan[future]
            try:
                plan_id, result = future.result()
                processed_plan_ids.append(plan_id)
                all_results.append({"plan_id": plan_id, "result": result})
            except Exception as e:
                logger.error(f"Plan {plan.get('plan_id')} failed: {e}")

    logger.info(f"Processed {len(processed_plan_ids)} plans")
    return processed_plan_ids, all_results


__all__ = [
    'process_pdf_with_mistral_ocr',
    'process_single_chunk_parallel',
    'process_single_pdf_url_worker',
    'process_pdfs_from_urls_in_parallel',
    'get_plan_and_payer_info',
    'get_all_plans_with_formulary_url',
    'deduplicate_dicts'
]