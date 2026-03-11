"""
pdf_processing.py - Main PDF Processing Pipeline

This is the main entry point module that orchestrates the PDF processing pipeline.
"""

import re
import json
import logging
import time
import traceback
import requests
import uuid   
import concurrent.futures
from io import BytesIO
from typing import Optional, List, Tuple
from collections import defaultdict
from urllib.parse import urlparse

from logger_setup import setup_logger
from mistralai import Mistral
from mistralai.models import DocumentURLChunk

from config import (
    MISTRAL_API_KEY, MAX_PAGES_PER_OCR_REQUEST, MAX_OCR_WORKERS,
    ENABLE_PAGE_PREFILTER, SKIP_INDEX_PAGES, MISTRAL_OCR_MODEL
)

# Import from extraction module (frequently changed functions)
from pdf_extraction import (
    OCR_ANNOTATION_SCHEMA,
    _build_requirements_from_item,
    _extract_drug_from_item,
    _extract_acronym_from_item,
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
    get_cached_result, get_cached_result_by_url, cache_result, update_plan_file_hash,
    insert_acronyms_to_ref_table, insert_drug_formulary_data,
    delete_drug_formulary_records_for_plan, log_audit_event,
    create_transaction, update_transaction, fetch_existing_coverage_by_file_hash
)

from utils import (
    similarity, clean_drug_name, calculate_file_hash, track_mistral_cost,
    calculate_bytes_hash, parse_complex_drug_name, normalize_requirement_code,
    transform_viewer_url
)

from coverage import (
    det_coverage_status, normalize_drug_tier, infer_drug_tier_from_text,
    detect_prior_authorization, detect_step_therapy
)
 
# from clasify import ml_predict_coverage_status

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
coverage_logger = setup_logger("coverage", "logs/coverage.log")
extraction_logger = setup_logger("extraction", "logs/extraction.log")
s3_logger = setup_logger("s3", "logs/s3.log")

# In-memory caches and counters (process scope)
_pdf_mem_cache_by_url = {}
_pdf_mem_cache_by_hash = {}
_s3_header_checks_total = 0
_s3_header_checks_direct = 0

def consolidate_drug_records(records: list) -> list:
    """
    Consolidates duplicate drug records by (plan_id, drug_name, drug_tier).
    Merges drug_requirements into a single string separated by "; ".
    Removes duplicate requirement values.
    """
    if not records:
        return []
    
    consolidated = {}
    for record in records:
        plan_id = record.get("plan_id")
        # Normalize name and tier for grouping
        drug_name = (record.get("drug_name") or "").strip().lower()
        drug_tier = (record.get("drug_tier") or "").strip()
        
        key = (plan_id, drug_name, drug_tier)
        
        if key not in consolidated:
            # First occurrence: Create a copy and initialize requirement set
            new_record = record.copy()
            reqs = record.get("drug_requirements")
            # Split existing requirements if they contain "; "
            if reqs:
                new_record["_req_set"] = {r.strip() for r in str(reqs).split(";") if r.strip()}
            else:
                new_record["_req_set"] = set()
            consolidated[key] = new_record
        else:
            # Subsequent occurrence: Merge requirements
            new_reqs = record.get("drug_requirements")
            if new_reqs:
                new_req_parts = {r.strip() for r in str(new_reqs).split(";") if r.strip()}
                consolidated[key]["_req_set"].update(new_req_parts)
            
            # Keep the record with the smaller page number if available
            current_page = consolidated[key].get("page_number")
            new_page = record.get("page_number")
            if new_page is not None and (current_page is None or new_page < current_page):
                consolidated[key]["page_number"] = new_page

    # Finalize records: Convert sets back to strings and remove temporary keys
    final_records = []
    for record in consolidated.values():
        req_set = record.pop("_req_set", set())
        if req_set:
            # Sort for deterministic output
            record["drug_requirements"] = "; ".join(sorted(list(req_set)))
        else:
            record["drug_requirements"] = None
        final_records.append(record)
        
    return final_records

def _log_s3_header_check_summary():
    try:
        s3_logger.info(f"s3.header_check.summary total={_s3_header_checks_total} direct={_s3_header_checks_direct}")
        logger.info(f"S3 header checks: total={_s3_header_checks_total}, direct={_s3_header_checks_direct}")
    except Exception:
        pass

# ✅ Thread-level lock to prevent duplicate URL processing race condition.
# When 2 plans share the same formulary_url and run in parallel, the 2nd worker
# waits until the 1st has finished and cached the result, then gets a cache hit.
import threading
_url_processing_locks: dict = {}
_url_locks_mutex = threading.Lock()

def _get_url_lock(url: str) -> threading.Lock:
    """Returns (or creates) a per-URL threading.Lock to prevent duplicate OCR calls."""
    with _url_locks_mutex:
        if url not in _url_processing_locks:
            _url_processing_locks[url] = threading.Lock()
        return _url_processing_locks[url]

try:
    import fitz
except ImportError:
    fitz = None

NON_LATIN_PATTERN = re.compile(r'[^\x00-\x7F]')

def contains_non_latin(text: str) -> bool:
    return bool(NON_LATIN_PATTERN.search(text))


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
        'raw_text':"",
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

        # Collect markdown from pages in this chunk
        chunk_text_parts = []
        for page in ocr_response.pages:
            if hasattr(page, 'markdown') and page.markdown:
                chunk_text_parts.append(page.markdown)
        result['raw_text'] = "\n\n".join(chunk_text_parts)
    

        if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
            chunk_json = ocr_response.document_annotation
            if isinstance(chunk_json, str):
                try:
                    chunk_json = json.loads(chunk_json)
                except json.JSONDecodeError:
                    chunk_json = {"DrugInformation": [], "FormularyAbbreviations": []}

            drug_info_list = chunk_json.get("DrugInformation", [])

            for drug_idx, item in enumerate(drug_info_list):
                if isinstance(item, dict):
                    drug_name = item.get("Drug Name", "")
                    if not drug_name or len(drug_name) < 2:
                        continue

                    drug_requirements = _build_requirements_from_item(item)

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
                        "drug_tier": item.get("drug tier"),
                        "drug_requirements": drug_requirements,
                        "category": item.get("category"),
                        "page_number": actual_pdf_page
                    })

            for item in chunk_json.get("FormularyAbbreviations", []):
                if isinstance(item, dict):
                    result['acronyms'].append(_extract_acronym_from_item(item))

        try:
            mistral_client.files.delete(file_id=chunk_uploaded.id)
        except:
            pass

        # Check if this chunk's data looks like it came from an index page
        if result['drugs'] and _is_extracted_data_from_index_page(result['drugs']):
            logger.warning(f"⚠️ Chunk {chunk_idx + 1}: Detected INDEX PAGE data, discarding {len(result['drugs'])} entries")
            result['drugs'] = []
            result['acronyms'] = []
        elif result['drugs']:
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
            all_raw_text = []
            
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
                            # Add raw text from this chunk
                            if result.get('raw_text'):
                                all_raw_text.append(result['raw_text'])

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
                "tiers": [],
                "raw_text": "\n\n".join(all_raw_text)
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

            # CRITICAL: Pass original_page_numbers for correct page mapping
            all_structured_data, all_acronyms, pages_processed, raw_text = _process_ocr_response(ocr_response, original_page_numbers)
            all_structured_data = _consolidate_and_clean_drug_table(all_structured_data)

            try:
                mistral_client.files.delete(file_id=uploaded_file.id)
            except:
                pass

            return {
                "drug_table": all_structured_data,
                "acronyms": all_acronyms,
                "tiers": [],
                "raw_text": raw_text
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
                FROM ebv_genai.plan_details p
                JOIN ebv_genai.payer_details pa ON p.payer_id = pa.payer_id
                WHERE LOWER(p.plan_name) = LOWER(%s) AND LOWER(pa.payer_name) = LOWER(%s)
                LIMIT 1
            """, (plan_name, payer))

            result = cursor.fetchone()
            if result:
                return result[0], result[1]

            cursor.execute("""
                SELECT payer_id, payer_name FROM ebv_genai.payer_details
                WHERE LOWER(payer_name) LIKE LOWER(%s) LIMIT 1
            """, (f"%{payer}%",))

            payer_result = cursor.fetchone()
            if payer_result:
                payer_id = payer_result[0]
                cursor.execute("""
                    SELECT plan_id FROM ebv_genai.plan_details
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


# def get_all_plans_with_formulary_url():
#     """Fetch all plans marked 'processing' with a non-null formulary_url."""
#     with get_db_connection() as conn:
#         cursor = conn.cursor()

#         cursor.execute("""
#             SELECT p.plan_id, p.plan_name, p.formulary_url, p.payer_id, p.state_name, pa.payer_name
#             FROM plan_details p
#             LEFT JOIN payer_details pa ON p.payer_id = pa.payer_id
#             WHERE p.status = 'processing' AND p.formulary_url IS NOT NULL
#         """)

#         plans = cursor.fetchall()
#         cursor.close()

#     return [{"plan_id": p[0], "plan_name": p[1], "formulary_url": p[2], "payer_id": p[3], 
#              "state_name": p[4], "payer_name": p[5]} for p in plans]

#Updated Codebase - 2024-06-20: Modified to fetch all active plans with non-null s3_frozen_pdf_url instead of formulary_url, as we want to process only those with frozen PDFs ready for OCR.
def get_all_plans_with_formulary_url():
    """
    Fetch all active plans with a non-null s3_frozen_pdf_url.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                p.plan_id,
                p.plan_name,
                p.s3_frozen_pdf_url,
                p.payer_id,
                p.state_name,
                pa.payer_name,
                p.file_hash
            FROM ebv_genai.plan_details p
            LEFT JOIN ebv_genai.payer_details pa 
                ON p.payer_id = pa.payer_id
            WHERE LOWER(p.status) = 'active'
              AND p.s3_frozen_pdf_url IS NOT NULL
        """)

        plans = cursor.fetchall()
        cursor.close()

    all_plans = [
        {
            "plan_id": p[0],
            "plan_name": p[1],
            "s3_frozen_pdf_url": p[2],  # keep key same
            "payer_id": p[3],
            "state_name": p[4],
            "payer_name": p[5],
            "file_hash": p[6]
        }
        for p in plans
    ]
    
    from config import PLAN_PROCESSING_LIMIT
    if str(PLAN_PROCESSING_LIMIT).strip().lower() != "all":
        try:
            limit = int(PLAN_PROCESSING_LIMIT)
            if limit > 0:
                logger.info(f"Limiting plan processing to first {limit} plans based on PLAN_PROCESSING_LIMIT.")
                all_plans = all_plans[:limit]
        except ValueError:
            pass
            
    return all_plans



def _insert_cached_data_for_plan(cached_data, plan_id, plan_name, payer_id, payer_name,
                                   state_name, s3_frozen_pdf_url, transaction_id, file_hash, acronym_cache=None):
    """
    Re-inserts drug formulary + acronym records for a NEW plan using previously cached data.
    Called when URL or hash cache hits to avoid skipping DB insertion for the new plan.
    """
    drug_table = cached_data.get("drug_table", [])
    acronyms = cached_data.get("acronyms", [])
    raw_text = cached_data.get("raw_text", "")

    # Log the cache hit for this specific plan
    logger.info(f"♻️ Plan {plan_id} ({plan_name}): Using file hash cache (Hash: {file_hash})")
    coverage_logger.info(f"♻️ Plan {plan_id}: Re-inserting {len(drug_table)} drugs and {len(acronyms)} acronyms from file hash cache hit.")

    # Delete existing records for this plan before re-inserting
    delete_drug_formulary_records_for_plan(plan_id)

    # --- ML REUSE OPTIMIZATION ---
    # Fetch existing coverage results for this file hash to avoid re-predicting
    cached_results = fetch_existing_coverage_by_file_hash(file_hash)
    existing_coverage = cached_results.get("drugs", {})
    existing_acronyms = cached_results.get("acronyms", {})
    ml_reused_count = 0
    acr_reused_count = 0
    # ---

    enriched_drug_records = []
    with get_db_connection() as conn:
        for drug in drug_table:
            drug_name = drug.get("drug_name")
            requirements_text = str(drug.get('drug_requirements', '') or '').strip()
            drug_tier = drug.get('drug_tier')

            # Normalize for lookup and for insertion/prediction
            requirements_text_norm = normalize_requirement_code(requirements_text)
            drug_tier_normalized = drug_tier or infer_drug_tier_from_text(requirements_text_norm)

            # Key for cache lookup
            lookup_key = (
                drug_name.lower() if drug_name else '',
                drug_tier,
                requirements_text if requirements_text else None
            )

            if existing_coverage and lookup_key in existing_coverage:
                cached_coverage = existing_coverage[lookup_key]
                coverage_status = cached_coverage["coverage_status"]
                confidence_score = cached_coverage["confidence_score"]
                manual_review = cached_coverage.get("manual_review", False)
                ml_reused_count += 1
                if ml_reused_count <= 20: # Log first few reuses
                    logger.info(f"[ML-REUSE] Reused coverage for '{drug_name}' from hash '{file_hash}'")

            else:
                # Fallback to prediction if not found in cache
                combined_acronym_parts = []
                if drug_tier_normalized:
                    combined_acronym_parts.append(str(drug_tier_normalized))
                if requirements_text_norm:
                    combined_acronym_parts.append(str(requirements_text_norm))
                combined_acronym = ", ".join(combined_acronym_parts) if combined_acronym_parts else None

                coverage_status, confidence_score, source, manual_review = det_coverage_status(
                    acronym=combined_acronym,
                    expansion=None,
                    explanation=None,
                    requirements_text=requirements_text_norm,
                    tier_text=drug_tier_normalized,
                    conn=conn,
                    state_name=state_name,
                    payer_name=payer_name,
                    # ml_predict_fn=ml_predict_coverage_status,
                    drug_name=drug.get('drug_name'),
                    acronym_cache=acronym_cache
                )
                # no ml used now ignore ---Manual verification if ML confidence is low - this is now handled within det_coverage_status
                # but we keep manual_review as returned from the function.

            enriched_drug_records.append({
                "id": str(uuid.uuid4()),
                "plan_id": plan_id,
                "payer_id": payer_id,
                "plan_name": plan_name,
                "payer_name": payer_name,
                "drug_name": drug.get("drug_name"),
                "drug_tier": drug_tier_normalized,
                "drug_cost": drug.get("drug_cost"),
                "drug_requirements": requirements_text or None,
                "page_number": drug.get("page_number"),
                "state_name": state_name,
                "coverage_status": coverage_status,
                "ndc_code": None,
                "jcode": None,
                "is_prior_authorization_required": detect_prior_authorization(requirements_text),
                "is_step_therapy_required": detect_step_therapy(requirements_text),
                "is_quantity_limit_applied": False,
                "coverage_details": None,
                "confidence_score": confidence_score,
                "manual_review": manual_review,
                "s3_frozen_pdf_url": s3_frozen_pdf_url,  # Updated key name
                "source_url": s3_frozen_pdf_url,
                "file_name": f"{plan_name}.pdf"
            })

    if ml_reused_count > 0:
        logger.info(f"[ML-REUSE] Reused {ml_reused_count} coverage predictions from hash '{file_hash}'")

    if enriched_drug_records:
        insert_drug_formulary_data(consolidate_drug_records(enriched_drug_records))
        logger.info(f"Cache re-insert: {len(enriched_drug_records)} drug records for plan {plan_id}")

    # ✅ Re-insert acronyms with ML coverage status
    if acronyms:
        with get_db_connection() as conn:
            for acronym in acronyms:
                # --- ML REUSE OPTIMIZATION FOR ACRONYMS ---
                acr_key = (
                    acronym.get('acronym'),
                    acronym.get('expansion'),
                    acronym.get('explanation')
                )
                
                if existing_acronyms and acr_key in existing_acronyms:
                    acr_coverage_status = existing_acronyms[acr_key]
                    acr_reused_count += 1
                    if acr_reused_count <= 10:
                        logger.info(f"[ML-REUSE] Reused coverage for acronym '{acr_key[0]}' from hash '{file_hash}'")
                else:
                    acr_coverage_status, _, _, _ = det_coverage_status(
                        acronym=acronym.get('acronym'),
                        expansion=acronym.get('expansion'),
                        explanation=acronym.get('explanation'),
                        requirements_text=f"{acronym.get('acronym', '')} {acronym.get('expansion', '')}",
                        tier_text=str(acronym.get('acronym', '')),
                        conn=conn,
                        state_name=state_name,
                        payer_name=payer_name,
                        # ml_predict_fn=ml_predict_coverage_status,
                        drug_name=None
                    )
                acronym['coverage_status'] = acr_coverage_status  # ✅ ML predicted (or reused)
        
        if acr_reused_count > 0:
            logger.info(f"[ML-REUSE] Reused {acr_reused_count} acronym coverage predictions from hash '{file_hash}'")

        insert_acronyms_to_ref_table(acronyms, state_name, payer_name, plan_name, "ebv_genai.pp_formulary_names")
        logger.info(f"Cache re-insert: {len(acronyms)} acronyms for plan {plan_id}")

    # Update plan file hash and statuses
    update_plan_file_hash(plan_id, file_hash)
    from database import update_drug_formulary_status, update_plan_and_payer_statuses
    try:
        update_drug_formulary_status([plan_id])
        update_plan_and_payer_statuses([plan_id], finalize_run=False)
        logger.info(f"✅ Cache re-insert complete for plan {plan_id} — drugs: {len(enriched_drug_records)}, acronyms: {len(acronyms)}")
    except Exception as e:
        logger.error(f"Failed to update statuses for cached plan {plan_id}: {e}")

# Updated Codebase
def download_pdf_bytes(pdf_url, log_prefix, proxies=None):
    """
    Download PDF from:
    - Public HTTP/HTTPS
    - S3 (public or private)
    - Pre-signed S3 URL
    Returns raw bytes
    """
 
    parsed = urlparse(pdf_url)

    # Header check logging
    global _s3_header_checks_total, _s3_header_checks_direct
    _s3_header_checks_total += 1
    is_direct_s3 = ("amazonaws.com" in parsed.netloc and "X-Amz-Signature" not in pdf_url)
    if is_direct_s3:
        _s3_header_checks_direct += 1
    s3_logger.info(f"{log_prefix} s3.header_check url={pdf_url} direct={is_direct_s3}")

    # --------------------------------------------------
    # 1️⃣ If S3 direct URL (not pre-signed)
    # --------------------------------------------------
    if is_direct_s3:
        logger.info(f"{log_prefix} Detected direct S3 URL")
        s3_logger.info(f"{log_prefix} s3.download.start url={pdf_url}")
 
        bucket = parsed.netloc.split(".")[0]
        key = parsed.path.lstrip("/")
 
        try:
            s3 = boto3.client("s3")  # Uses IAM role or env creds
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            s3_logger.info(f"{log_prefix} s3.download.success bytes={len(content)}")
            return content
        except ClientError as e:
            logger.error(f"{log_prefix} S3 download failed: {e}")
            s3_logger.error(f"{log_prefix} s3.download.failure error={e}")
            raise
 
    # --------------------------------------------------
    # 2️⃣ HTTP / Pre-signed URL
    # --------------------------------------------------
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
 
    for attempt in range(2):
        try:
            with requests.get(
                pdf_url,
                timeout=120,
                headers=headers,
                stream=True,
                verify=(attempt == 0),
                proxies=proxies
            ) as resp:
 
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
 
                if "text/html" in content_type.lower():
                    raise Exception(f"HTML returned instead of PDF ({content_type})")
 
                content = resp.content
                if "amazonaws.com" in parsed.netloc:
                    s3_logger.info(f"{log_prefix} s3.download.success bytes={len(content)} via_http")
                return content
 
        except requests.exceptions.SSLError:
            logger.warning(f"{log_prefix} SSL error. Retrying without verification.")
        except Exception as e:
            logger.warning(f"{log_prefix} Attempt {attempt+1} failed: {e}")
            if "amazonaws.com" in parsed.netloc:
                s3_logger.warning(f"{log_prefix} s3.download.retry attempt={attempt+1} error={e}")
 
    raise Exception("Download failed after retries")


def process_single_pdf_url_worker(plan_info):
    """Worker: Download PDF from URL and process it entirely in-memory."""
    
    plan_id = plan_info['plan_id']
    plan_name = plan_info['plan_name']
    s3_frozen_pdf_url = plan_info['s3_frozen_pdf_url']  # Updated key name
    payer_id = plan_info.get('payer_id')
    state_name = plan_info.get('state_name', 'Unknown')
    payer_name = plan_info.get('payer_name', plan_name)
    
    # CREATE TRANSACTION
    transaction_id = str(uuid.uuid4())
    create_transaction(
        transaction_id=transaction_id,
        job_type="single_pdf_realtime",
        plan_id=plan_id,
        payer_id=payer_id,
        file_name=f"{plan_name}.pdf",
        request_summary={"plan_name": plan_name, "s3_frozen_pdf_url": s3_frozen_pdf_url},
        status="in_progress"
    )

    logger.info(f"Processing plan: {plan_name} (ID: {plan_id}) [Transaction: {transaction_id}]")

    # ✅ Optimization: Pre-load all acronyms for this Payer and State once
    from database import fetch_acronym_cache
    acronym_cache = fetch_acronym_cache(payer_name, state_name)

    try:
        # Validate URL before attempting download
        if not s3_frozen_pdf_url or not isinstance(s3_frozen_pdf_url, str):
            logger.error(f"Plan {plan_id}: Invalid or missing s3_frozen_pdf_url")
            update_transaction(transaction_id=transaction_id, status="failed",
                completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                response_summary={"error": "Invalid or missing s3_frozen_pdf_url"})
            log_audit_event(transaction_id=transaction_id, event_type="validation.invalid_url",
                service="pdf_processing", error_message="Invalid or missing s3_frozen_pdf_url",
                payload={"plan_id": plan_id})
            return plan_id, {"error": "Invalid or missing s3_frozen_pdf_url", "drug_table": [], "acronyms": []}
    # try:
        pdf_url = transform_viewer_url(s3_frozen_pdf_url)
        log_prefix = f"Plan {plan_id}:"

        # Use browser-like headers to avoid 406 errors from websites blocking bots
        download_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        } 
        # Check if URL has a valid scheme (http:// or https://)
        # if not formulary_url.startswith(('http://', 'https://')):
        #     logger.error(f"Plan {plan_id}: Invalid URL format '{formulary_url[:100]}' - not a valid http/https URL. Skipping.")
        #     update_transaction(transaction_id=transaction_id, status="failed",
        #         completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        #         response_summary={"error": f"Invalid URL format — not http/https: {formulary_url[:120]}"})
        #     log_audit_event(transaction_id=transaction_id, event_type="validation.invalid_url",
        #         service="pdf_processing", error_message=f"formulary_url is not a valid URL: {formulary_url[:120]}",
        #         payload={"plan_id": plan_id, "bad_value": formulary_url[:200]})
        #     return plan_id, {"error": f"Invalid URL format: {formulary_url[:100]}", "drug_table": [], "acronyms": []}

        # Acquire URL-level lock BEFORE any network I/O to avoid duplicate downloads
        url_lock = _get_url_lock(s3_frozen_pdf_url)
        url_lock.acquire()
        try:
            # STEP 1: Check cache by URL (inside lock — handles case where another worker finished while this one was queued)
            cached_file_hash, cached_data, cached_content = get_cached_result_by_url(s3_frozen_pdf_url)
            if cached_data:
                logger.info(f"🎯 URL Cache HIT for plan {plan_id} - Re-inserting from cache")
                log_audit_event(
                    transaction_id=transaction_id,
                    event_type="cache.hit",
                    service="pdf_processing",
                    payload={"plan_id": plan_id, "file_hash": cached_file_hash, "s3_frozen_pdf_url": s3_frozen_pdf_url[:100]}
                )
                _insert_cached_data_for_plan(
                    cached_data=cached_data,
                    plan_id=plan_id, plan_name=plan_name, payer_id=payer_id,
                    payer_name=payer_name, state_name=state_name,
                    s3_frozen_pdf_url=s3_frozen_pdf_url, transaction_id=transaction_id,
                    file_hash=cached_file_hash,
                    acronym_cache=acronym_cache
                )
                update_transaction(
                    transaction_id=transaction_id,
                    status="completed",
                    completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    file_hash=cached_file_hash,
                    rows_inserted=len(cached_data.get('drug_table', [])),
                    ocr_pages_processed=0,
                    response_summary={"cache_hit": True, "source": "url"}
                )
                return plan_id, cached_data

            # STEP 2: In-memory URL cache (within this process run)
            pdf_content_bytes = _pdf_mem_cache_by_url.get(s3_frozen_pdf_url)
            if pdf_content_bytes is None:
                try:
                    log_audit_event(transaction_id=transaction_id, event_type="s3.download.start", service="pdf_processing",
                                    payload={"plan_id": plan_id, "url": pdf_url})
                    pdf_content_bytes = download_pdf_bytes(
                        pdf_url,
                        log_prefix,
                        proxies=None
                    )
                    logger.info(f"{log_prefix} Download successful")
                    log_audit_event(transaction_id=transaction_id, event_type="s3.download.success", service="pdf_processing",
                                    payload={"plan_id": plan_id, "bytes": len(pdf_content_bytes)})
                    _pdf_mem_cache_by_url[s3_frozen_pdf_url] = pdf_content_bytes
                except Exception as e:
                    logger.warning(f"{log_prefix} Download failed: {e}")
                    update_transaction(transaction_id=transaction_id, status="failed",
                        completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        response_summary={"error": f"Download failed: {e}"})
                    log_audit_event(transaction_id=transaction_id, event_type="download.failed",
                        service="pdf_processing", error_message=f"Download failed: {e}",
                        payload={"plan_id": plan_id, "s3_frozen_pdf_url": s3_frozen_pdf_url})
                    return plan_id, {"error": f"Download failed: {e}", "drug_table": [], "acronyms": []}

            # Compute hash and store in in-memory hash cache
            file_hash = calculate_bytes_hash(pdf_content_bytes)
            _pdf_mem_cache_by_hash[file_hash] = pdf_content_bytes

            # STEP 3: Check cache by file hash (if identical PDF already processed under a different URL)
            if not cached_data:
                # ✅ STEP 2: Check cache by file hash (in case URL changed but PDF is the same)
                cached_data, cached_content = get_cached_result(file_hash)
                cached_file_hash = file_hash if cached_data else None

            if cached_data:
                logger.info(f"🎯 Cache HIT (URL/Hash) for plan {plan_id} - Using hash: {cached_file_hash}")
                coverage_logger.info(f"🎯 Plan {plan_id}: Cache HIT found for hash {cached_file_hash}. Re-inserting cached data.")
                
                log_audit_event(
                    transaction_id=transaction_id,
                    event_type="cache.hit",
                    service="pdf_processing",
                    payload={"plan_id": plan_id, "file_hash": cached_file_hash, "s3_frozen_pdf_url": s3_frozen_pdf_url[:100]}
                )
                
                # ✅ Re-insert drugs and acronyms for THIS plan from cached data
                _insert_cached_data_for_plan(
                    cached_data=cached_data,
                    plan_id=plan_id, plan_name=plan_name, payer_id=payer_id,
                    payer_name=payer_name, state_name=state_name,
                    s3_frozen_pdf_url=s3_frozen_pdf_url, transaction_id=transaction_id,
                    file_hash=cached_file_hash,
                    acronym_cache=acronym_cache
                )
                
                update_transaction(
                    transaction_id=transaction_id,
                    status="completed",
                    completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    file_hash=cached_file_hash,
                    rows_inserted=len(cached_data.get('drug_table', [])),
                    ocr_pages_processed=0,
                    response_summary={"cache_hit": True, "source": "hash",
                                       "drugs_reinserted": len(cached_data.get('drug_table', [])),
                                       "acronyms_reinserted": len(cached_data.get('acronyms', []))}
                )
                
                return plan_id, cached_data
            
            # ✅ STEP 4: Proceed with OCR (no cache hit)
            pdf_bytes = BytesIO(pdf_content_bytes)
            structured_data, method, costs = process_pdf_with_mistral_ocr(
                pdf_bytes,
                payer_name=plan_name,
                filename=f"{plan_name}.pdf"
            )

            drug_table = structured_data.get("drug_table", [])
            acronyms = structured_data.get("acronyms", [])
            raw_text = structured_data.get("raw_text", "")

            if _is_extracted_data_from_index_page(drug_table):
                logger.warning(f"Plan {plan_id}: Detected index page, skipping")
                
                # Log index page detection
                log_audit_event(
                    transaction_id=transaction_id,
                    event_type="validation.index_page_detected",
                    service="pdf_processing",
                    payload={"plan_id": plan_id}
                )
                
                update_transaction(
                    transaction_id=transaction_id,
                    status="completed",
                    completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    response_summary={"status": "index_page", "drugs_extracted": 0}
                )
                
                return plan_id, {"drug_table": [], "acronyms": [], "status": "index_page"}
            filtered_by_lang = 0
            removed_drugs = []
            filtered_drugs = []
            for drug in drug_table:
                remove_row = False
                for key, value in drug.items():
                    if isinstance(value, str) and value.strip():
                        if key == "drug_name":
                            continue
                        if contains_non_latin(value):
                            remove_row = True
                            break
                if remove_row:
                    removed_drugs.append(drug)
                else:
                    filtered_drugs.append(drug)
            drug_table = filtered_drugs
            filtered_by_lang = len(removed_drugs)
            if filtered_by_lang > 0:
                logger.warning(
                    f"⚠️ Language filter removed {filtered_by_lang} drugs "
                    f"(kept {len(drug_table)})"
                )
                for drug in removed_drugs:
                    drug_name = drug.get("drug_name", "UNKNOWN_DRUG")
                    logger.warning(
                        f"❌ Removed drug due to non-Latin characters: {drug_name}"
                    )
            else:
                logger.info("🌐 Language filter applied — no drugs removed")


            # Clean and normalize drug data
            for drug in drug_table:
                if drug.get("drug_name"):
                    cleaned_name, extracted_reqs = clean_drug_name(drug["drug_name"])
                    drug["drug_name"] = cleaned_name
                    
                    # Append extracted requirements to existing requirements if they exist
                    if extracted_reqs:
                        existing_reqs = drug.get("drug_requirements")
                        if existing_reqs:
                            drug["drug_requirements"] = f"{existing_reqs}, {extracted_reqs}"
                        else:
                            drug["drug_requirements"] = extracted_reqs

                if drug.get("drug_tier"):
                    drug["drug_tier"] = normalize_drug_tier(drug["drug_tier"])

            acronyms = deduplicate_dicts(acronyms, 'acronym')
            
            # Log OCR completion
            log_audit_event(
                transaction_id=transaction_id,
                event_type="ocr.completed",
                service="mistral_ocr",
                payload={
                    "plan_id": plan_id,
                    "method": method,
                    "drugs_extracted": len(drug_table),
                    "acronyms_extracted": len(acronyms)
                }
            )
            extraction_logger.info(json.dumps({
                "event": "ocr.completed",
                "plan_id": plan_id,
                "method": method,
                "drugs_extracted": len(drug_table),
                "acronyms_extracted": len(acronyms)
            }))

            # Delete existing records for this plan
            delete_drug_formulary_records_for_plan(plan_id)
            
            # Enrich drug records with plan metadata for database insertion
            enriched_drug_records = []

            with get_db_connection() as conn:
                for drug in drug_table:
                    requirements_text = str(drug.get('drug_requirements', '') or '').strip()
                    requirements_text_norm = normalize_requirement_code(requirements_text)
                    
                    # Tier normalization and inference
                    drug_tier = drug.get('drug_tier')
                    drug_tier_normalized = drug_tier or infer_drug_tier_from_text(requirements_text_norm) or infer_drug_tier_from_text(drug.get("drug_name"))
                    
                    # Combine tier and requirements for ML model input (passed as acronym)
                    combined_acronym_parts = []
                    if drug_tier_normalized:
                        combined_acronym_parts.append(str(drug_tier_normalized))
                    if requirements_text_norm:
                        combined_acronym_parts.append(str(requirements_text_norm))
                    
                    combined_acronym = ", ".join(combined_acronym_parts) if combined_acronym_parts else None

                    # Determine coverage status using ML-powered logic
                    coverage_status, confidence_score, source, manual_review = det_coverage_status(
                        acronym=combined_acronym,
                        expansion=None,
                        explanation=None,
                        requirements_text=requirements_text_norm,
                        tier_text=drug_tier_normalized,
                        conn=conn,
                        state_name=state_name,
                        payer_name=payer_name,
                        # ml_predict_fn=ml_predict_coverage_status,
                        drug_name=drug.get("drug_name"),
                        acronym_cache=acronym_cache
                    )
                    coverage_logger.info(json.dumps({
                        "plan_id": plan_id,
                        "drug_name": drug.get("drug_name"),
                        "tier": drug_tier_normalized,
                        "requirements": requirements_text_norm,
                        "coverage_status": coverage_status,
                        "confidence": confidence_score
                    }))
                    
                    # Update the drug dict with final status
                    drug["coverage_status"] = coverage_status
                    
                    # Manual verification if ML confidence is low - this is now handled within det_coverage_status
                    # but we keep manual_review as returned from the function.

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
                        "state_name": state_name,
                        "coverage_status": coverage_status,
                        "ndc_code": None,
                        "jcode": None,
                        "is_prior_authorization_required": detect_prior_authorization(requirements_text),
                        "is_step_therapy_required": detect_step_therapy(requirements_text),
                        "is_quantity_limit_applied": "Yes" if "ql" in (requirements_text or "").lower() else "No",
                        "coverage_details": None,
                        "confidence_score": confidence_score,
                        "manual_review": manual_review,
                        "source_url": s3_frozen_pdf_url,
                        "file_name": f"{plan_name}.pdf"
                    }
                    enriched_drug_records.append(enriched_record)
            
            # Insert enriched records into database
            if enriched_drug_records:
                insert_drug_formulary_data(consolidate_drug_records(enriched_drug_records))
                logger.info(f"Inserted {len(enriched_drug_records)} drug records for plan {plan_id}")
                log_audit_event(transaction_id=transaction_id, event_type="database.insert.details",
                                service="pdf_processing",
                    payload={"plan_id": plan_id, "records": len(enriched_drug_records)})
            
            # Insert Acronyms into pp_formulary_names with coverage status
            if acronyms:
                # ✅ Enrich acronyms with coverage_status before insertion
                with get_db_connection() as conn:
                    for acronym in acronyms:
                        # Determine coverage status for each acronym
                        coverage_status, confidence_score, source, _ = det_coverage_status(
                            acronym=acronym.get("acronym"),
                            expansion=acronym.get("expansion"),
                            explanation=acronym.get("explanation"),
                            requirements_text=f"{acronym.get('acronym', '')} {acronym.get('expansion', '')}", # Pass acronym and expansion as pseudo requirements
                            tier_text=str(acronym.get('acronym', '')), # Pass acronym as pseudo tier
                            conn=conn,
                            state_name=state_name,
                            payer_name=payer_name,
                            # ml_predict_fn=ml_predict_coverage_status,
                            drug_name=None  # Acronyms are not drugs
                        )
                        
                        # Add coverage_status to acronym dict
                        acronym["coverage_status"] = coverage_status
                
                insert_acronyms_to_ref_table(acronyms, state_name, payer_name, plan_name, "ebv_genai.pp_formulary_names")
                logger.info(f"Inserted {len(acronyms)} acronyms with coverage status into pp_formulary_names for plan {plan_id}")
                log_audit_event(transaction_id=transaction_id, event_type="database.insert.acronyms",
                                service="pdf_processing",
                                payload={"plan_id": plan_id, "acronyms": len(acronyms)})
            
            # Log database insertion
            log_audit_event(
                transaction_id=transaction_id,
                event_type="database.insert",
                service="pdf_processing",
                payload={
                    "plan_id": plan_id,
                    "records_inserted": len(enriched_drug_records),
                    "acronyms_inserted": len(acronyms)
                }
            )

            result = {
                "drug_table": drug_table,
                "acronyms": acronyms,
                "method": method,
                "drug_count": len(drug_table),
                "raw_text": raw_text
            }
            cache_result(file_hash, result, None, formulary_url=s3_frozen_pdf_url)
            
            update_plan_file_hash(plan_id, file_hash)
            
            # Update transaction as completed
            update_transaction(
                transaction_id=transaction_id,
                status="completed",
                completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                rows_inserted=len(enriched_drug_records),
                ocr_pages_processed=costs.get("pages_processed", 0) if isinstance(costs, dict) else 0,
                mistral_cost=costs.get("total_cost", 0) if isinstance(costs, dict) else 0,
                file_hash=file_hash,
                file_name=f"{plan_name}.pdf",
                response_summary={
                    "drugs_extracted": len(drug_table),
                    "acronyms_extracted": len(acronyms),
                    "method": method
                }
            )

            logger.info(f"Plan {plan_id}: Extracted {len(drug_table)} drugs")
            return plan_id, result

        finally:
            try:
                url_lock.release()
            except (RuntimeError, UnboundLocalError):
                pass

    except Exception as e:
        # Log error audit event
        log_audit_event(
            transaction_id=transaction_id,
            event_type="processing.error",
            service="pdf_processing",
            error_message=str(e),
            error_stack=traceback.format_exc(),
            payload={"plan_id": plan_id}
        )
        
        # Update transaction as failed
        update_transaction(
            transaction_id=transaction_id,
            status="failed",
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            file_hash=file_hash if 'file_hash' in dir() else None,
            response_summary={"error": str(e), "plan_id": plan_id}
        )
        
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
    
    # -------------------------------------------------------------------------
    # OPTIMIZATION: Check for cache hits BEFORE spawning workers (saves memory/bandwidth)
    # -------------------------------------------------------------------------
    plans_to_submit = []
    for plan in plans:
        plan_id = plan['plan_id']
        s3_frozen_pdf_url = plan['s3_frozen_pdf_url']
        plan_file_hash = plan.get('file_hash')
        
        # 1. Check by plan_file_hash (best case - no download/OCR needed)
        if plan_file_hash:
            cached_data, _ = get_cached_result(plan_file_hash)
            if cached_data:
                logger.info(f"♻️ Pre-dispatch Cache HIT (hash) for plan {plan_id} - Handling immediately")
                # Handle re-insertion here (sequentially or in a light thread)
                # For simplicity and correctness, we use a separate small pool or the worker pool but with a flag
                # However, the user wants "before the worker is sent".
                # We'll just submit it to the pool anyway but the worker will handle it.
                # Wait, if we submit it to the pool, we are still spawning a worker.
                # To really save memory, we handle it here.
                try:
                    # We still need a transaction for audit/history
                    transaction_id = str(uuid.uuid4())
                    create_transaction(
                        transaction_id=transaction_id,
                        job_type="cache_hit_realtime",
                        plan_id=plan_id,
                        payer_id=plan.get('payer_id'),
                        file_name=f"{plan['plan_name']}.pdf",
                        request_summary={"plan_name": plan['plan_name'], "s3_frozen_pdf_url": s3_frozen_pdf_url},
                        status="in_progress"
                    )
                    
                    log_audit_event(
                        transaction_id=transaction_id,
                        event_type="cache.hit",
                        service="pdf_dispatcher",
                        payload={"plan_id": plan_id, "file_hash": plan_file_hash, "source": "pre_dispatch_hash"}
                    )
                    
                    _insert_cached_data_for_plan(
                        cached_data=cached_data,
                        plan_id=plan_id, plan_name=plan['plan_name'], payer_id=plan.get('payer_id'),
                        payer_name=plan.get('payer_name', plan['plan_name']), state_name=plan.get('state_name', 'Unknown'),
                        s3_frozen_pdf_url=s3_frozen_pdf_url, transaction_id=transaction_id,
                        file_hash=plan_file_hash
                    )
                    
                    update_transaction(
                        transaction_id=transaction_id,
                        status="completed",
                        completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        file_hash=plan_file_hash,
                        rows_inserted=len(cached_data.get('drug_table', [])),
                        ocr_pages_processed=0,
                        response_summary={"cache_hit": True, "source": "pre_dispatch_hash"}
                    )
                    
                    processed_plan_ids.append(plan_id)
                    all_results.append({"plan_id": plan_id, "result": cached_data})
                    continue
                except Exception as e:
                    logger.error(f"Failed handling pre-dispatch cache hit for plan {plan_id}: {e}")
                    # Fallback to submitting it to the pool if pre-dispatch handling fails
        
        # 2. Check by URL (no download needed if PDF already cached for this URL)
        cached_hash, cached_data, _ = get_cached_result_by_url(s3_frozen_pdf_url)
        if cached_data:
            logger.info(f"🎯 Pre-dispatch Cache HIT (url) for plan {plan_id} - Handling immediately")
            try:
                transaction_id = str(uuid.uuid4())
                create_transaction(
                    transaction_id=transaction_id,
                    job_type="cache_hit_realtime",
                    plan_id=plan_id,
                    payer_id=plan.get('payer_id'),
                    file_name=f"{plan['plan_name']}.pdf",
                    request_summary={"plan_name": plan['plan_name'], "s3_frozen_pdf_url": s3_frozen_pdf_url},
                    status="in_progress"
                )
                
                log_audit_event(
                    transaction_id=transaction_id,
                    event_type="cache.hit",
                    service="pdf_dispatcher",
                    payload={"plan_id": plan_id, "file_hash": cached_hash, "source": "pre_dispatch_url"}
                )
                
                _insert_cached_data_for_plan(
                    cached_data=cached_data,
                    plan_id=plan_id, plan_name=plan['plan_name'], payer_id=plan.get('payer_id'),
                    payer_name=plan.get('payer_name', plan['plan_name']), state_name=plan.get('state_name', 'Unknown'),
                    s3_frozen_pdf_url=s3_frozen_pdf_url, transaction_id=transaction_id,
                    file_hash=cached_hash
                )
                
                update_transaction(
                    transaction_id=transaction_id,
                    status="completed",
                    completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    file_hash=cached_hash,
                    rows_inserted=len(cached_data.get('drug_table', [])),
                    ocr_pages_processed=0,
                    response_summary={"cache_hit": True, "source": "pre_dispatch_url"}
                )
                
                processed_plan_ids.append(plan_id)
                all_results.append({"plan_id": plan_id, "result": cached_data})
                continue
            except Exception as e:
                logger.error(f"Failed handling pre-dispatch cache hit for plan {plan_id}: {e}")
        
        plans_to_submit.append(plan)

    if not plans_to_submit:
        logger.info(f"All {len(plans)} plans were handled via pre-dispatch cache hits.")
        _log_s3_header_check_summary()
        return processed_plan_ids, all_results

    logger.info(f"Submitting {len(plans_to_submit)} plans to the worker pool for processing.")
    max_workers = min(6, len(plans_to_submit))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plan = {executor.submit(process_single_pdf_url_worker, plan): plan for plan in plans_to_submit}

        for future in concurrent.futures.as_completed(future_to_plan):
            plan = future_to_plan[future]
            try:
                plan_id, result = future.result()
                processed_plan_ids.append(plan_id)
                all_results.append({"plan_id": plan_id, "result": result})
            except Exception as e:
                logger.error(f"Plan {plan.get('plan_id')} failed: {e}")

    logger.info(f"Processed {len(processed_plan_ids)} plans")
    _log_s3_header_check_summary()
    return processed_plan_ids, all_results


def _upload_chunk_for_batch(mistral_client, chunk_bytes, chunk_idx, plan_id, original_chunk_pages):
    """Helper to upload a single chunk to Mistral concurrently."""
    try:
        uploaded_file = _upload_pdf_to_mistral(
            mistral_client, 
            chunk_bytes, 
            f"{plan_id}_chunk_{chunk_idx}.pdf"
        )
        
        if not uploaded_file:
            raise Exception(f"Failed to upload chunk {chunk_idx} for plan {plan_id} after retries.")

        # Get signed URL (24 hours expiry)
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=24)
        
        custom_id_data = {
            "plan_id": str(plan_id),
            "original_pages": original_chunk_pages
        }
        
        request_item = {
            "custom_id": json.dumps(custom_id_data),
            "body": {
                "model": MISTRAL_OCR_MODEL,
                "document": {
                    "type": "document_url",
                    "document_url": signed_url.url
                },
                "document_annotation_format": OCR_ANNOTATION_SCHEMA,
                "include_image_base64": False
            }
        }
        return request_item, uploaded_file.id
    except Exception as e:
        logger.error(f"Error in chunk upload {chunk_idx} for plan {plan_id}: {e}")
        raise

def process_single_plan_for_batch(plan: dict, mistral_client) -> Tuple[List[dict], List[str]]:
    """
    Helper function to process a single plan for batch preparation.
    Downloads, extracts pages, chunks, and uploads to Mistral in PARALLEL.
    """
    requests_payload = []
    uploaded_file_ids =[]
    
    plan_id = plan['plan_id']
    plan_name = plan['plan_name']
    s3_frozen_pdf_url = plan['s3_frozen_pdf_url']
    
    try:
        # 1. Download PDF
        pdf_url = transform_viewer_url(s3_frozen_pdf_url)
        pdf_bytes = download_pdf_bytes(pdf_url, log_prefix=f"[Plan {plan_id}]")
        
        # Calculate Hash and Check Cache
        file_hash = calculate_bytes_hash(pdf_bytes)
        update_plan_file_hash(plan_id, file_hash)
        
        cached_data, cached_content = get_cached_result(file_hash)
        
        if cached_data:
            logger.info(f"Plan {plan_id}: Hash {file_hash} found in cache. Using cached data (Skipping Batch Submission).")
            # Reuse cached data logic (same as your original)
            drug_table = cached_data.get("drug_table",[])
            raw_text = cached_data.get("raw_text", "")
            delete_drug_formulary_records_for_plan(plan_id)
            
            enriched_drug_records =[]
            state_name_local = plan.get('state_name', 'Unknown')
            payer_name_local = plan.get('payer_name', plan_name)
            with get_db_connection() as conn:
                for drug in drug_table:
                    requirements_text = str(drug.get('drug_requirements', '') or '').strip()
                    requirements_text_norm = normalize_requirement_code(requirements_text)
                    drug_tier = drug.get('drug_tier')
                    drug_tier_normalized = drug_tier or infer_drug_tier_from_text(requirements_text_norm)
                    combined_acronym_parts =[]
                    if drug_tier_normalized:
                        combined_acronym_parts.append(str(drug_tier_normalized))
                    if requirements_text_norm:
                        combined_acronym_parts.append(str(requirements_text_norm))
                    combined_acronym = ", ".join(combined_acronym_parts) if combined_acronym_parts else None
                    
                    coverage_status, confidence_score, source, manual_review = det_coverage_status(
                        acronym=combined_acronym,
                        expansion=None, explanation=None,
                        requirements_text=requirements_text_norm,
                        tier_text=drug_tier_normalized,
                        conn=conn, state_name=state_name_local, payer_name=payer_name_local,
                        drug_name=drug.get('drug_name')
                    )

                    enriched_record = {
                        "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": plan.get('payer_id'),
                        "plan_name": plan_name, "payer_name": payer_name_local,
                        "drug_name": drug.get("drug_name"), "drug_tier": drug_tier_normalized,
                        "drug_cost": drug.get("drug_cost"), "drug_requirements": requirements_text or None,
                        "page_number": drug.get("page_number"), "state_name": state_name_local,
                        "coverage_status": coverage_status, "ndc_code": None, "jcode": None,
                        "is_prior_authorization_required": detect_prior_authorization(requirements_text),
                        "is_step_therapy_required": detect_step_therapy(requirements_text),
                        "is_quantity_limit_applied": False, "coverage_details": None,
                        "confidence_score": confidence_score, "manual_review": manual_review,
                        "source_url": s3_frozen_pdf_url, "file_name": f"{plan_name}.pdf"
                    }
                    enriched_drug_records.append(enriched_record)
            
            if enriched_drug_records:
                insert_drug_formulary_data(consolidate_drug_records(enriched_drug_records))
                logger.info(f"Plan {plan_id}: Inserted {len(enriched_drug_records)} cached records.")
                
            return [],[] # Handled via cache
        
        # 2. Extract/Prefilter Pages
        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(src_doc)
        src_doc.close()
        
        if total_pages > MAX_PDF_PAGES:
            logger.warning(f"Plan {plan_id}: PDF too large ({total_pages} pages), skipping.")
            return[],[]

        page_indices_0_based = _get_pages_to_process(f"{plan_name}.pdf", total_pages)
        if not page_indices_0_based:
            page_indices_0_based = list(range(total_pages))
            
        pages_to_process = [p + 1 for p in page_indices_0_based]
        original_page_numbers = pages_to_process.copy()
        
        if ENABLE_PAGE_PREFILTER:
            pages_to_process = prefilter_pages_with_pymupdf(BytesIO(pdf_bytes), pages_to_process)
            original_page_numbers = pages_to_process.copy()
            
        if not pages_to_process:
            return [],[]
            
        extracted_pdf = _extract_pages_from_pdf(BytesIO(pdf_bytes), pages_to_process)
        if not extracted_pdf:
            return [],[]
            
        final_pdf_bytes = extracted_pdf.getvalue()
        
        # 3. Create Chunk Payloads in memory FIRST
        src_doc = fitz.open(stream=final_pdf_bytes, filetype="pdf")
        sequential_pages = list(range(1, len(src_doc) + 1))
        
        chunk_tasks =[]
        for i in range(0, len(sequential_pages), MAX_PAGES_PER_OCR_REQUEST):
            chunk_pages = sequential_pages[i:i + MAX_PAGES_PER_OCR_REQUEST]
            original_chunk_pages = original_page_numbers[i:i + len(chunk_pages)]
            
            chunk_doc = fitz.open()
            for page_num in chunk_pages:
                chunk_doc.insert_pdf(src_doc, from_page=page_num-1, to_page=page_num-1)
            
            chunk_tasks.append({
                "bytes": chunk_doc.tobytes(),
                "idx": i,
                "original_pages": original_chunk_pages
            })
            chunk_doc.close()
            
        src_doc.close()

        # 4. Upload Chunks CONCURRENTLY (Massive Speedup 🚀)
        logger.info(f"Plan {plan_id}: Uploading {len(chunk_tasks)} chunks concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures =[
                executor.submit(_upload_chunk_for_batch, mistral_client, t["bytes"], t["idx"], plan_id, t["original_pages"])
                for t in chunk_tasks
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    req_item, file_id = future.result()
                    requests_payload.append(req_item)
                    uploaded_file_ids.append(file_id)
                except Exception as e:
                    logger.error(f"Failed concurrent chunk upload for Plan {plan_id}: {e}")
                    
        logger.info(f"Plan {plan_id} completely prepared with {len(requests_payload)} chunks.")
        
    except Exception as e:
        logger.error(f"Failed to prepare plan {plan_id}: {e}")
        traceback.print_exc()
        
    return requests_payload, uploaded_file_ids


def prepare_batch_requests(plans: List[dict]) -> Tuple[List[dict], List[str]]:
    """
    Prepares batch OCR requests for a list of plans in parallel.
    Returns:
        - requests_payload: List of dicts representing the JSONL lines for the batch request.
        - uploaded_file_ids: List of file IDs uploaded to Mistral.
    """
    all_requests_payload = []
    all_uploaded_file_ids = []
    
    mistral_client = create_resilient_mistral_client()
    
    logger.info(f"Preparing batch requests for {len(plans)} plans in parallel (Workers: {MAX_OCR_WORKERS})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
        # Submit all plans
        future_to_plan = {
            executor.submit(process_single_plan_for_batch, plan, mistral_client): plan 
            for plan in plans
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_plan)):
            plan = future_to_plan[future]
            try:
                requests, file_ids = future.result()
                if requests:
                    all_requests_payload.extend(requests)
                if file_ids:
                    all_uploaded_file_ids.extend(file_ids)
                logger.info(f"[{i+1}/{len(plans)}] Completed preparation for plan {plan['plan_name']}")
            except Exception as e:
                logger.error(f"Plan {plan['plan_name']} generated an exception: {e}")
                
    return all_requests_payload, all_uploaded_file_ids


def process_plan_batch_result(plan_id: str, chunks: list):
    """Processes Mistral Batch Output for a single plan and saves to DB."""
    import uuid
    transaction_id = str(uuid.uuid4())
    create_transaction(
        transaction_id=transaction_id,
        job_type="batch_process_plan",
        plan_id=plan_id,
        status="in_progress"
    )
    
    logger.info(f"Processing results for plan {plan_id} ({len(chunks)} chunks)... [Transaction: {transaction_id}]")
    
    try:
        all_drugs = []
        all_acronyms = []
        all_raw_text = []
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.plan_name, p.payer_id, p.state_name, pa.payer_name, p.s3_frozen_pdf_url
                FROM ebv_genai.plan_details p
                LEFT JOIN ebv_genai.payer_details pa ON p.payer_id = pa.payer_id
                WHERE p.plan_id = %s
            """, (plan_id,))
            plan_data = cursor.fetchone()
            cursor.close()
            
        if not plan_data:
            logger.error(f"Plan {plan_id} not found in DB, skipping insertion.")
            update_transaction(transaction_id=transaction_id, status="failed", completed_at=time.strftime('%Y-%m-%d %H:%M:%S'), response_summary={"error": "Plan not found"})
            return
            
        plan_name, payer_id, state_name, payer_name, s3_frozen_pdf_url = plan_data
        payer_name = payer_name or plan_name
        state_name = state_name or 'Unknown'

        from database import fetch_acronym_cache
        acronym_cache = fetch_acronym_cache(payer_name, state_name)
        update_transaction(transaction_id=transaction_id, file_name=f"{plan_name}.pdf")
        
        # Compile chunks
        class MockResponse:
            def __init__(self, data):
                self.pages = data.get('pages',[])
                self.document_annotation = data.get('document_annotation')
                
        all_pages_processed = set()
        for chunk in chunks:
            original_pages = chunk['original_pages']
            all_pages_processed.update(original_pages)
            
            mock_response = MockResponse(chunk['ocr_response'])
            chunk_drugs, chunk_acronyms, _, chunk_raw_text = _process_ocr_response(mock_response, original_pages)
            all_drugs.extend(chunk_drugs)
            all_acronyms.extend(chunk_acronyms)
            
            if chunk_raw_text:
                all_raw_text.append(chunk_raw_text)
            
        track_mistral_cost(payer_name, len(all_pages_processed))
        all_drugs = _consolidate_and_clean_drug_table(all_drugs)
        
        if _is_extracted_data_from_index_page(all_drugs):
            logger.warning(f"Plan {plan_id}: Detected index page data after merge, discarding.")
            all_drugs =[]
            
        delete_drug_formulary_records_for_plan(plan_id)
        
        enriched_drug_records =[]
        with get_db_connection() as conn:
            for drug in all_drugs:
                requirements_text = str(drug.get('drug_requirements', '') or '').strip()
                requirements_text_norm = normalize_requirement_code(requirements_text)
                drug_tier = drug.get('drug_tier')
                drug_tier_normalized = drug_tier or infer_drug_tier_from_text(requirements_text_norm)
                combined_acronym_parts =[str(t) for t in [drug_tier_normalized, requirements_text_norm] if t]
                combined_acronym = ", ".join(combined_acronym_parts) if combined_acronym_parts else None
                
                coverage_status, confidence_score, source, manual_review = det_coverage_status(
                    acronym=combined_acronym, expansion=None, explanation=None,
                    requirements_text=requirements_text_norm, tier_text=drug_tier_normalized,
                    conn=conn, state_name=state_name, payer_name=payer_name,
                    drug_name=drug.get('drug_name'), acronym_cache=acronym_cache
                )

                enriched_drug_records.append({
                    "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
                    "plan_name": plan_name, "payer_name": payer_name,
                    "drug_name": drug.get("drug_name"), "drug_tier": drug_tier_normalized,
                    "drug_cost": drug.get("drug_cost"), "drug_requirements": requirements_text or None,
                    "page_number": drug.get("page_number"), "state_name": state_name,
                    "coverage_status": coverage_status, "confidence_score": confidence_score,
                    "manual_review": manual_review, "source_url": s3_frozen_pdf_url,
                    "is_prior_authorization_required": detect_prior_authorization(requirements_text),
                    "is_step_therapy_required": detect_step_therapy(requirements_text),
                    "is_quantity_limit_applied": False, "file_name": f"{plan_name}.pdf"
                })
            
        if enriched_drug_records:
            insert_drug_formulary_data(consolidate_drug_records(enriched_drug_records))
            
        if all_acronyms:
            from database import insert_acronyms_to_ref_table
            with get_db_connection() as conn:
                for acronym in all_acronyms:
                    acr_coverage_status, _, _, _ = det_coverage_status(
                        acronym=acronym.get('acronym'), expansion=acronym.get('expansion'),
                        explanation=acronym.get('explanation'),
                        requirements_text=f"{acronym.get('acronym', '')} {acronym.get('expansion', '')}",
                        tier_text=str(acronym.get('acronym', '')),
                        conn=conn, state_name=state_name, payer_name=payer_name,
                        acronym_cache=acronym_cache
                    )
                    acronym['coverage_status'] = acr_coverage_status
            insert_acronyms_to_ref_table(all_acronyms, state_name, payer_name, plan_name, "ebv_genai.pp_formulary_names")
             
        file_hash = None
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_hash FROM ebv_genai.plan_details WHERE plan_id = %s", (plan_id,))
            res = cursor.fetchone()
            if res and res[0]:
                file_hash = res[0]
                
        if file_hash:
            cache_result(file_hash, {
                "drug_table": all_drugs, 
                "acronyms": all_acronyms, 
                "method": "MISTRAL_BATCH", 
                "drug_count": len(all_drugs),
                "raw_text": "\n\n".join(all_raw_text) if all_raw_text else ""
            }, None, formulary_url=s3_frozen_pdf_url)
            
        from database import update_drug_formulary_status, update_plan_and_payer_statuses, log_audit_event
        update_drug_formulary_status([plan_id])
        update_plan_and_payer_statuses([plan_id], finalize_run=False)
        
        log_audit_event(
            transaction_id=transaction_id,
            event_type="batch_processing_complete",
            service="MistralOCRBatch",
            payload={
                "message": f"Successfully processed OCR batch results for plan {plan_id}",
                "totals": {
                    "drugs_extracted": len(all_drugs),
                    "acronyms_extracted": len(all_acronyms),
                    "pages_processed": len(all_pages_processed)
                }
            }
        )
        
        update_transaction(
            transaction_id=transaction_id, status="completed", completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            rows_inserted=len(enriched_drug_records), ocr_pages_processed=len(all_pages_processed),
            file_hash=file_hash, response_summary={"drugs": len(all_drugs), "acronyms": len(all_acronyms)}
        )
        
    except Exception as e:
        try:
            from database import log_audit_event
            log_audit_event(transaction_id=transaction_id, event_type="batch_processing_failed", service="MistralOCRBatch", payload={"error": str(e)})
        except:
            pass
        update_transaction(transaction_id=transaction_id, status="failed", completed_at=time.strftime('%Y-%m-%d %H:%M:%S'), response_summary={"error": str(e)})
        logger.error(f"Failed to process plan {plan_id}: {e}")
        traceback.print_exc()

def process_batch_output(output_file_path: str):
    """
    Reads the downloaded Mistral Batch output JSONL file and delegates
    the processing to a ThreadPool to process all plans concurrently.
    """
    logger.info(f"Processing batch output from: {output_file_path}")
    
    plan_results = defaultdict(list)
    
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                response_item = json.loads(line)
                custom_id_str = response_item.get('custom_id')
                if not custom_id_str:
                    continue
                    
                custom_id = json.loads(custom_id_str)
                plan_id = custom_id.get('plan_id')
                
                if response_item.get('error'):
                    logger.error(f"Error in batch item for plan {plan_id}: {response_item['error']}")
                    continue
                    
                plan_results[plan_id].append({
                    "original_pages": custom_id.get('original_pages'),
                    "ocr_response": response_item.get('response', {}).get('body', {})
                })
            except Exception as e:
                logger.error(f"Failed to parse batch output line: {e}")
                
    logger.info(f"Starting CONCURRENT processing for {len(plan_results)} plans...")
    
    # Process DB inserts and ML checks concurrently for all returned plans
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
        futures =[
            executor.submit(process_plan_batch_result, plan_id, chunks)
            for plan_id, chunks in plan_results.items()
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Catch any unhandled thread exceptions
            except Exception as e:
                logger.error(f"Exception raised in parallel plan worker: {e}")



__all__ = [
    'process_pdf_with_mistral_ocr',
    'process_single_chunk_parallel',
    'process_single_pdf_url_worker',
    'process_pdfs_from_urls_in_parallel',
    'get_plan_and_payer_info',
    'get_all_plans_with_formulary_url',
    'deduplicate_dicts',
    'prepare_batch_requests',
    'process_batch_output'
]
