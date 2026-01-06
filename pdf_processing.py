import os
import re
import json
import logging
import traceback
import requests
import httpx
import uuid
import time
from io import BytesIO
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

# Mistral SDK
from mistralai import Mistral
from mistralai.models import DocumentURLChunk

# Internal Imports
from config import (
    PROCESS_COUNT, MISTRAL_API_KEY, 
    MISTRAL_OCR_COST_PER_1K_PAGES, PDF_PAGE_PROCESSING_CONFIG,
    CLIENT_TIMEOUT, CONNECT_TIMEOUT, MAX_RETRIES, BACKOFF_MULTIPLIER,
    LLM_PAGE_WORKERS
)
from database import (
    get_db_connection, batch_determine_coverage_status, get_cached_result, 
    cache_result, update_plan_file_hash, 
    insert_drug_formulary_data
)
from utils import (
    clean_drug_name, detect_prior_authorization,
    detect_step_therapy, calculate_bytes_hash, 
    track_mistral_cost, transform_viewer_url, similarity
)

import fitz # PyMuPDF

logger = logging.getLogger(__name__)

MISTRAL_OCR_MODEL = "mistral-ocr-2512"

# --- REFINED SCHEMA (Removed Page Number hallucination) ---
DRUG_EXTRACTION_SCHEMA = {
  "type": "object",
  "properties": {
    "DrugInformation": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Drug Name": {
              "type": "string",
              "description": "The actual name of the drug. IMPORTANT: Do NOT extract Section Headers like 'Analgesics', or Tier Definitions like 'Tier 1 Preferred Generic'.Do NOT extract section headers or sentences."
          },
          "drug tier": {"type": "string", 
          "description": "The value from the 'Tier Designation' column."
          },
          "requirements": {"type": "string",
          "description": "Look at the columns 'Specialty', 'Prior Approval', 'Step Therapy', and 'Dispensing Limits'. If a dot (•), bullet, or symbol is present in a column, add that column's name to this field (e.g., 'Specialty, Prior Approval, Dispensing Limits')."
          },
          "relative_page_number": {
              "type": "integer", 
              "description": "The page number within this provided document (1-8) where the drug was found."
          }
        },
        "required": ["Drug Name"]
      }
    },
    "FormularyAbbreviations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Acronym": {"type": "string", "description": "The short symbol only (e.g. PA, ST, QL)."},
          "Expansion": {"type": "string"},
          "Explanation": {"type": "string"}
        },
        "required": ["Acronym", "Expansion", "Explanation"]
      }
    }
  }
}


def is_junk_record(drug_name, tier, reqs):
    """
    Clean the specific junk seen in the user CSV (Page 95-108).
    """
    name = str(drug_name or "").strip()
    name_lower = name.lower()
    tier_val = str(tier or "").strip()
    reqs_val = str(reqs or "").strip()

    is_medical_product = any(unit in name for unit in ['mg', 'ml', 'mcg', '%', 'soln', 'tabs'])

    if not tier and not reqs and not is_medical_product:
        return True

    # 1. Block "None" records (Seen in your Row 317)
    if name_lower in ["none", "null", "n/a", ""]:
        return True

    # 2. Block Single Letters (Index headers like "A", "B")
    if len(name) <= 1:
        return True

    # 3. Numeric Hallucination / Index Page Check
    # Block if tier is a high number or list of numbers, regardless of requirements.
    # Real tiers are usually 1-6, sometimes "NC" or "NF".
    # Index pages often have "Drug Name ... 85, 89" where 85, 89 is captured as tier.
    
    # Clean tier to remove "Tier " prefix if present for checking
    clean_tier = tier_val.lower().replace("tier", "").strip()
    
    # Check for comma-separated numbers (e.g. "85, 89" or "142,148")
    if "," in clean_tier:
        parts = [p.strip() for p in clean_tier.split(",")]
        # If all parts are digits (or empty strings from trailing commas)
        if all(p.isdigit() for p in parts if p):
             # If any part is > 6, it's likely page numbers
             if any(int(p) > 6 for p in parts if p):
                 return True
                 
    # Check for single high number
    if clean_tier.isdigit() and int(clean_tier) > 6:
        return True

    # 4. Sentence/Description Check
    # "interchangeable biosimilar" (Row 6) or "original biological product" (Row 3)
    if len(name.split()) > 5:
        # Check if it's a real drug with strengths (e.g. "Drug name 10 mg tab")
        if not any(unit in name_lower for unit in ['mg', 'ml', 'mcg', '%']):
            return True

    # 5. Introductory keywords / Table of Contents / Language Assistance Check
    # Blocks "Network Health Preferred Drug list", "Non-preferred medications", "Hindi", "Spanish", etc.
    intro_junk = [
        "priority health", "medicare formulary", "abbreviations", "how to use",
        "preferred drug list", "preferred medications", "non-preferred medications",
        "split-fill program", "non-prescription medications", "compounded prescriptions",
        "smoking cessation products", "morphine milligram equivalent", "immediate-release formulation",
        "language assistance", "interpreter", "hindi", "spanish", "vietnamese", "russian", "german",
        "korean", "tagalog", "pennsylvania dutch", "hmong", "laotian", "french", "arabic", "chinese",
        "your drug", "my drug", "the drug", "this drug"
    ]
    if any(kw in name_lower for kw in intro_junk):
        return True
        
    # Block if tier is "Not specified" or "Tier of the primary ingredient"
    if "not specified" in tier_val.lower() or "primary ingredient" in tier_val.lower():
        return True
        
    # Block Language Assistance rows where requirements contain "speak" or "language"
    if "speak" in reqs_val.lower() or "language" in reqs_val.lower():
        return True

    # 6. Tier Definition/Example Check
    # Blocks "Tier 1 Preferred Generic", "Tier 2 Generic" in EITHER name or tier column.
    # This catches the "Example" table on Page 1.
    if re.match(r'^tier\s*\d+', name_lower) or re.match(r'^tier\s*\d+\s+[a-z]+', tier_val.lower()):
        return True

    # 7. Legend/Glossary Check (New)
    # Blocks "generic equivalent", "biosimilar", "interchangeable biosimilar"
    legend_keywords = [
        "generic equivalent", 
        "biosimilar", 
        "interchangeable biosimilar", 
        "original biological product", 
        "brand name", 
        "preferred brand",
        "non-preferred drug",
        "preferred generic",
        "drugs removed from the market",
        "drug that is being changed",
        "brand name drug"
    ]
    if any(kw in name_lower for kw in legend_keywords):
        return True
    
    # 8. Document ID / Footer Check
    # Blocks long alphanumeric strings like "Y0056NCMS10010852503DC 07102024"
    # Logic: > 15 chars, contains digits, no spaces in the first 10 chars
    if len(name) > 15 and any(c.isdigit() for c in name) and " " not in name[:10]:
         # Ensure it's not a chemical name like "2-methyl..." by checking for high digit count or specific ID format
         if sum(c.isdigit() for c in name) > 5:
             return True

    # 9. Generic Terms Check
    # Blocks "Drugs", "Generic", "Specialty" if they appear at the start
    generic_starts = ["drugs", "drug", "generic", "specialty"]
    if any(name_lower.startswith(s) for s in generic_starts):
        # 1. Exact match or short phrase (e.g. "drug", "generic tier")
        if name_lower in generic_starts or len(name.split()) < 4:
             return True
             
        # 2. Parenthetical junk (e.g. "Specialty (30-day supply only)")
        if "(" in name_lower:
             return True
             
        # 3. Long sentence starting with generic term (e.g. "drug you have been taking")
        # If it has no dosage info, it's likely junk.
        if not any(unit in name_lower for unit in ['mg', 'ml', 'mcg', '%']):
             return True

    # Also block if the name contains these phrases AND has no dosage info, 
    # to catch "we may remove a brand name..." type sentences if they slip through rule 4
    if any(kw in name_lower for kw in legend_keywords) and len(name.split()) > 3:
         if not any(unit in name_lower for unit in ['mg', 'ml', 'mcg', '%']):
            return True

    return False

def is_index_page(markdown: str) -> bool:
    """
    Strictest Index Detection: Detects if a page is an Alphabetical Index
    by checking for the absence of headers and the presence of page references.
    """
    lower = markdown.lower().strip()
    lines = lower.splitlines()
    
    # --- RULE 1: Direct Header Check ---
    # Actual drug pages MUST have headers. Index pages usually don't.
    # We check the first 15 lines for the word "Tier" or "Requirements"
    header_lines = lines[:15]
    has_formulary_headers = False
    has_index_headers = False
    
    for line in header_lines:
        # Check for Index headers
        if "page" in line and ("no" in line or "number" in line or "#" in line):
            has_index_headers = True
        
        # Check for column header combinations (e.g. "Drug Name | Tier | Requirements")
        if "tier" in line and ("requirement" in line or "limit" in line or "restriction" in line):
            has_formulary_headers = True
            break
        if "drug" in line and ("tier" in line or "requirement" in line):
            has_formulary_headers = True
            break
        # Check for specific column headers on their own line
        # REMOVED "drug name" from single-line check as it appears in Index pages too
        if line.strip() in ["tier", "requirements", "limits", "restrictions", "drug tier"]:
            has_formulary_headers = True
            break
            
    # If we found explicit index headers like "Page No", it's likely an index
    if has_index_headers and not has_formulary_headers:
        return True
    
    # If the page literally says "Index" or "Index of Drugs" at the top
    # Check first 10 lines for "Index" title
    for i in range(min(10, len(lines))):
        line_clean = lines[i].strip('# ').strip()
        if any(kw in line_clean for kw in ['index', 'index of drugs', 'alphabetical index', 'drug index']):
            # Extra safety: only skip if it doesn't have the data headers
            if not has_formulary_headers:
                return True

    # --- RULE 2: Lack of requirements density ---
    # On a real drug page, Mistral OCR 3 markdown will have pipes | or tabs.
    # In your CSV, index pages have 100% empty requirements.
    # We will check the markdown for specific index patterns.
    
    # Pattern: Drug Name followed by 5+ dots or just a trailing number
    # Example: "abacavir sulfate ..................... 43" or "abacavir sulfate 43"
    index_pattern_dots = re.compile(r'\.{3,}\s*\d+')
    index_pattern_num = re.compile(r'\s+\d{1,3}$')
    
    matches_dots = sum(1 for line in lines if index_pattern_dots.search(line.strip()))
    matches_num = sum(1 for line in lines if index_pattern_num.search(line.strip()))
    
    if len(lines) > 5:
        # If dots are present, we can be more confident with a lower ratio
        if matches_dots > 0:
            match_ratio = matches_dots / len(lines)
            if match_ratio > 0.4 and not has_formulary_headers:
                logger.info(f"Detected Index page via dot pattern ratio: {match_ratio:.2%}")
                return True
        
        # REMOVED: Trailing number check (match_ratio > 0.5) caused false positives on pages 3-16

    # --- RULE 3: Alphabetical Section Headers ---
    # Indexes have single letters (A, B, C) as headers. 
    # If we see a single letter on a line by itself frequently:
    alpha_headers = sum(1 for line in lines if len(line.strip('# ')) == 1 and line.strip('# ').isalpha())
    if alpha_headers > 3 and not has_formulary_headers:
        return True

    return False

def is_aca_drug_list_page(markdown: str) -> bool:
    """
    Detects if a page is an ACA/Preventative summary page.
    """
    lower_markdown = markdown.lower()
    score = 0
    
    # Feature 1: BRAND/GENERIC column structure (very common in ACA lists)
    if re.search(r'\|\s*brand\s*\|\s*generic\s*\|', lower_markdown):
        score += 8
    
    # Feature 2: Specific Titles
    titles = ["aca drug list", "preventive medications", "contraceptives", "tobacco cessation"]
    if any(t in lower_markdown for t in titles):
        score += 5
        
    # Feature 3: Keywords
    if "affordable care act" in lower_markdown:
        score += 5

    return score >= 10

def create_resilient_mistral_client():
    timeout = httpx.Timeout(CLIENT_TIMEOUT, connect=CONNECT_TIMEOUT)
    client = httpx.Client(timeout=timeout, transport=httpx.HTTPTransport(retries=MAX_RETRIES))
    return Mistral(api_key=MISTRAL_API_KEY, client=client)

# --- NEW: CHUNKED PAGE EXTRACTION ---

def process_chunk(client, chunk_bytes, chunk_indices, filename):
    """
    Processes a batch of up to 8 pages in parallel using Mistral OCR.
    chunk_indices: list of 0-based absolute page indices.
    """
    try:
        # Upload the chunk PDF
        uploaded_file = client.files.upload(
            file={"file_name": f"chunk_{chunk_indices[0]}_{filename}", "content": chunk_bytes}, 
            purpose="ocr"
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)

        # NO RETRY LOOP - Single Attempt
        ocr_response = client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document=DocumentURLChunk(document_url=signed_url.url),
            document_annotation_format={
                "type": "json_schema", 
                "json_schema": {"name": "drug_extraction", "schema": DRUG_EXTRACTION_SCHEMA, "strict": True}
            }
        )
        
        # Identify Index Pages within the chunk
        # Mistral returns pages in order.
        index_pages_relative = set()
        all_markdowns = []
        
        for i, page in enumerate(ocr_response.pages):
            md = page.markdown
            all_markdowns.append(md)
            if is_index_page(md):
                abs_p = chunk_indices[i] + 1
                logger.warning(f"Page {abs_p} detected as Index Page.")
                index_pages_relative.add(i + 1) # 1-based relative index

        raw_json = getattr(ocr_response, 'document_annotation', {})
        if isinstance(raw_json, str):
            raw_json = json.loads(raw_json)

        # Filter records
        valid_drugs = []
        if "DrugInformation" in raw_json:
            for drug in raw_json["DrugInformation"]:
                d_name = drug.get("Drug Name")
                d_tier = drug.get("drug tier")
                d_reqs = drug.get("requirements")

                # 1. Check Junk
                if is_junk_record(d_name, d_tier, d_reqs):
                    continue
                
                # 1.5 Fix Tier/Requirements Swap
                # If Tier contains only requirements codes (PA, QL, SP, ST) and Requirements is empty
                # Move Tier to Requirements.
                tier_upper = str(d_tier or "").upper().strip()
                reqs_upper = str(d_reqs or "").upper().strip()
                
                # Check if tier looks like requirements (contains PA, QL, SP, ST and NO "Tier" or "Generic" or "Brand")
                req_codes = ["PA", "QL", "SP", "ST", "LIMIT", "RESTRICTION"]
                is_req_code = any(code in tier_upper for code in req_codes)
                is_tier_name = any(x in tier_upper for x in ["TIER", "GENERIC", "BRAND", "PREFERRED"])
                
                if is_req_code and not is_tier_name:
                    # If requirements is empty, or also looks like requirements (merge them)
                    if not d_reqs:
                        d_reqs = d_tier
                        d_tier = None # Set to None so it can be backfilled if needed, or stay empty
                        # Update the dict
                        drug["drug tier"] = d_tier
                        drug["requirements"] = d_reqs
                        logger.info(f"Fixed Tier/Req swap for {d_name}: Tier='{d_tier}', Reqs='{d_reqs}'")

                # 2. Check Index Page
                rel_p = drug.get("relative_page_number", 1)
                if rel_p in index_pages_relative:
                    continue
                
                # 3. Calculate Absolute Page
                # rel_p is 1-based. chunk_indices is 0-based list of absolute page numbers.
                try:
                    abs_p = chunk_indices[rel_p - 1] + 1 
                except IndexError:
                    abs_p = chunk_indices[0] + 1
                
                drug["absolute_page"] = abs_p
                valid_drugs.append(drug)
        
        # Extract acronyms
        acronyms = []
        if "FormularyAbbreviations" in raw_json:
            acronyms = [{"acronym": i.get("Acronym"), "expansion": i.get("Expansion"), "explanation": i.get("Explanation")} 
                        for i in raw_json["FormularyAbbreviations"]]

        client.files.delete(file_id=uploaded_file.id)
        return valid_drugs, acronyms, "\n\n".join(all_markdowns)

    except Exception as e:
        logger.error(f"Chunk starting at page {chunk_indices[0]+1} failed: {e}")
        return [], [], ""

def process_pdf_with_mistral_ocr_v3(pdf_input_bytes, payer_name=None, filename=None):
    total_costs = {'mistral_pages': 0, 'mistral_cost': 0.0}
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    src_doc = fitz.open(stream=pdf_input_bytes, filetype="pdf")
    total_pages = len(src_doc)
    
    target_indices = _get_pages_to_process(filename, total_pages)
    
    if not target_indices:
        src_doc.close()
        return {"drug_table": [], "acronyms": [], "tiers": []}, "", total_costs

    # Divide indices into chunks of 8 (Mistral's limit for annotations)
    CHUNK_SIZE = 8
    chunks = [target_indices[i:i + CHUNK_SIZE] for i in range(0, len(target_indices), CHUNK_SIZE)]
    
    all_data = {"drug_table": [], "acronyms": [], "tiers": []}
    all_markdown = []
    raw_acronyms = []

    with ThreadPoolExecutor(max_workers=LLM_PAGE_WORKERS) as executor:
        futures = []
        for chunk in chunks:
            # Create a mini-PDF for this chunk
            chunk_doc = fitz.open()
            for p_idx in chunk:
                chunk_doc.insert_pdf(src_doc, from_page=p_idx, to_page=p_idx)
            
            futures.append(executor.submit(process_chunk, client, chunk_doc.tobytes(), chunk, filename))
            chunk_doc.close()

        for future in as_completed(futures):
            chunk_drugs, chunk_acronyms, chunk_md = future.result()
            
            # Convert to internal format
            for drug in chunk_drugs:
                all_data["drug_table"].append({
                    "drug_name": drug.get("Drug Name"),
                    "drug_tier": drug.get("drug tier"),
                    "drug_requirements": drug.get("requirements"),
                    "page_number": drug.get("absolute_page")
                })
            
            raw_acronyms.extend(chunk_acronyms)
            all_markdown.append(chunk_md)

    src_doc.close()
    
    # Reclassify acronyms vs tiers
    ca, ct = _reclassify_definitions(raw_acronyms, [])
    all_data["acronyms"] = ca
    all_data["tiers"] = ct

    total_costs['mistral_pages'] = len(target_indices)
    total_costs['mistral_cost'] = (len(target_indices) / 1000.0) * MISTRAL_OCR_COST_PER_1K_PAGES
    return all_data, "\n\n".join(all_markdown), total_costs

# --- REFINED ACRONYM FILTERING ---

def is_valid_acronym(item):
    acr = str(item.get('acronym') or '').strip()
    exp = str(item.get('expansion') or '').strip()
    
    if not acr or not exp: return False
    
    # 1. Strict uppercase/symbol check (PA, ST, QL, B/D, HI, 90DS)
    # Allows letters, numbers, and slashes, max 5 chars.
    if re.match(r'^[A-Z0-9/]{1,5}$', acr):
        # Prevent acronyms that are just numbers (e.g. "1")
        if acr.isdigit(): return False
        return True
        
    # 2. Specifically allow Tier definitions
    if acr.lower().startswith("tier") or acr.lower().startswith("t "):
        return True

    return False

def process_single_pdf_url_worker(plan_info):
    state_name, payer_name, plan_name, plan_id, payer_id, formulary_url, old_file_hash = plan_info
    log_prefix = f"[{plan_name}]"
    try:
        formulary_url = transform_viewer_url(formulary_url)
        resp = requests.get(formulary_url, timeout=90)
        resp.raise_for_status()
        pdf_bytes = resp.content
        new_hash = calculate_bytes_hash(pdf_bytes)

        cached_data, raw_content = get_cached_result(new_hash)
        if cached_data:
            full_structured_data = cached_data
            costs = {'mistral_pages': 0, 'mistral_cost': 0}
        else:
            full_structured_data, raw_content, costs = process_pdf_with_mistral_ocr_v3(pdf_bytes, payer_name, f"{state_name}_{payer_name}_{plan_name}.pdf")
            cache_result(new_hash, full_structured_data, raw_content)

        update_plan_file_hash(plan_id, new_hash)

        # 1. Acronym Processing (pp_formulary_names)
        raw_defs = full_structured_data.get('acronyms', []) + full_structured_data.get('tiers', [])
        filtered_defs = [d for d in raw_defs if is_valid_acronym(d)]
        all_defs = deduplicate_definitions(filtered_defs)
        if all_defs:
            from database import insert_acronyms_to_ref_table
            insert_acronyms_to_ref_table(all_defs, state_name, payer_name, plan_name, "pp_formulary_names")

        # 2. Drug Table Processing
        drug_table = full_structured_data.get('drug_table', [])
        if drug_table:
            drug_table = _clean_and_propagate_drug_groups(drug_table)
            drug_table = _consolidate_and_clean_drug_table(drug_table)

            processed_records = []
            unique_pairs = set((str(r.get('drug_requirements', '')), str(r.get('drug_tier', ''))) for r in drug_table)
            with get_db_connection() as conn:
                coverage_map = batch_determine_coverage_status(unique_pairs, conn, state_name, payer_name)

            for row in drug_table:
                name = clean_drug_name(str(row.get('drug_name', '')))
                if not name: continue
                reqs = str(row.get('drug_requirements', '') or '').strip()
                tier = str(row.get('drug_tier', '') or '').strip()
                processed_records.append({
                    "id": str(uuid.uuid4()), "plan_id": plan_id, "payer_id": payer_id,
                    "drug_name": name, "state_name": state_name, 
                    "coverage_status": coverage_map.get((reqs, tier), "Covered"),
                    "drug_tier": tier, "drug_requirements": reqs,
                    "page_number": row.get('page_number'),
                    "is_prior_authorization_required": "Yes" if detect_prior_authorization(reqs) else "No",
                    "is_step_therapy_required": "Yes" if detect_step_therapy(reqs) else "No",
                    "is_quantity_limit_applied": "Yes" if "ql" in reqs.lower() else "No",
                    "confidence_score": 0.98, "source_url": formulary_url,
                    "plan_name": plan_name, "payer_name": payer_name,
                    "file_name": f"{state_name}_{payer_name}_{plan_name}.pdf"
                })

            if processed_records:
                unique_records = {}
                for rec in processed_records:
                    key = (rec['plan_id'], rec['drug_name'].lower(), rec['drug_tier'].lower(), rec['drug_requirements'].lower())
                    if key not in unique_records: unique_records[key] = rec
                return 'SUCCESS', plan_name, {"processed_records": list(unique_records.values()), "db_payer_name": payer_name}, costs
        
        return 'SUCCESS', plan_name, {"processed_records": [], "db_payer_name": payer_name}, costs

    except Exception as e:
        logger.error(f"{log_prefix} Worker failed: {e}")
        return 'ERROR', plan_name, str(e), {'mistral_pages': 0, 'mistral_cost': 0}

# --- HELPERS (KEEP UNCHANGED) ---

def _get_pages_to_process(filename: Optional[str], total_pages: int) -> List[int]:
    # IF YOU WANT TO SPEED THIS UP: Change config to skip pages 1-12
    config = PDF_PAGE_PROCESSING_CONFIG
    selected_rule = config.get("default", "all")
    if filename:
        for key, val in config.items():
            if key != "default" and key.lower() in filename.lower():
                selected_rule = val
                break
    if selected_rule == "all": return list(range(total_pages))
    
    pages = set()
    rule_list = [selected_rule] if not isinstance(selected_rule, list) else selected_rule
    for item in rule_list:
        item_str = str(item).strip()
        if '-' in item_str:
            start, end = map(int, item_str.split('-'))
            pages.update(range(start-1, end))
        else: pages.add(int(item_str)-1)
    return [p for p in sorted(list(pages)) if 0 <= p < total_pages]


def _consolidate_and_clean_drug_table(drug_table):
    cons = []; parts = []; last = None
    for d in drug_table:
        name = str(d.get("drug_name") or "").strip()
        tier = str(d.get("drug_tier") or "").strip()
        is_frag = (not tier and (not re.match(r'^[a-zA-Z]', name) or re.match(r'^(oral|tablet|capsule|mg|ml)\b', name, re.IGNORECASE)))
        if not is_frag:
            if last: last['drug_name'] = ' '.join(parts); cons.append(last)
            parts = [name]; last = d
        else:
            if name: parts.append(name)
    if last: last['drug_name'] = ' '.join(parts); cons.append(last)
    return cons

def _clean_and_propagate_drug_groups(drug_table):
    ctx = {}
    for d in drug_table:
        name = str(d.get('drug_name', '')).strip()
        if not name: continue
        base = name.split()[0].lower()
        if d.get('drug_tier') or d.get('drug_requirements'):
            ctx[base] = {'t': d.get('drug_tier'), 'r': d.get('drug_requirements'), 'p': d.get('page_number')}
    for d in drug_table:
        name = str(d.get('drug_name', '')).strip()
        if not name: continue
        base = name.split()[0].lower()
        if base in ctx:
            if not d.get('drug_tier'): d['drug_tier'] = ctx[base]['t']
            if not d.get('drug_requirements'): d['drug_requirements'] = ctx[base]['r']
            if not d.get('page_number'): d['page_number'] = ctx[base]['p']
    return drug_table

def _reclassify_definitions(acronyms_list: list, tiers_list: list) -> Tuple[list, list]:
    ca, ct = [], []
    TIER_KWS = {'aca', 'preventive', 'specialty', 'preferred', 'generic', 'brand'}
    for item in acronyms_list + tiers_list:
        acr = str(item.get('acronym') or '').strip().lower()
        if acr.startswith('tier') or any(kw in acr for kw in TIER_KWS): ct.append(item)
        else: ca.append(item)
    return ca, ct

def deduplicate_definitions(dicts, primary_key='acronym'):
    if not dicts: return []
    merged = {}
    for item in dicts:
        key = str(item.get(primary_key, '')).strip().lower()
        if not key: continue
        if key not in merged or len(str(item.get('expansion', ''))) > len(str(merged[key].get('expansion', ''))):
            merged[key] = item
    return list(merged.values())

def process_pdfs_from_urls_in_parallel():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    plans = get_all_plans_with_formulary_url()
    if not plans: return [], {}
    successfully_processed_plan_ids = []
    with ProcessPoolExecutor(max_workers=PROCESS_COUNT, initializer=initialize_worker) as executor:
        future_to_plan = {executor.submit(process_single_pdf_url_worker, plan): plan for plan in plans}
        for future in as_completed(future_to_plan):
            try:
                status, plan_name, result_data, costs = future.result()
                if status == 'SUCCESS':
                    recs = result_data.get("processed_records", [])
                    if recs:
                        insert_drug_formulary_data(recs)
                        successfully_processed_plan_ids.append(recs[0]['plan_id'])
                    track_mistral_cost(result_data['db_payer_name'], costs['mistral_pages'])
            except Exception as e: logger.error(f"Execution failed: {e}")
    return successfully_processed_plan_ids, {}

def get_all_plans_with_formulary_url():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pd.state_name, py.payer_name, pd.plan_name, pd.plan_id, py.payer_id, pd.formulary_url, pd.file_hash FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id WHERE pd.formulary_url IS NOT NULL AND pd.formulary_url != '' AND pd.status = 'processing'")
        return cursor.fetchall()

def initialize_worker(): pass