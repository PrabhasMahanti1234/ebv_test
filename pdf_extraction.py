"""
pdf_extraction.py - Drug and Formulary Data Extraction Functions

This module contains functions that are frequently modified when:
- New PDF template formats are added
- OCR schema needs updating for new table structures
- Drug/acronym extraction logic needs tweaking

Functions in this file:
- _build_requirements_from_item() - Build requirements from multiple formats
- _extract_drug_from_item() - Standardized drug extraction
- _extract_acronym_from_item() - Standardized acronym extraction
- is_index_content() - Unified index detection
- OCR_ANNOTATION_SCHEMA - OCR schema constant
- robust_json_repair() - JSON parsing and repair
- _is_extracted_data_from_index_page() - Index detection for extracted data
- _consolidate_and_clean_drug_table() - Drug table cleaning
- _clean_and_propagate_drug_groups() - Tier/requirement propagation
- _sanitize_output() - Output sanitization
- is_index_page() - Index page detection for markdown
- extract_metadata_from_filename() - Filename parsing
- _parse_and_split_tier_definitions() - Tier parsing
- _reclassify_definitions() - Definition classification
- is_valid_formulary_definition() - Definition validation
"""

import re
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for index/drug detection (PERFORMANCE OPTIMIZATION)
RE_PAGE_NUMBER_END = re.compile(r'[\.\s]+\d{1,3}\s*$')
RE_DOT_LEADERS = re.compile(r'\.{3,}')
RE_DOSAGE_FORM = re.compile(r'\d+\s*(mg|ml|mcg|unit|%|tablet|capsule|solution|cream|gel|patch|spray)', re.IGNORECASE)

# json_repair for robust LLM output parsing
try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logger.warning("json_repair not available. Using fallback JSON parsing.")


# =============================================================================
# OCR ANNOTATION SCHEMA - Supports Multiple PDF Formats
# =============================================================================
# FORMAT 1 (Traditional): Drug Name | Drug Tier | Requirements columns
# FORMAT 2 (PDL): B,G,O | Comment | P,N,R,NR | Therapeutic Category
# FORMAT 3 (Tier Designation): Drug Name | Tier Designation | dot-marked columns
# FORMAT 4 (Hennepin/Product): PRODUCT DESCRIPTION | TIER | LIMITS & RESTRICTIONS
# =============================================================================
OCR_ANNOTATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "drug_extraction_schema",
        "schema": {
            "type": "object",
            "title": "StructuredData",
            "properties": {
                "DrugInformation": {
                    "type": "array",
                    "description": "Extract drugs from formulary tables. Supports multiple formats: 1) Traditional: Drug Name|Drug Tier|Requirements. 2) PDL: B,G,O|P,N,R,NR. 3) Tier Designation with dots. 4) Product format: PRODUCT DESCRIPTION|TIER|LIMITS & RESTRICTIONS where TIER can be 'Generics', 'Preferred Generics', 'Non-Preferred', etc. SKIP index pages and TOC.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Drug Name": {"type": "string", "description": "Complete drug name with form/dosage. Map from PRODUCT DESCRIPTION column if present."},
                            "drug tier": {"type": ["string", "null"], "description": "From Drug Tier, TIER column (Generics, Preferred Generics, Non-Preferred, Preferred Brand), or Tier Designation (NP, P, NC)."},
                            "requirements": {"type": ["string", "null"], "description": "From Requirements, LIMITS & RESTRICTIONS (e.g. 'QPD 6.0 per day', 'PA', 'ST', 'QL'), or combined dot-marked columns."},
                            "BGO": {"type": ["string", "null"], "description": "PDL format: B=Brand, G=Generic, O=OTC."},
                            "PNRNR": {"type": ["string", "null"], "description": "PDL format: P=Preferred, N=Non-Preferred, R/NR."},
                            "Specialty": {"type": ["boolean", "null"], "description": "True if Specialty column has dot."},
                            "PriorAuthorization": {"type": ["boolean", "null"], "description": "True if Prior Authorization column has dot or PA in limits."},
                            "StepTherapy": {"type": ["boolean", "null"], "description": "True if Step Therapy column has dot or ST in limits."},
                            "DispensingLimits": {"type": ["boolean", "null"], "description": "True if Dispensing Limits column has dot or QL/QPD in limits."},
                            "category": {"type": ["string", "null"], "description": "Therapeutic category, section header, or drug class (e.g. SALICYLATES)."},
                            "page_number": {"type": ["integer", "null"], "description": "Page number where drug is found."},
                            "pa_form_link": {"type": ["string", "null"], "description": "PA Form Link URL if present."}
                        }
                    }
                },
                "FormularyAbbreviations": {
                    "type": "array",
                    "description": "Extract abbreviation/legend definitions. Include QPD, QL, PA, ST definitions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Acronym": {"type": "string", "description": "The abbreviation code (e.g., QPD, QL, PA, ST)."},
                            "Expansion": {"type": "string", "description": "What it stands for (e.g., Quantity Per Day, Quantity Limit)."},
                            "Explanation": {"type": ["string", "null"], "description": "Additional explanation if available."}
                        }
                    }
                }
            }
        }
    }
}


def _build_requirements_from_item(item):
    """
    Build drug_requirements from multiple format types:
    - Tier Designation format (boolean dot columns)
    - PDL format (BGO + PNRNR)
    - Traditional requirements text
    """
    # Check for Tier Designation format (dot-marked columns)
    specialty = item.get("Specialty")
    prior_auth = item.get("PriorAuthorization")
    step_therapy = item.get("StepTherapy")
    dispensing_limits = item.get("DispensingLimits")

    # Priority 1: Tier Designation format
    tier_parts = []
    if specialty is True:
        tier_parts.append("Specialty")
    if prior_auth is True:
        tier_parts.append("Prior Authorization")
    if step_therapy is True:
        tier_parts.append("Step Therapy")
    if dispensing_limits is True:
        tier_parts.append("Dispensing Limits")

    if tier_parts:
        return ", ".join(tier_parts)

    # Priority 2: PDL format (BGO + PNRNR)
    bgo = item.get("BGO", "").strip() if item.get("BGO") else ""
    pnrnr = item.get("PNRNR", "").strip() if item.get("PNRNR") else ""
    if bgo or pnrnr:
        parts = [p for p in [bgo, pnrnr] if p]
        return "; ".join(parts) if parts else None

    # Priority 3: Traditional requirements text
    return item.get("requirements") or None


def _extract_drug_from_item(item: dict, page_number: int) -> dict:
    """
    Extract drug data from an OCR item into a standardized dictionary format.
    Centralizes the drug extraction logic used in multiple places.
    """
    return {
        "drug_name": item.get("Drug Name"),
        "drug_tier": item.get("drug tier"),
        "drug_requirements": _build_requirements_from_item(item),
        "category": item.get("category"),
        "page_number": page_number
    }


def _extract_acronym_from_item(item: dict) -> dict:
    """Extract acronym data from an OCR item into a standardized dictionary format."""
    return {
        "acronym": item.get("Acronym"),
        "expansion": item.get("Expansion"),
        "explanation": item.get("Explanation")
    }


def is_index_content(content, content_type='markdown'):
    """
    Unified index/TOC detection that works with both raw markdown and extracted drug data.

    Args:
        content: Either markdown string or list of drug dictionaries
        content_type: 'markdown' or 'drug_table'

    Returns:
        True if content appears to be from an index/TOC page
    """
    if content_type == 'drug_table':
        return _is_extracted_data_from_index_page(content)
    else:
        return is_index_page(content)


def _sanitize_output(parsed_data, default_output):
    """
    Ensures the parsed output conforms to the expected dictionary structure
    with the correct keys, returning empty lists for any missing keys.
    """
    if not isinstance(parsed_data, dict):
        return default_output

    return {
        "drug_table": parsed_data.get("drug_table", parsed_data.get("DrugInformation", [])),
        "acronyms": parsed_data.get("acronyms", parsed_data.get("FormularyAbbreviations", [])),
        "tiers": parsed_data.get("tiers", [])
    }


def robust_json_repair(json_string: str):
    """
    Parse and repair malformed JSON from LLM outputs.
    Uses json_repair library if available, with fallback to basic cleanup.
    """
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}

    if not isinstance(json_string, str) or not json_string.strip():
        return default_output

    # Remove markdown code fences
    json_string = re.sub(r'^```(?:json)?\s*', '', json_string.strip())
    json_string = re.sub(r'\s*```$', '', json_string.strip())

    # Try json_repair library first (most robust)
    if JSON_REPAIR_AVAILABLE:
        try:
            result = repair_json(json_string, return_objects=True)
            if isinstance(result, dict):
                return _sanitize_output(result, default_output)
            elif isinstance(result, list) and len(result) > 0:
                first_item = result[0] if isinstance(result[0], dict) else {}
                return _sanitize_output(first_item, default_output)
        except Exception as e:
            logger.debug(f"json_repair failed: {e}, trying fallback...")

    # Fallback: Basic JSON parsing with cleanup
    try:
        start_idx = json_string.find('{')
        if start_idx == -1:
            return default_output

        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(json_string)):
            if json_string[i] == '{':
                brace_count += 1
            elif json_string[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            return default_output

        json_str = json_string[start_idx:end_idx + 1]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        result = json.loads(json_str)
        return _sanitize_output(result, default_output)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        return default_output


def _is_extracted_data_from_index_page(drug_table: List[dict]) -> bool:
    """
    Detect if extracted drug data appears to come from an index/table of contents page.
    Returns True if the data looks like an index, False otherwise.

    Index page indicators:
    - Drug names contain page numbers (e.g., "BACITRACIN...13" or "BACLOFEN 260")
    - Drug names contain dot leaders (........)
    - High percentage have same tier (hallucinated uniformly)
    - No requirements but uniform tier
    - Drug names are just drug names without form/dosage info
    """
    if not drug_table or len(drug_table) < 5:
        return False

    total = len(drug_table)
    
    # Count various index indicators
    page_number_at_end = 0
    dot_leader_pattern = 0
    no_dosage_form = 0
    tier_counts = {}
    no_requirements = 0
    
    for item in drug_table:
        drug_name = item.get("drug_name", "") or ""
        tier = item.get("drug_tier", "") or ""
        req = item.get("drug_requirements")
        
        # Pattern 1: Drug name ends with page number - USE COMPILED REGEX
        if RE_PAGE_NUMBER_END.search(drug_name):
            page_number_at_end += 1
        
        # Pattern 2: Dot leaders in name - USE COMPILED REGEX
        if RE_DOT_LEADERS.search(drug_name):
            dot_leader_pattern += 1
        
        # Pattern 3: No dosage/form info - USE COMPILED REGEX
        if not RE_DOSAGE_FORM.search(drug_name):
            no_dosage_form += 1
        
        # Pattern 4: Count tier values to detect uniform hallucination
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Pattern 5: No requirements
        if not req:
            no_requirements += 1

    # Detection Rule 1: High percentage of page numbers at end
    if page_number_at_end / total >= 0.2:
        logger.info(f"🚫 INDEX PAGE DETECTED: {page_number_at_end}/{total} names have page numbers at end")
        return True
    
    # Detection Rule 2: Dot leaders present
    if dot_leader_pattern / total >= 0.1:
        logger.info(f"🚫 INDEX PAGE DETECTED: {dot_leader_pattern}/{total} names have dot leaders")
        return True
    
    # Detection Rule 3: Very uniform tier (likely hallucinated) + no requirements + no dosage info
    if tier_counts:
        most_common_tier = max(tier_counts.values())
        tier_uniformity = most_common_tier / total
        
        # If >80% have same tier, no requirements, and no dosage info → likely index page
        if tier_uniformity >= 0.8 and no_requirements / total >= 0.9 and no_dosage_form / total >= 0.7:
            logger.info(f"🚫 INDEX PAGE DETECTED: {tier_uniformity*100:.0f}% uniform tier, "
                       f"{no_requirements}/{total} no requirements, {no_dosage_form}/{total} no dosage info")
            return True
    
    # Detection Rule 4: Almost all entries have no dosage/form info
    if no_dosage_form / total >= 0.85:
        logger.info(f"🚫 INDEX PAGE DETECTED: {no_dosage_form}/{total} entries have no dosage/form info")
        return True

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

    initial_count = len(drug_table)
    logger.info(f"🧹 Cleaning drug table: Starting with {initial_count} raw entries")

    # Stage 1: Consolidate fragmented entries
    # NOTE: We're now being less aggressive about merging to avoid losing valid drugs
    consolidated = []
    i = 0
    merged_count = 0
    while i < len(drug_table):
        current = drug_table[i].copy()
        drug_name = current.get("drug_name", "") or ""

        # Look ahead for fragments (lines that are continuations)
        j = i + 1
        while j < len(drug_table):
            next_item = drug_table[j]
            next_name = next_item.get("drug_name", "") or ""

            # A fragment is an entry that:
            # - Has NO tier AND NO requirements (clearly part of previous line)
            # - Is short (like "aspirin" continuation)
            # - Doesn't look like a real drug name (no dosage info, no form)
            is_fragment = (
                not next_item.get("drug_tier") and
                not next_item.get("drug_requirements") and
                len(next_name) < 30 and  # Reduced from 50
                next_name and
                not re.search(r'\d+\s*(mg|ml|mcg|unit|%)', next_name, re.IGNORECASE) and  # No dosage
                not re.match(r'^[a-z]+\s+\d', next_name)  # Not "drug 100mg" pattern
            )

            if is_fragment and next_name:
                drug_name = drug_name + " " + next_name
                merged_count += 1
                j += 1
            else:
                break

        current["drug_name"] = drug_name.strip()
        consolidated.append(current)
        i = j

    logger.info(f"🧹 After consolidation: {len(consolidated)} entries (merged {merged_count} fragments)")

    # Stage 2: Propagate tier/requirements
    result = _clean_and_propagate_drug_groups(consolidated)

    # Stage 3: Filter invalid entries
    filtered = []
    filtered_out = []
    for item in result:
        name = item.get("drug_name", "") or ""
        # Keep if name is substantial
        if len(name) >= 3 and not name.isdigit():
            filtered.append(item)
        else:
            filtered_out.append(name)

    if filtered_out:
        logger.debug(f"🧹 Filtered out {len(filtered_out)} invalid entries: {filtered_out[:5]}")

    logger.info(f"🧹 Final drug count: {len(filtered)} (filtered {len(result) - len(filtered)} invalid entries)")

    return filtered


def _clean_and_propagate_drug_groups(drug_table: List[dict]) -> List[dict]:
    """
    Corrected function that fills in missing context (tier/requirements) for
    fragmented drug entries without incorrectly overwriting valid, extracted data.
    """
    if not drug_table:
        return []

    result = []
    current_tier = None
    current_category = None

    for item in drug_table:
        new_item = item.copy()

        # Update current tier if this item has one
        if new_item.get("drug_tier"):
            current_tier = new_item["drug_tier"]
        elif current_tier and not new_item.get("drug_tier"):
            # Propagate tier if missing
            new_item["drug_tier"] = current_tier

        # Update current category if this item has one
        if new_item.get("category"):
            current_category = new_item["category"]
        elif current_category and not new_item.get("category"):
            new_item["category"] = current_category

        result.append(new_item)

    return result


def is_index_page(markdown: str) -> bool:
    """
    Detect if a page is an index/table of contents with enhanced detection logic.
    Returns True if index, False otherwise.

    NOTE: Consider consolidating with _is_extracted_data_from_index_page()
    which does similar detection on extracted drug data.
    """
    if not markdown or len(markdown.strip()) < 50:
        return False

    markdown_lower = markdown.lower()
    lines = markdown.split('\n')

    # Quick check for explicit index/TOC indicators
    explicit_indicators = [
        "table of contents", "alphabetical index", "drug index",
        "index of drugs", "formulary index"
    ]
    for indicator in explicit_indicators:
        if indicator in markdown_lower[:500]:
            logger.info(f"Detected index page: Found '{indicator}'")
            return True

    # Check for page number pattern at end of lines
    page_number_pattern = re.compile(r'\.{2,}\s*\d+\s*$|\s+\d{2,3}\s*$')
    total_lines = 0
    page_number_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('|') or stripped.startswith(':'):
            continue
        total_lines += 1
        if page_number_pattern.search(stripped):
            page_number_lines += 1

    if total_lines > 0 and page_number_lines / total_lines >= 0.30:
        logger.info(f"Detected index page: {page_number_lines}/{total_lines} lines have page numbers")
        return True

    return False


def extract_metadata_from_filename(filename):
    """Extract state, payer, and plan name from filename"""
    if not filename:
        return None, None, None

    parts = filename.replace('.pdf', '').split('_')
    if len(parts) >= 3:
        return parts[0], parts[1], '_'.join(parts[2:])
    return None, None, None


def _parse_and_split_tier_definitions(tier_list: list) -> list:
    """
    Parses tier definitions where the acronym and expansion might be combined in one field.
    This corrects LLM outputs like {"acronym": "Tier 1 - Generic", "expansion": None}
    into {"acronym": "Tier 1", "expansion": "Generic"}.
    """
    result = []
    for item in tier_list:
        if not isinstance(item, dict):
            continue

        acronym = item.get("acronym", "") or ""
        expansion = item.get("expansion", "") or ""

        # Check if acronym contains the expansion
        if " - " in acronym and not expansion:
            parts = acronym.split(" - ", 1)
            acronym = parts[0].strip()
            expansion = parts[1].strip() if len(parts) > 1 else ""

        result.append({
            "acronym": acronym,
            "expansion": expansion,
            "explanation": item.get("explanation")
        })

    return result


def _reclassify_definitions(acronyms_list: list, tiers_list: list) -> tuple:
    """
    Sorts definitions into acronyms or tiers based on heuristics to correct LLM misclassifications.
    """
    final_acronyms = []
    final_tiers = []

    tier_patterns = re.compile(r'^tier\s*\d|^level\s*\d|^t\d', re.IGNORECASE)

    for item in acronyms_list + tiers_list:
        if not isinstance(item, dict):
            continue

        acronym = item.get("acronym", "") or ""

        if tier_patterns.match(acronym):
            final_tiers.append(item)
        else:
            final_acronyms.append(item)

    return final_acronyms, final_tiers


def is_valid_formulary_definition(item: dict) -> bool:
    """
    Automatically detects if an extracted item is a valid formulary definition.
    """
    if not isinstance(item, dict):
        return False

    acronym = item.get("acronym", "") or ""
    expansion = item.get("expansion", "") or ""

    # Must have both acronym and expansion
    if not acronym or not expansion:
        return False

    # Acronym should be short
    if len(acronym) > 30:
        return False

    return True
