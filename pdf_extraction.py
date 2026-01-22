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
# FORMAT 1 (CareSource/Standard): Drug Name | Tier | Restrictions/Limits
# FORMAT 2 (Traditional): Drug Name | Drug Tier | Requirements
# FORMAT 3 (PDL): B,G,O | Comment | P,N,R,NR | Therapeutic Category
# FORMAT 4 (Tier Designation): Drug Name | Tier Designation | dot-marked columns
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
                    "description": """Extract ALL drugs from the formulary drug table. 

CRITICAL: Look for tables with these column headers:
- "Drug Name" | "Tier" | "Drug Requirements/Limits"
- OR "Drug Name" | "Drug Tier" | "Requirements"
- OR "PREFERRED" | "NON-PREFERRED" (two-column format)
- OR "Preferred Agents" | "Non-preferred Agents" with sub-headers "No PA Required" | "PA Required"
- OR similar layouts

FORMAT 1 - THREE-COLUMN TABLE:
- Drug Name (left), Tier (middle - often just numbers like 2, 3, 4, 5), Drug Requirements/Limits (right)
- CATEGORY HEADERS appear as shaded/bold rows (e.g., "ANTIFUNGALS", "ANTIMALARIALS")

FORMAT 2 - TWO-COLUMN PREFERRED/NON-PREFERRED TABLE:
- Left column: "PREFERRED" or "Preferred Agents" (may have sub-header "No PA Required")
- Right column: "NON-PREFERRED" or "Non-preferred Agents" (may have sub-header "PA Required")
- For drugs in PREFERRED column: set preferred_agent="yes", non_preferred_agent="no"
- For drugs in NON-PREFERRED column: set preferred_agent="no", non_preferred_agent="yes"
- IMPORTANT for requirements field:
  * If the column header explicitly says "PA Required": set requirements="PA" for those drugs
  * If the column header does NOT mention PA: leave requirements=null
  * Only set requirements based on what is EXPLICITLY shown in the PDF headers
- Remove asterisks (*) from drug names if present
- IGNORE the 3rd column (Prior Authorization Criteria) - do NOT extract text from it

EXTRACTION RULES:
1. Extract EVERY drug row - do NOT skip any
2. Category headers go in the 'category' field, NOT 'Drug Name'
3. Set 'page_number' to the actual PDF page number where this drug appears""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Drug Name": {
                                "type": "string", 
                                "description": """The COMPLETE drug name from the drug column - extract every single word. This column may contain EITHER:
1. Just the drug name (e.g., 'AMOXICILLIN', 'AMPICILLIN')
2. Drug name WITH chemical salt/form information (e.g., 'fentanyl citrate', 'carbinoxamine maleate', 'tramadol tartrate')
3. Drug name WITH brand name in parentheses (e.g., 'ACTIQ (fentanyl citrate) lozenge', 'FENTORA (fentanyl citrate) buccal tablet')

CRITICAL: Extract the FULL TEXT exactly as shown. DO NOT truncate or omit ANY part of the drug name:
- Include salt forms: citrate, maleate, tartrate, sulfate, hydrochloride, etc.
- Include brand/generic info in parentheses
- Include dosage form words: lozenge, tablet, capsule, solution, etc.
- REMOVE only trailing asterisks (*)

Example: 'ACTIQ (fentanyl citrate) lozenge' should be extracted as 'ACTIQ (fentanyl citrate) lozenge' - NOT truncated to 'ACTIQ (fentanyl' or 'ACTIQ'."""
                            },
                            "Dosage Form/Strength": {
                                "type": ["string", "null"],
                                "description": "The dosage form and strength IF it appears in a SEPARATE second column (between Drug Name and Tier). Examples: 'TAB 250MG', 'CAP 500MG', 'SUS 200/5ML'. In many PDFs, this information is ALREADY included in the Drug Name column, so this field will be null. Only fill this if there's a distinct second column."
                            },
                            "drug tier": {
                                "type": ["string", "null"], 
                                "description": "The tier value from the 'Tier' column. Can be a plain number ('1', '2', '3', '4', '5') OR text ('Tier 1', 'Tier 2', 'Generic', 'Brand', 'Specialty'). Copy EXACTLY as shown in the PDF - if it shows just '2', extract '2'. If it shows 'Tier 2', extract 'Tier 2'. Leave null ONLY for category header rows OR for PREFERRED/NON-PREFERRED format tables."
                            },
                            "requirements": {
                                "type": ["string", "null"], 
                                "description": "The restrictions/requirements. For standard tables: extract from 'Drug Requirements/Limits' column. For PREFERRED/NON-PREFERRED tables: set to 'PA' if the drug is in the 'PA Required' or 'Non-preferred Agents' column, set to null if the drug is in the 'No PA Required' or 'Preferred Agents' column."
                            },
                            "preferred_agent": {
                                "type": ["string", "null"],
                                "enum": ["yes", "no", None],
                                "description": "ONLY USE VALUES: 'yes', 'no', or null. NO OTHER VALUES ALLOWED. For PREFERRED/NON-PREFERRED format tables: Set to 'yes' if the drug is in the PREFERRED or 'No PA Required' column. Set to 'no' if the drug is in the NON-PREFERRED or 'PA Required' column. Leave null for standard tier-based tables. NEVER use '[default]' or any other placeholder text."
                            },
                            "non_preferred_agent": {
                                "type": ["string", "null"],
                                "enum": ["yes", "no", None],
                                "description": "ONLY USE VALUES: 'yes', 'no', or null. NO OTHER VALUES ALLOWED. For PREFERRED/NON-PREFERRED format tables: Set to 'yes' if the drug is in the NON-PREFERRED or 'PA Required' column. Set to 'no' if the drug is in the PREFERRED or 'No PA Required' column. Leave null for standard tier-based tables. NEVER use '[default]' or any other placeholder text."
                            },
                            "BGO": {"type": ["string", "null"], "description": "PDL format only: B=Brand, G=Generic, O=OTC. Leave null for standard formulary tables."},
                            "PNRNR": {"type": ["string", "null"], "description": "PDL format only: P=Preferred, N=Non-Preferred, R/NR. Leave null for standard formulary tables."},
                            "Specialty": {"type": ["boolean", "null"], "description": "True if marked as Specialty drug. Leave null if not indicated."},
                            "PriorAuthorization": {"type": ["boolean", "null"], "description": "True if 'PA' appears in requirements column."},
                            "StepTherapy": {"type": ["boolean", "null"], "description": "True if 'ST' appears in requirements column."},
                            "DispensingLimits": {"type": ["boolean", "null"], "description": "True if 'QL' appears in requirements column."},
                            "category": {
                                "type": ["string", "null"], 
                                "description": "Category header text from shaded/bold rows that span all columns. Examples: 'ANTICONVULSANTS – CARBAMAZEPINE DERIVATIVES', 'ANTICONVULSANTS – FIRST GENERATION', 'ANTIFUNGALS'. These are NOT drug names."
                            },
                            "page_number": {
                                "type": "integer", 
                                "description": """CRITICAL: Carefully determine which page each drug appears on within this document. 
Pages are numbered 1, 2, 3, 4 based on their position in this document chunk.
- Look for visual page breaks, page footers, or page headers to identify where pages end
- If a table continues across pages, drugs AFTER a page break should have the NEXT page number
- Count pages from the start of this document (first page = 1, second page = 2, etc.)
- Do NOT default all drugs to page 1 - carefully identify the actual page for each drug"""
                            },
                            "pa_form_link": {"type": ["string", "null"], "description": "PA Form Link URL if present in the table."}
                        },
                        "required": ["Drug Name", "page_number"]
                    }
                },
                "FormularyAbbreviations": {
                    "type": "array",
                    "description": """Extract ALL abbreviation/legend definitions from ANYWHERE in the document.
                    
Look for legends in: page headers, footers, sidebar text, or dedicated sections.
Common patterns: 'ST = Step Therapy', 'PA = Prior Authorization', 'QL = Quantity Limit'
                    
Extract EVERY abbreviation definition found.""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Acronym": {"type": "string", "description": "The abbreviation code. Examples: 'ST', 'PA', 'QL', 'SP', 'Tier 1', 'Generic'."},
                            "Expansion": {"type": "string", "description": "What the abbreviation stands for. Examples: 'Step Therapy', 'Prior Authorization', 'Quantity Limit'."},
                            "Explanation": {"type": ["string", "null"], "description": "Additional explanation if provided in the legend."}
                        },
                        "required": ["Acronym", "Expansion"]
                    }
                }
            },
            "required": ["DrugInformation"]
        }
    }
}


def _build_requirements_from_item(item):
    """
    Build drug_requirements from multiple format types:
    - Traditional requirements text (PRIORITY - exact values like "QL (2 EA per 30 days)")
    - Tier Designation format (boolean dot columns - fallback)
    - PDL format (BGO + PNRNR)
    """
    # PRIORITY 1: Traditional requirements text (contains exact QL/PA/ST values)
    requirements_text = item.get("requirements")
    if requirements_text and requirements_text.strip():
        return requirements_text.strip()
    
    # PRIORITY 2: Check for Tier Designation format (dot-marked columns) as FALLBACK
    specialty = item.get("Specialty")
    prior_auth = item.get("PriorAuthorization")
    step_therapy = item.get("StepTherapy")
    dispensing_limits = item.get("DispensingLimits")

    tier_parts = []
    if specialty is True:
        tier_parts.append("Specialty")
    if prior_auth is True:
        tier_parts.append("PA")  # Shortened for consistency
    if step_therapy is True:
        tier_parts.append("ST")  # Shortened for consistency
    if dispensing_limits is True:
        tier_parts.append("QL")  # Shortened - actual value should be in requirements text

    if tier_parts:
        return ", ".join(tier_parts)

    # PRIORITY 3: PDL format (BGO + PNRNR)
    bgo = item.get("BGO", "").strip() if item.get("BGO") else ""
    pnrnr = item.get("PNRNR", "").strip() if item.get("PNRNR") else ""
    if bgo or pnrnr:
        parts = [p for p in [bgo, pnrnr] if p]
        return "; ".join(parts) if parts else None

    return None


def _extract_drug_from_item(item: dict, page_number: int) -> dict:
    """
    Extract drug data from an OCR item into a standardized dictionary format.
    Centralizes the drug extraction logic used in multiple places.
    
    Combines Drug Name + Dosage Form/Strength into a single drug_name field.
    Example: "AMOXICILLIN" + "TAB 875MG" → "AMOXICILLIN TAB 875MG"
    """
    drug_name = item.get("Drug Name") or ""
    dosage_form = item.get("Dosage Form/Strength") or ""
    
    # Combine drug name and dosage form/strength if both present
    if drug_name and dosage_form:
        combined_name = f"{drug_name.strip()} {dosage_form.strip()}"
    else:
        combined_name = drug_name.strip() if drug_name else ""
    
    # Remove trailing asterisks from drug names (common in PREFERRED/NON-PREFERRED format)
    if combined_name:
        combined_name = combined_name.rstrip('*').strip()
    
    # Sanitize preferred_agent and non_preferred_agent - ONLY allow "yes" or "no"
    # Convert any other values (like "[default]", "default", etc.) to None
    def sanitize_agent_value(value):
        if value is None:
            return None
        value_str = str(value).strip().lower()
        if value_str == "yes":
            return "yes"
        elif value_str == "no":
            return "no"
        else:
            # Any other value (including "[default]", "default", etc.) becomes None
            return None
    
    return {
        "drug_name": combined_name if combined_name else None,
        "drug_tier": item.get("drug tier"),
        "drug_requirements": _build_requirements_from_item(item),
        "category": item.get("category"),
        "page_number": page_number,
        "preferred_agent": sanitize_agent_value(item.get("preferred_agent")),
        "non_preferred_agent": sanitize_agent_value(item.get("non_preferred_agent"))
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
    Parse and repair malformed JSON from LLM/OCR outputs.
    Uses json_repair library if available, with fallback to basic cleanup.
    Handles common issues like:
    - Unquoted values with special characters ($$$, etc.)
    - Truncated JSON arrays
    - Missing closing brackets
    """
    default_output = {"drug_table": [], "acronyms": [], "tiers": []}

    if not isinstance(json_string, str) or not json_string.strip():
        return default_output

    # Remove markdown code fences
    json_string = re.sub(r'^```(?:json)?\s*', '', json_string.strip())
    json_string = re.sub(r'\s*```$', '', json_string.strip())

    # Pre-process: Fix common malformed JSON patterns
    # Fix unquoted values starting with $ (like $$$ Non-preferred)
    json_string = re.sub(r':\s*(\$+[^"}\],]+)"', r': "\1"', json_string)
    json_string = re.sub(r':\s*(\$+[^"}\],\n]+)\s*([,}\]])', r': "\1"\2', json_string)

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

    # Fallback 1: Try to extract DrugInformation array directly
    drug_table = []
    try:
        # Find DrugInformation array and extract individual objects
        drug_info_match = re.search(r'"DrugInformation"\s*:\s*\[', json_string)
        if drug_info_match:
            start = drug_info_match.end()
            # Extract all complete JSON objects from the array
            obj_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
            for match in obj_pattern.finditer(json_string[start:]):
                try:
                    obj = json.loads(match.group())
                    if obj.get("Drug Name"):
                        # Combine Drug Name + Dosage Form/Strength
                        drug_name = obj.get("Drug Name", "")
                        dosage_form = obj.get("Dosage Form/Strength", "")
                        if drug_name and dosage_form:
                            combined_name = f"{drug_name.strip()} {dosage_form.strip()}"
                        else:
                            combined_name = drug_name.strip() if drug_name else ""
                        
                        drug_table.append({
                            "drug_name": combined_name,
                            "drug_tier": obj.get("drug tier"),
                            "drug_requirements": obj.get("requirements"),
                            "category": obj.get("category"),
                            "page_number": obj.get("page_number")
                        })
                except json.JSONDecodeError:
                    continue
            
            if drug_table:
                logger.info(f"✅ Fallback extraction recovered {len(drug_table)} drugs from truncated JSON")
                return {"drug_table": drug_table, "acronyms": [], "tiers": []}
    except Exception as e:
        logger.debug(f"Fallback drug extraction failed: {e}")

    # Fallback 2: Basic JSON parsing with cleanup
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
            # JSON is truncated - try to close it
            json_str = json_string[start_idx:]
            # Close any open arrays and objects
            open_brackets = json_str.count('[') - json_str.count(']')
            open_braces = json_str.count('{') - json_str.count('}')
            json_str = json_str + (']' * open_brackets) + ('}' * open_braces)
        else:
            json_str = json_string[start_idx:end_idx + 1]
        
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        result = json.loads(json_str)
        return _sanitize_output(result, default_output)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        return default_output


def _is_index_entry(item: dict) -> bool:
    """
    Check if a single drug entry looks like it came from an index page.
    Returns True if the entry looks like an index entry, False otherwise.
    """
    drug_name = item.get("drug_name", "") or ""
    
    # Index entry indicators:
    # 1. Drug name ends with page number (e.g., "BACITRACIN...13")
    if RE_PAGE_NUMBER_END.search(drug_name):
        return True
    
    # 2. Drug name contains dot leaders (.......)
    if RE_DOT_LEADERS.search(drug_name):
        return True
    
    return False


def _is_extracted_data_from_index_page(drug_table: List[dict]) -> bool:
    """
    Detect if extracted drug data appears to come from an index/table of contents page.
    Returns True ONLY if the MAJORITY of data looks like an index, False otherwise.
    
    NOTE: This function now uses higher thresholds to avoid discarding valid drug data
    when a chunk contains a mix of index and drug pages.
    """
    if not drug_table or len(drug_table) < 10:  # Increased minimum from 5 to 10
        return False

    total = len(drug_table)
    
    # Count various index indicators
    index_entries = 0
    tier_counts = {}
    no_requirements = 0
    no_dosage_form = 0
    
    for item in drug_table:
        drug_name = item.get("drug_name", "") or ""
        tier = item.get("drug_tier", "") or ""
        req = item.get("drug_requirements")
        
        # Check if this specific entry looks like an index entry
        if _is_index_entry(item):
            index_entries += 1
        
        # Pattern: No dosage/form info
        if not RE_DOSAGE_FORM.search(drug_name):
            no_dosage_form += 1
        
        # Count tier values
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # No requirements
        if not req:
            no_requirements += 1

    # Detection Rule 1: Very high percentage of obvious index entries (>50% instead of 20%)
    if total > 0 and index_entries / total >= 0.50:
        logger.info(f"🚫 INDEX PAGE DETECTED: {index_entries}/{total} entries are obvious index entries")
        return True
    
    # Detection Rule 2: Very uniform tier + no requirements + no dosage (stricter thresholds)
    if tier_counts and total > 20:  # Only check for larger datasets
        most_common_tier = max(tier_counts.values())
        tier_uniformity = most_common_tier / total
        
        # If >95% have same tier AND >98% no requirements AND >95% no dosage info
        if tier_uniformity >= 0.95 and no_requirements / total >= 0.98 and no_dosage_form / total >= 0.95:
            logger.info(f"🚫 INDEX PAGE DETECTED: {tier_uniformity*100:.0f}% uniform tier, "
                       f"{no_requirements}/{total} no requirements, {no_dosage_form}/{total} no dosage info")
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
            # - Has NO preferred_agent/non_preferred_agent (not from PREFERRED/NON-PREFERRED format)
            # - Is VERY short and looks like a simple continuation (not a new drug)
            # - Doesn't contain common drug identifiers
            has_preference_info = next_item.get("preferred_agent") or next_item.get("non_preferred_agent")
            
            # More conservative checks - only merge if it's truly a fragment
            looks_like_complete_drug = (
                re.search(r'\d+\s*(mg|ml|mcg|unit|%)', next_name, re.IGNORECASE) or  # Has dosage
                re.search(r'\b(tablet|capsule|solution|suspension|patch|cream|gel|spray|injection|syrup|liquid|powder|ER|XR|SR|CR|IR)\b', next_name, re.IGNORECASE) or  # Has dosage form
                re.search(r'\([^)]+\)', next_name) or  # Has parentheses (likely brand/generic info)
                re.match(r'^[A-Z]', next_name) or  # Starts with capital letter (new drug name)
                len(next_name) > 15  # Too long to be a fragment
            )
            
            is_fragment = (
                not next_item.get("drug_tier") and
                not next_item.get("drug_requirements") and
                not has_preference_info and  # Don't merge if it has preferred/non-preferred info
                next_name and
                not looks_like_complete_drug  # Only merge if it doesn't look like a complete drug
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

    # Stage 3: Filter invalid entries AND individual index entries
    filtered = []
    filtered_out = []
    index_entries_removed = 0
    for item in result:
        name = item.get("drug_name", "") or ""
        # Skip if name is too short or just a number
        if len(name) < 3 or name.isdigit():
            filtered_out.append(name)
            continue
        # Skip individual index entries (page numbers, dot leaders)
        if _is_index_entry(item):
            index_entries_removed += 1
            continue
        filtered.append(item)

    if filtered_out:
        logger.debug(f"🧹 Filtered out {len(filtered_out)} invalid entries: {filtered_out[:5]}")
    if index_entries_removed > 0:
        logger.info(f"🧹 Removed {index_entries_removed} individual index entries from drug table")

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
