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
- filter_index_entries_from_mistral_response() - Filter index entries from OCR response
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
                    "description": """Extract ALL drugs from the page.

🎨 CRITICAL COLOR DETECTION INSTRUCTION:
IF you see COLORED BADGES or COLORED TEXT for requirements (QL, PA, ST, etc.), you MUST extract the colors in the 'badge_colors' field.
Look for: purple badges, brown/orange badges, red badges, green badges, blue badges, etc.
Example: If "QL" appears in a purple pill-shaped badge, and "PA" in a brown badge → badge_colors: {"QL": "purple", "PA": "brown"}
If ALL text is standard BLACK with NO colored badges → leave badge_colors as NULL.

KEY INSTRUCTION: This page contains a list of drugs, tiers, and requirements. 
IT MAY NOT LOOK LIKE A TRADITIONAL TABLE.
IT MAY NOT HAVE HEADERS.
IT MAY LOOK LIKE A VERTICAL LIST.

YOU MUST EXTRACT EVERY DRUG ENTRY YOU SEE.

RECOGNITION PATTERNS:
1. Standard Table: "Drug Name" | "Tier" | "Restrictions"
2. List Format: Drug Name on one line, Tier/Requirements on next line.
3. Compact List: Drug Name followed by Tier (e.g. "5") and Requirements (e.g. "QL")

EXAMPLE OF LIST FORMAT (Common in this document):
  TREMFYA SOSY 100mg/ml
  QL (1 syringe / 28 days)
  5
  QL NM PA
  ==> Extract as: Drug Name="TREMFYA SOSY 100mg/ml", Tier="5", Requirements="QL (1 syringe / 28 days); QL NM PA"

EXTRACTION RULES:
1. Extract EVERY drug row - do NOT skip any.
2. Category headers go in the 'category' field.
3. Tier values: Look for "Tier 1", "Tier 2" OR just numbers "1", "2", "3", "4", "5".
4. Restrictions: Look for "QL", "PA", "ST", "NM", "LA", "B/D".
5. DO NOT SKIP PAGES. Extract whatever looks like a drug list.
6. ALWAYS check for colored badges/text and extract colors if present.""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Drug Name": {
                                "type": "string", 
                                "description": """The complete drug name from the first/left-most column. This column may contain EITHER:
1. Just the drug name (e.g., 'AMOXICILLIN', 'AMPICILLIN')
2. Drug name WITH dosage form inline (e.g., 'carbinoxamine maleate oral liquid', 'carbinoxamine maleate oral tablet 4 mg', 'azelastine nasal spray non-aerosol 137 mcg (0.1 %)')

CRITICAL EXCLUSIONS:
- DO NOT include "QL", "PA", "ST", "Limits", or "(... per ... days)" in the Drug Name field, even if they appear in the first column or directly below the drug name.
- Example: If text is "TREMFYA SOSY 100mg/ml\nQL (1 syringe / 28 days)", extract ONLY "TREMFYA SOSY 100mg/ml" as Drug Name. The "QL..." part goes to 'requirements'.
- Example: If text is "XYZAL\nQL", extract ONLY "XYZAL".
- EXTRACT THE FULL TEXT from the first column, including any dosage information, BUT STOP before any coverage restrictions.
- IF THE COLUMN HEADER IS "pub drug name", EXTRACT DATA FROM THAT COLUMN.
- IF THERE IS A SEPARATE COLUMN FOR DOSAGE/STRENGTH (e.g. "pub strength", "strength"), AND IT IS NOT EXTRACTED IN THE 'Dosage Form/Strength' FIELD, APPEND IT TO THIS FIELD."""
                            },
                            "Dosage Form/Strength": {
                                "type": ["string", "null"],
                                "description": "The dosage form and strength IF it appears in a SEPARATE second column (between Drug Name and Tier). Examples: 'TAB 250MG', 'CAP 500MG', 'SUS 200/5ML'. In many PDFs, this information is ALREADY included in the Drug Name column, so this field will be null. Only fill this if there's a distinct second column. IF THE COLUMN HEADER IS 'pub strength' OR 'pub dosage', EXTRACT DATA FROM THAT COLUMN."
                            },
                            "BrandOrGeneric": {
                                "type": ["string", "null"],
                                "description": "The value from the 'Brand or Generic' column if present (often the 2nd column). Values like 'B', 'G', 'Brand', 'Generic'. EXTRACT THIS SEPARATELY so it does not get mixed into Drug Name or Tier."
                            },
                            "drug tier": {
                                "type": ["string", "null"], 
                                "description": "The tier/drug type value. Copy exactly as shown. Can be: 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5', OR 'Generic', 'Brand', 'Specialty'. If the column contains costs (e.g., '$0/$1.60'), IGNORE the costs and extract the tier classification (e.g., 'Tier 1 - Generic', 'Tier 2'). IF THE COLUMN HEADER IS 'pub tier', EXTRACT DATA FROM THAT COLUMN. Leave null ONLY for category header rows. DO NOT INCLUDE 'B' or 'G' from the Brand/Generic column here."
                            },
                            "requirements": {
                                "type": ["string", "null"], 
                                "description": "The restrictions from the Restrictions/Limits column (right column). Copy EXACTLY as shown. Examples: 'ST' (Step Therapy), 'PA' (Prior Authorization), 'QL (60 ML per 30 days)' (Quantity Limit with details), 'PA, QL', empty cells should be null. ALSO INCLUDE any limits (QL, PA, ST) that appear in the first column under the drug name. IF THE COLUMN HEADER IS 'drug edit', EXTRACT DATA FROM THAT COLUMN. DO NOT INCLUDE 'B' or 'G' here. For PREFERRED/NON-PREFERRED tables: set to 'PA' if the drug is in the 'PA Required' or 'Non-preferred Agents' column, set to null if the drug is in the 'No PA Required' or 'Preferred Agents' column."
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
                                "description": "Category header text from gray/shaded rows that span all columns. Examples: '*ADHD/ANTI-NARCOLEPSY...', '*AMPHETAMINE MIXTURES*'. These are NOT drug names."
                            },
                            "page_number": {"type": ["integer", "null"], "description": "Page number in the PDF where this drug is found."},
                            "pa_form_link": {"type": ["string", "null"], "description": "PA Form Link URL if present in the table."},
                            "badge_colors": {
                                "type": ["object", "null"], 
                                "description": """🎨 LOOK FOR COLORS! Extract colors of requirement badges (QL, PA, ST, etc.) if they are NOT black.
Examples: {"QL": "purple"}, {"PA": "brown"}, {"ST": "olive"}, {"HYB": "green"}, {"HNB": "red"}
NULL if all text is black.""",
                                "properties": {},
                                "additionalProperties": {"type": "string"}
                            }
                        },
                        "required": ["Drug Name"]
                    }
                },

                "FormularyAbbreviations": {
                    "type": "array",
                    "description": """Extract ALL abbreviation/legend definitions AND tier definitions from ANYWHERE in the document.

🎨 CRITICAL COLOR DETECTION INSTRUCTION:
When extracting acronyms/abbreviations, LOOK AT THE VISUAL APPEARANCE:
- If the acronym appears in a COLORED BADGE (purple, brown, green, red, orange, blue, etc.), extract that color in 'badge_color'
- If the acronym text has COLOR (not black), extract that color in 'badge_color'
- Example: "QL" in purple pill badge → badge_color: "purple"
- Example: "PA" in brown/orange badge → badge_color: "brown"
- Example: "HYB" in green badge → badge_color: "green"  
- Example: "HNB" in red badge → badge_color: "red"
- If text is standard BLACK with no colored badge → badge_color: NULL

THIS IS VERY IMPORTANT: Two acronyms may be the SAME text but DIFFERENT colors mean DIFFERENT things!
Example: "HYB" (green) = Preferred, "HNB" (red) = Non-Preferred

EXTRACT TWO TYPES OF DEFINITIONS:

TYPE 1 - ABBREVIATION CODES:
Look for legends in: page headers, footers, sidebar text, or dedicated sections.
Common patterns: 'ST = Step Therapy', 'PA = Prior Authorization', 'QL = Quantity Limit', 'B/D = Brand/Drug'
Examples: 'NDS' = 'Non-Dispensing Supply', 'LA' = 'Limitation on Age', 'EX' = 'Excluded Drug'

TYPE 2 - TIER DEFINITIONS (CRITICAL - EXTRACT THESE):
Look for tier explanation sections with text like:
- 'Tier 1 - Preferred Generic Drugs: This tier includes commonly prescribed generic drugs...'
- 'Tier 2 - Generic Drugs: This tier includes generic drugs...'
- 'Tier 3 - Preferred Brand Drugs: This tier includes preferred brand-name drugs...'
- 'Tier 4 - Non-Preferred Drugs: This tier includes higher-priced brand name drugs...'
- 'Tier 5 - Specialty Tier drugs: This tier includes high-cost drugs...'
- 'Tier 6 - Select Care Diabetic Drugs: This tier includes...'

For TIER DEFINITIONS:
- Acronym = 'Tier 1', 'Tier 2', 'Tier 3', etc.
- Expansion = The tier name like 'Preferred Generic Drugs', 'Generic Drugs', etc. 
- Explanation = The full description text 'This tier includes commonly prescribed generic drugs...'

Extract EVERY abbreviation definition AND tier definition found WITH THEIR COLORS IF PRESENT.""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Acronym": {"type": "string", "description": "The abbreviation code OR tier identifier. Examples: 'ST', 'PA', 'QL', 'SP', 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5', 'Tier 6'. EXTRACT ONLY THE CODE/IDENTIFIER HERE."},
                            "Expansion": {"type": "string", "description": "What the abbreviation stands for OR the tier name. Examples: 'Step Therapy', 'Prior Authorization', 'Preferred Generic Drugs', 'Non-Preferred Drugs'. EXTRACT THE NAME HERE."},
                            "Explanation": {"type": ["string", "null"], "description": "Additional explanation if provided. For tier definitions, this is the full description like 'This tier includes commonly prescribed generic drugs. Drugs in Tier 1 will typically be your most affordable option.'"},
                            "badge_color": {"type": ["string", "null"], "description": "CONDITIONAL - Extract ONLY if the acronym/abbreviation appears in COLORED text or badge (NOT black). Specify color name: 'purple', 'brown', 'green', 'red', 'orange', 'blue', etc. NULL if black text."}
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
        "drug_tier": drug_tier,
        "drug_requirements": drug_requirements,
        "category": item.get("category"),
        "page_number": page_number,
        "badge_colors": item.get("badge_colors"),
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


def filter_index_entries_from_mistral_response(drug_info_list: List[dict]) -> List[dict]:
    """
    Filter out index page entries from Mistral OCR response.
    
    Index entries are identified by:
    - Having a page_number field (pointing to where the drug is listed)
    - BUT missing both Tier and Requirements fields
    - These are drug names extracted from an index/TOC page
    
    Real drug entries have:
    - Tier field (e.g., "1", "2", "Tier 1")
    - Requirements field (e.g., "PA", "QL", "ST")
    - May or may not have page_number
    
    Args:
        drug_info_list: List of drug dictionaries from Mistral OCR
        
    Returns:
        Filtered list with index entries removed
    """
    filtered = []
    index_entries_removed = 0
    
    for item in drug_info_list:
        # Check for page_number field
        has_page_number = item.get("page_number") is not None
        
        # Check for tier (multiple possible field names)
        has_tier = bool(item.get("Tier") or item.get("drug tier") or item.get("drug_tier"))
        
        # Check for requirements (multiple possible field names)
        has_requirements = bool(item.get("Requirements") or item.get("drug_requirements") or item.get("requirements"))
        
        # Index entry pattern: has page_number but NO tier and NO requirements
        if has_page_number and not has_tier and not has_requirements:
            index_entries_removed += 1
            logger.debug(f"🚫 Filtered index entry: {item.get('Drug Name', 'Unknown')} (page {item.get('page_number')})")
            continue
        
        filtered.append(item)
    
    if index_entries_removed > 0:
        logger.info(f"🚫 Filtered {index_entries_removed} index entries from Mistral response (entries with page_number but no tier/requirements)")
    
    return filtered


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


def _is_valid_tier_format(tier_value: str) -> bool:
    """
    Check if a tier value represents valid formulary tiers vs page numbers.
    
    Valid tier formats:
    - Single digit: "2", "4", "5"
    - Comma-separated: "2, 4", "2, 4, 5", "4, 5"
    - All tiers must be 1-6
    
    Invalid (page numbers):
    - Large numbers: "115", "180", "214"
    - Mixed: "24", "245" (multi-tier parsing artifacts)
    
    Returns True if this looks like a valid tier, False if it looks like a page number.
    """
    if not tier_value or not str(tier_value).strip():
        return False
    
    tier_str = str(tier_value).strip()
    
    # Split by comma to handle multi-tier values
    parts = [p.strip() for p in tier_str.split(',')]
    
    # Check if all parts are single-digit valid tiers (1-6)
    for part in parts:
        if not part.isdigit():
            return False
        tier_num = int(part)
        # Valid formulary tiers are 1-6
        if tier_num < 1 or tier_num > 6:
            return False
    
    return True


def _extract_number_from_any_field(item: dict) -> Optional[int]:
    """
    Extract the first number found in tier, requirements, or name fields.
    Used for statistical analysis to detect if numbers are page numbers.
    """
    # Check tier field first
    tier = item.get("drug_tier", "") or ""
    if tier:
        cleaned = re.sub(r'[^0-9]', '', str(tier))
        if cleaned and cleaned.isdigit():
            return int(cleaned)
    
    # Check requirements
    reqs = item.get("drug_requirements", "") or ""
    if reqs:
        cleaned = re.sub(r'[^0-9]', '', str(reqs))
        if cleaned and cleaned.isdigit():
            return int(cleaned)
    
    # Check end of drug name for numbers
    name = item.get("drug_name", "") or ""
    # Extract trailing number
    match = re.search(r'(\d{2,3})\s*$', name)
    if match:
        return int(match.group(1))
    
    return None


def _is_index_entry(item: dict) -> bool:
    """
    Check if a single drug entry looks like it came from an index page.
    Uses multiple heuristics to catch various OCR parsing patterns.
    """
    drug_name = item.get("drug_name", "") or ""
    tier = item.get("drug_tier", "") or ""
    reqs = item.get("drug_requirements", "") or ""
    
    # Heuristic 1: Tier is a high number (Page number)
    if str(tier).strip().isdigit():
        tier_val = int(str(tier).strip())
        if tier_val > 6:  # Tiers are usually 1-6
            return True
             
    # Heuristic 2: Requirements is just a number (Page number)
    if str(reqs).strip().isdigit():
        req_val = int(str(reqs).strip())
        if req_val > 6:
            return True
    
    # Heuristic 3: Drug name ends with page number
    if RE_PAGE_NUMBER_END.search(drug_name):
        return True
    
    # Heuristic 4: Drug name contains dot leaders
    if RE_DOT_LEADERS.search(drug_name):
        return True
        
    # Heuristic 5: Name ends with isolated number > 10
    # e.g., "AMOXICILLIN 270" where 270 got merged into name
    match = re.search(r'\s+(\d{2,3})\s*$', drug_name)
    if match:
        num = int(match.group(1))
        if num > 10:  # Unlikely to be a valid part of drug name
            return True
    
    # Heuristic 6: Drug name is ONLY a drug name with NO dosage info
    # AND tier/requirements are missing or suspicious
    # This catches index entries like "AMOXICILLIN" with no tier/requirements
    has_dosage = bool(RE_DOSAGE_FORM.search(drug_name))
    if not has_dosage and not tier and not reqs:
        # If drug name is just a plain name with no dosage, tier, or requirements
        # it's likely from an index page
        if len(drug_name) > 3 and drug_name.replace(' ', '').replace('-', '').isalpha():
            return True
        
    return False


def _is_extracted_data_from_index_page(drug_table: List[dict]) -> bool:
    """
    Detect if extracted drug data appears to come from an index/table of contents page.
    Uses STATISTICAL VARIANCE ANALYSIS instead of simple thresholds.
    
    CRITICAL: Handles multi-tier formats like "2, 4, 5" correctly by validating 
    tier format before including in variance analysis.
    
    Returns True if the page looks like an index, False otherwise.
    """
    if not drug_table or len(drug_table) < 3:  # Lowered from 5 to catch smaller index pages
        return False

    total = len(drug_table)
    
    # CRITICAL CHECK: PREFERRED/NON-PREFERRED Format Detection
    # If drugs have preferred_agent or non_preferred_agent fields, this is a valid
    # PREFERRED/NON-PREFERRED format table, NOT an index page!
    # These tables don't have tiers or requirements, which would normally trigger index detection.
    preferred_agent_count = 0
    for item in drug_table:
        if item.get("preferred_agent") or item.get("non_preferred_agent"):
            preferred_agent_count += 1
    
    # If >50% of entries have preferred/non-preferred agent info, it's definitely a valid table
    if preferred_agent_count / total >= 0.50:
        logger.info(f"✅ PREFERRED/NON-PREFERRED format detected: {preferred_agent_count}/{total} entries have agent classification. NOT an index page.")
        return False
    
    # Extract numbers from all entries
    numbers = []
    index_entries = 0
    no_dosage_count = 0
    no_requirements_count = 0
    no_tier_count = 0
    valid_tier_format_count = 0  # Track entries with valid tier formats
    
    for item in drug_table:
        # Check if individual entry looks like index
        if _is_index_entry(item):
            index_entries += 1
        
        # Check if tier value is a valid tier format (e.g., "2, 4, 5")
        tier_value = item.get("drug_tier", "")
        if tier_value and _is_valid_tier_format(tier_value):
            valid_tier_format_count += 1
            # DO NOT include this in variance analysis - it's a valid tier, not a page number
        else:
            # Extract number for variance analysis ONLY if NOT a valid tier format
            num = _extract_number_from_any_field(item)
            if num is not None:
                numbers.append(num)
        
        # Check for dosage info
        drug_name = item.get("drug_name", "") or ""
        if not RE_DOSAGE_FORM.search(drug_name):
            no_dosage_count += 1
        
        # Check for requirements
        if not item.get("drug_requirements"):
            no_requirements_count += 1
        
        # Check for tier
        if not item.get("drug_tier"):
            no_tier_count += 1

    # RULE 1: High percentage of obvious index entries (LOWERED THRESHOLD)
    if total > 0 and index_entries / total >= 0.20:  # Lowered from 25% to 20%
        logger.info(f"🚫 INDEX PAGE DETECTED (Rule 1): {index_entries}/{total} ({index_entries/total*100:.1f}%) entries are index entries")
        return True
    
    # NEW RULE 1.5: If majority have valid tier formats, this is NOT an index page
    if total > 0 and valid_tier_format_count / total >= 0.50:
        logger.info(f"✅ VALID DRUG PAGE: {valid_tier_format_count}/{total} ({valid_tier_format_count/total*100:.1f}%) entries have valid tier formats (1-6)")
        return False
    
    # RULE 2: STATISTICAL VARIANCE ANALYSIS
    # Real drug tiers: [1, 1, 2, 1, 3, 2, 1] → low variance, mean ~1.5
    # Page numbers: [301, 115, 143, 180, 214] → high variance, mean ~190
    # NOTE: Now only analyzes numbers from entries WITHOUT valid tier formats
    if len(numbers) >= 3:  # Lowered from 5 to catch smaller samples
        import statistics
        
        mean_val = statistics.mean(numbers)
        
        # Calculate standard deviation if we have enough values
        if len(numbers) >= 2:
            try:
                stdev_val = statistics.stdev(numbers)
            except:
                stdev_val = 0
        else:
            stdev_val = 0
        
        # If mean > 10 AND stdev > 30, these are page numbers
        if mean_val > 10 and stdev_val > 30:
            logger.info(f"🚫 INDEX PAGE DETECTED (Rule 2 - Variance): mean={mean_val:.1f}, stdev={stdev_val:.1f} (page numbers, not tiers)")
            return True
        
        # If mean > 50, definitely page numbers regardless of stdev
        if mean_val > 50:
            logger.info(f"🚫 INDEX PAGE DETECTED (Rule 2 - High Mean): mean={mean_val:.1f} (page numbers)")
            return True

    # RULE 3: Structural uniformity (no dosage + no requirements + no tier)
    # LOWERED THRESHOLDS to be more aggressive
    if (no_dosage_count / total >= 0.80 and  # Lowered from 85%
        no_requirements_count / total >= 0.80 and  # Lowered from 85%
        no_tier_count / total >= 0.80):  # NEW: Also check for missing tiers
        logger.info(f"🚫 INDEX PAGE DETECTED (Rule 3 - Uniformity): {no_dosage_count}/{total} no dosage, {no_requirements_count}/{total} no requirements, {no_tier_count}/{total} no tier")
        return True

    # RULE 4: If >80% have no tier AND no requirements AND >50% have no dosage
    # This combination is very strong evidence of an index page
    if (no_tier_count / total >= 0.80 and 
        no_requirements_count / total >= 0.80 and
        no_dosage_count / total >= 0.50):
        logger.info(f"🚫 INDEX PAGE DETECTED (Rule 4 - Missing Everything): {no_tier_count}/{total} no tier, {no_requirements_count}/{total} no requirements, {no_dosage_count}/{total} no dosage")
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
            # A fragment is an entry that is clearly a continuation of the previous line.
            
            # Check if line seems to be just dosage/form information
            is_dosage_or_form = bool(re.match(r'^(tablet|capsule|cap|tab|sol|susp|inj|cream|oint|gel|patch|kit|dev|\d)', next_name, re.IGNORECASE))
            
            # Check for explicit continuation indicators
            is_continuation = bool(re.match(r'^(and|with|w/|\+|&)', next_name, re.IGNORECASE))

            # Check if line is very short (likely a stray character or number)
            is_short = len(next_name) < 4
            
            # It is a fragment ONLY if:
            # 1. No Tier/Requirements (empty columns)
            # 2. AND (Is dosage/form OR Is explicit continuation OR Is very short)
            # NOTE: usage of lowercase start as a signal is REMOVED/AVOIDED because many valid generics start with lowercase.
            is_fragment = (
                not next_item.get("drug_tier") and
                not next_item.get("drug_requirements") and
                (is_dosage_or_form or is_continuation or is_short)
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
    header_rows_removed = 0
    
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
        # Skip category headers/sub-headers incorrectly extracted as drugs
        if _is_header_row(name):
            header_rows_removed += 1
            logger.debug(f"🧹 Filtered header row: '{name}'")
            continue
            
        filtered.append(item)

    if filtered_out:
        logger.debug(f"🧹 Filtered out {len(filtered_out)} invalid entries: {filtered_out[:5]}")
    if index_entries_removed > 0:
        logger.info(f"🧹 Removed {index_entries_removed} individual index entries from drug table")
    if header_rows_removed > 0:
        logger.info(f"🧹 Removed {header_rows_removed} category header rows from drug table")

    logger.info(f"🧹 Final drug count: {len(filtered)} (filtered {len(result) - len(filtered)} invalid entries)")

    return filtered


def _is_header_row(drug_name: str) -> bool:
    """
    Detects if a row is likely a therapeutic category header rather than a drug.
    """
    if not drug_name:
        return False
        
    name_lower = drug_name.lower()
    
    # Pattern 1: Explicit "Drugs to Treat" phrase (very strong signal)
    if "drugs to treat" in name_lower:
        return True
        
    # Pattern 2: Category-like structure (Hyphens) + NO Dosage Info
    # Headers often have " - " but NO strength/form info (mg, ml, tab, cap)
    
    # Regex for dosage/form - reused from RE_DOSAGE_FORM but broader
    has_dosage = re.search(r'\d+\s*(mg|ml|mcg|unit|%|tab|cap|sol|cream|gel|patch|spray)', name_lower)
    
    if not has_dosage:
        # Normalize dashes: replace en-dash, em-dash with standard hyphen
        normalized_name = name_lower.replace('–', '-').replace('—', '-').replace('−', '-')
        
        # If it has a hyphen and no dosage, it's suspicious.
        # Check for " - " (spaced hyphen) OR just a hyphen if it looks like a compound category
        if "-" in normalized_name:
            # If it has " - " (spaced), it's very likely a header if no dosage
            if " - " in normalized_name:
                return True
            
            # If it has a hyphen but no spaces (e.g. "Anti-Infective"), we need to be careful.
            # But if it's "Category - Subcategory" with weird spacing like "Category-Subcategory",
            # we might want to catch it.
            # Let's check if the hyphen is surrounded by letters, which might be a valid drug name (e.g. "Anti-Infective").
            # But "Urinary Antispasmodics-Direct Muscle Relaxants" is a header.
            # Heuristic: If it's long (>20 chars) and has a hyphen, and NO dosage, it's likely a header
            if len(normalized_name) > 20:
                return True
            
        # Check for all caps words at start (common in headers)
        # Only if the name is reasonably long (avoid filtering "ASPIRIN")
        parts = drug_name.split(' ')
        if len(parts) > 0 and parts[0].isupper() and len(parts[0]) > 4 and len(drug_name) > 20:
             return True
             
    return False


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
