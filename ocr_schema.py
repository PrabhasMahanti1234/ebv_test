O_A_S = {
    "type": "json_schema",
    "json_schema": {
        "name": "drug_extraction_schema",
        "schema": {
            "type": "object",
            "title": "StructuredData",
            "properties": {

                # ──────────────────────────────────────────────────────────────
                #  DRUG INFORMATION
                # ──────────────────────────────────────────────────────────────
                "DrugInformation": {
                    "type": "array",
                    "description": """Extract ALL drugs from the formulary drug table / page.

KEY INSTRUCTION: The source may NOT look like a traditional table.
It may be a vertical list, a compact list, or a multi-column table.
YOU MUST EXTRACT EVERY DRUG ENTRY YOU SEE regardless of format.

──────────────────────────────────────────────────────────────
RECOGNITION PATTERNS
──────────────────────────────────────────────────────────────
1. Standard Table  : "Drug Name" | "Tier" | "Restrictions/Limits"
2. List Format     : Drug Name on one line, Tier/Requirements on next line.
3. Compact List    : Drug Name followed by Tier (e.g. "5") and Requirements (e.g. "QL").

EXAMPLE OF LIST FORMAT:
  TREMFYA SOSY 100mg/ml
  QL (1 syringe / 28 days)
  5
  QL NM PA
  ==> Drug Name="TREMFYA SOSY 100mg/ml", Tier="5", Requirements="QL (1 syringe / 28 days); QL NM PA"

──────────────────────────────────────────────────────────────
SUPPORTED COLUMN HEADER PATTERNS
──────────────────────────────────────────────────────────────
- "Drug Name" | "Tier" | "Restrictions/Limits"
- "Drug Name" | "Drug Tier" | "Requirements"
- "Drug Name" | "Step Therapy Requirements"                          → SPECIAL TEMPLATE 12
- "Drug Name" | "Quantity Limit" (or "Quantity Limit (QL)")          → SPECIAL TEMPLATE 13
- "Drug Name" | "Dosage" | "Strength/Package Size" | "Billing Unit" | "UM Type" | "Code 1"   → SPECIAL TEMPLATE 1
- "HCPCS/CPT Code" | "Drug Name" | "HCPCS/CPT Code Description" | "Coverage Level" | "Notes & Restrictions"  → SPECIAL TEMPLATE 2
- "Drug Class/Drug Name" | "Reference Brand Name" | "Brand Only/Generic Notes" | "Preferred Drug Status" | "PA Status" | "Step Therapy Requirements" | "Quantity Limit (QL)" | "QL Days"  → SPECIAL TEMPLATE 3
- "Drug name" | "Brand or Generic" | "Drug tier" | "Coverage rules or limits on use"  → SPECIAL TEMPLATE 4
- "Drug Name" | "Special Code" | "Tier Category"                     → SPECIAL TEMPLATE 5
- Flag-column tables (e.g. PA | QL | ST as indicator columns)        → SPECIAL TEMPLATE 6
- "Therapeutic Class" | "Label Name" | "Generic Name" | "Tier Value" | "Prior Auth" | "Quantity Limit" | "Step Therapy" | "Specialty"  → SPECIAL TEMPLATE 7
- "PDL DRUG CATEGORY" | "PREFERRED" | "PREFERRED WITH PA" | "NON-PREFERRED"  → SPECIAL TEMPLATE 8
- Two-column "PREFERRED" / "NON-PREFERRED" (or similar)              → SPECIAL TEMPLATE 9
- Page-header-only tier/requirement indication                        → SPECIAL TEMPLATE 10
- "Drug Name" + multiple tier columns (e.g. "Generic", "Preferred Brand", "Specialty")  → SPECIAL TEMPLATE 11
- "pub drug name" | "pub strength" | "pub tier" | "drug edit"        → extract from those columns directly

──────────────────────────────────────────────────────────────
TABLE STRUCTURE NOTES
──────────────────────────────────────────────────────────────
- CATEGORY HEADERS appear as gray/shaded rows spanning all columns
  (e.g. "FIRST GENERATION ANTIHISTAMINES", "PHENOTHIAZINE DERIVATIVES").
  These go into the 'category' field, NOT 'Drug Name'.
- Drug rows contain: full drug name (with optional dosage) | tier | restrictions.

──────────────────────────────────────────────────────────────
SPECIAL TEMPLATE RULES
──────────────────────────────────────────────────────────────

SPECIAL TEMPLATE 1 — UM Type / Code 1:
  Headers: "Drug Name" | "Dosage" | "Strength/Package Size" | "Billing Unit" | "UM Type" | "Code 1"
  - NO Tier column → set 'drug tier' to null.
  - Concatenate "UM Type" and "Code 1" into 'requirements' (e.g. "AL * Restricted to...").

SPECIAL TEMPLATE 2 — HCPCS/CPT:
  Headers: "HCPCS/CPT Code" | "Drug Name" | "HCPCS/CPT Code Description" | "Coverage Level" | "Notes & Restrictions"
  - Concatenate "Drug Name" + "HCPCS/CPT Code Description" into 'Drug Name'.
  - Map "Coverage Level" → 'drug tier'.
  - Map "Notes & Restrictions" → 'requirements'.

SPECIAL TEMPLATE 3 — AHCCCS:
  Headers: "Drug Class/Drug Name" | "Reference Brand Name" | "Brand Only/Generic Notes" | "Preferred Drug Status" | "PA Status" | "Step Therapy Requirements" | "Quantity Limit (QL)" | "QL Days"
  - Concatenate "Drug Class/Drug Name" + "Reference Brand Name" (in parentheses) → 'Drug Name'.
  - Concatenate "Brand Only/Generic Notes" + "Preferred Drug Status" → 'drug tier'.
  - Concatenate ALL NON-EMPTY values from "PA Status", "Step Therapy Requirements", "Quantity Limit (QL)", "QL Days" → 'requirements'.
    DO NOT OMIT ANY VALUE. If both PA and QL exist, include BOTH.
    Format QL columns as: "QL limit: [value], QL days: [value]".
    Example: "PA, QL limit: 30, QL days: 30".

SPECIAL TEMPLATE 4 — Brand or Generic column:
  Headers: "Drug name" | "Brand or Generic" | "Drug tier" | "Coverage rules or limits on use"
  - Use ONLY "Drug tier" for 'drug tier'. Do NOT include "Brand or Generic" in tier.
  - If "Drug tier" column is missing, set 'drug tier' to null.
  - Map "Coverage rules or limits on use" → 'requirements'.

SPECIAL TEMPLATE 5 — Special Code / Tier Category:
  Headers: "Drug Name" | "Special Code" | "Tier Category"
  - Map "Special Code" → 'requirements'.
  - Map "Tier Category" → 'drug tier'. Extract ONLY the tier value (e.g. "3", "NC", "4");
    IGNORE any appended category text (e.g. "3 ANTIVIRALS" → extract "3").

SPECIAL TEMPLATE 6 — Flag/Indicator columns:
  If the table has columns that serve as flags (e.g. PA, QL, ST columns where "X" means the flag is set):
  - If "X" or ANY value is present in such a column, use the COLUMN NAME as the requirement.
  - Concatenate all triggered column names → 'requirements'. If none triggered, set null.
  - If headers include "Generic medication name" and "Medication name", concatenate as
    "Generic medication name (Medication name)" → 'Drug Name'.

SPECIAL TEMPLATE 7 — Therapeutic Class / Label Name / Generic Name:
  Headers: "Therapeutic Class" | "Label Name" | "Generic Name" | "Tier Value" | "Prior Auth" | "Quantity Limit" | "Step Therapy" | "Specialty"
  - Concatenate "Generic Name" + "Label Name" (in parentheses) → 'Drug Name'.
  - Map "Therapeutic Class" → 'category'.
  - Map "Tier Value" → 'drug tier'.
  - For 'requirements': if "Prior Auth" = "X" add "PA"; "Quantity Limit" = "X" add "QL";
    "Step Therapy" = "X" add "ST"; "Specialty" = "X" add "SP". Join with commas (e.g. "PA, QL, SP").

SPECIAL TEMPLATE 8 — PDL PREFERRED / NON-PREFERRED / PREFERRED WITH PA:
  Headers: "PDL DRUG CATEGORY" | "PREFERRED" | "PREFERRED WITH PA" | "NON-PREFERRED"
  - Drugs in "PREFERRED"         → 'drug tier' = "Preferred",     'requirements' = null.
  - Drugs in "PREFERRED WITH PA" → 'drug tier' = "Preferred",     'requirements' = "PA".
  - Drugs in "NON-PREFERRED"     → 'drug tier' = "Non-Preferred", 'requirements' = null.
  - Map "PDL DRUG CATEGORY" → 'category'.

SPECIAL TEMPLATE 9 — Two-column Preferred / Non-Preferred:
  If the table has only 2 columns "PREFERRED" and "NON-PREFERRED"
  (or similar, e.g. "Preferred Agents" / "Non-Preferred Agents"):
  - Drugs in "PREFERRED" column    → 'drug tier' = "Preferred".
  - Drugs in "NON-PREFERRED" column → 'drug tier' = "Non-Preferred".

SPECIAL TEMPLATE 10 — Page Header Inheritance:
  If the table LACKS specific 'Requirements' or 'Tier' columns, check the PAGE HEADER or SECTION TITLE.
  - If the table has "Drug name" and "Reference Name" columns, concatenate as
    "Drug name (Reference Name)" → 'Drug Name'.
  - If a 'Tier' or 'Drug Tier' column exists, map drugs to 'drug tier'; otherwise set null.
  - If the page header contains requirement codes (e.g. "OTC", "PA", "QL"), map ALL drugs
    in that table to 'requirements'; otherwise set null.

SPECIAL TEMPLATE 11 — Drug Name + multiple tier columns:
  If headers are "Drug Name" plus multiple tier-labeled columns (e.g. "Generic", "Preferred Brand", "Specialty"):
  - The "Drug Name" column may contain the drug name AND requirement codes in bold
    (e.g. "DrugName PA, QL"). Extract the name → 'Drug Name'; extract codes → 'requirements'.
  - The tier value is the numeric/text cell found in the applicable tier column
    (e.g. if "Generic" column has "1", 'drug tier' = "1").

SPECIAL TEMPLATE 12 — Drug Name + Step Therapy Requirements only:
  Headers: "Drug Name" | "Step Therapy Requirements"
  - NO Tier column → set 'drug tier' to null.
  - Copy full text from "Step Therapy Requirements" → 'requirements'.

SPECIAL TEMPLATE 13 — Drug Name + Quantity Limit only:
  Headers: "Drug Name" | "Quantity Limit" (or "Quantity Limit (QL)")
  - NO Tier column → set 'drug tier' to null.
  - Copy full text from the "Quantity Limit" column → 'requirements'.

──────────────────────────────────────────────────────────────
GENERAL EXTRACTION RULES
──────────────────────────────────────────────────────────────
1.  Extract EVERY drug row — do NOT skip any, even if dosage form is in the drug name.
2.  Category headers go in 'category', NOT 'Drug Name'.
3.  Tier values are exactly: "Tier 1" … "Tier 5", OR plain numbers "1"…"5",
    OR "Generic", "Brand", "Specialty", "Preferred", "Non-Preferred".
    Copy exact text as shown. IF THE TIER COLUMN IS MISSING, SET 'drug tier' TO NULL.
4.  Restrictions column may contain: "ST", "PA", "QL", "QL (60 ML per 30 days)", "NM", "LA", "B/D", "^", etc.
5.  If a Restrictions/Limits cell is empty, set 'requirements' to null.
6.  Map each drug to its own tier and requirements. Do NOT carry over tier/requirements
    from a previous row when a row has no explicit value.
7.  SKIP index/table-of-contents pages (pages with just drug names and page numbers).
8.  For drugs with inline dosage (e.g. 'carbinoxamine maleate oral tablet 4 mg'),
    extract the COMPLETE text into 'Drug Name' and leave 'Dosage Form/Strength' as null.
9.  Set 'page_number' to the actual PDF page number where the drug appears.
10. Do NOT include "QL", "PA", "ST", or limit expressions like "(X per Y days)"
    inside the 'Drug Name' field — those belong in 'requirements'.
11. If a column is headed "pub drug name", extract from that column for Drug Name.
    If a column is headed "pub strength" or "pub dosage", extract into 'Dosage Form/Strength'.
    If a column is headed "pub tier", extract into 'drug tier'.
    If a column is headed "drug edit", extract into 'requirements'.
12. If the tier column contains cost information (e.g. "$0 (Tier 1)", "$0/$1.60"),
    extract the FULL TEXT including the cost into 'drug tier'.""",

                    "items": {
                        "type": "object",
                        "properties": {

                            "Drug Name": {
                                "type": "string",
                                "description": """The complete drug name from the first / left-most column (or 'pub drug name' column).

May contain:
  1. Just the drug name (e.g. 'AMOXICILLIN')
  2. Drug name WITH inline dosage (e.g. 'carbinoxamine maleate oral tablet 4 mg',
     'azelastine nasal spray non-aerosol 137 mcg (0.1 %)')

EXTRACT THE FULL TEXT including any dosage information. Copy EXACTLY as shown.
STOP before any coverage restrictions (QL, PA, ST, limit expressions).

SPECIAL CONCATENATION CASES:
  SC-1: Table has "HCPCS/CPT Code Description" column →
        Concatenate "Drug Name" + "HCPCS/CPT Code Description" here.
  SC-2: Table has "Reference Brand Name" column →
        Concatenate "Drug Class/Drug Name" + "Reference Brand Name" (in parentheses) here.
        Example: "AMPHETAMINE... (ADDERALL XR)"
  SC-3: Table has "Label Name" and "Generic Name" columns →
        Concatenate "Generic Name" + "Label Name" (in parentheses) here.
  SC-4: Table has "Generic medication name" and "Medication name" columns →
        Concatenate "Generic medication name" + "Medication name" (in parentheses) here.
  SC-5: Table has "Drug name" and "Reference Name" columns (Template 10) →
        Concatenate as "Drug name (Reference Name)" here.
  SC-6: Table has a separate "pub strength" / "pub dosage" column not captured elsewhere →
        Append that value to this field."""
                            },

                            "Dosage Form/Strength": {
                                "type": ["string", "null"],
                                "description": """The dosage form and strength ONLY if it appears in a SEPARATE column
(e.g. a distinct 'Dosage', 'Strength/Package Size', 'Billing Unit', 'pub strength', or 'pub dosage' column).
Examples: 'TAB 250MG', 'CAP 500MG', 'SUS 200/5ML'.
Leave null when dosage is already embedded in the Drug Name column."""
                            },

                            "BrandOrGeneric": {
                                "type": ["string", "null"],
                                "description": """Value from a 'Brand or Generic' column when one exists (typically the 2nd column).
Common values: 'B', 'G', 'Brand', 'Generic'.
Extract separately here — do NOT mix into Drug Name or drug tier."""
                            },

                            "drug tier": {
                                "type": ["string", "null"],
                                "description": """The tier value. Copy exactly as shown in the tier column.

Valid forms: 'Tier 1'–'Tier 5', plain numbers '1'–'5', 'Generic', 'Brand',
'Specialty', 'Preferred', 'Non-Preferred'.

Special rules:
- If the tier column contains cost text (e.g. '$0 (Tier 1)', '$0/$1.60'),
  extract the FULL TEXT including the cost.
- If the column is headed 'pub tier', extract from that column.
- Do NOT include 'B' or 'G' from a Brand/Generic column here.
- If NO tier column exists (Templates 1, 3, 12, 13, or tables with only
  Drug Name + Step Therapy / Quantity Limit), set to null.
- For Template 5 ('Tier Category' column): extract ONLY the tier portion
  (e.g. '3', 'NC') — ignore any appended category text."""
                            },

                            "requirements": {
                                "type": ["string", "null"],
                                "description": """Restrictions/limits for the drug. Copy EXACTLY as shown.

Common values: 'ST', 'PA', 'QL', 'QL (60 ML per 30 days)', 'PA, QL', 'NM', 'LA', 'B/D'.
Empty cells → null.

Also include any QL/PA/ST codes that appear in the first column BELOW the drug name.
If column is headed 'drug edit', extract from that column.
Do NOT include 'B' or 'G' (Brand/Generic indicator) here.

SPECIAL CONCATENATION CASES:
  SC-1: 'UM Type' + 'Code 1' columns exist → concatenate both here.
  SC-2: 'PA Status', 'Step Therapy Requirements', 'Quantity Limit (QL)', 'QL Days' columns exist →
        concatenate ALL NON-EMPTY values. Do not omit any.
        Format QL as: 'QL limit: [value], QL days: [value]'.
        Example: 'PA, QL limit: 30, QL days: 30'.
  SC-3: 'Special Code' column exists → include its value here.
  SC-4: Table has only 'Drug Name' + 'Step Therapy Requirements' → copy full ST text here.
  SC-5: Table has only 'Drug Name' + 'Quantity Limit'/'Quantity Limit (QL)' → copy full QL text here.
  SC-6: PREFERRED/NON-PREFERRED format → set 'PA' if drug is in PA-required / non-preferred column;
        null if in preferred / no-PA-required column."""
                            },

                            "preferred_agent": {
                                "type": ["string", "null"],
                                "enum": ["yes", "no", None],
                                "description": """ONLY USE VALUES: 'yes', 'no', or null. NO OTHER VALUES ALLOWED.
For PREFERRED / NON-PREFERRED format tables:
  'yes' → drug is in the PREFERRED or 'No PA Required' column.
  'no'  → drug is in the NON-PREFERRED or 'PA Required' column.
Leave null for standard tier-based tables."""
                            },

                            "non_preferred_agent": {
                                "type": ["string", "null"],
                                "enum": ["yes", "no", None],
                                "description": """ONLY USE VALUES: 'yes', 'no', or null. NO OTHER VALUES ALLOWED.
For PREFERRED / NON-PREFERRED format tables:
  'yes' → drug is in the NON-PREFERRED or 'PA Required' column.
  'no'  → drug is in the PREFERRED or 'No PA Required' column.
Leave null for standard tier-based tables."""
                            },

                            "BGO": {
                                "type": ["string", "null"],
                                "description": "PDL format only: B=Brand, G=Generic, O=OTC. Leave null for standard formulary tables."
                            },

                            "PNRNR": {
                                "type": ["string", "null"],
                                "description": "PDL format only: P=Preferred, N=Non-Preferred, R/NR. Leave null for standard formulary tables."
                            },

                            "Specialty": {
                                "type": ["boolean", "null"],
                                "description": "True if the drug is marked as a Specialty drug. Leave null if not indicated."
                            },

                            "PriorAuthorization": {
                                "type": ["boolean", "null"],
                                "description": "True if 'PA' appears in the requirements for this drug."
                            },

                            "StepTherapy": {
                                "type": ["boolean", "null"],
                                "description": "True if 'ST' appears in the requirements for this drug."
                            },

                            "DispensingLimits": {
                                "type": ["boolean", "null"],
                                "description": "True if 'QL' appears in the requirements for this drug."
                            },

                            "category": {
                                "type": ["string", "null"],
                                "description": """Category header text from gray/shaded rows that span all columns.
Examples: 'FIRST GENERATION ANTIHISTAMINES', 'PHENOTHIAZINE DERIVATIVES',
'*ADHD/ANTI-NARCOLEPSY...', 'SECOND GENERATION ANTIHISTAMINES'.
These are drug class/group labels — NOT drug names."""
                            },

                            "page_number": {
                                "type": ["integer", "null"],
                                "description": "Actual PDF page number where this drug entry appears."
                            },

                            "pa_form_link": {
                                "type": ["string", "null"],
                                "description": "PA Form Link URL if present in the table row, otherwise null."
                            }
                        },
                        "required": ["Drug Name"]
                    }
                },

                # ──────────────────────────────────────────────────────────────
                #  FORMULARY ABBREVIATIONS
                # ──────────────────────────────────────────────────────────────
                "FormularyAbbreviations": {
                    "type": "array",
                    "description": """Extract ALL abbreviation/legend definitions AND tier definitions
found ANYWHERE in the document (page headers, footers, sidebars, dedicated legend sections).

──────────────────────────────────────────────────────────────
TYPE 1 — ABBREVIATION CODES
──────────────────────────────────────────────────────────────
Look for patterns like: 'ST = Step Therapy', 'PA = Prior Authorization', 'QL = Quantity Limit'.
Also look for: 'NDS = Non-Dispensing Supply', 'LA = Limitation on Age', 'EX = Excluded Drug',
'B/D = Brand/Drug', 'NM', 'SP', etc.

TYPE 2 — TIER DEFINITIONS (CRITICAL — ALWAYS EXTRACT THESE)
──────────────────────────────────────────────────────────────
Look for tier explanation sections with patterns like:
  'Tier 1 - Preferred Generic Drugs: This tier includes commonly prescribed generic drugs...'
  'Tier 2 - Generic Drugs: This tier includes...'
  'Tier 3 - Preferred Brand Drugs: This tier includes preferred brand-name drugs...'
  'Tier 4 - Non-Preferred Drugs: This tier includes higher-priced brand name drugs...'
  'Tier 5 - Specialty Tier drugs: This tier includes high-cost drugs...'
  'Tier 6 - Select Care Diabetic Drugs: This tier includes...'

For tier definitions:
  - Acronym     = 'Tier 1', 'Tier 2', etc.
  - Expansion   = The tier name (e.g. 'Preferred Generic Drugs', 'Non-Preferred Drugs').
  - Explanation = The full description text.

Extract EVERY abbreviation definition AND tier definition found in the document.""",

                    "items": {
                        "type": "object",
                        "properties": {
                            "Acronym": {
                                "type": "string",
                                "description": "The abbreviation code OR tier identifier. Examples: 'ST', 'PA', 'QL', 'SP', 'NM', 'LA', 'B/D', 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5', 'Tier 6'. Extract ONLY the code/identifier here."
                            },
                            "Expansion": {
                                "type": "string",
                                "description": "What the abbreviation stands for OR the tier name. Examples: 'Step Therapy', 'Prior Authorization', 'Quantity Limit', 'Preferred Generic Drugs', 'Non-Preferred Drugs'."
                            },
                            "Explanation": {
                                "type": ["string", "null"],
                                "description": "Additional explanation if provided. For tier definitions, this is the full description text (e.g. 'This tier includes commonly prescribed generic drugs. Drugs in Tier 1 will typically be your most affordable option.'). Null if no further explanation is given."
                            }
                        },
                        "required": ["Acronym", "Expansion"]
                    }
                }

            },
            "required": ["DrugInformation"]
        }
    }
}
