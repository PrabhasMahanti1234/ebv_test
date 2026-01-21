# PDF Processing Fix Summary - Select 4 Tier CA IND

## Problems Identified

### 1. **Page Number Mapping Wrong**
The OCR returns page numbers relative to processed chunks (1-4), not the original PDF. The code already has the correct mapping logic in `pdf_processing.py` lines 216-229, which maps OCR chunk pages back to original PDF page numbers.

### 2. **Skipping Pages**  
Two causes:
- **Limited page range**: Config was set to only process pages 1-40, but drug data extends much further
- **Over-aggressive index detection**: The index page filters were incorrectly marking valid drug pages as index pages

### 3. **Skipping Drugs on Some Pages**
**Schema mismatch**: The template expected a separate "Dosage Form/Strength" column, but in the Select 4 Tier PDF:
- Drug names and dosages are COMBINED in the first column (e.g., "carbinoxamine maleate oral tablet 4 mg")
- The schema was looking for drugs in this format:
  ```
  Drug Name | Dosage Form/Strength | Tier | Requirements
  ```
- But the actual PDF has:
  ```
  Drug Name (with inline dosage) | Tier | Restrictions/Limits
  ```

## Solutions Applied

### Fix 1: Updated OCR Schema (pdf_extraction.py)
Modified the schema to handle both formats:
- Updated "Drug Name" description to accept inline dosages
- Made "Dosage Form/Strength" truly optional
- Added explicit instructions to extract COMPLETE text from first column
- Now supports both formats:
  - Format A: `AMOXICILLIN | TAB 875MG | Tier 1 | PA`
  - Format B: `carbinoxamine maleate oral tablet 4 mg | Tier 1 | ST`

### Fix 2: Increased Page Range (config.py)
Changed from `["1-40"]` to `["1-100"]` to ensure complete drug table capture.

### Fix 3: Page Number Mapping (Already Working)
The existing code in `pdf_processing.py` (lines 216-229) already correctly maps:
- OCR chunk page numbers (1,2,3,4) → Original PDF page numbers (e.g., 270,271,272,273)
- Uses the `original_pages` array passed to each chunk

## Testing Recommendation

Test the fix with:
```bash
python your_script.py --url "https://fm.formularynavigator.com/FBO/143/2026_Select_4_Tier_CA_IND.pdf"
```

### What to Verify:
1. ✅ All drug pages are processed (check logs for "📄 Extracted X pages")
2. ✅ Drugs with inline dosages are captured (e.g., "carbinoxamine maleate oral liquid")
3. ✅ Page numbers in database match actual PDF page numbers
4. ✅ No valid drug data is skipped due to index detection

## Additional Notes

### Index Detection Logic (Already Optimized)
The code has already been optimized to avoid false positives:
- Chunk-level index detection is DISABLED (lines 255-261)
- Only individual index entries are filtered out during cleaning
- Thresholds raised to 30-50% (from 20-40%) to reduce false positives

### Schema Flexibility
The schema now handles these variations:
- Category headers (gray rows): `FIRST GENERATION ANTIHISTAMINES`
- Combined drug+dosage: `azelastine nasal spray non-aerosol 137 mcg (0.1 %)`
- Tier formats: Both "Tier 1" and "Generic"/"Brand"
- Requirements: Exact text like "QL (60 ML per 30 days)" instead of boolean flags

## Key Code Sections

### Schema Change (pdf_extraction.py:87-96)
```python
"Drug Name": {
    "type": "string", 
    "description": """The complete drug name from the first/left-most column. This column may contain EITHER:
1. Just the drug name (e.g., 'AMOXICILLIN', 'AMPICILLIN')
2. Drug name WITH dosage form inline (e.g., 'carbinoxamine maleate oral liquid')
EXTRACT THE FULL TEXT from the first column, including any dosage information."""
},
"Dosage Form/Strength": {
    "type": ["string", "null"],
    "description": "Only fill this if there's a distinct second column."
}
```

### Page Range (config.py:104-110)
```python
PDF_PAGE_PROCESSING_CONFIG = {
    "default": ["1-100"]  # Increased from ["1-40"]
}
```

### Page Mapping (pdf_processing.py:216-229)
```python
ocr_page_num = item.get("page_number")
# Map OCR page number (1,2,3,4) to original PDF page numbers
if ocr_page_num and 1 <= ocr_page_num <= len(original_pages):
    actual_pdf_page = original_pages[ocr_page_num - 1]
```
