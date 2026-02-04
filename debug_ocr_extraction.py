#!/usr/bin/env python
"""Debug script to see raw OCR output before consolidation"""

from pdf_processing import process_single_chunk_parallel
from pdf_core import _extract_pages_from_pdf
from io import BytesIO
import fitz
import json

pdf_path = r'C:\Users\VH0000547\Downloads\ebv_test\download_pdf.pdf'

# Extract pages 31-35 to see the drug table
with open(pdf_path, 'rb') as f:
    pdf_bytes = BytesIO(f.read())

# Extract pages 31-35
extracted_pdf = _extract_pages_from_pdf(pdf_bytes, [31, 32, 33, 34, 35])
if extracted_pdf:
    extracted_bytes = extracted_pdf.getvalue()
    
    # Create chunk info
    chunk_info = {
        'chunk_idx': 0,
        'chunk_pages': [1],  # First page of extracted PDF
        'original_pages': [31],
        'pdf_bytes': extracted_bytes,
        'ocr_schema': None  # Will be imported
    }
    
    # Import schema
    from pdf_extraction import OCR_ANNOTATION_SCHEMA
    chunk_info['ocr_schema'] = OCR_ANNOTATION_SCHEMA
    
    # Process the chunk
    result = process_single_chunk_parallel(chunk_info)
    
    print("=== RAW OCR EXTRACTION RESULTS ===\n")
    print(f"Drugs extracted: {len(result['drugs'])}\n")
    
    # Show first 20 drugs as extracted
    for i, drug in enumerate(result['drugs'][:20]):
        print(f"{i+1}. Drug Name: {drug.get('drug_name')}")
        print(f"   Tier: {drug.get('drug_tier')}")
        print(f"   Requirements: {drug.get('drug_requirements')}")
        print(f"   Category: {drug.get('category')}")
        print()
