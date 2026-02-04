#!/usr/bin/env python
"""Quick test to verify the fixes"""

from pdf_processing import process_pdf_with_mistral_ocr
from io import BytesIO

pdf_path = r'C:\Users\VH0000547\Downloads\ebv_test\download_pdf.pdf'

print("Processing PDF...")
with open(pdf_path, 'rb') as f:
    pdf_bytes = BytesIO(f.read())

result, method, costs = process_pdf_with_mistral_ocr(pdf_bytes, payer_name='Test', filename='test.pdf')

drugs = result.get("drug_table", [])
print(f"\nTotal drugs extracted: {len(drugs)}\n")

# Show first 15 drugs
for i, drug in enumerate(drugs[:15]):
    print(f"{i+1}. {drug.get('drug_name')}")
    tier = drug.get('drug_tier')
    req = drug.get('drug_requirements')
    if tier:
        print(f"   Tier: {tier}")
    if req:
        print(f"   Requirements: {req}")
    print()

print(f"\n=== SUMMARY ===")
print(f"Total drugs: {len(drugs)}")
print(f"Drugs with tier: {sum(1 for d in drugs if d.get('drug_tier'))}")
print(f"Drugs with requirements: {sum(1 for d in drugs if d.get('drug_requirements'))}")
