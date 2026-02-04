#!/usr/bin/env python
"""Test script to verify PDF extraction fixes"""

from pdf_processing import process_pdf_with_mistral_ocr
from io import BytesIO

pdf_path = r'C:\Users\VH0000547\Downloads\ebv_test\download_pdf.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = BytesIO(f.read())

result, method, costs = process_pdf_with_mistral_ocr(pdf_bytes, payer_name='Test', filename='test.pdf')

drugs = result.get('drug_table', [])
print(f'Total drugs extracted: {len(drugs)}')
print(f'Method: {method}')
print()

# Show first 15 drugs
for i, drug in enumerate(drugs[:15]):
    print(f'{i+1}. {drug.get("drug_name")}')
    print(f'   Tier: {drug.get("drug_tier")}')
    print(f'   Requirements: {drug.get("drug_requirements")}')
    print()

# Check for tier mapping issues
print("\n=== TIER MAPPING ANALYSIS ===")
tier_counts = {}
for drug in drugs:
    tier = drug.get('drug_tier')
    if tier:
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

print("Tier distribution:")
for tier, count in sorted(tier_counts.items()):
    print(f"  {tier}: {count} drugs")

# Check for requirements issues
print("\n=== REQUIREMENTS ANALYSIS ===")
req_counts = {}
for drug in drugs:
    req = drug.get('drug_requirements')
    if req:
        req_counts[req] = req_counts.get(req, 0) + 1

print("Top 10 requirement codes:")
for req, count in sorted(req_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  '{req}': {count} drugs")
