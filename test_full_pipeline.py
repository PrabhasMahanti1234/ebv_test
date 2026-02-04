#!/usr/bin/env python
"""Test the full pipeline to identify where drugs are being merged"""

from pdf_processing import process_pdf_with_mistral_ocr
from utils import clean_drug_name, normalize_drug_tier
from io import BytesIO
import json

pdf_path = r'C:\Users\VH0000547\Downloads\ebv_test\download_pdf.pdf'

print("=" * 80)
print("TESTING FULL PIPELINE - DRUG MERGING ISSUE")
print("=" * 80)

# Step 1: Process PDF with OCR
print("\n[STEP 1] Processing PDF with OCR...")
with open(pdf_path, 'rb') as f:
    pdf_bytes = BytesIO(f.read())

result, method, costs = process_pdf_with_mistral_ocr(pdf_bytes, payer_name='Test', filename='test.pdf')

drug_table = result.get("drug_table", [])
print(f"Extracted {len(drug_table)} drugs from OCR")

# Show first 10 drugs BEFORE cleaning
print("\n[STEP 2] Drugs BEFORE clean_drug_name():")
print("-" * 80)
for i, drug in enumerate(drug_table[:10]):
    print(f"{i+1}. {drug.get('drug_name')}")
    print(f"   Tier: {drug.get('drug_tier')}")
    print(f"   Requirements: {drug.get('drug_requirements')}")
    print()

# Step 3: Apply clean_drug_name
print("\n[STEP 3] Applying clean_drug_name()...")
for drug in drug_table:
    if drug.get("drug_name"):
        original = drug["drug_name"]
        cleaned = clean_drug_name(drug["drug_name"])
        drug["drug_name"] = cleaned
        if original != cleaned:
            print(f"  Changed: '{original}' -> '{cleaned}'")

# Show first 10 drugs AFTER cleaning
print("\n[STEP 4] Drugs AFTER clean_drug_name():")
print("-" * 80)
for i, drug in enumerate(drug_table[:10]):
    print(f"{i+1}. {drug.get('drug_name')}")
    print(f"   Tier: {drug.get('drug_tier')}")
    print(f"   Requirements: {drug.get('drug_requirements')}")
    print()

# Step 4: Apply normalize_drug_tier
print("\n[STEP 5] Applying normalize_drug_tier()...")
for drug in drug_table:
    if drug.get("drug_tier"):
        original = drug["drug_tier"]
        normalized = normalize_drug_tier(drug["drug_tier"])
        drug["drug_tier"] = normalized
        if original != normalized:
            print(f"  Changed: '{original}' -> '{normalized}'")

# Final check
print("\n[STEP 6] Final drug list (first 10):")
print("-" * 80)
for i, drug in enumerate(drug_table[:10]):
    print(f"{i+1}. {drug.get('drug_name')}")
    print(f"   Tier: {drug.get('drug_tier')}")
    print(f"   Requirements: {drug.get('drug_requirements')}")
    print()

# Check for merged drugs
print("\n[STEP 7] Checking for merged drugs...")
print("-" * 80)
merged_count = 0
for drug in drug_table:
    name = drug.get('drug_name', '')
    # Check if drug name contains multiple drug names (heuristic: multiple "oral" or "tablet" mentions)
    if name.count(' oral ') > 1 or name.count(' tablet ') > 2:
        merged_count += 1
        print(f"WARNING POSSIBLE MERGE: {name[:100]}...")

if merged_count == 0:
    print("No merged drugs detected!")
else:
    print(f"\nFound {merged_count} potentially merged drugs")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
