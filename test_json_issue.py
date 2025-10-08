#!/usr/bin/env python3

import json
import re
from pdf_processing import robust_json_repair

# Test with a valid JSON that the LLM is returning
test_json = '''{
  "drug_table": [
    {
      "drug_name": "abacavir",
      "drug_tier": "Tier 3",
      "drug_requirements": null
    },
    {
      "drug_name": "abacavir-lamivudine",
      "drug_tier": "Tier 4",
      "drug_requirements": null
    }
  ],
  "acronyms": [],
  "tiers": []
}'''

print("Original JSON:")
print(test_json)
print("\n" + "="*50 + "\n")

# Test if the original JSON is valid
try:
    original_parsed = json.loads(test_json)
    print(f"Original JSON is valid. drug_table length: {len(original_parsed['drug_table'])}")
    print(f"First drug: {original_parsed['drug_table'][0]}")
except Exception as e:
    print(f"Original JSON is invalid: {e}")

print("\n" + "="*50 + "\n")

# Test robust_json_repair
result = robust_json_repair(test_json)
print("After robust_json_repair:")
print(f"Result type: {type(result)}")
print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
if isinstance(result, dict) and 'drug_table' in result:
    print(f"drug_table length: {len(result['drug_table'])}")
    if result['drug_table']:
        print(f"First drug: {result['drug_table'][0]}")
    else:
        print("drug_table is empty!")
