"""Debug script to capture OCR markdown content"""
import os
import json
from pdf_extraction import OCR_ANNOTATION_SCHEMA
from pdf_extraction import OCR_ANNOTATION_SCHEMA
from mistralai import Mistral
import os
import fitz  # PyMuPDF
import io

# Get a sample PDF
import glob
# Find any PDF in the directory
pdf_files = glob.glob(r"c:\Users\VH0000547\Downloads\ebv_test\*.pdf")
if not pdf_files:
    print("No PDF files found!")
    exit(1)
pdf_path = pdf_files[0]

print(f"Loading PDF: {pdf_path}")
doc = fitz.open(pdf_path)
print(f"PDF has {len(doc)} pages")

# Extract specific pages (8-10)
pages_to_extract = [7, 8, 9]  # 0-indexed, so 8,9,10 in 1-indexed
subdoc = fitz.open()
for page_num in pages_to_extract:
    if page_num < len(doc):
        subdoc.insert_pdf(doc, from_page=page_num, to_page=page_num)

print(f"Extracted {len(subdoc)} pages for OCR")

# Save to bytes
pdf_bytes = subdoc.write()

# Upload to Mistral and process
from mistralai import Mistral
from mistralai.models import DocumentURLChunk

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("MISTRAL_API_KEY not set!")
    exit(1)

client = Mistral(api_key=api_key)

# Upload file
import base64
uploaded = client.files.upload(
    file={"file_name": "test.pdf", "content": pdf_bytes},
    purpose="ocr"
)
print(f"Uploaded file: {uploaded.id}")

# Get signed URL
signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=300)

# Run OCR
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document=DocumentURLChunk(document_url=signed_url.url),
    document_annotation_format=OCR_ANNOTATION_SCHEMA,
    include_image_base64=False
)

print(f"\n{'='*80}")
print("OCR RESPONSE - RAW MARKDOWN FROM EACH PAGE")
print(f"{'='*80}\n")

for page_idx, page in enumerate(ocr_response.pages):
    print(f"\n--- PAGE {page_idx + 1} MARKDOWN ---\n")
    if hasattr(page, 'markdown') and page.markdown:
        print(page.markdown[:5000])  # First 5000 chars
    else:
        print("No markdown content")

print(f"\n{'='*80}")
print("OCR RESPONSE - DOCUMENT ANNOTATION (STRUCTURED DATA)")
print(f"{'='*80}\n")

if hasattr(ocr_response, 'document_annotation') and ocr_response.document_annotation:
    import json
    annotation = ocr_response.document_annotation
    if isinstance(annotation, str):
        annotation = json.loads(annotation)
    
    drugs = annotation.get("DrugInformation", [])
    print(f"Found {len(drugs)} drugs in annotation")
    print("\nFirst 5 drugs:")
    for i, drug in enumerate(drugs[:5]):
        print(f"  {i+1}. {drug}")
else:
    print("No document annotation found")

# Cleanup
client.files.delete(file_id=uploaded.id)
print("\nDone!")
