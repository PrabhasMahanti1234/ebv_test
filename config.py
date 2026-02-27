import os
import logging
from dotenv import load_dotenv
from mistralai import Mistral
import nest_asyncio
import json
from collections import defaultdict
load_dotenv()
# Setup
nest_asyncio.apply()

# Configure Logging
def setup_logging():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler("pipeline_log.txt", mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Failed LLM JSON Logger
    llm_logger = logging.getLogger("failed_llm_json")
    llm_logger.setLevel(logging.ERROR)
    llm_handler = logging.FileHandler("failed_llm_json.log", mode='a', encoding='utf-8')
    llm_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    llm_logger.addHandler(llm_handler)
    llm_logger.propagate = False # Prevent double logging to root

    return root_logger, llm_logger

logger, llm_fail_logger = setup_logging()

# -----------------------------
# Configuration
# -----------------------------


EXCEL_FILE_PATH = "errors.xlsx"
PDF_FOLDER = "druglist1"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# PROCESS_COUNT = 16 #No of PDFS
# LLM_PAGE_WORKERS = 8 #No of pages
MAX_PAGES_PER_OCR_REQUEST = 4 # Mistral's HARD LIMIT for structured output (document_annotations)
MAX_OCR_WORKERS = 6  # Increased from 4 to 6 for faster parallel processing


DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}
print(DB_CONFIG)
# Target fields for structured extraction
TARGET_FIELDS = ["drug_name", "drug_tier", "drug_requirements"]
DB_FIELDS = ["drug_name", "drug_tier", "drug_requirements"]

# Global storage for processed data
ALL_PROCESSED_DATA = []
ALL_RAW_CONTENT = {}

# Initialize clients
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Add these constants after the existing configuration
MAX_RETRIES = 5
BACKOFF_MULTIPLIER = 2

# -----------------------------
# Optimization Settings
# -----------------------------
# Parallel OCR chunk processing (Optimization 1)
#OCR_CHUNK_WORKERS = 10  # Number of OCR chunks to process in parallel


# Mistral-specific rate limiting (Optimization 7) - Mistral has higher limits than Bedrock
MISTRAL_OCR_RATE_LIMIT = 0.05  # 50ms between Mistral OCR calls (faster)

# Smart page pre-filtering (Optimization 5)
ENABLE_PAGE_PREFILTER = True  # Enable/disable page pre-filtering before OCR
MIN_PAGE_TEXT_LENGTH = 100  # Minimum text characters to consider a page worth processing
SKIP_INDEX_PAGES = True  # Skip pages that look like index/TOC pages
MISTRAL_OCR_MODEL = "mistral-ocr-latest"

# Add these constants after your existing configuration
MISTRAL_OCR_COST_PER_1K_PAGES = 3.0   # $3.00 per 1000 pages (with document_annotation)

CLIENT_TIMEOUT = 60.0  # 60 seconds for general read/write timeouts (was 300s, reduced for faster failure detection)
CONNECT_TIMEOUT = 10.0  # 10 seconds for establishing a connection (was 15s)

# Global cost tracking dictionary
COST_TRACKER = {
    'payer_costs': defaultdict(lambda: {
        'mistral_ocr_pages': 0,
        'mistral_cost': 0.0,
        'total_cost': 0.0,
        'pdfs_processed': 0,
        'llm_calls': 0
    }),
    'total_pages': 0,
    'total_cost': 0.0,
    'total_llm_calls': 0,
    'total_pdfs_processed': 0
}

# -----------------------------
# PDF Page Processing Control
# -----------------------------
#
# This setting allows you to control which pages of a PDF are processed.
#
# How it works:
# - Keys are unique substrings of filenames (e.g., "Cigna", "UnitedHealthcare").
# - Values can be:
#   - "all": Processes every page.
#   - A list containing numbers and/or strings for ranges.
#     Example: [1, 5, "10-20", 35] will process pages 1, 5, 10 through 20, and 35.
# - The special key "default" applies to any file NOT matched by other keys.

PDF_PAGE_PROCESSING_CONFIG = {
    # Process pages to find the drug table
    # Increased range to capture full drug lists (some PDFs have 60+ pages of drugs)
    # Use "all" to process everything, or specific ranges like ["1-100"]
    "default": "all"  # Only process page 313 (drug data page), skip 314 (index page)
    # Override for specific payers if needed:
    # "Cigna": "all",
    # "Select": ["1-80"],
}