import os
import logging
from dotenv import load_dotenv
from mistralai import Mistral
import nest_asyncio
import boto3
import json
from collections import defaultdict
load_dotenv()
# Setup
nest_asyncio.apply()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------


EXCEL_FILE_PATH = "errors.xlsx"
PDF_FOLDER = "druglist1"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PROCESS_COUNT = 16 #No of PDFS
LLM_PAGE_WORKERS = 8 #No of pages
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


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
RATE_LIMIT_DELAY = 1.0  # Minimum seconds between API calls
MAX_RETRIES = 5
BACKOFF_MULTIPLIER = 2

# Add these constants after your existing configuration
BEDROCK_COST_PER_1K_TOKENS = 0.00022  # $0.00022 per 1000 tokens
MISTRAL_OCR_COST_PER_1K_PAGES = 1.0   # $1.00 per 1000 pages

CLIENT_TIMEOUT = 300.0  # 5 minutes for general read/write timeouts
CONNECT_TIMEOUT = 15.0  # 15 seconds for establishing a connection

# Global cost tracking dictionary
COST_TRACKER = {
    'payer_costs': defaultdict(lambda: {
        'bedrock_tokens': 0,
        'mistral_ocr_pages': 0,
        'bedrock_cost': 0.0,
        'mistral_cost': 0.0,
        'total_cost': 0.0,
        'pdfs_processed': 0,
        'llm_calls': 0
    }),
    'total_tokens': 0,
    'total_pages': 0,
    'total_cost': 0.0,
    'total_llm_calls': 0,
    'total_pdfs_processed': 0
}

# Initialize AWS Bedrock client
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bedrock_region = os.getenv('AWS_BEDROCK_REGION', 'us-east-1')

bedrock = boto3.client(
    service_name="bedrock-runtime", 
    region_name=bedrock_region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)
