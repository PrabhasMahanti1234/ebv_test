import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from urllib.parse import urljoin
import time
import requests

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup

from database import get_db_connection

logger = logging.getLogger(__name__)
DOWNLOAD_DIR = Path("druglist1")
MAX_WORKERS = 4 # Reduced for stability with Selenium

def sanitize_filename(filename):
    s = str(filename).strip()
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    s = s.replace(' ', '_')
    if not s.lower().endswith('.pdf'):
        s += ".pdf"
    return s[:240]

def download_file_with_requests(url, filepath, cookies):
    """Uses requests to download the final file, using cookies from Selenium."""
    logger.info(f"    -> Downloading final PDF content with requests...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    cookie_jar = requests.cookies.RequestsCookieJar()
    for cookie in cookies:
        cookie_jar.set(cookie['name'], cookie['value'], domain=cookie['domain'])

    with requests.get(url, headers=headers, cookies=cookie_jar, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info(f"✅ Downloaded and saved: {filepath}")

def download_and_save_pdf_with_selenium(plan_info):
    """
    Uses Selenium to navigate to a URL and robustly download the PDF.
    It handles direct PDF links, HTML pages, and dynamic JS pages.
    """
    state, payer, plan, url = plan_info
    filename_raw = f"{state}_{payer}_{plan}"
    filename = sanitize_filename(filename_raw)
    filepath = DOWNLOAD_DIR / filename

    logger.info(f"📥 Processing with Selenium for: {filename}")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = None
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(5) # Allow time for any redirects or JS to load

        current_url = driver.current_url
        
        # Case 1: The URL is a direct PDF link, and the browser has loaded it.
        if current_url.lower().endswith('.pdf'):
            logger.info("    -> Direct PDF link confirmed by browser.")
            download_file_with_requests(current_url, filepath, driver.get_cookies())
            return filepath

        # Case 2: It's an HTML page. We need to find the link.
        logger.info("    -> HTML page detected. Scraping for PDF link...")
        soup = BeautifulSoup(driver.page_source, 'lxml')
        links = soup.find_all('a', href=True)
        keywords = ['.pdf', 'formulary', 'drug-list', 'comprehensive', 'druglist']
        
        for link in links:
            href = link['href'].lower()
            if any(keyword in href for keyword in keywords):
                pdf_url = urljoin(driver.current_url, link['href'])
                logger.info(f"    -> Found potential PDF link: {pdf_url}")
                download_file_with_requests(pdf_url, filepath, driver.get_cookies())
                return filepath
        
        logger.error(f"❌ Failed to find a PDF link on the page for {filename}")
        return None

    except WebDriverException as e:
        logger.error(f"❌ Selenium WebDriver error for {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred in Selenium downloader for {filename}: {e}", exc_info=True)
        return None
    finally:
        if driver:
            driver.quit()

def download_and_scrape_pdfs():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("--- Starting PDF Download and Scrape Phase (Selenium-Powered) ---")
    
    plans_to_process = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pd.state_name, py.payer_name, pd.plan_name, pd.formulary_url
            FROM plan_details pd JOIN payer_details py ON pd.payer_id = py.payer_id
            WHERE pd.status = 'processing' AND pd.formulary_url IS NOT NULL AND pd.formulary_url != ''
        """)
        plans_to_process = cursor.fetchall()

    if not plans_to_process:
        logger.warning("No plans with formulary URLs found to download.")
        return []

    logger.info(f"📊 Found {len(plans_to_process)} plans to process for download.")
    
    successful_downloads = []
    failed_downloads = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for plan_info in plans_to_process:
            state, payer, plan, _ = plan_info
            filename = sanitize_filename(f"{state}_{payer}_{plan}")
            filepath = DOWNLOAD_DIR / filename
            if filepath.exists() and filepath.stat().st_size > 0:
                logger.info(f"⏭️ Already exists: {filename}")
                successful_downloads.append(filepath)
                continue
            
            # Submit the Selenium-based downloader for each plan
            futures.append(executor.submit(download_and_save_pdf_with_selenium, plan_info))

        for future in as_completed(futures):
            result_path = future.result()
            if result_path:
                successful_downloads.append(result_path)
            else:
                failed_downloads += 1

    logger.info("--- Download Phase Complete ---")
    logger.info(f"📈 Summary: {len(successful_downloads)} successful downloads (including existing), {failed_downloads} failed.")
    return successful_downloads