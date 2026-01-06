import re
import pandas as pd
import uuid
import logging
from datetime import datetime, date
from database import get_db_connection
from utils import generate_filename, validate_required_files
from config import EXCEL_FILE_PATH

logger = logging.getLogger(__name__)

def get_or_create_payer(cursor, payer_data):
    """Get existing payer or create new one, preventing duplicates and handling status"""
    payer_name = payer_data['payer_name']
    state = payer_data.get('state', '')
    

    # First, try to find existing payer
    cursor.execute("""
        SELECT payer_id FROM payer_details 
        WHERE LOWER(TRIM(payer_name)) = LOWER(TRIM(%s)) 
        AND LOWER(TRIM(COALESCE(state, ''))) = LOWER(TRIM(%s))
    """, (payer_name, state))
    
    result = cursor.fetchone()
    if result:
        logger.debug(f"Found existing payer: {payer_name}")
        # Update status to processing for existing payer
        cursor.execute("""
            UPDATE payer_details 
            SET status = 'processing', last_updated_at = CURRENT_TIMESTAMP
            WHERE payer_id = %s
        """, (result[0],))
        return result[0]
    
    # Create new payer
    payer_id = str(uuid.uuid4())
    try:
        cursor.execute("""
            INSERT INTO payer_details (
                payer_id, payer_name, contact_phone, address_line_1, 
                address_line_2, city, state, zip_code, status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (payer_name, state) DO UPDATE SET
                contact_phone = EXCLUDED.contact_phone,
                address_line_1 = EXCLUDED.address_line_1,
                address_line_2 = EXCLUDED.address_line_2,
                city = EXCLUDED.city,
                zip_code = EXCLUDED.zip_code,
                status = 'processing',
                last_updated_at = CURRENT_TIMESTAMP
            RETURNING payer_id
        """, (
            payer_id, payer_data['payer_name'], payer_data.get('contact_phone'),
            payer_data.get('address_line_1'), payer_data.get('address_line_2'),
            payer_data.get('city'), payer_data.get('state'), payer_data.get('zip_code'),
            'processing',  # NEW: Set initial status to processing
            payer_data.get('created_at')
        ))
        
        result = cursor.fetchone()
        if result:
            if result[0] == payer_id:
                logger.info(f"Created new payer: {payer_name}")
            else:
                logger.debug(f"Updated existing payer: {payer_name}")
            return result[0]
        else:
            # Fallback - find the existing record
            cursor.execute("""
                SELECT payer_id FROM payer_details 
                WHERE LOWER(TRIM(payer_name)) = LOWER(TRIM(%s)) 
                AND LOWER(TRIM(COALESCE(state, ''))) = LOWER(TRIM(%s))
            """, (payer_name, state))
            result = cursor.fetchone()
            return result[0] if result else None
            
    except Exception as e:
        logger.error(f"Error creating/updating payer {payer_name}: {e}")
        # Try to find existing record
        cursor.execute("""
            SELECT payer_id FROM payer_details 
            WHERE LOWER(TRIM(payer_name)) = LOWER(TRIM(%s)) 
            AND LOWER(TRIM(COALESCE(state, ''))) = LOWER(TRIM(%s))
        """, (payer_name, state))
        result = cursor.fetchone()
        return result[0] if result else None

def get_date_for_db(date_obj):
    """
    Converts a pandas Timestamp, datetime.datetime, or NaT into a 
    Python datetime.date object for the database, or None if the date is missing.
    
    This function expects an already-converted date object, NOT a string.
    """
    # Handles NaT, None, or other "empty" pandas values
    if pd.isna(date_obj):
        return None

    # For Timestamp or datetime objects, extract just the date part.
    # This is safe because both types have a .date() method.
    if isinstance(date_obj, (datetime)):
        return date_obj.date()
    
    # If it's already a date object, just return it.
    if isinstance(date_obj, date):
        return date_obj

    # If we received something else (like a string that slipped through),
    # log a warning and return None. It's better to fix the upstream
    # data cleaning than to attempt a risky parse here.
    logger.warning(f"get_date_for_db received an unexpected type: {type(date_obj)}. Value: '{date_obj}'. Returning None.")
    return None

def get_or_create_plan(cursor, plan_data, payer_id):
    """Get existing plan or create new one, preventing duplicates and handling status"""
    plan_name = plan_data['plan_name']
    state_name = plan_data['state_name']
    payer_name = plan_data.get('payer_name', '')  # <-- Add this line

    # First, try to find existing plan
    cursor.execute("""
        SELECT plan_id FROM plan_details 
        WHERE payer_id = %s 
        AND LOWER(TRIM(plan_name)) = LOWER(TRIM(%s))
        AND LOWER(TRIM(state_name)) = LOWER(TRIM(%s))
    """, (payer_id, plan_name, state_name))
    
    result = cursor.fetchone()
    if result:
        logger.debug(f"Found existing plan: {plan_name}")
        # Update status to processing for existing plan
        cursor.execute("""
            UPDATE plan_details 
            SET status = 'processing', last_updated_date = CURRENT_TIMESTAMP
            WHERE plan_id = %s
        """, (result[0],))
        return result[0]
    
    # Create new plan
    plan_id = str(uuid.uuid4())
    try:
        cursor.execute("""
            INSERT INTO plan_details (
                plan_id, payer_id, payer_name, plan_name, state_name, formulary_url,
                source_link, formulary_date, status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (payer_id, plan_name, state_name) DO UPDATE SET
                formulary_url = EXCLUDED.formulary_url,
                source_link = EXCLUDED.source_link,
                formulary_date = EXCLUDED.formulary_date,
                status = 'processing',
                last_updated_date = CURRENT_TIMESTAMP
            RETURNING plan_id
        """, (
            plan_id, payer_id, payer_name, plan_data['plan_name'], plan_data['state_name'],
            plan_data.get('formulary_url'), plan_data.get('source_link'),
            plan_data.get('formulary_date'), 'processing',
            plan_data.get('created_at')
        ))
        
        result = cursor.fetchone()
        if result:
            if result[0] == plan_id:
                logger.info(f"Created new plan: {plan_name}")
            else:
                logger.debug(f"Updated existing plan: {plan_name}")
            return result[0]
        else:
            # Fallback - find the existing record
            cursor.execute("""
                SELECT plan_id FROM plan_details 
                WHERE payer_id = %s 
                AND LOWER(TRIM(plan_name)) = LOWER(TRIM(%s))
                AND LOWER(TRIM(state_name)) = LOWER(TRIM(%s))
            """, (payer_id, plan_name, state_name))
            result = cursor.fetchone()
            return result[0] if result else None
            
    except Exception as e:
        logger.error(f"Error creating/updating plan {plan_name}: {e}")
        # Try to find existing record
        cursor.execute("""
            SELECT plan_id FROM plan_details 
            WHERE payer_id = %s 
            AND LOWER(TRIM(plan_name)) = LOWER(TRIM(%s))
            AND LOWER(TRIM(state_name)) = LOWER(TRIM(%s))
        """, (payer_id, plan_name, state_name))
        result = cursor.fetchone()
        return result[0] if result else None

def populate_payer_and_plan_tables():
    """Populate payer_details and plan_details tables from Excel file with duplicate prevention"""
    logger.info("STEP 1: Populating Payer and Plan Tables from Excel")
    
    validate_required_files()
    
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name="Sheet1")
    logger.info(f"Loaded {len(df)} records from Excel")

    if df.empty:
        logger.warning(f"Excel file {EXCEL_FILE_PATH} is empty. No records to process.")
        return
    
    # Clean the column names
    df.columns = df.columns.astype(str).str.replace('\n', ' ').str.strip()
    logger.info("Cleaned DataFrame column names.")
    
    # Rename the misspelled column to the correct name if it exists
    if 'Formulory date' in df.columns:
        logger.warning("Found misspelled column 'Formulory date'. Renaming to 'Formulary Date'.")
        df.rename(columns={'Formulory date': 'Formulary Date'}, inplace=True)

    # <<< THE ROBUST FIX: A smarter date conversion loop >>>
# In populate_payer_and_plan_tables()

    # <<< THE ROBUST FIX: Let pandas infer the date format >>>
    date_cols = ['Formulary Date', 'Captured Date']

    for col in date_cols:
        if col in df.columns:
            # If the column is already a datetime type (pandas read it correctly), great!
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info(f"Column '{col}' was already a datetime type. No conversion needed.")
                continue

            # If it's an object/string, convert it, letting pandas figure out the format.
            logger.info(f"Column '{col}' is not a datetime type. Attempting to auto-parse to datetime.")
            
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors='coerce') # REMOVED the format string
            
            # Check if the conversion created new nulls, which indicates parsing problems.
            new_nulls = df[col].isna().sum() - original_nulls
            if new_nulls > 0:
                logger.warning(f"{new_nulls} values in '{col}' could not be parsed and were set to NULL.")
    
    # Add filename column (using the cleaned column names)
    df['Name'] = df.apply(lambda row: generate_filename(
        row['States Covered'], row['Company Name'], row['Plan Name']
    ), axis=1)
    
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:

            def clean_excel_string(value):
                """
                A robust function to clean strings read from Excel. It removes
                all common non-printable characters, control characters, and the
                Byte Order Mark (BOM) that cause URL truncation issues.
                """
                if pd.isna(value):
                    return None
                
                # Convert to string first
                cleaned_str = str(value).encode('ascii', 'ignore').decode('ascii') 
                
                # Regex to remove control characters and other non-printing chars,
                # including the BOM (\ufeff). This is more comprehensive.
                # This pattern matches characters in the C0 and C1 control blocks, plus BOM.
                cleaned_str = re.sub(r'[\x00-\x1F\x7F-\x9F\ufeff]', '', cleaned_str)
                
                # Finally, strip leading/trailing whitespace
                cleaned_str = cleaned_str.strip()
                
                return cleaned_str if cleaned_str else None
            
            payers_added = 0
            plans_added = 0
            
            for _, row in df.iterrows():
                # Extract payer data
                payer_data = {
                    'payer_name': str(row['Company Name']).strip() if pd.notna(row['Company Name']) else '',
                    'contact_phone': str(row['Contact Phone']).strip() if pd.notna(row['Contact Phone']) else None,
                    'address_line_1': str(row['Communication Address Line 1']).strip() if pd.notna(row['Communication Address Line 1']) else None,
                    'address_line_2': str(row['Communication Address Line 2']).strip() if pd.notna(row['Communication Address Line 2']) else None,
                    'city': str(row['City']).strip() if pd.notna(row['City']) else None,
                    'state': str(row['States Covered']).strip() if pd.notna(row['States Covered']) else None,
                    'zip_code': str(row['Zip']).strip() if pd.notna(row['Zip']) else None,
                    # This now works because the column is a proper Timestamp
                    'created_at': str(row.get('Captured Date', '')).strip() or None,
                
                    'payer_name': str(row.get('Company Name', '')).strip() or None,
                    'contact_phone': str(row.get('Contact Phone', '')).strip() or None,
                    'address_line_1': str(row.get('Communication Address Line 1', '')).strip() or None,
                    'address_line_2': str(row.get('Communication Address Line 2', '')).strip() or None,
                    'city': str(row.get('City', '')).strip() or None,
                    'state': str(row.get('States Covered', '')).strip() or None,
                    'zip_code': str(row.get('Zip', '')).strip() or None,
                    'created_at': get_date_for_db(row.get('Captured Date')), # CORRECTED
                }
                
                # Skip if essential data is missing
                if not payer_data['payer_name']:
                    logger.warning(f"Skipping row with missing payer name: {row}")
                    continue
                
                # Get or create payer
                payer_id = get_or_create_payer(cursor, payer_data)
                if not payer_id:
                    logger.error(f"Failed to get or create payer: {payer_data['payer_name']}")
                    continue

                # Extract plan data
                plan_data = {
                    'plan_name': str(row['Plan Name']).strip() if pd.notna(row['Plan Name']) else '',
                    'state_name': str(row['States Covered']).strip() if pd.notna(row['States Covered']) else '',
                    'payer_name': payer_data['payer_name'],  # <-- Add this line
                    'formulary_url': str(row['Formulary URL']).strip() if pd.notna(row['Formulary URL']) else None,
                    'source_link': str(row['Source Link']).strip() if pd.notna(row['Source Link']) else None,
                    'formulary_date': str(row.get('Formulary Date', '')).strip() or None,
                    'created_at': str(row.get('Captured Date', '')).strip() or None,
                    'plan_name': str(row.get('Plan Name', '')).strip() or None,
                    'state_name': str(row.get('States Covered', '')).strip() or None,
                    'payer_name': payer_data['payer_name'],
                    'formulary_url': str(row.get('Formulary URL', '')).strip() or None,
                    'source_link': str(row.get('Source Link', '')).strip() or None,
                    'formulary_date': get_date_for_db(row.get('Formulary Date')), # CORRECTED
                    'created_at': get_date_for_db(row.get('Captured Date')), # CORRECTED
                }
                
                # Skip if essential plan data is missing
                if not plan_data['plan_name'] or not plan_data['state_name']:
                    logger.warning(f"Skipping row with missing plan data: {row}")
                    continue
                
                # Get or create plan
                plan_id = get_or_create_plan(cursor, plan_data, payer_id)
                if plan_id:
                    plans_added += 1
                    logger.debug(f"Processed plan: {plan_data['plan_name']}")
            
            conn.commit()
            logger.info(f"Step 1 Complete! Processed {plans_added} plans")
            
            # Get final counts
            cursor.execute("SELECT COUNT(*) FROM payer_details")
            total_payers = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM plan_details")
            total_plans = cursor.fetchone()[0]
            
            logger.info(f"Database now contains {total_payers} payers and {total_plans} plans")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in Step 1: {e}")
            raise