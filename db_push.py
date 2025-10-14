#!/usr/bin/env python3
"""
db_push_chunked.py

- Streams a large CSV file.
- Processes rows in configurable chunks.
- For each chunk, it performs a set-based "upsert" using psycopg2's fast execute_values.
- The ON CONFLICT target is now explicitly defined by the business key columns.
- Generates UUID for missing IDs, normalizes booleans, and sets a last_updated_date.
"""

import csv
import os
import logging
import uuid
from datetime import datetime
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# ---------- CONFIG ----------
DB_CONFIG = {
    'host': '127.0.0.1',
    'database': 'ebvdb',
    'user': 'postgres',
    'password': 'm2I6k7aMRiys8FchvPlE',
    'port': 5433
}

CSV_FILE_PATH = 'missed_drug_formulary.csv'
TARGET_TABLE = "drug_formulary_details"

# The number of rows to process and push to the DB in a single batch.
# Adjust this based on your system's memory and database performance.
CHUNK_SIZE = 10000

# Must match target table column names and include last_updated_date
COLUMNS = [
    'id', 'plan_id', 'payer_id', 'drug_name', 'ndc_code', 'jcode',
    'state_name', 'coverage_status', 'drug_tier', 'drug_requirements',
    'is_prior_authorization_required', 'is_step_therapy_required',
    'coverage_details', 'confidence_score', 'source_url', 'plan_name',
    'payer_name', 'file_name', 'is_quantity_limit_applied', 'status',
    'last_updated_date'
]

BOOLEAN_COLUMNS = [
    'is_prior_authorization_required', 'is_step_therapy_required',
    'is_quantity_limit_applied'
]

# *** IMPORTANT ***
# These are the columns that define a unique record in your table, based on the error message.
# This tells the database how to identify duplicates.
CONFLICT_TARGET_COLUMNS = ['plan_id', 'drug_name', 'drug_tier', 'drug_requirements']


# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_push_chunked")

# ---------- DB UTIL ----------
def connect_to_db():
    """Establishes a connection to the database."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False  # We will manage transactions manually
    logger.info("Successfully connected to the database.")
    return conn

# ---------- DATA TRANSFORMATION ----------
def transform_row(row_dict, now_str):
    """
    Takes a dictionary from the CSV reader, cleans it, and returns a tuple
    in the correct order for insertion.
    """
    record = []
    for col in COLUMNS:
        val = row_dict.get(col, "")

        if col == 'id':
            # Generate a UUID if the 'id' is missing.
            record.append(str(uuid.uuid4()) if not val else val)
        elif col == 'last_updated_date':
            record.append(now_str if not val else val)
        elif col in BOOLEAN_COLUMNS:
            # Normalize boolean values
            if val is None or str(val).strip() == "":
                record.append(None)
            else:
                v_lower = str(val).strip().lower()
                if v_lower in ('yes', 'y', 'true', 't', '1'):
                    record.append(True)
                elif v_lower in ('no', 'n', 'false', 'f', '0'):
                    record.append(False)
                else:
                    record.append(None) # Or handle as an error
        else:
            # Use None for empty strings so they are inserted as NULL in the DB
            record.append(val if val != "" else None)
    return tuple(record)


# ---------- MAIN ----------
def main():
    """Main execution function."""
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"CSV file not found: {CSV_FILE_PATH}")
        return

    conn = None
    try:
        conn = connect_to_db()

        # Build the SQL query for the upsert operation
        # This is done once and reused for every chunk.
        cols_sql = sql.SQL(', ').join(map(sql.Identifier, COLUMNS))
        
        # Define the columns to use for conflict detection
        conflict_cols_sql = sql.SQL(', ').join(map(sql.Identifier, CONFLICT_TARGET_COLUMNS))
        
        # Define which columns should be updated if a conflict occurs.
        # We update all columns except the ones that define the conflict.
        update_cols = [c for c in COLUMNS if c not in CONFLICT_TARGET_COLUMNS]
        update_assignments = sql.SQL(', ').join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c)) for c in update_cols
        )

        # The complete UPSERT statement
        upsert_sql = sql.SQL("""
            INSERT INTO {target_table} ({cols})
            VALUES %s
            ON CONFLICT ({conflict_cols}) DO UPDATE SET
            {update_assignments};
        """).format(
            target_table=sql.Identifier(TARGET_TABLE),
            cols=cols_sql,
            conflict_cols=conflict_cols_sql,
            update_assignments=update_assignments
        )
        
        logger.info(f"Starting chunked upsert of '{CSV_FILE_PATH}' into '{TARGET_TABLE}'.")
        logger.info(f"Processing in chunks of {CHUNK_SIZE} rows.")

        chunk = []
        rows_processed = 0
        now_utc_str = datetime.utcnow().isoformat()

        with open(CSV_FILE_PATH, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                transformed = transform_row(row, now_utc_str)
                chunk.append(transformed)
                
                if len(chunk) >= CHUNK_SIZE:
                    with conn.cursor() as cur:
                        execute_values(cur, upsert_sql, chunk)
                    conn.commit()
                    rows_processed += len(chunk)
                    logger.info(f"Successfully pushed chunk. Total rows processed: {rows_processed}")
                    chunk = [] # Reset chunk

            # Process any remaining rows in the last chunk
            if chunk:
                with conn.cursor() as cur:
                    execute_values(cur, upsert_sql, chunk)
                conn.commit()
                rows_processed += len(chunk)
                logger.info(f"Successfully pushed final chunk. Total rows processed: {rows_processed}")

        logger.info("All data has been successfully upserted.")

    except psycopg2.Error as e:
        logger.error("A database error occurred:", exc_info=True)
        if conn:
            conn.rollback() # Rollback the failed transaction
    except Exception as e:
        logger.error("An unexpected error occurred:", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
        logger.info("Done.")

if __name__ == "__main__":
    main()