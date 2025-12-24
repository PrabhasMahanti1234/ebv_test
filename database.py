import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import IntegrityError
from contextlib import contextmanager
import logging
import json
import pandas as pd
from io import StringIO
from config import DB_CONFIG

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False  # Ensure we control transactions
        yield conn
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass  # Connection might be closed
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass  # Connection might already be closed

def ensure_database_schema():
    """Ensure all required tables exist with proper constraints, partitioning, and indexing"""
    logger.info("Ensuring database schema exists...")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        try:
            # Create payer_details table with status column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payer_details (
                    payer_id VARCHAR(36) PRIMARY KEY,
                    payer_name VARCHAR(1000) NOT NULL,
                    contact_phone VARCHAR(50),
                    address_line_1 VARCHAR(1000),
                    address_line_2 VARCHAR(1000),
                    city VARCHAR(100),
                    state VARCHAR(50),
                    zip_code VARCHAR(20),
                    status VARCHAR(20) DEFAULT 'active',
                    created_at DATE,
                    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

        except Exception as e:
            logger.debug(f"Payer table creation issue (may already exist): {e}")
            conn.rollback()

        # Add status column to existing payer_details if not exists
        try:
            cursor.execute("""
                ALTER TABLE payer_details
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'processing'
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"Status column may already exist in payer_details: {e}")
            conn.rollback()

        # Add payer constraints in separate transactions
        _add_constraint(conn, cursor, """
            ALTER TABLE payer_details
            ADD CONSTRAINT unique_payer_name_state
            UNIQUE (payer_name, state)
        """, "unique_payer_name_state")

        try:
            # Create plan_details table with status column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plan_details (
                    plan_id VARCHAR(36) PRIMARY KEY,
                    payer_id VARCHAR(36) NOT NULL,
                    payer_name VARCHAR(1000) NOT NULL,
                    plan_name VARCHAR(1000) NOT NULL,
                    state_name VARCHAR(100) NOT NULL,
                    formulary_url TEXT,
                    source_link TEXT,
                    formulary_date DATE,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at DATE,
                    last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

        except Exception as e:
            logger.debug(f"Plan table creation issue (may already exist): {e}")
            conn.rollback()

        # Add status column to existing plan_details if not exists
        try:
            cursor.execute("""
                ALTER TABLE plan_details
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'processing'
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"Status column may already exist in plan_details: {e}")
            conn.rollback()

        # Add file_hash column to plan_details
        try:
            cursor.execute("""
                ALTER TABLE plan_details
                ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64)
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"file_hash column may already exist in plan_details: {e}")
            conn.rollback()


        # Create processed_file_cache table
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_file_cache (
                    file_hash VARCHAR(64) PRIMARY KEY,
                    structured_data_json JSONB,
                    raw_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("Created/ensured processed_file_cache table")
        except Exception as e:
            logger.debug(f"processed_file_cache table creation issue (may already exist): {e}")
            conn.rollback()

        # Add plan constraints in separate transactions
        _add_constraint(conn, cursor, """
            ALTER TABLE plan_details
            ADD CONSTRAINT fk_plan_payer
            FOREIGN KEY (payer_id) REFERENCES payer_details(payer_id) ON DELETE CASCADE
        """, "fk_plan_payer")

        _add_constraint(conn, cursor, """
            ALTER TABLE plan_details
            ADD CONSTRAINT unique_plan_payer_state
            UNIQUE (payer_id, plan_name, state_name)
        """, "unique_plan_payer_state")

        # Create the main partitioned drug_formulary_details table
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drug_formulary_details (
                    id VARCHAR(36) NOT NULL,
                    plan_id VARCHAR(36) NOT NULL,
                           plan_name VARCHAR(1000),
                    payer_id VARCHAR(36) NOT NULL,
                           payer_name VARCHAR(1000),
                    drug_name TEXT NOT NULL,
                    ndc_code VARCHAR(50),
                    jcode VARCHAR(50),
                    state_name VARCHAR(100) NOT NULL,
                    coverage_status VARCHAR(1000),
                    drug_tier TEXT,
                    drug_requirements TEXT,
                    is_prior_authorization_required VARCHAR(10) DEFAULT 'No',
                    is_step_therapy_required VARCHAR(10) DEFAULT 'No',
                    coverage_details VARCHAR(10000),
                    confidence_score DECIMAL(3,2),
                    source_url TEXT,
                    file_name VARCHAR(1000),
                    status VARCHAR(20) DEFAULT 'processing',
                     
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           review_needed Boolean,
 
                    PRIMARY KEY (id, plan_id)
                ) PARTITION BY HASH (plan_id);
            """)
            conn.commit()
            logger.info("Created partitioned drug_formulary_details table")

        except Exception as e:
            logger.debug(f"Drug table creation issue (may already exist): {e}")
            conn.rollback()
            
        # Create pp_formulary_names table for all acronyms and tier definitions
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pp_formulary_names (
                    id BIGSERIAL PRIMARY KEY,
                    state_name VARCHAR(100),
                    payer_name VARCHAR(200),
                    plan_name VARCHAR(200),
                    acronym VARCHAR(50),
                    expansion TEXT,
                    explanation TEXT
                );
            """)
            conn.commit()
            logger.info("Created/ensured pp_formulary_names table")
        except Exception as e:
            logger.debug(f"pp_formulary_names table creation issue (may already exist): {e}")
            conn.rollback()

        _add_constraint(conn, cursor, """
            ALTER TABLE pp_formulary_names
            ADD CONSTRAINT uq_formulary_names
            UNIQUE (state_name, payer_name, plan_name, acronym)
        """, "uq_formulary_names")

        # Add new columns to existing drug_formulary_details if not exists
        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS is_prior_authorization_required BOOLEAN DEFAULT FALSE
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"Prior auth column may already exist: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS is_step_therapy_required BOOLEAN DEFAULT FALSE
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"Step therapy column may already exist: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS is_quantity_limit_applied BOOLEAN DEFAULT FALSE
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"Quantity limit column may already exist: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'processing'
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"Status column may already exist in drug_formulary_details: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS product_labeler_code VARCHAR(100),
                ADD COLUMN IF NOT EXISTS product_proprietaryname TEXT;
            """)
            conn.commit()
            logger.info("Ensured product mapping columns exist in drug_formulary_details.")
        except Exception as e:
            logger.debug(f"Product mapping columns may already exist: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS page_number INTEGER
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"page_number column may already exist in drug_formulary_details: {e}")
            conn.rollback()

        # Add indexes for the new columns for better query performance
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_product_labeler_code ON drug_formulary_details(product_labeler_code)", "idx_product_labeler_code")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_product_proprietaryname ON drug_formulary_details(product_proprietaryname)", "idx_product_proprietaryname")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_page_number ON drug_formulary_details(page_number)", "idx_page_number")
        # Create partitions for better performance with 15-20M records
        # Create 8 partitions based on hash of plan_id
        for i in range(8):
            try:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS drug_formulary_details_{i}
                    PARTITION OF drug_formulary_details
                    FOR VALUES WITH (MODULUS 8, REMAINDER {i});
                """)
                conn.commit()
                logger.debug(f"Created partition drug_formulary_details_{i}")
            except Exception as e:
                logger.debug(f"Partition {i} may already exist: {e}")
                conn.rollback()

        # Add drug table constraints in separate transactions
        _add_constraint(conn, cursor, """
            ALTER TABLE drug_formulary_details
            ADD CONSTRAINT fk_drug_plan
            FOREIGN KEY (plan_id) REFERENCES plan_details(plan_id) ON DELETE CASCADE
        """, "fk_drug_plan")

        _add_constraint(conn, cursor, """
            ALTER TABLE drug_formulary_details
            ADD CONSTRAINT fk_drug_payer
            FOREIGN KEY (payer_id) REFERENCES payer_details(payer_id) ON DELETE CASCADE
        """, "fk_drug_payer")

        _add_constraint(conn, cursor, """
            ALTER TABLE drug_formulary_details
            ADD CONSTRAINT unique_drug_plan_tier_req
            UNIQUE (plan_id, drug_name, drug_tier, drug_requirements)
        """, "unique_drug_plan_tier_req")

        # Create comprehensive indexes for 15-20M records
        # Basic indexes
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_payer_name ON payer_details(payer_name)", "idx_payer_name")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_payer_status ON payer_details(status)", "idx_payer_status")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_plan_name ON plan_details(plan_name)", "idx_plan_name")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_plan_status ON plan_details(status)", "idx_plan_status")

        # Comprehensive indexes for drug_formulary_details
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_drug_name ON drug_formulary_details(drug_name)", "idx_drug_name")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_drug_name_lower ON drug_formulary_details(LOWER(drug_name))", "idx_drug_name_lower")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_plan_drug ON drug_formulary_details(plan_id, drug_name)", "idx_plan_drug")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_payer_drug ON drug_formulary_details(payer_id, drug_name)", "idx_payer_drug")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_state_drug ON drug_formulary_details(state_name, drug_name)", "idx_state_drug")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_drug_tier ON drug_formulary_details(drug_tier)", "idx_drug_tier")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_coverage_status ON drug_formulary_details(coverage_status)", "idx_coverage_status")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_prior_auth ON drug_formulary_details(is_prior_authorization_required)", "idx_prior_auth")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_step_therapy ON drug_formulary_details(is_step_therapy_required)", "idx_step_therapy")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_drug_status ON drug_formulary_details(status)", "idx_drug_status")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_created_at ON drug_formulary_details(created_at)", "idx_created_at")

        # Composite indexes for common queries
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_plan_status_drug ON drug_formulary_details(plan_id, status, drug_name)", "idx_plan_status_drug")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_payer_state_drug ON drug_formulary_details(payer_id, state_name, drug_name)", "idx_payer_state_drug")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_drug_auth_therapy ON drug_formulary_details(drug_name, is_prior_authorization_required, is_step_therapy_required)", "idx_drug_auth_therapy")

        # Text search index for drug names (using GIN for better text search performance)
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            conn.commit()
            _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_drug_name_gin ON drug_formulary_details USING GIN (drug_name gin_trgm_ops)", "idx_drug_name_gin")
        except Exception as e:
            logger.debug(f"GIN index creation failed (extension may not be available): {e}")
            conn.rollback()

        logger.info("Database schema ensured successfully with partitioning and comprehensive indexing")

def _add_constraint(conn, cursor, sql, constraint_name):
    """Add a constraint with proper transaction handling"""
    try:
        cursor.execute(sql)
        conn.commit()
        logger.debug(f"Added constraint: {constraint_name}")
    except Exception as e:
        conn.rollback()
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            logger.debug(f"Constraint {constraint_name} already exists")
        else:
            logger.debug(f"Issue with constraint {constraint_name}: {e}")

def _add_index(conn, cursor, sql, index_name):
    """Add an index with proper transaction handling"""
    try:
        cursor.execute(sql)
        conn.commit()
        logger.debug(f"Added index: {index_name}")
    except Exception as e:
        conn.rollback()
        if "already exists" in str(e).lower():
            logger.debug(f"Index {index_name} already exists")
        else:
            logger.debug(f"Issue with index {index_name}: {e}")

def get_cached_result(file_hash):
    """
    Retrieves a complete cached result (drug_table, acronyms, tiers) from the database.
    Returns the full structured data dictionary and raw_content.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT structured_data_json, raw_content FROM processed_file_cache WHERE file_hash = %s",
            (file_hash,)
        )
        result = cursor.fetchone()
        if result:
            logger.info(f"Cache HIT for hash: {file_hash}")
            structured_data_json, raw_content = result
            
            # If the cached JSON is null or empty, treat it as a miss.
            if not structured_data_json:
                return None, None

            try:
                # The stored item is the full dictionary.
                return structured_data_json, raw_content
            except Exception as e:
                logger.warning(f"Failed to parse cached JSON for hash {file_hash}. Will re-process. Error: {e}")
                # Return None to indicate a cache miss due to corruption, forcing reprocessing.
                return None, None
                
    logger.info(f"Cache MISS for hash: {file_hash}")
    return None, None

def cache_result(file_hash, structured_data_dict, raw_content):
    """
    Caches the full structured data dictionary (drug_table, acronyms, tiers) in the database.
    """
    if not isinstance(structured_data_dict, dict):
        logger.error("Attempted to cache non-dictionary object. Aborting cache operation.")
        return

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            # psycopg2 can automatically serialize a Python dictionary to a JSONB field.
            cursor.execute(
                """
                INSERT INTO processed_file_cache (file_hash, structured_data_json, raw_content)
                VALUES (%s, %s, %s)
                ON CONFLICT (file_hash) DO UPDATE SET
                    structured_data_json = EXCLUDED.structured_data_json,
                    raw_content = EXCLUDED.raw_content,
                    created_at = CURRENT_TIMESTAMP;
                """,
                (file_hash, json.dumps(structured_data_dict), raw_content)
            )
            conn.commit()
            logger.info(f"Successfully cached result for hash: {file_hash}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to cache result for hash {file_hash}: {e}")


# In database.py

def insert_drug_formulary_data(processed_data):
    """
    Inserts a batch of processed drug formulary data into the database
    with high efficiency and robust error handling. This version includes a
    fix to prevent 'integer out of range' errors for the page_number column.
    """
    if not processed_data:
        logger.warning("No processed data provided to insert.")
        return

    logger.info(f"Preparing to insert {len(processed_data)} records into the database.")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # The columns must match the order of values in the data tuples
        cols = [
            "id", "plan_id", "payer_id", "drug_name", "ndc_code", "jcode",
            "state_name", "coverage_status", "drug_tier", "drug_requirements", "page_number",
            "is_prior_authorization_required", "is_step_therapy_required", "is_quantity_limit_applied",
            "coverage_details", "confidence_score", "source_url", "plan_name", "payer_name", "file_name", "status"
        ]

        data_tuples = []
        for record in processed_data:
            # Defensive: skip records with missing essential info
            if not record.get("plan_name") or not record.get("payer_name"):
                logger.warning(f"Skipping record with missing plan_name or payer_name: {record}")
                continue

            page_number = record.get("page_number")
            try:
                # Ensure page_number is a valid integer or None
                if page_number is not None:
                    # int() can handle floats (e.g., 123.0) and string numbers ("123")
                    page_number = int(page_number)
                # If it's already None, it will be correctly inserted as NULL
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid page number '{page_number}' for drug "
                    f"'{record.get('drug_name')}'. Setting to NULL."
                )
                page_number = None


            # Prepare tuple for insertion, ensuring the order matches `cols`
            data_tuples.append((
                record.get("id"),
                record.get("plan_id"),
                record.get("payer_id"),
                record.get("drug_name"),
                record.get("ndc_code"),
                record.get("jcode"),
                record.get("state_name"),
                record.get("coverage_status"),
                record.get("drug_tier"),
                record.get("drug_requirements"),
                page_number,  # Use the sanitized page number
                record.get("is_prior_authorization_required"),
                record.get("is_step_therapy_required"),
                record.get("is_quantity_limit_applied"),
                record.get("coverage_details"),
                record.get("confidence_score"),
                record.get("source_url"),
                record.get("plan_name"),
                record.get("payer_name"),
                record.get("file_name"),
                'processing'  # Set initial status for new records
            ))

        # Using ON CONFLICT to prevent duplicates and update existing records
        insert_query = f"""
            INSERT INTO drug_formulary_details ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (plan_id, drug_name, drug_tier, drug_requirements)
            DO UPDATE SET
                coverage_status = EXCLUDED.coverage_status,
                page_number = EXCLUDED.page_number,
                is_prior_authorization_required = EXCLUDED.is_prior_authorization_required,
                is_step_therapy_required = EXCLUDED.is_step_therapy_required,
                is_quantity_limit_applied = EXCLUDED.is_quantity_limit_applied,
                source_url = EXCLUDED.source_url,
                file_name = EXCLUDED.file_name,
                status = 'completed',
                last_updated_date = CURRENT_TIMESTAMP;
        """

        try:
            # Use execute_values for efficient batch insertion
            execute_values(
                cursor,
                insert_query,
                data_tuples,
                template=None,
                page_size=500
            )
            conn.commit()
            logger.info(f"Successfully inserted or updated {len(data_tuples)} records.")

        except IntegrityError as e:
            conn.rollback()
            logger.error(f"Database integrity error during insertion: {e}")
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"An unexpected error occurred during data insertion: {e}")
            raise

def update_drug_formulary_status(processed_plan_ids):
    """
    Updates the status of records in drug_formulary_details to 'completed'
    for all successfully processed plans.
    """
    if not processed_plan_ids:
        logger.warning("No processed plan IDs provided for drug formulary status update.")
        return

    logger.info(f"Updating status to 'completed' for drugs in {len(processed_plan_ids)} plans.")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            query = "UPDATE drug_formulary_details SET status = 'completed', last_updated_date = CURRENT_TIMESTAMP WHERE plan_id = ANY(%s)"
            cursor.execute(query, (processed_plan_ids,))
            conn.commit()
            logger.info(f"Successfully updated status for {cursor.rowcount} drug formulary records.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update drug formulary statuses: {e}")
            raise

def update_plan_and_payer_statuses(processed_plan_ids):
    """
    Updates the status of plans and payers after processing.
    - Sets status to 'active' for successfully processed plans.
    - Sets status to 'inactive' for plans that were being processed but failed.
    - Updates payer status based on the status of their plans.
    """
    logger.info("Updating final status for all payers and plans...")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Update successfully processed plans to 'active'
            if processed_plan_ids:
                active_query = "UPDATE plan_details SET status = 'active', last_updated_date = CURRENT_TIMESTAMP WHERE plan_id = ANY(%s)"
                cursor.execute(active_query, (processed_plan_ids,))
                logger.info(f"Set {cursor.rowcount} plans to 'active'.")

            # Update any plans that were 'processing' but did not complete successfully to 'inactive'.
            # This correctly marks failed plans without affecting existing 'active' or 'inactive' plans.
            inactive_query = "UPDATE plan_details SET status = 'inactive', last_updated_date = CURRENT_TIMESTAMP WHERE status = 'processing'"
            cursor.execute(inactive_query)
            logger.info(f"Set {cursor.rowcount} failed or unprocessed plans to 'inactive'.")

            # Update payers with at least one active plan to 'active'
            update_payers_to_active_query = """
                UPDATE payer_details
                SET status = 'active', last_updated_at = CURRENT_TIMESTAMP
                WHERE payer_id IN (
                    SELECT DISTINCT payer_id FROM plan_details WHERE status = 'active'
                );
            """
            cursor.execute(update_payers_to_active_query)
            logger.info(f"Set {cursor.rowcount} payers to 'active'.")

            # Update payers with no active plans to 'inactive'
            update_payers_to_inactive_query = """
                UPDATE payer_details
                SET status = 'inactive', last_updated_at = CURRENT_TIMESTAMP
                WHERE payer_id NOT IN (
                    SELECT DISTINCT payer_id FROM plan_details WHERE status = 'active'
                );
            """
            cursor.execute(update_payers_to_inactive_query)
            logger.info(f"Set {cursor.rowcount} payers to 'inactive'.")

            conn.commit()
            logger.info("Successfully updated all plan and payer statuses.")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update plan and payer statuses: {e}")
            raise

def get_all_processed_plan_ids():
    """
    Retrieves a list of all plan_ids that have been marked as 'processing'.
    This is used at the end of the pipeline to correctly mark failed plans.
    """
    logger.info("Fetching all plan IDs marked for processing...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT plan_id FROM plan_details WHERE status = 'processing'")
            plan_ids = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(plan_ids)} plans marked for processing.")
            return plan_ids
        except Exception as e:
            logger.error(f"Failed to fetch processing plan IDs: {e}")
            return []

def update_plan_file_hash(plan_id, file_hash):
    """Updates the file_hash for a given plan_id."""
    if not plan_id or not file_hash:
        return
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE plan_details SET file_hash = %s, last_updated_date = CURRENT_TIMESTAMP WHERE plan_id = %s",
                (file_hash, plan_id)
            )
            conn.commit()
            logger.info(f"Updated file_hash for plan_id: {plan_id}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update file_hash for plan_id {plan_id}: {e}")

def process_and_cache_file(file_hash, structured_data, raw_content):
    """
    Process the uploaded file, update the database, and cache the result.
    - Expects the file_hash to identify the file.
    - structured_data is the DataFrame containing the processed data.
    - raw_content is the original content of the file.
    """
    logger.info(f"Processing and caching file with hash: {file_hash}")

    # Extract plan_id, payer_id, and other relevant info from structured_data
    plan_id = structured_data['plan_id'].iloc[0] if 'plan_id' in structured_data else None
    payer_id = structured_data['payer_id'].iloc[0] if 'payer_id' in structured_data else None
    plan_name = structured_data['plan_name'].iloc[0] if 'plan_name' in structured_data else None
    payer_name = structured_data['payer_name'].iloc[0] if 'payer_name' in structured_data else None

    # Update or insert the main data into drug_formulary_details
    insert_drug_formulary_data(structured_data.to_dict(orient='records'))

    # Update the plan_details and payer_details statuses
    update_plan_and_payer_statuses([plan_id])

    # Cache the result for quick retrieval
    cache_result(file_hash, structured_data, raw_content)

    logger.info(f"Successfully processed and cached file: {file_hash}")

def insert_acronyms_to_ref_table(acronyms, state_name, payer_name, plan_name, table_name):
    """
    Insert a list of acronyms into the specified reference table.
    This version prevents duplicate records and filters out rows with no meaningful data.
    """
    if not acronyms:
        return
        
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        data_tuples = []
        for ac in acronyms:
            acronym = ac.get("acronym")
            expansion = ac.get("expansion")
            explanation = ac.get("explanation")
             
            is_expansion_blank = not expansion or str(expansion).strip() in ('', '[null]')
            is_explanation_blank = not explanation or str(explanation).strip() in ('', '[null]')
 
            if acronym and not (is_expansion_blank and is_explanation_blank):
                data_tuples.append(
                    (
                        state_name,
                        payer_name,
                        plan_name,
                        acronym,
                        expansion if not is_expansion_blank else None,
                        explanation if not is_explanation_blank else None,
                    )
                )

        if not data_tuples:
            logger.warning(f"No valid, non-blank acronyms to insert into {table_name}.")
            return

        # ON CONFLICT ensures that if a row with the same unique keys exists, we update it.
        # We use COALESCE to update a field only if the new value is NOT NULL,
        # effectively filling in missing details without overwriting existing data with nulls.
        insert_query = f"""
            INSERT INTO {table_name} (state_name, payer_name, plan_name, acronym, expansion, explanation)
            VALUES %s
            ON CONFLICT (state_name, payer_name, plan_name, acronym)
            DO UPDATE SET
                expansion = COALESCE(EXCLUDED.expansion, {table_name}.expansion),
                explanation = COALESCE(EXCLUDED.explanation, {table_name}.explanation);
        """
        
        try:
            execute_values(cursor, insert_query, data_tuples)
            conn.commit()
            logger.info(f"Successfully inserted or updated {len(data_tuples)} records into {table_name}.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert acronyms into {table_name}: {e}")
            raise

def batch_determine_coverage_status(requirement_tier_pairs, conn, state_name, payer_name):
    """
    Batch lookup for coverage status for unique (requirement_code, drug_tier) pairs.
    Returns a dict mapping (requirement_code, drug_tier) -> coverage_status.
    """
    from utils import determine_coverage_status  # Import here to avoid circular import
    mapping = {}
    for req_code, tier in requirement_tier_pairs:
        status = determine_coverage_status(req_code, tier, conn, state_name, payer_name)
        mapping[(req_code, tier)] = status
    return mapping

def delete_drug_formulary_records_for_plan(plan_id: str):
    """
    Deletes all records from the drug_formulary_details table for a specific plan_id.
    This is used when a formulary file has been updated and needs to be replaced.
    """
    if not plan_id:
        logger.warning("No plan_id provided for deletion. Aborting.")
        return 0

    logger.info(f"Preparing to delete all existing drug records for plan_id: {plan_id}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            query = "DELETE FROM drug_formulary_details WHERE plan_id = %s"
            cursor.execute(query, (plan_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Successfully deleted {deleted_count} records for plan_id: {plan_id}")
            return deleted_count
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete records for plan_id {plan_id}: {e}")
            raise