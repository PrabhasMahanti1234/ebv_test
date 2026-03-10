import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import IntegrityError
from contextlib import contextmanager
import logging
from logger_setup import setup_logger
import json
import pandas as pd
from io import StringIO
from config import DB_CONFIG

logger = logging.getLogger(__name__)
db_logger = setup_logger("database", "logs/database.log")

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

#Updated Codebase
def fetch_existing_coverage_by_file_hash(file_hash):
    """
    Fetch existing coverage results for a given file_hash.
    Returns a dictionary with 'drugs' and 'acronyms' keys.
    """

    drug_coverage = {}
    acronym_coverage = {}

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Fetch drug coverage
            cursor.execute("""
                SELECT 
                    LOWER(d.drug_name),
                    d.drug_tier,
                    d.drug_requirements,
                    d.coverage_status,
                    d.confidence_score,
                    d.manual_review
                FROM drug_formulary_details d
                JOIN plan_details p ON d.plan_id = p.plan_id
                WHERE p.file_hash = %s
                AND d.coverage_status IS NOT NULL
            """, (file_hash,))

            rows = cursor.fetchall()
            for row in rows:
                key = (
                    row[0],  # drug_name lower
                    row[1],  # tier
                    row[2]   # requirements
                )
                drug_coverage[key] = {
                    "coverage_status": row[3],
                    "confidence_score": row[4],
                    "manual_review": row[5]
                }

            # Fetch acronym coverage
            # We join with plan_details to find plans that have this file_hash,
            # then join with pp_formulary_names on plan/payer/state.
            cursor.execute("""
                SELECT DISTINCT
                    a.acronym,
                    a.expansion,
                    a.explanation,
                    a.coverage_status
                FROM pp_formulary_names a
                JOIN plan_details p ON a.plan_name = p.plan_name 
                    AND a.payer_name = p.payer_name 
                    AND a.state_name = p.state_name
                WHERE p.file_hash = %s
                AND a.coverage_status IS NOT NULL
            """, (file_hash,))

            a_rows = cursor.fetchall()
            for row in a_rows:
                # Key for acronym lookup: (acronym, expansion, explanation)
                key = (
                    row[0],  # acronym
                    row[1],  # expansion
                    row[2]   # explanation
                )
                acronym_coverage[key] = row[3]

            if drug_coverage or acronym_coverage:
                logger.info(f"[REUSE] Found {len(drug_coverage)} drugs and {len(acronym_coverage)} acronyms for hash {file_hash}")

        except Exception as e:
            logger.error(f"Error fetching coverage by file_hash: {e}")

    return {"drugs": drug_coverage, "acronyms": acronym_coverage}

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
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active'
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
                    s3_frozen_pdf_url TEXT,
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
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active'
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

        # Ensure s3_frozen_pdf_url exists for S3-based processing
        try:
            cursor.execute("""
                ALTER TABLE plan_details
                ADD COLUMN IF NOT EXISTS s3_frozen_pdf_url TEXT
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"s3_frozen_pdf_url column may already exist in plan_details: {e}")
            conn.rollback()

        # Index for faster selection of S3 URLs
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_plan_s3_url ON plan_details(s3_frozen_pdf_url)", "idx_plan_s3_url")


        # Create processed_file_cache table
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_file_cache (
                    file_hash VARCHAR(64) PRIMARY KEY,
                    formulary_url TEXT,  -- ✅ Added for URL-based caching
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

        # Add formulary_url column if it doesn't exist (for existing tables)
        try:
            cursor.execute("""
                ALTER TABLE processed_file_cache
                ADD COLUMN IF NOT EXISTS formulary_url TEXT
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"formulary_url column may already exist in processed_file_cache: {e}")
            conn.rollback()

        # Add index on formulary_url for fast lookups
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_cache_url ON processed_file_cache(formulary_url)", "idx_cache_url")

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

        # ---------------------------------------------------------
        # Transaction Table
        # ---------------------------------------------------------
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transaction (
                    transaction_id UUID PRIMARY KEY,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    started_at TIMESTAMP WITH TIME ZONE NULL,
                    completed_at TIMESTAMP WITH TIME ZONE NULL,
                    status VARCHAR(32), -- queued, submitted, in_progress, completed, failed, cancelled
                    job_type VARCHAR(64), -- ocr_batch, single_pdf, cache_lookup, manual_reprocess
                    plan_id VARCHAR(36) NULL, -- FK to plan_details
                    payer_id VARCHAR(36) NULL, -- FK to payer_details
                    file_hash VARCHAR(64) NULL, -- link to processed_file_cache
                    file_name VARCHAR(1000) NULL,
                    request_summary JSONB NULL,
                    response_summary JSONB NULL,
                    rows_inserted INTEGER NULL,
                    ocr_pages_processed INTEGER NULL,
                    mistral_cost NUMERIC(10,4) NULL,
                    batch_job_id VARCHAR(200) NULL,
                    last_updated TIMESTAMP DEFAULT now()
                );
            """)
            conn.commit()
            logger.info("Created/ensured transaction table")
        except Exception as e:
            logger.debug(f"Transaction table creation issue (may already exist): {e}")
            conn.rollback()
        
        # Add Indexes for Transaction
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_txn_plan_id ON transaction(plan_id)", "idx_txn_plan_id")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_txn_payer_id ON transaction(payer_id)", "idx_txn_payer_id")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_txn_status ON transaction(status)", "idx_txn_status")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_txn_created_at ON transaction(created_at)", "idx_txn_created_at")
        _add_index(conn, cursor, "CREATE INDEX IF NOT EXISTS idx_txn_plan_date ON transaction(plan_id, created_at)", "idx_txn_plan_date")


        # ---------------------------------------------------------
        # Audit Table
        # ---------------------------------------------------------
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit (
                    audit_id BIGSERIAL PRIMARY KEY,
                    transaction_id UUID NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    event_type VARCHAR(64),
                    event_subtype VARCHAR(64) NULL,
                    service VARCHAR(64) NULL,
                    payload JSONB NULL,
                    error_message TEXT NULL,
                    error_stack TEXT NULL,
                    meta JSONB NULL,
                    CONSTRAINT fk_audit_transaction FOREIGN KEY (transaction_id) 
                        REFERENCES transaction(transaction_id) ON DELETE CASCADE
                );
            """)
            conn.commit()
            logger.info("Created/ensured audit table")
        except Exception as e:
            logger.debug(f"Audit table creation issue (may already exist): {e}")
            conn.rollback()

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
                    confidence_score DECIMAL(5,2),
                    source_url TEXT,
                    file_name VARCHAR(1000),
                    status VARCHAR(20) DEFAULT 'processing',
                     
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    manual_review BOOLEAN DEFAULT FALSE,
 
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
                    payer_name VARCHAR(1000),
                    plan_name VARCHAR(1000),
                    acronym VARCHAR(50),
                    expansion TEXT,
                    explanation TEXT,
                    coverage_status VARCHAR(1000)
                );
            """)
            conn.commit()
            logger.info("Created/ensured pp_formulary_names table")
            
            # Ensure columns are wide enough if table existed with smaller columns
            cursor.execute("ALTER TABLE pp_formulary_names ALTER COLUMN payer_name TYPE VARCHAR(1000);")
            cursor.execute("ALTER TABLE pp_formulary_names ALTER COLUMN plan_name TYPE VARCHAR(1000);")
            conn.commit()
        except Exception as e:
            logger.debug(f"pp_formulary_names table creation/alter issue (may already exist): {e}")
            conn.rollback()

        # Add coverage_status column to existing pp_formulary_names if not exists
        try:
            cursor.execute("""
                ALTER TABLE pp_formulary_names
                ADD COLUMN IF NOT EXISTS coverage_status VARCHAR(1000)
            """)
            conn.commit()
            logger.info("Ensured coverage_status column exists in pp_formulary_names.")
        except Exception as e:
            logger.debug(f"coverage_status column may already exist in pp_formulary_names: {e}")
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
                ALTER COLUMN confidence_score TYPE DECIMAL(5,2)
            """)
            conn.commit()
            logger.info("Updated confidence_score column type to DECIMAL(5,2).")
        except Exception as e:
            logger.debug(f"confidence_score column type update issue: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS manual_review BOOLEAN DEFAULT FALSE
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"manual_review column may already exist: {e}")
            conn.rollback()

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

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS badge_colors JSONB
            """)
            conn.commit()
        except Exception as e:
            logger.debug(f"badge_colors column may already exist in drug_formulary_details: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS preferred_agent VARCHAR(10)
            """)
            conn.commit()
            logger.info("Added preferred_agent column to drug_formulary_details.")
        except Exception as e:
            logger.debug(f"preferred_agent column may already exist: {e}")
            conn.rollback()

        try:
            cursor.execute("""
                ALTER TABLE drug_formulary_details
                ADD COLUMN IF NOT EXISTS non_preferred_agent VARCHAR(10)
            """)
            conn.commit()
            logger.info("Added non_preferred_agent column to drug_formulary_details.")
        except Exception as e:
            logger.debug(f"non_preferred_agent column may already exist: {e}")
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

def create_transaction(transaction_id, job_type, plan_id=None, payer_id=None, file_hash=None, file_name=None, request_summary=None, status='queued'):
    """Creates a new transaction record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO transaction (
                    transaction_id, job_type, plan_id, payer_id, file_hash, file_name, request_summary, status, created_at, last_updated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (transaction_id, job_type, plan_id, payer_id, file_hash, file_name, json.dumps(request_summary) if request_summary else None, status))
            conn.commit()
            logger.info(f"Transaction {transaction_id} created.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create transaction {transaction_id}: {e}")

def update_transaction(transaction_id, status=None, started_at=None, completed_at=None, response_summary=None, rows_inserted=None, ocr_pages_processed=None, mistral_cost=None, batch_job_id=None, error=None, file_hash=None, file_name=None):
    """Updates an existing transaction record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            updates = []
            params = []
            
            if status:
                updates.append("status = %s")
                params.append(status)
            if started_at:
                updates.append("started_at = %s")
                params.append(started_at)
            if completed_at:
                updates.append("completed_at = %s")
                params.append(completed_at)
            if response_summary:
                updates.append("response_summary = %s")
                params.append(json.dumps(response_summary))
            if rows_inserted is not None:
                updates.append("rows_inserted = %s")
                params.append(rows_inserted)
            if ocr_pages_processed is not None:
                updates.append("ocr_pages_processed = %s")
                params.append(ocr_pages_processed)
            if mistral_cost is not None:
                updates.append("mistral_cost = %s")
                params.append(mistral_cost)
            if file_hash:
                updates.append("file_hash = %s")
                params.append(file_hash)
            if file_name:
                updates.append("file_name = %s")
                params.append(file_name)
            if batch_job_id:
                updates.append("batch_job_id = %s")
                params.append(batch_job_id)

            updates.append("last_updated = NOW()")
            params.append(transaction_id) # For the WHERE clause

            if updates:
                query = f"UPDATE transaction SET {', '.join(updates)} WHERE transaction_id = %s"
                cursor.execute(query, tuple(params))
                conn.commit()
                # logger.info(f"Transaction {transaction_id} updated.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update transaction {transaction_id}: {e}")

def log_audit_event(transaction_id, event_type, event_subtype=None, service=None, payload=None, error_message=None, error_stack=None, meta=None):
    """Logs an event to the audit table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO audit (
                    transaction_id, event_type, event_subtype, service, payload, error_message, error_stack, meta, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                transaction_id, 
                event_type, 
                event_subtype, 
                service, 
                json.dumps(payload) if payload else None, 
                error_message, 
                error_stack, 
                json.dumps(meta) if meta else None
            ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log audit event check if transaction id is present in transaction table: {e}")


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

def get_cached_result_by_url(formulary_url):
    """
    Retrieves cached result by formulary_url instead of file_hash.
    This allows checking if a PDF has been processed before downloading it.
    Returns (file_hash, structured_data, raw_content) or (None, None, None) if not found.
    """
    if not formulary_url:
        return None, None, None
        
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_hash, structured_data_json, raw_content FROM processed_file_cache WHERE formulary_url = %s",
            (formulary_url,)
        )
        result = cursor.fetchone()
        if result:
            file_hash, structured_data_json, raw_content = result
            logger.info(f"✅ Cache HIT by URL: {formulary_url[:80]}... (hash: {file_hash})")
            
            # If the cached JSON is null or empty, treat it as a miss.
            if not structured_data_json:
                return None, None, None

            try:
                return file_hash, structured_data_json, raw_content
            except Exception as e:
                logger.warning(f"Failed to parse cached JSON for URL {formulary_url[:80]}. Will re-process. Error: {e}")
                return None, None, None
                
    logger.info(f"Cache MISS by URL: {formulary_url[:80]}...")
    return None, None, None

def cache_result(file_hash, structured_data_dict, raw_content, formulary_url=None):
    """
    Caches the full structured data dictionary (drug_table, acronyms, tiers) in the database.
    Now includes formulary_url for URL-based caching.
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
                INSERT INTO processed_file_cache (file_hash, formulary_url, structured_data_json, raw_content)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (file_hash) DO UPDATE SET
                    formulary_url = EXCLUDED.formulary_url,
                    structured_data_json = EXCLUDED.structured_data_json,
                    raw_content = EXCLUDED.raw_content,
                    created_at = CURRENT_TIMESTAMP;
                """,
                (file_hash, formulary_url, json.dumps(structured_data_dict), raw_content)
            )
            conn.commit()
            logger.info(f"Successfully cached result for hash: {file_hash} (URL: {formulary_url[:80] if formulary_url else 'N/A'}...)")
            db_logger.info(f"cache.store hash={file_hash} url={formulary_url} size_drugs={len(structured_data_dict.get('drug_table', []))} size_acronyms={len(structured_data_dict.get('acronyms', []))}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to cache result for hash {file_hash}: {e}")
            db_logger.error(f"cache.error hash={file_hash} error={e}")


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

    # DEDUPLICATION: Remove duplicates based on conflict key (plan_id, drug_name, drug_tier)
    # Keep the last occurrence (which may have more complete data)
    seen_keys = {}
    for record in processed_data:
        key = (
            record.get("plan_id"),
            record.get("drug_name", "").strip().lower() if record.get("drug_name") else "",
            record.get("drug_tier", "").strip() if record.get("drug_tier") else None,
            record.get("drug_requirements", "").strip() if record.get("drug_requirements") else None
        )
        seen_keys[key] = record  # Later records overwrite earlier ones
    
    deduplicated_data = list(seen_keys.values())
    
    if len(deduplicated_data) < len(processed_data):
        logger.info(f"🔄 Deduplicated: {len(processed_data)} → {len(deduplicated_data)} records (removed {len(processed_data) - len(deduplicated_data)} duplicates)")
    
    logger.info(f"Preparing to insert {len(deduplicated_data)} records into the database.")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # The columns must match the order of values in the data tuples
        cols = [
            "id", "plan_id", "payer_id", "drug_name", "ndc_code", "jcode",
            "state_name", "coverage_status", "drug_tier", "drug_requirements", "page_number", "badge_colors",
            "preferred_agent", "non_preferred_agent",
            "is_prior_authorization_required", "is_step_therapy_required", "is_quantity_limit_applied",
            "coverage_details", "confidence_score", "manual_review", "source_url", "plan_name", "payer_name", "file_name", "status"
        ]

        data_tuples = []
        for record in deduplicated_data:  # Use deduplicated data
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

            # Extract and serialize badge_colors if present
            badge_colors = record.get("badge_colors")
            if badge_colors and isinstance(badge_colors, dict):
                badge_colors_json = json.dumps(badge_colors)
            else:
                badge_colors_json = None

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
                badge_colors_json,  # Serialized badge colors
                record.get("preferred_agent"),  # Add preferred_agent
                record.get("non_preferred_agent"),  # Add non_preferred_agent
                record.get("is_prior_authorization_required"),
                record.get("is_step_therapy_required"),
                record.get("is_quantity_limit_applied"),
                record.get("coverage_details"),
                record.get("confidence_score"),
                record.get("manual_review", False),
                record.get("source_url"),
                record.get("plan_name"),
                record.get("payer_name"),
                record.get("file_name"),
                'processing'  # Set initial status for new records
            ))

        # Using standard INSERT since previous records are deleted prior to this step
         
        insert_query = f"""
            INSERT INTO drug_formulary_details ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (plan_id, drug_name, drug_tier, drug_requirements)
            DO UPDATE SET
                drug_requirements = EXCLUDED.drug_requirements,
                coverage_status = EXCLUDED.coverage_status,
                page_number = EXCLUDED.page_number,
                badge_colors = EXCLUDED.badge_colors,
                preferred_agent = EXCLUDED.preferred_agent,
                non_preferred_agent = EXCLUDED.non_preferred_agent,
                is_prior_authorization_required = EXCLUDED.is_prior_authorization_required,
                is_step_therapy_required = EXCLUDED.is_step_therapy_required,
                is_quantity_limit_applied = EXCLUDED.is_quantity_limit_applied,
                confidence_score = EXCLUDED.confidence_score,
                manual_review = EXCLUDED.manual_review,
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
            db_logger.info(f"db.insert drug_formulary_details count={len(data_tuples)}")

        except IntegrityError as e:
            conn.rollback()
            logger.error(f"Database integrity error during insertion: {e}")
            db_logger.error(f"db.integrity_error table=drug_formulary_details error={e}")
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"An unexpected error occurred during data insertion: {e}")
            db_logger.error(f"db.insert_error table=drug_formulary_details error={e}")
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
            db_logger.info(f"db.update drug_formulary_details set=status='completed' plans={len(processed_plan_ids)} rows={cursor.rowcount}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update drug formulary statuses: {e}")
            db_logger.error(f"db.update_error table=drug_formulary_details error={e}")
            raise

def update_plan_and_payer_statuses(processed_plan_ids, finalize_run=True):
    """
    Updates the status of plans and payers after processing.
    - Sets status to 'active' for successfully processed plans.
    - If finalize_run=True: Sets status to 'inactive' for plans that were being processed but failed.
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
                db_logger.info(f"db.update plan_details set=status='active' rows={cursor.rowcount}")

            # Update any plans that were 'processing' but did not complete successfully to 'inactive'.
            # This correctly marks failed plans without affecting existing 'active' or 'inactive' plans.
            if finalize_run:
                inactive_query = "UPDATE plan_details SET status = 'inactive', last_updated_date = CURRENT_TIMESTAMP WHERE status = 'processing'"
                cursor.execute(inactive_query)
                logger.info(f"Set {cursor.rowcount} failed or unprocessed plans to 'inactive'.")
                db_logger.info(f"db.update plan_details set=status='inactive' rows={cursor.rowcount}")

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
            db_logger.info(f"db.update payer_details set=status='active' rows={cursor.rowcount}")

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
            db_logger.info(f"db.update payer_details set=status='inactive' rows={cursor.rowcount}")

            conn.commit()
            logger.info("Successfully updated all plan and payer statuses.")
            db_logger.info("db.update statuses committed")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update plan and payer statuses: {e}")
            db_logger.error(f"db.update_error statuses error={e}")
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
    Also stores coverage_status if provided.
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

            # ✅ NEW: coverage_status support
            coverage_status = ac.get("coverage_status")

            is_expansion_blank = not expansion or str(expansion).strip() in ('', '[null]')
            is_explanation_blank = not explanation or str(explanation).strip() in ('', '[null]')

            # Keep existing behavior: only insert meaningful acronym rows
            if acronym and not (is_expansion_blank and is_explanation_blank):
                data_tuples.append(
                    (
                        state_name,
                        payer_name,
                        plan_name,
                        acronym,
                        expansion if not is_expansion_blank else None,
                        explanation if not is_explanation_blank else None,
                        coverage_status,  # ✅ NEW
                    )
                )

        if not data_tuples:
            logger.warning(f"No valid, non-blank acronyms to insert into {table_name}.")
            return

        # Deduplicate by conflict key (state_name, payer_name, plan_name, acronym) — keep last occurrence
        dedup_map = {}
        for tup in data_tuples:
            key = (tup[0], tup[1], tup[2], tup[3])  # state, payer, plan, acronym
            dedup_map[key] = tup
        data_tuples = list(dedup_map.values())
        logger.debug(f"Deduplicated acronyms to {len(data_tuples)} unique entries before insert.")

        insert_query = f"""
            INSERT INTO {table_name} (
                state_name, payer_name, plan_name, acronym, expansion, explanation, coverage_status
            )
            VALUES %s
            ON CONFLICT (state_name, payer_name, plan_name, acronym)
            DO UPDATE SET
                expansion = COALESCE(EXCLUDED.expansion, {table_name}.expansion),
                explanation = COALESCE(EXCLUDED.explanation, {table_name}.explanation),
                coverage_status = COALESCE(EXCLUDED.coverage_status, {table_name}.coverage_status);
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
    Returns a dict mapping (requirement_code, drug_tier) -> (coverage_status, confidence_score, source).
    """
    from coverage import det_coverage_status  # Import here to avoid circular import

    acronym_cache = fetch_acronym_cache(payer_name, state_name)

    mapping = {}
    for req_code, tier in requirement_tier_pairs:
        status, confidence, source, _ = det_coverage_status(
            acronym=req_code,
            expansion=None,
            explanation=None,
            requirements_text=req_code,
            tier_text=tier,
            conn=conn,
            state_name=state_name,
            payer_name=payer_name,
            drug_name=None,
            acronym_cache=acronym_cache
        )
        mapping[(req_code, tier)] = (status, confidence, source)
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
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Successfully deleted {count} records for plan_id: {plan_id}")
            return count
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete records for plan_id: {plan_id}: {e}")
            raise

def fetch_acronym_cache(payer_name, state_name):
    """
    Fetch all acronym definitions for a specific payer and state once.
    This cache will be used to avoid redundant DB lookups for every drug entry.
    Returns a dictionary mapping acronym -> (expansion, explanation, coverage_status)
    """
    cache = {}
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Fetch from the master table (tier_requirement_expansion)
        # Order by specificity: (state, payer) > (state) > (payer) > (global)
        cursor.execute("""
            SELECT acronym, expansion, explanation, coverage_status, state_name, payer_name
            FROM tier_requirement_expansion
            WHERE (state_name IS NULL OR UPPER(state_name) = UPPER(%s))
              AND (payer_name IS NULL OR UPPER(payer_name) = UPPER(%s))
            ORDER BY (state_name IS NOT NULL)::int DESC, (payer_name IS NOT NULL)::int DESC
        """, (state_name, payer_name))
        
        rows = cursor.fetchall()
        for row in rows:
            # ✅ Normalize cache keys: Strip and Uppercase
            acr = str(row[0]).strip().upper()
            # Since we ordered by specificity, the first time we see an acronym, it's the most specific one
            if acr not in cache:
                cache[acr] = (row[1], row[2], row[3])

        # 2. Fetch from the plan-specific table (pp_formulary_names)
        # These might overwrite or supplement the master table for this specific payer/state
        cursor.execute("""
            SELECT acronym, expansion, explanation, coverage_status
            FROM pp_formulary_names
            WHERE (state_name IS NULL OR UPPER(state_name) = UPPER(%s))
              AND (payer_name IS NULL OR UPPER(payer_name) = UPPER(%s))
        """, (state_name, payer_name))
        
        rows = cursor.fetchall()
        for row in rows:
            # ✅ Normalize cache keys: Strip and Uppercase
            acr = str(row[0]).strip().upper()
            # Plan-specific definitions usually take precedence or fill gaps
            if acr not in cache or (row[1] or row[2] or row[3]):
                cache[acr] = (row[1], row[2], row[3])
                
    logger.info(f"Loaded {len(cache)} acronyms into memory cache for {payer_name} ({state_name})")
    return cache
