import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os
import re
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --- DB CONNECTION ---
load_dotenv()

def get_postgres_connection_string():
    """Builds the connection string from environment variables."""
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    user = os.environ.get("DB_USER")
    database = os.environ.get("DB_NAME")
    password = os.environ.get("DB_PASSWORD")
    
    connection_string = f"postgresql+psycopg2://{user}:%s@{host}:{port}/{database}"
    return connection_string % quote_plus(password)


def map_products_to_formulary():
    """
    Main function to map product master data to drug formulary details.
    This function can be called from the main pipeline.
    """
    logger.info("========================================")
    logger.info("STARTING PRODUCT MAPPING")
    logger.info("========================================")
    
    try:
        engine = create_engine(get_postgres_connection_string())
        logger.info("✅ Database engine created successfully.")
        
        # --- 1. LOAD ALL DATA ---
        logger.info("Loading all records from database...")
        
        # Load all records from the drug formulary
        formulary_query = "SELECT id, drug_name FROM drug_formulary_details;"
        df_formulary = pd.read_sql(text(formulary_query), engine)
        logger.info(f"Loaded {len(df_formulary):,} records from drug_formulary_details.")
        
        # Load all records from the product master table
        product_master_query = "SELECT product_labeler_code, product_ndc, proprietaryname FROM product_master;"
        df_product_master = pd.read_sql(text(product_master_query), engine)
        logger.info(f"Loaded {len(df_product_master):,} records from product_master.")
        
        # --- 2. PRE-PROCESS PRODUCT MASTER DATA ---
        logger.info("Pre-processing product master data...")
        
        # Drop rows where proprietaryname is null, as they cannot be mapped
        df_product_master.dropna(subset=['proprietaryname'], inplace=True)
        
        # Normalize proprietaryname for consistent matching
        df_product_master['proprietaryname_normalized'] = df_product_master['proprietaryname'].str.upper().str.strip()
        
        # Group by the normalized name to aggregate NDCs and get a single labeler code
        df_product_mapping = df_product_master.groupby('proprietaryname_normalized').agg(
            product_labeler_code=('product_labeler_code', 'first'),
            proprietaryname=('proprietaryname', 'first')  # Keep one of the original proprietary names
        ).reset_index()
        
        logger.info(f"Created a mapping table with {len(df_product_mapping):,} unique product names.")
        
        # --- 3. EFFICIENT MAPPING USING REGEX ---
        logger.info("Starting efficient mapping process...")
        
        # Normalize the formulary drug names for matching
        df_formulary['drug_name_normalized'] = df_formulary['drug_name'].str.upper()
        
        # Create a single regex pattern from all unique product names.
        # We sort by length descending to match longer names first (e.g., "XOLAIR PEN" before "XOLAIR").
        # We also escape special regex characters in names.
        unique_names = sorted(df_product_mapping['proprietaryname_normalized'].unique(), key=len, reverse=True)
        escaped_names = [re.escape(name) for name in unique_names]
        pattern = '|'.join(escaped_names)
        
        # Use str.extract to find the first matching product name in each formulary drug_name.
        # This is a highly efficient, vectorized operation.
        df_formulary['matched_proprietary_name'] = df_formulary['drug_name_normalized'].str.extract(f'({pattern})')
        
        # Merge the formulary data with the product mapping data based on the match
        df_merged = pd.merge(
            df_formulary,
            df_product_mapping,
            left_on='matched_proprietary_name',
            right_on='proprietaryname_normalized',
            how='left'
        )
        
        # Filter down to only the rows that successfully found a match
        df_to_update = df_merged.dropna(subset=['matched_proprietary_name']).copy()
        logger.info(f"Found {len(df_to_update):,} potential matches to update in the database.")
        
        # --- 4. PREPARE AND EXECUTE BULK UPDATE ---
        if not df_to_update.empty:
            def bulk_update(conn, updates):
                """
                Performs a bulk update to the database using a CASE statement for efficiency.
                """
                if not updates:
                    logger.info("No updates to perform.")
                    return
            
                params = {}
                labeler_case = " ".join([f"WHEN :id{i} THEN :labeler{i}" for i, _ in enumerate(updates)])
                prop_name_case = " ".join([f"WHEN :id{i} THEN :prop_name{i}" for i, _ in enumerate(updates)])
            
                query = f"""
                    UPDATE drug_formulary_details
                    SET 
                        product_labeler_code = CASE id {labeler_case} END,
                        product_proprietaryname = CASE id {prop_name_case} END
                    WHERE id IN ({",".join([f':id{i}' for i in range(len(updates))])})
                """
                
                for i, (id_, labeler, prop_name) in enumerate(updates):
                    params[f"id{i}"] = id_
                    params[f"labeler{i}"] = labeler
                    params[f"prop_name{i}"] = prop_name
                    
                conn.execute(text(query), params)
            
            # Prepare the list of updates from the dataframe
            # We use 'proprietaryname' from the mapping df to get the original casing
            update_columns = ['id', 'product_labeler_code', 'proprietaryname']
            updates = [tuple(x) for x in df_to_update[update_columns].to_numpy()]
            
            # --- Main Execution Loop ---
            batch_size = 5000  # Adjust batch size based on your system's memory and performance
            total_updated = 0
            
            for start in range(0, len(updates), batch_size):
                batch_updates = updates[start:start+batch_size]
                
                with engine.begin() as conn:
                    bulk_update(conn, batch_updates)
                
                total_updated += len(batch_updates)
                logger.info(f"📝 Updated batch. Total records updated so far: {total_updated:,}")
            
            logger.info(f"\n✅ Database update process complete. Total records updated: {total_updated:,}")
        else:
            logger.warning("No matching records found to update.")
        
        logger.info("========================================")
        logger.info("PRODUCT MAPPING COMPLETE")
        logger.info("========================================")
        
        return total_updated if not df_to_update.empty else 0
        
    except Exception as e:
        logger.error(f"Error in product mapping: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    map_products_to_formulary()