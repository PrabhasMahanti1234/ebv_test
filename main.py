import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import uuid
import multiprocessing

from config import ALL_PROCESSED_DATA, COST_TRACKER
from database import ensure_database_schema, insert_drug_formulary_data, update_plan_and_payer_statuses, update_drug_formulary_status
from excel_processing import populate_payer_and_plan_tables
from pdf_processing import process_pdfs_from_urls_in_parallel
from utils import validate_required_files, detect_step_therapy
from product_mapper import map_products_to_formulary

logger = logging.getLogger(__name__)

def safe_create_directory(path):
    """Safely create a directory if it does not exist."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False
 
def save_cumulative_exports(all_processed_data):
    """
    Save all processed data to Excel and CSV files with enhanced error handling and new columns.
    NOTE: This function is no longer called in the main PDF-by-PDF flow but is kept for potential future use.
    """
    if not all_processed_data:
        logger.warning("No data to export")
        return
    
    full_df = pd.DataFrame(all_processed_data)
    
    # Ensure all required boolean/status columns exist with a default value
    required_columns = {
        "is_prior_authorization_required": "No",
        "is_step_therapy_required": "No", 
        "is_quantity_limit_applied": "No",
        "status": "processing"
    }
    
    for col, default_value in required_columns.items():
        if col not in full_df.columns:
            logger.warning(f"Adding missing column '{col}' with default value '{default_value}'")
            full_df[col] = default_value
    
    # Normalize boolean columns to "Yes" or "No"
    for col in ["is_prior_authorization_required", "is_step_therapy_required", "is_quantity_limit_applied"]:
         full_df[col] = full_df[col].apply(lambda x: "Yes" if str(x).strip().lower() in ["yes", "true", "1"] else "No")

    output_dir = Path("output_exports")
    if not safe_create_directory(output_dir):
        logger.error("Failed to create output directory")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        excel_path = output_dir / f"drug_formulary_complete_{timestamp}.xlsx"
        
        # Define the desired column order for the Excel export
        column_order = [
            "payer_name", "plan_name", "state_name", "drug_name", 
            "drug_tier", "drug_requirements", "coverage_status",
            "is_prior_authorization_required", "is_step_therapy_required",
            "is_quantity_limit_applied",
            "status", "file_name", "id", "plan_id", "payer_id",
            "product_labeler_code", "product_proprietaryname"
        ]
        
        # Reorder dataframe columns, adding any missing ones
        excel_df = full_df.reindex(columns=column_order)
        
        excel_df.to_excel(excel_path, index=False)
        logger.info(f"Successfully exported data to Excel: {excel_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        
    try:
        csv_path = output_dir / f"drug_formulary_complete_{timestamp}.csv"
        full_df.to_csv(csv_path, index=False)
        logger.info(f"Successfully exported data to CSV: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")


def main():
    """Main function to run the entire data processing pipeline"""
    try:
        logger.info("========================================")
        logger.info("STARTING DRUG FORMULARY PROCESSING")
        logger.info("========================================")
        
        # Step 0: Initial Setup
        ensure_database_schema()
        validate_required_files()
        
        # Step 1: Populate Payer and Plan tables from Excel
        populate_payer_and_plan_tables()
        
        # Step 2: Process PDFs in parallel from URLs (now handles its own DB insertion)
        # This function now returns a list of successfully processed plan IDs.
        processed_plan_ids, _ = process_pdfs_from_urls_in_parallel()
        
        # Step 3 is now integrated into Step 2. We just need to update statuses.
        if processed_plan_ids:
            logger.info("STEP 3: Database insertion was handled in real-time during PDF processing.")
            
            # We still need to update the final status for the drugs that were inserted
            logger.info("Updating status to 'completed' for all inserted drug records.")
            update_drug_formulary_status(processed_plan_ids)
            
            # The export step is skipped as the primary goal is real-time DB insertion.
            logger.info("Skipping cumulative data export as data is now in the database.")
            # Step 4: Save cumulative data after all processing and DB operations
            # logger.info("STEP 4: Saving Cumulative Data")
            # save_cumulative_exports(deduplicated_data)

        else:
            logger.warning("No data was processed or inserted into the database.")

        # Step 5: Update final statuses in the database (This uses processed_plan_ids)
        logger.info("STEP 5: Updating final plan and payer statuses.")
        update_plan_and_payer_statuses(processed_plan_ids)
        
        # Step 6: Map products to formulary
        logger.info("STEP 6: Mapping products to formulary")
        try:
            records_mapped = map_products_to_formulary()
            logger.info(f"Successfully mapped {records_mapped:,} drug records to product master data.")
        except Exception as e:
            logger.error(f"Product mapping failed, but continuing: {e}")
            # Don't raise - allow the pipeline to complete even if mapping fails

        logger.info("========================================")
        logger.info("DRUG FORMULARY PROCESSING COMPLETE")
        logger.info("========================================")
        
    except Exception as e:
        logger.critical(f"A critical error occurred in the main pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cost report
        logger.info("\n--- FINAL COST & USAGE REPORT ---")
        for payer, costs in COST_TRACKER['payer_costs'].items():
            logger.info(f"\nPayer: {payer}")
            logger.info(f"  - PDFs Processed: {costs['pdfs_processed']}")
            logger.info(f"  - Mistral Pages: {costs['mistral_ocr_pages']}")
            logger.info(f"  - Mistral Cost: ${costs['mistral_cost']:.4f}")
            logger.info(f"  - Bedrock LLM Calls: {costs['llm_calls']}")
            logger.info(f"  - Bedrock Tokens: {costs['bedrock_tokens']}")
            logger.info(f"  - Bedrock Cost: ${costs['bedrock_cost']:.8f}")
            logger.info(f"  - Payer Total Cost: ${costs['total_cost']:.8f}")
        
        logger.info("\n--- OVERALL TOTALS ---")
        logger.info(f"Total PDFs Processed: {COST_TRACKER['total_pdfs_processed']}")
        logger.info(f"Total Mistral Pages: {COST_TRACKER['total_pages']}")
        logger.info(f"Total LLM Calls: {COST_TRACKER['total_llm_calls']}")
        logger.info(f"Total Bedrock Tokens: {COST_TRACKER['total_tokens']}")
        logger.info(f"GRAND TOTAL COST: ${COST_TRACKER['total_cost']:.8f}")

if __name__ == "__main__":

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # The start method can only be set once. This is to prevent errors
        # in environments like Jupyter notebooks where the script might be re-run.
        pass

    main()