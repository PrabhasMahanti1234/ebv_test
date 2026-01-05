import logging
import json
import time
from database import ensure_database_schema, update_drug_formulary_status, update_plan_and_payer_statuses
from excel_processing import populate_payer_and_plan_tables
from pdf_processing import process_pdfs_from_urls_in_parallel
from config import COST_TRACKER


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def print_cost_summary():
    """Print a detailed cost summary for the run."""
    print("\n" + "=" * 80)
    print("                         COST SUMMARY REPORT")
    print("=" * 80)
    
    # Global totals
    print(f"\n{'GLOBAL TOTALS':^80}")
    print("-" * 80)
    print(f"  Total Pages Processed (Mistral OCR): {COST_TRACKER['total_pages']:,}")
    print(f"  Total Tokens Used (Bedrock LLM):     {COST_TRACKER['total_tokens']:,}")
    print(f"  Total LLM Calls:                     {COST_TRACKER['total_llm_calls']:,}")
    print(f"  Total PDFs Processed:                {COST_TRACKER['total_pdfs_processed']:,}")
    print("-" * 80)
    print(f"  TOTAL ESTIMATED COST:                ${COST_TRACKER['total_cost']:.6f}")
    print("=" * 80)
    
    # Per-payer breakdown
    if COST_TRACKER['payer_costs']:
        print(f"\n{'PER-PAYER COST BREAKDOWN':^80}")
        print("-" * 80)
        print(f"{'Payer Name':<30} {'Pages':>8} {'Tokens':>10} {'LLM Calls':>10} {'Cost':>12}")
        print("-" * 80)
        
        for payer_name, costs in sorted(COST_TRACKER['payer_costs'].items()):
            print(f"{payer_name[:30]:<30} {costs['mistral_ocr_pages']:>8,} {costs['bedrock_tokens']:>10,} {costs['llm_calls']:>10,} ${costs['total_cost']:>10.6f}")
        
        print("-" * 80)
    
    print("\n")
    
    # Return cost data as dictionary for JSON output
    return {
        "total_pages": COST_TRACKER['total_pages'],
        "total_tokens": COST_TRACKER['total_tokens'],
        "total_llm_calls": COST_TRACKER['total_llm_calls'],
        "total_pdfs_processed": COST_TRACKER['total_pdfs_processed'],
        "total_cost_usd": round(COST_TRACKER['total_cost'], 6),
        "payer_breakdown": {
            payer: {
                "mistral_ocr_pages": costs['mistral_ocr_pages'],
                "bedrock_tokens": costs['bedrock_tokens'],
                "bedrock_cost_usd": round(costs['bedrock_cost'], 6),
                "mistral_cost_usd": round(costs['mistral_cost'], 6),
                "total_cost_usd": round(costs['total_cost'], 6),
                "llm_calls": costs['llm_calls'],
                "pdfs_processed": costs['pdfs_processed']
            }
            for payer, costs in COST_TRACKER['payer_costs'].items()
        }
    }




def main():
	start_time = time.time()
	logger.info("Starting pipeline")
	ensure_database_schema()
	populate_payer_and_plan_tables()
	processed_plan_ids, all_processed_data = process_pdfs_from_urls_in_parallel()

	if processed_plan_ids:
		update_drug_formulary_status(processed_plan_ids)
		update_plan_and_payer_statuses(processed_plan_ids)

	end_time = time.time()
	elapsed_time = end_time - start_time
	
	logger.info("Pipeline finished")
	
	# Print cost summary
	cost_data = print_cost_summary()
	
	# Prepare final output JSON
	final_output = {
		"run_summary": {
			"status": "completed",
			"total_processing_time_seconds": round(elapsed_time, 2),
			"total_plans_processed": len(processed_plan_ids),
			"processed_plan_ids": processed_plan_ids
		},
		"cost_summary": cost_data
	}
	
	# Print final JSON output to terminal
	print("\n" + "=" * 80)
	print("                         FINAL OUTPUT JSON")
	print("=" * 80)
	print(json.dumps(final_output, indent=2))
	print("=" * 80 + "\n")
	
	return final_output

if __name__ == '__main__':
	main()