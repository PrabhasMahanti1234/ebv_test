import logging
import json
import time
import argparse
import sys
import uuid
from database import (
    ensure_database_schema, update_drug_formulary_status, 
    update_plan_and_payer_statuses, create_transaction, update_transaction
)
from pdf_processing import (
    process_pdfs_from_urls_in_parallel, 
    get_all_plans_with_formulary_url, 
    prepare_batch_requests,
    process_batch_output
)
from product_mapper import map_products_to_formulary
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
    print(f"  Total PDFs Processed:                {COST_TRACKER['total_pdfs_processed']:,}")
    print("-" * 80)
    print(f"  TOTAL ESTIMATED COST:                ${COST_TRACKER['total_cost']:.6f}")
    print("=" * 80)
    
    # Per-payer breakdown
    if COST_TRACKER['payer_costs']:
        print(f"\n{'PER-PAYER COST BREAKDOWN':^80}")
        print("-" * 80)
        print(f"{'Payer Name':<30} {'Pages':>8} {'Cost':>12}")
        print("-" * 80)
        
        for payer_name, costs in sorted(COST_TRACKER['payer_costs'].items()):
            print(f"{payer_name[:30]:<30} {costs['mistral_ocr_pages']:>8,} ${costs['total_cost']:>10.6f}")
        
        print("-" * 80)
    
    print("\n")
    
    # Return cost data as dictionary for JSON output
    return {
        "total_pages": COST_TRACKER['total_pages'],
        "total_pdfs_processed": COST_TRACKER['total_pdfs_processed'],
        "total_cost_usd": round(COST_TRACKER['total_cost'], 6),
        "payer_breakdown": {
            payer: {
                "mistral_ocr_pages": costs['mistral_ocr_pages'],
                "mistral_cost_usd": round(costs['mistral_cost'], 6),
                "total_cost_usd": round(costs['total_cost'], 6),
                "pdfs_processed": costs['pdfs_processed']
            }
            for payer, costs in COST_TRACKER['payer_costs'].items()
        }
    }


def run_realtime_pipeline():
    start_time = time.time()
    transaction_id = str(uuid.uuid4())
    logger.info(f"Starting pipeline (REAL-TIME MODE) Transaction ID: {transaction_id}")
    
    create_transaction(
        transaction_id=transaction_id,
        job_type="realtime_pipeline",
        status="running",
        request_summary={"mode": "realtime"}
    )
    
    try:
        ensure_database_schema()
        
        processed_plan_ids, all_processed_data = process_pdfs_from_urls_in_parallel()

        if processed_plan_ids:
            update_drug_formulary_status(processed_plan_ids)
            update_plan_and_payer_statuses(processed_plan_ids)

        # Step 6: Map products to formulary
        logger.info("STEP 6: Mapping products to formulary")
        try:
            records_mapped = map_products_to_formulary()
            logger.info(f"Successfully mapped {records_mapped:,} drug records to product master data.")
        except Exception as e:
            logger.error(f"Product mapping failed, but continuing: {e}")
            # Don't raise - allow the pipeline to complete even if mapping fails

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info("Pipeline finished")
        
        # Print cost summary
        cost_data = print_cost_summary()
        
        update_transaction(
            transaction_id=transaction_id,
            status="completed",
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            response_summary={
                "processing_time": elapsed_time,
                "plans_processed": len(processed_plan_ids),
                "cost_data": cost_data
            },
            mistral_cost=cost_data['total_cost_usd'],
            ocr_pages_processed=cost_data['total_pages']
        )
        
        return {
            "run_summary": {
                "status": "completed",
                "mode": "realtime",
                "total_processing_time_seconds": round(elapsed_time, 2),
                "total_plans_processed": len(processed_plan_ids),
                "processed_plan_ids": processed_plan_ids
            },
            "cost_summary": cost_data
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        update_transaction(
            transaction_id=transaction_id,
            status="failed",
            error=str(e),
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        raise


def run_batch_submit_pipeline(output_jsonl="batch_requests.jsonl"):
    transaction_id = str(uuid.uuid4())
    logger.info(f"Starting pipeline (BATCH SUBMIT MODE) Transaction ID: {transaction_id}")
    
    create_transaction(
        transaction_id=transaction_id,
        job_type="batch_submit",
        status="running",
        request_summary={"mode": "batch_submit", "output_file": output_jsonl}
    )

    try:
        ensure_database_schema()
        
        plans = get_all_plans_with_formulary_url()
        if not plans:
            logger.warning("No plans processing.")
            update_transaction(transaction_id=transaction_id, status="completed", response_summary={"message": "No plans found"})
            return
            
        requests_payload, file_ids = prepare_batch_requests(plans)
        
        if requests_payload:
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for req in requests_payload:
                    f.write(json.dumps(req) + '\n')
            logger.info(f"Successfully wrote {len(requests_payload)} requests to {output_jsonl}")
            logger.info(f"Uploaded {len(file_ids)} files to Mistral.")
            logger.info("NEXT STEP: Upload this JSONL file to Mistral Batch API manually (or implement auto-submit).")
            
            update_transaction(
                transaction_id=transaction_id,
                status="completed",
                completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                response_summary={
                    "requests_count": len(requests_payload),
                    "files_uploaded": len(file_ids),
                    "output_file": output_jsonl
                }
            )
        else:
            logger.warning("No batch requests generated.")
            update_transaction(transaction_id=transaction_id, status="completed", response_summary={"message": "No requests generated"})

    except Exception as e:
        logger.error(f"Batch submit failed: {e}")
        update_transaction(
            transaction_id=transaction_id,
            status="failed",
            error=str(e),
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        raise


def run_batch_process_pipeline(input_jsonl):
    transaction_id = str(uuid.uuid4())
    logger.info(f"Starting pipeline (BATCH PROCESS MODE) with input: {input_jsonl} Transaction ID: {transaction_id}")
    
    create_transaction(
        transaction_id=transaction_id,
        job_type="batch_process",
        status="running",
        request_summary={"mode": "batch_process", "input_file": input_jsonl}
    )

    try:
        ensure_database_schema()
        
        process_batch_output(input_jsonl)
        
        # Step 6: Map products to formulary
        logger.info("STEP 6: Mapping products to formulary")
        try:
            records_mapped = map_products_to_formulary()
            logger.info(f"Successfully mapped {records_mapped:,} drug records to product master data.")
        except Exception as e:
            logger.error(f"Product mapping failed, but continuing: {e}")

        # Note: Cost tracking might need to be retrieved from the batch output metadata if available, 
        # or calculated based on pages processed.
        # For now, we update with status.
        
        update_transaction(
            transaction_id=transaction_id,
            status="completed",
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    except Exception as e:
        logger.error(f"Batch process failed: {e}")
        update_transaction(
            transaction_id=transaction_id,
            status="failed",
            error=str(e),
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        raise


def run_batch_auto_pipeline(output_jsonl="batch_requests.jsonl", result_jsonl="batch_results.jsonl"):
    transaction_id = str(uuid.uuid4())
    logger.info(f"Starting pipeline (BATCH AUTO MODE) Transaction ID: {transaction_id}")
    
    create_transaction(
        transaction_id=transaction_id,
        job_type="batch_auto",
        status="running",
        request_summary={"mode": "batch_auto"}
    )

    try:
        ensure_database_schema()
        
        # Step 1: Prepare batch requests
        plans = get_all_plans_with_formulary_url()
        if not plans:
            logger.warning("No plans processing.")
            update_transaction(transaction_id=transaction_id, status="completed", response_summary={"message": "No plans found"})
            return
            
        requests_payload, file_ids = prepare_batch_requests(plans)
        
        if not requests_payload:
            logger.warning("No batch requests generated.")
            update_transaction(transaction_id=transaction_id, status="completed", response_summary={"message": "No requests generated"})
            return

        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for req in requests_payload:
                f.write(json.dumps(req) + '\n')
        logger.info(f"Successfully wrote {len(requests_payload)} requests to {output_jsonl}")
        logger.info(f"Uploaded {len(file_ids)} files to Mistral.")

        # Step 2: Upload batch file
        from pdf_processing import create_resilient_mistral_client
        mistral_client = create_resilient_mistral_client()
        logger.info(f"Uploading batch file {output_jsonl} to Mistral...")
        batch_data = mistral_client.files.upload(
            file={
                "file_name": output_jsonl,
                "content": open(output_jsonl, "rb")
            },
            purpose="batch"
        )

        # Step 3: Create batch job
        logger.info("Creating Mistral batch job...")
        created_job = mistral_client.batch.jobs.create(
            input_files=[batch_data.id],
            model="mistral-ocr-latest",
            endpoint="/v1/ocr",
            metadata={"job_type": "automated_batch"}
        )

        # Step 4: Wait for job completion
        logger.info(f"Batch job {created_job.id} created. Waiting for completion...")
        while True:
            retrieved_job = mistral_client.batch.jobs.get(job_id=created_job.id)
            status = retrieved_job.status
            logger.info(f"Job Status: {status} | Total: {retrieved_job.total_requests} | Success: {retrieved_job.succeeded_requests} | Failed: {retrieved_job.failed_requests}")
            
            if status not in ["QUEUED", "RUNNING"]:
                break
                
            time.sleep(10)

        if status != "SUCCESS":
            logger.error(f"Batch job {created_job.id} finished with status: {status}")
            if status == "FAILED" or retrieved_job.failed_requests == retrieved_job.total_requests:
                 raise Exception(f"Batch job failed. Check mistral dashboard for details.")

        # Step 5: Download results
        logger.info("Downloading batch results...")
        file_response = mistral_client.files.download(file_id=retrieved_job.output_file)
        
        # Save content to file (handle both bytes and objects that have .content depending on the mistralai client version)
        with open(result_jsonl, "wb") as f:
            if hasattr(file_response, 'read'):
                f.write(file_response.read())
            elif hasattr(file_response, 'content'):
                f.write(file_response.content)
            elif isinstance(file_response, bytes):
                f.write(file_response)
            else:
                f.write(str(file_response).encode('utf-8'))
                
        logger.info(f"Saved results to {result_jsonl}")

        # Step 6: Process output
        logger.info(f"Processing output file {result_jsonl}...")
        process_batch_output(result_jsonl)
        
        # Step 7: Map products to formulary
        logger.info("STEP 7: Mapping products to formulary")
        try:
            records_mapped = map_products_to_formulary()
            logger.info(f"Successfully mapped {records_mapped:,} drug records to product master data.")
        except Exception as e:
            logger.error(f"Product mapping failed, but continuing: {e}")
            
        update_transaction(
            transaction_id=transaction_id,
            status="completed",
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            response_summary={
                "requests_count": len(requests_payload),
                "files_uploaded": len(file_ids),
                "job_status": status,
                "succeeded_requests": retrieved_job.succeeded_requests,
                "failed_requests": retrieved_job.failed_requests
            }
        )
        logger.info("Automated batch pipeline finished successfully.")

    except Exception as e:
        logger.error(f"Batch auto pipeline failed: {e}")
        update_transaction(
            transaction_id=transaction_id,
            status="failed",
            error=str(e),
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="PDF Processing Pipeline")
    parser.add_argument('--mode', type=str, choices=['realtime', 'batch_submit', 'batch_process', 'batch_auto'], default='batch_auto', help='Pipeline mode')
    parser.add_argument('--output', type=str, default='batch_requests.jsonl', help='Output JSONL file for batch submit')
    parser.add_argument('--input', type=str, help='Input JSONL file for batch process (output from Mistral)', required=False)
    
    args = parser.parse_args()
    
    if args.mode == 'realtime':
        run_realtime_pipeline()
    elif args.mode == 'batch_auto':
        run_batch_auto_pipeline()
    elif args.mode == 'batch_submit':
        run_batch_submit_pipeline(args.output)
    elif args.mode == 'batch_process':
        if not args.input:
            logger.error("--input argument is required for batch_process mode")
            sys.exit(1)
        run_batch_process_pipeline(args.input)


if __name__ == "__main__":
    main()
