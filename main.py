import logging
from database import ensure_database_schema, update_drug_formulary_status, update_plan_and_payer_statuses
from excel_processing import populate_payer_and_plan_tables
from pdf_processing import process_pdfs_from_urls_in_parallel


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)




def main():
	logger.info("Starting pipeline")
	ensure_database_schema()
	populate_payer_and_plan_tables()
	processed_plan_ids, _ = process_pdfs_from_urls_in_parallel()

	if processed_plan_ids:
		update_drug_formulary_status(processed_plan_ids)
		update_plan_and_payer_statuses(processed_plan_ids)

	logger.info("Pipeline finished")

if __name__ == '__main__':
	main()