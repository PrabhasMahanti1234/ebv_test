
import logging
import sys
from database import get_db_connection
from coverage import det_coverage_status

# Setup basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_drug_coverage():
    # Test Cases
    test_cases = [
        {
            "drug_name": "acyclovir oral capsule 200 mg",
            "tier": "Tier 1",
            "requirements": "",
            "payer_name": "Wellcare",
            "state_name": "Arkansas"
        },
        {
            "drug_name": "Test Drug 5^",
            "tier": "5^",
            "requirements": "",
            "payer_name": "Wellcare",
            "state_name": "Arkansas"
        },
        {
            "drug_name": "Test Drug Tier 5^",
            "tier": "^",
            "requirements": "",
            "payer_name": "Wellcare",
            "state_name": "Arkansas"
        },
        {
            "drug_name": "Test Drug 1",
            "tier": "1",
            "requirements": "",
            "payer_name": "Wellcare",
            "state_name": "Arkansas"
        },
        {
            "drug_name": "Test Drug tier-1",
            "tier": "tier-1",
            "requirements": "",
            "payer_name": "Wellcare",
            "state_name": "Arkansas"
        }
    ]

    for case in test_cases:
        drug_name = case["drug_name"]
        tier = case["tier"]
        requirements = case["requirements"]
        payer_name = case["payer_name"]
        state_name = case["state_name"]
        
        # In the pipeline, acronym is usually combined tier and requirements
        acronym = tier
        if requirements:
            acronym = f"{tier}, {requirements}"

        print(f"\n{'='*60}")
        print(f"TESTING COVERAGE STATUS FOR:")
        print(f"Drug Name:    {drug_name}")
        print(f"Tier:         {tier}")
        print(f"Requirements: {requirements}")
        print(f"Payer:        {payer_name}")
        print(f"State:        {state_name}")
        print(f"{'='*60}")

        try:
            with get_db_connection() as conn:
                # Call the function
                status, confidence, source, manual_review = det_coverage_status(
                    acronym=acronym,
                    requirements_text=requirements,
                    tier_text=tier,
                    conn=conn,
                    state_name=state_name,
                    payer_name=payer_name,
                    drug_name=drug_name
                )

                print(f"RESULT:")
                print(f"Coverage Status: {status}")
                print(f"Confidence Score: {confidence}")
                print(f"Source:          {source}")
                print(f"Manual Review:   {manual_review}")
                print(f"{'*'*60}")

        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_drug_coverage()
