-- Query to list all extracted drugs in order
SELECT 
    drug_name,
    preferred_agent,
    non_preferred_agent,
    drug_requirements
FROM drug_formulary_details
WHERE payer_name = 'Hennepin Health_12'
ORDER BY id;
