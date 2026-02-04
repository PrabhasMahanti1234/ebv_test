import json

# Expected drugs from PDF (manually counted from the image)
expected_drugs = {
    "Alpha-Adrenergic Agonists": {
        "Preferred": [
            "Alphagan® P (brimonidine) 0.1%",
            "Brimonidine 0.2%",
            "Iopidine® (apraclonidine)"
        ],
        "Non-Preferred, PA Required": [
            "Alphagan® P (brimonidine) 0.15%"
        ]
    },
    "Antihistamines/Mast Cell Stabilizers": {
        "Preferred": [
            "Alaway® (ketotifen)",
            "Cromolyn® (cromolyn)",
            "Optivar® (azelastine)",
            "Pataday® 0.1%, 0.2% (olopatadine)",
            "Patanol® (olopatadine)",
            "Refresh® (ketotifen)",
            "Zaditor® (ketotifen)"
        ],
        "Non-Preferred, PA Required": [
            "Alocril® (nedocromil)",
            "Alomide® (lodoxamide)",
            "Bepreve® (bepotastine)",
            "Elestat® (epinastine)",
            "Emadine® (emedastine)",
            "Lastacaft® (alcaftadine)",
            "Pataday® 0.7% (olopatadine)",
            "Pazeo® (olopatadine)",
            "Zerviate™ (cetirizine)"
        ]
    },
    "Anti-Infective/Steroid Combinations": {
        "Preferred": [
            "Blephamide® (sulfacetamide/prednisolone)",
            "Maxitrol® (neomycin/polymyxin/dexamethasone)",
            "Pred-G® (prednisolone/gentamicin",
            "Pred-G S.O.P.® (prednisolone/gentamicin)"
        ],
        "Non-Preferred, PA Required": [
            "Blephamide S.O.P.® (sulfacetamide/prednisolone)",
            "TobraDex® (tobramycin/dexamethasone)",
            "TobraDex® ST (tobramycin/dexamethasone)",
            "Zylet® (loteprednol/tobramycin)"
        ]
    },
    "Beta-Blockers": {
        "Preferred": [
            "Betagart® (levobunolol)",
            "Betimol® (timolol)",
            "Betoptic® (betaxolol)",
            "Betoptic®-S (betaxolol)",
            "Carteolol",
            "OptiPranolol® (metipranolol)",
            "Timoptic® (timolol)",
            "Timoptic-XE® (timolol)"
        ],
        "Non-Preferred, PA Required": [
            "Istalol® (timolol)",
            "Timoptic® Ocudose® (timolol)"
        ]
    },
    "Carbonic Anhydrase Inhibitors": {
        "Preferred": [
            "Azopt® (brinzolamide)"
        ],
        "Non-Preferred, PA Required": [
            "Trusopt® (dorzolamide)"
        ]
    }
}

# Count expected total
total_expected = 0
for category, columns in expected_drugs.items():
    for column, drugs in columns.items():
        total_expected += len(drugs)

print(f"Expected total drugs from PDF: {total_expected}")
print()

# Read debug_response.json
try:
    with open('debug_response.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_drugs = data.get('DrugInformation', [])
    print(f"Extracted drugs in debug_response.json: {len(extracted_drugs)}")
    print()
    
    # List all extracted drug names
    print("=" * 80)
    print("EXTRACTED DRUGS:")
    print("=" * 80)
    for i, drug in enumerate(extracted_drugs, 1):
        name = drug.get('Drug Name', '')
        pref = drug.get('preferred_agent', '')
        non_pref = drug.get('non_preferred_agent', '')
        reqs = drug.get('requirements', '')
        category = drug.get('category', '')
        
        agent_status = ""
        if pref == 'yes':
            agent_status = "[PREFERRED]"
        elif non_pref == 'yes':
            agent_status = f"[NON-PREFERRED{' + PA' if reqs == 'PA' else ''}]"
        
        print(f"{i:3}. {agent_status:25} {name[:60]:<60} | {category}")
    
    print()
    print("=" * 80)
    
    # Build a set of extracted drug names (normalized)
    def normalize(name):
        return name.lower().replace('®', '').replace('™', '').replace(' ', '').strip()
    
    extracted_set = {normalize(d.get('Drug Name', '')) for d in extracted_drugs}
    
    # Check which expected drugs are missing
    missing = []
    for category, columns in expected_drugs.items():
        for column, drugs in columns.items():
            for drug in drugs:
                if normalize(drug) not in extracted_set:
                    missing.append(f"{drug} ({category} - {column})")
    
    if missing:
        print(f"\n🚨 MISSING {len(missing)} DRUGS:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\n✅ All expected drugs were extracted!")
        
except FileNotFoundError:
    print("debug_response.json not found!")
except Exception as e:
    print(f"Error: {e}")
