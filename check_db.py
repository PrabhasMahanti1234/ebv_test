import psycopg2
from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
    SELECT drug_name, preferred_agent, non_preferred_agent, badge_colors 
    FROM drug_formulary_details 
    WHERE preferred_agent IS NOT NULL OR non_preferred_agent IS NOT NULL
    LIMIT 10
""")

print("Drugs with preferred_agent/non_preferred_agent data:")
print("=" * 100)
for row in cursor.fetchall():
    print(f"Drug: {row[0][:50]:<50} | Preferred: {row[1]:<5} | Non-Preferred: {row[2]:<5} | Badge: {row[3]}")

cursor.execute("SELECT COUNT(*) FROM drug_formulary_details WHERE preferred_agent IS NOT NULL")
pref_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM drug_formulary_details WHERE non_preferred_agent IS NOT NULL")
non_pref_count = cursor.fetchone()[0]

print("\n" + "=" * 100)
print(f"Total drugs with preferred_agent: {pref_count}")
print(f"Total drugs with non_preferred_agent: {non_pref_count}")

cursor.close()
conn.close()
