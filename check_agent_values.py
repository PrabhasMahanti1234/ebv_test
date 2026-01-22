"""
Check the status of preferred_agent and non_preferred_agent values in the database.
"""
from database import get_db_connection

def check_agent_values():
    print("=" * 70)
    print("CHECKING AGENT VALUES IN DATABASE")
    print("=" * 70)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Count records by preferred_agent / non_preferred_agent values
        print("\n1. Count by preferred_agent and non_preferred_agent:")
        cursor.execute("""
            SELECT 
                preferred_agent, 
                non_preferred_agent, 
                COUNT(*) as count
            FROM drug_formulary_details 
            GROUP BY preferred_agent, non_preferred_agent
            ORDER BY count DESC
        """)
        
        results = cursor.fetchall()
        print(f"\n   {'Preferred Agent':<20} {'Non-Preferred Agent':<20} {'Count':<10}")
        print(f"   {'-'*20} {'-'*20} {'-'*10}")
        for row in results:
            pref = str(row[0]) if row[0] is not None else "NULL"
            non_pref = str(row[1]) if row[1] is not None else "NULL"
            count = row[2]
            print(f"   {pref:<20} {non_pref:<20} {count:<10}")
        
        # Check for [default] values specifically
        print("\n2. Checking for '[default]' values:")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM drug_formulary_details 
            WHERE preferred_agent = '[default]' OR non_preferred_agent = '[default]'
        """)
        default_count = cursor.fetchone()[0]
        
        if default_count > 0:
            print(f"   ❌ Found {default_count} records with '[default]' values!")
        else:
            print(f"   ✅ No '[default]' values found!")
        
        # Total drug count
        cursor.execute("SELECT COUNT(*) FROM drug_formulary_details")
        total = cursor.fetchone()[0]
        print(f"\n3. Total drug records: {total}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_agent_values()
