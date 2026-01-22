"""
Clear cache and drug records to force reprocessing with updated code.
"""
from database import get_db_connection

def clear_all_cache():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear file cache
        cursor.execute("DELETE FROM processed_file_cache")
        cache_count = cursor.rowcount
        print(f"Deleted {cache_count} cached file entries")
        
        # Clear drug formulary details
        cursor.execute("DELETE FROM drug_formulary_details")
        drug_count = cursor.rowcount
        print(f"Deleted {drug_count} drug records")
        
        conn.commit()
        print("✅ Cache and drug records cleared! Re-run main.py to reprocess.")

if __name__ == "__main__":
    clear_all_cache()
