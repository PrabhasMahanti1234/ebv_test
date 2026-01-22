"""
Deep clean - removes ALL data and cache, forces complete reprocessing.
"""
import os
import glob
from database import get_db_connection

def deep_clean():
    print("=" * 70)
    print("DEEP CLEAN - Removing ALL data and cache")
    print("=" * 70)
    
    # 1. Clear database tables
    print("\n1. Clearing database tables...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear cache table
        cursor.execute("DELETE FROM processed_file_cache")
        cache_count = cursor.rowcount
        print(f"   ✓ Deleted {cache_count} cached file entries")
        
        # Clear drug records
        cursor.execute("DELETE FROM drug_formulary_details")
        drug_count = cursor.rowcount
        print(f"   ✓ Deleted {drug_count} drug records")
        
        # Clear acronyms
        cursor.execute("DELETE FROM pp_formulary_names")
        acronym_count = cursor.rowcount
        print(f"   ✓ Deleted {acronym_count} acronym records")
        
        conn.commit()
    
    # 2. Clear Python bytecode cache
    print("\n2. Clearing Python bytecode cache...")
    removed = 0
    for root, dirs, files in os.walk('.'):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_dir = os.path.join(root, '__pycache__')
            for f in os.listdir(pycache_dir):
                os.remove(os.path.join(pycache_dir, f))
                removed += 1
            os.rmdir(pycache_dir)
        
        # Remove .pyc files
        for f in files:
            if f.endswith('.pyc'):
                os.remove(os.path.join(root, f))
                removed += 1
    
    print(f"   ✓ Removed {removed} bytecode cache files")
    
    print("\n" + "=" * 70)
    print("✅ DEEP CLEAN COMPLETE!")
    print("=" * 70)
    print("\nNow run: python -B main.py")
    print("")

if __name__ == "__main__":
    deep_clean()
