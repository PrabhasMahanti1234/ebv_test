"""
clean_db.py - Database Cleanup Script

Simple script to clean database tables for testing and development.
Run with: python clean_db.py [option]

Options:
    --all           : Clean ALL tables (full reset)
    --drugs         : Clean only drug_formulary_details
    --drugs-plan    : Clean drugs for a specific plan (prompts for plan_id)
    --cache         : Clean processed_file_cache only
    --acronyms      : Clean pp_formulary_names only
    --help          : Show this help message
"""

import sys
import logging
from database import get_db_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_table_counts():
    """Get current record counts for all tables."""
    tables = [
        'drug_formulary_details',
        'pp_formulary_names', 
        'processed_file_cache',
        'plan_details',
        'payer_details'
    ]
    
    counts = {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except Exception as e:
                counts[table] = f"Error: {e}"
    
    return counts


def print_table_counts(counts, title="Current Database State"):
    """Pretty print table counts."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for table, count in counts.items():
        print(f"  {table:<30} : {count:>10}")
    print(f"{'='*50}\n")


def clean_drug_formulary():
    """Clean all records from drug_formulary_details."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM drug_formulary_details")
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted} records from drug_formulary_details")
        return deleted


def clean_drugs_for_plan(plan_id):
    """Clean drug records for a specific plan."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM drug_formulary_details WHERE plan_id = %s", (plan_id,))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted} records for plan_id: {plan_id}")
        return deleted


def clean_cache():
    """Clean the processed_file_cache table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM processed_file_cache")
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted} records from processed_file_cache")
        return deleted


def clean_acronyms():
    """Clean the pp_formulary_names table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pp_formulary_names")
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted} records from pp_formulary_names")
        return deleted


def clean_plans():
    """Clean all plans (will cascade delete drugs due to FK)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM plan_details")
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted} records from plan_details")
        return deleted


def clean_payers():
    """Clean all payers (will cascade delete plans and drugs due to FK)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM payer_details")
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted} records from payer_details")
        return deleted


def clean_all():
    """Clean ALL tables - full database reset."""
    logger.info("🧹 Starting full database cleanup...")
    
    # Order matters due to foreign key constraints
    # Clean in reverse order of dependencies
    clean_drug_formulary()
    clean_acronyms()
    clean_cache()
    clean_plans()
    clean_payers()
    
    logger.info("✅ Full database cleanup complete!")


def reset_plan_status():
    """Reset all plans to 'processing' status for reprocessing."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE plan_details SET status = 'processing'")
        updated = cursor.rowcount
        conn.commit()
        logger.info(f"✅ Reset {updated} plans to 'processing' status")
        return updated


def show_help():
    """Show help message."""
    print(__doc__)


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ['--help']
    
    # Show current state before any operation
    if args[0] != '--help':
        print_table_counts(get_table_counts(), "BEFORE Cleanup")
    
    if '--help' in args or '-h' in args:
        show_help()
        return
    
    if '--all' in args:
        confirm = input("⚠️  This will DELETE ALL DATA. Type 'YES' to confirm: ")
        if confirm == 'YES':
            clean_all()
        else:
            print("❌ Aborted.")
            return
    
    elif '--drugs' in args:
        confirm = input("⚠️  This will delete ALL drug records. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            clean_drug_formulary()
        else:
            print("❌ Aborted.")
            return
    
    elif '--drugs-plan' in args:
        plan_id = input("Enter plan_id to clean: ").strip()
        if plan_id:
            clean_drugs_for_plan(plan_id)
        else:
            print("❌ No plan_id provided.")
            return
    
    elif '--cache' in args:
        clean_cache()
    
    elif '--acronyms' in args:
        clean_acronyms()
    
    elif '--reset-status' in args:
        reset_plan_status()
    
    else:
        print(f"❌ Unknown option: {args[0]}")
        show_help()
        return
    
    # Show state after cleanup
    print_table_counts(get_table_counts(), "AFTER Cleanup")


if __name__ == "__main__":
    main()

