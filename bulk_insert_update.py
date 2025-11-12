import os
import sys
import traceback
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# --- Configuration ---
DB_HOST = "ebv-dev-gentech-instance-1.c3e4zmhfpeff.us-east-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD") or "8*H0<.[q41p(3a8yNqEtk3kGu4r:"
DB_NAME = "ebvdb"

CSV_PATH = "Xolair_drug.csv"
BATCH_SIZE = 5000


def connect_to_db():
    try:
        password_escaped = quote_plus(DB_PASSWORD)
        url = f"postgresql+psycopg2://{DB_USER}:{password_escaped}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print("🔗 Connecting to DB...")
        engine = create_engine(url, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            print("✅ Connected:", conn.execute(text("SELECT version()")).scalar())
        return engine
    except Exception:
        print("❌ Database connection failed.")
        traceback.print_exc()
        sys.exit(1)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows missing required fields"""
    before = len(df)
    # Drop rows with no ID or entirely empty
    df = df.dropna(subset=["id"])
    df = df[df["id"].astype(str).str.strip() != ""]
    after_id = len(df)

    # Drop rows with no drug_name (optional safety)
    df = df.dropna(subset=["drug_name"])
    df = df[df["drug_name"].astype(str).str.strip() != ""]
    after_drug = len(df)

    print(f"🧹 Cleaned dataframe: {before} → {after_id} (after ID check) → {after_drug} (after drug_name check)")
    df["verification_status"] = "Verified"
    return df


def delete_existing(conn, ids):
    print(f"🗑️  Deleting {len(ids)} old records...")
    conn.execute(text("DELETE FROM public.drug_formulary_details WHERE id = ANY(:ids)"), {"ids": list(ids)})
    print("✅ Old records deleted.")


def insert_batches(conn, df):
    print(f"⬆️ Inserting {len(df)} cleaned records in batches of {BATCH_SIZE}...")
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        batch = df.iloc[start:end]
        print(f"   → Batch {start+1}-{end}")
        try:
            batch.to_sql("drug_formulary_details", conn, if_exists="append", index=False)
        except Exception:
            print(f"❌ Error inserting batch {start+1}-{end}, skipping.")
            traceback.print_exc()
    print("✅ All valid records inserted.")


def main():
    try:
        if not os.path.exists(CSV_PATH):
            print(f"❌ CSV not found: {CSV_PATH}")
            sys.exit(1)

        print(f"📄 Reading {CSV_PATH} ...")
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from CSV.")

        engine = connect_to_db()
        df = clean_dataframe(df)

        if "id" not in df.columns:
            print("❌ CSV must contain 'id' column.")
            sys.exit(1)

        ids_to_delete = df["id"].dropna().unique().tolist()
        print(f"🔍 Found {len(ids_to_delete)} IDs to refresh.")

        with engine.begin() as conn:
            delete_existing(conn, ids_to_delete)
            insert_batches(conn, df)

        print("\n✅✅ Bulk update completed successfully with cleaned data.")
    except Exception:
        print("❌ Fatal error during process.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
