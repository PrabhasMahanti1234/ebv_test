import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ebv",
    "user": "postgres",
    "password": "1234"
}

FILE_PATH = r"C:\Users\VH0000540\Downloads\Latest_eBV (2)\table creation\Payer_Master Table - Copy(Payer Master).csv"


def import_data():

    print("🔄 Starting Payer_Master import...")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # ===============================
    # READ CSV
    # ===============================
    df = pd.read_csv(FILE_PATH, dtype=str)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.where(pd.notnull(df), None)

    column_order = [
        "Payer_ID",
        "Payer_Name",
        "State_Name",
        "Mailing_Address1",
        "Mailing_Address2",
        "Mailing_City",
        "Mailing_State",
        "Mailing_Zip",
        "FAX",
        "Main_Phone",
        "BV_Phone",
        "Claims_Phone"
    ]

    df = df[column_order]
    values = [tuple(row) for row in df.to_numpy()]

    # ===============================
    # INSERT ONLY NEW ROWS
    # ===============================
    insert_query = """
    INSERT INTO public."payer_master"
    ("Payer_ID","Payer_Name","State_Name",
     "Mailing_Address1","Mailing_Address2","Mailing_City",
     "Mailing_State","Mailing_Zip","FAX",
     "Main_Phone","BV_Phone","Claims_Phone")
    VALUES %s
    ON CONFLICT ("Payer_Name","State_Name") DO NOTHING
    """

    execute_values(cursor, insert_query, values)
    inserted_rows = cursor.rowcount
    conn.commit()

    print(f"✅ Newly inserted rows: {inserted_rows}")

    # ===============================
    # GENERATE UUID ONLY FOR NEW PAYER_NAMES
    # ===============================
    print("🔄 Generating UUID for new payer names...")

    cursor.execute("""
        WITH new_ids AS (
            SELECT
                "Payer_Name",
                gen_random_uuid()::text AS new_uuid
            FROM public."payer_master"
            WHERE "Payer_ID" IS NULL
            GROUP BY "Payer_Name"
        )
        UPDATE public."payer_master" pm
        SET "Payer_ID" = n.new_uuid
        FROM new_ids n
        WHERE pm."Payer_Name" = n."Payer_Name"
          AND pm."Payer_ID" IS NULL;
    """)

    generated_ids = cursor.rowcount
    conn.commit()

    cursor.close()
    conn.close()

    print(f"✅ New UUIDs generated: {generated_ids}")
    print("✅ Payer import completed successfully.")


if __name__ == "__main__":
    import_data()