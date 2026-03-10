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

FILE_PATH = r"C:\Users\VH0000540\Downloads\Latest_eBV (2)\table creation\2026 Plan Master - Copy(January).csv"


def import_data():

    print("🔄 Starting Plan_Master import...")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # ==========================================
    # READ & CLEAN CSV
    # ==========================================
    df = pd.read_csv(FILE_PATH, dtype=str, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()

    df.rename(columns={
        "PLAN_TYPECommercial,Medicare,Medicaid": "PLAN_TYPE",
        "PLAN_SUB_TYPEHMO,PPO,etc": "PLAN_SUB_TYPE",
        "PLAN_CATEGORYIndividual,Group,Small Group, Large Group, Employee Based, etc": "PLAN_CATEGORY",
        "Mailing_Address1-": "Mailing_Address1",
        "CLAIMS_PHONE ": "CLAIMS_PHONE"
    }, inplace=True)

    df = df.where(pd.notnull(df), None)

    # ==========================================
    # CLEAN PLAN_NAME
    # ==========================================
    if "PLAN_NAME" in df.columns:
        df["PLAN_NAME"] = (
            df["PLAN_NAME"]
            .astype(str)
            .str.strip()                       # remove leading/trailing spaces
            .str.replace(r"\s*-\s*", "-", regex=True)  # remove spaces around "-"
        )

    # Normalize payer name (trim + lowercase for matching)
    df["PAYER_ID"] = df["PAYER_ID"].str.strip()

    # ==========================================
    # AUTO CREATE MISSING PAYERS
    # ==========================================
    cursor.execute("""
        SELECT LOWER(TRIM("Payer_Name")), "Payer_ID"
        FROM public."payer_master"
    """)
    payer_map = dict(cursor.fetchall())

    new_payers = set()

    for payer in df["PAYER_ID"].unique():
        # Skip empty/None values so it doesn't break
        if payer is None or pd.isna(payer):
            continue
            
        key = str(payer).lower().strip()
        if key not in payer_map:
            new_payers.add(payer)

    if new_payers:
        print(f"🔄 Creating {len(new_payers)} new payer(s)...")

        insert_query = """
        INSERT INTO public."payer_master"
        ("Payer_ID","Payer_Name","State_Name")
        VALUES %s
        """

        new_rows = [(None, payer, None) for payer in new_payers]
        execute_values(cursor, insert_query, new_rows)
        conn.commit()

        # Generate UUID for new payers
        cursor.execute("""
            WITH new_ids AS (
                SELECT "Payer_Name", gen_random_uuid()::text AS new_uuid
                FROM public."payer_master"
                WHERE "Payer_ID" IS NULL
            )
            UPDATE public."payer_master" pm
            SET "Payer_ID" = n.new_uuid
            FROM new_ids n
            WHERE pm."Payer_Name" = n."Payer_Name"
              AND pm."Payer_ID" IS NULL;
        """)
        conn.commit()

        # Reload payer map
        cursor.execute("""
            SELECT LOWER(TRIM("Payer_Name")), "Payer_ID"
            FROM public."payer_master"
        """)
        payer_map = dict(cursor.fetchall())

    # ==========================================
    # MAP payer_name → UUID
    # ==========================================
    df["PAYER_ID"] = df["PAYER_ID"].str.lower().str.strip()
    df["PAYER_ID"] = df["PAYER_ID"].map(payer_map)

    # ==========================================
    # PREVENT RE-INSERT (IDENTITY CHECK)
    # ==========================================
    cursor.execute("""
        SELECT "PAYER_ID","PLAN_NAME","STATE_NAME"
        FROM public."payer_plan_master"
    """)
    existing_identity = set(cursor.fetchall())

    def identity(row):
        return (
            row["PAYER_ID"],
            row["PLAN_NAME"],
            row["STATE_NAME"]
        )

    new_rows_df = df[
        ~df.apply(lambda r: identity(r) in existing_identity, axis=1)
    ]

    column_order = [
        "PP_ID","PAYER_ID","PLAN_NAME","PLAN_TYPE",
        "PLAN_SUB_TYPE","PLAN_CATEGORY","STATE_NAME",
        "Mailing_Address1","Mailing_Address2","Mailing_City",
        "Mailing_State","Mailing_Zip","MAIN_FAX",
        "Main_Phone","Prior Authorization_FAX","CLAIMS_PHONE"
    ]

    new_rows_df = new_rows_df[column_order]
    values = [tuple(row) for row in new_rows_df.to_numpy()]

    # ==========================================
    # INSERT NEW ROWS
    # ==========================================
    if values:
        insert_query = """
        INSERT INTO public."payer_plan_master"
        ("PP_ID","PAYER_ID","PLAN_NAME","PLAN_TYPE",
         "PLAN_SUB_TYPE","PLAN_CATEGORY","STATE_NAME",
         "Mailing_Address1","Mailing_Address2","Mailing_City",
         "Mailing_State","Mailing_Zip","MAIN_FAX",
         "Main_Phone","Prior Authorization_FAX","CLAIMS_PHONE")
        VALUES %s
        """
        execute_values(cursor, insert_query, values)
        conn.commit()

    print(f"✅ Newly inserted rows: {len(values)}")

    # ==========================================
    # GENERATE STABLE PP_ID
    # ==========================================
    cursor.execute("""
        WITH unique_combos AS (
            SELECT
                "PAYER_ID",
                "PLAN_NAME",
                "STATE_NAME",
                gen_random_uuid()::text AS new_uuid
            FROM public."payer_plan_master"
            WHERE "PP_ID" IS NULL
            GROUP BY "PAYER_ID","PLAN_NAME","STATE_NAME"
        )
        UPDATE public."payer_plan_master" ppm
        SET "PP_ID" = uc.new_uuid
        FROM unique_combos uc
        WHERE ppm."PAYER_ID" = uc."PAYER_ID"
          AND ppm."PLAN_NAME" = uc."PLAN_NAME"
          AND ppm."STATE_NAME" = uc."STATE_NAME"
          AND ppm."PP_ID" IS NULL;
    """)

    generated = cursor.rowcount
    conn.commit()

    cursor.close()
    conn.close()

    print(f"✅ PP_ID generated for: {generated}")
    print("✅ Plan import completed successfully.")


if __name__ == "__main__":
    import_data()