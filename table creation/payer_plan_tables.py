import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ebv",
    "user": "postgres",
    "password": "1234"
}

def create_tables():

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # ===============================
    # ENABLE pgcrypto (for gen_random_uuid)
    # ===============================
    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
    """)

    # ===============================
    # PAYER_MASTER
    # ===============================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS public."payer_master" (
        "Payer_ID" VARCHAR(255),
        "Payer_Name" VARCHAR(255),
        "State_Name" VARCHAR(100),
        "Mailing_Address1" VARCHAR(255),
        "Mailing_Address2" VARCHAR(255),
        "Mailing_City" VARCHAR(100),
        "Mailing_State" VARCHAR(100),
        "Mailing_Zip" VARCHAR(20),
        "FAX" VARCHAR(50),
        "Main_Phone" VARCHAR(50),
        "BV_Phone" VARCHAR(50),
        "Claims_Phone" VARCHAR(50),

        CONSTRAINT uq_payer_identity
        UNIQUE ("Payer_Name","State_Name")
    );
    """)

    # Optional index for faster search
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_payer_name
        ON public."payer_master" ("Payer_Name");
    """)

    # ===============================
    # PLAN_MASTER
    # ===============================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS public."payer_plan_master" (
        "PP_ID" VARCHAR(255),
        "PAYER_ID" VARCHAR(255),
        "PLAN_NAME" VARCHAR(255),
        "PLAN_TYPE" VARCHAR(100),
        "PLAN_SUB_TYPE" VARCHAR(100),
        "PLAN_CATEGORY" VARCHAR(255),
        "STATE_NAME" VARCHAR(100),
        "Mailing_Address1" VARCHAR(255),
        "Mailing_Address2" VARCHAR(255),
        "Mailing_City" VARCHAR(100),
        "Mailing_State" VARCHAR(100),
        "Mailing_Zip" VARCHAR(20),
        "MAIN_FAX" VARCHAR(50),
        "Main_Phone" VARCHAR(50),
        "Prior Authorization_FAX" VARCHAR(50),
        "CLAIMS_PHONE" VARCHAR(50)
    );
    """)

    # Optional index for faster lookup
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_plan_identity
        ON public."payer_plan_master"
        ("PAYER_ID","PLAN_NAME","STATE_NAME");
    """)

    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Tables created successfully (clean master structure).")


if __name__ == "__main__":
    create_tables()