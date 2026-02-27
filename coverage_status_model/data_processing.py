import pandas as pd

def clean_coverage_data(input_path, output_path=None):
    """
    Load CSV data and remove rows where 'Coverage Status' is null.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the cleaned CSV (optional, if None returns DataFrame only)
    
    Returns:
        Cleaned DataFrame
    """
    print("=" * 60)
    print("DATA PREPROCESSING STARTED")
    print("=" * 60)
    
    # Load the data (try multiple encodings for compatibility)
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin-1')
    
    # Standardize column names (strip whitespace and handle case variations)
    df.columns = df.columns.str.strip()
    
    # Map column names to standard format
    column_mapping = {
        'COVERAGE STATUS': 'Coverage Status',
        'EXPLANATION': 'Explanation'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Drop unnamed/empty columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"Dropping {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)
    
    # Print initial row count
    initial_count = len(df)
    print(f"\nRows before processing: {initial_count}")
    
    # Print null values per column
    print("-" * 60)
    print("NULL VALUES PER COLUMN:")
    print("-" * 60)
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"  {col}: {count} null values")
    
    total_nulls = null_counts.sum()
    print(f"\nTotal null values across all columns: {total_nulls}")
    
    # Remove rows where 'Coverage Status' is null
    print("-" * 60)
    print("REMOVING NULL COVERAGE STATUS ROWS")
    print("-" * 60)
    
    coverage_null_count = df['Coverage Status'].isnull().sum()
    print(f"Null values in 'Coverage Status' column: {coverage_null_count}")
    
    df_cleaned = df.dropna(subset=['Coverage Status'])
    
    # Normalize Coverage Status - strip whitespace
    df_cleaned['Coverage Status'] = df_cleaned['Coverage Status'].str.strip()
    
    # Define valid Coverage Status values
    valid_statuses = ['Covered', 'Covered with Condition', 'Not Covered']
    
    # Find and remove rows with invalid Coverage Status
    invalid_rows = df_cleaned[~df_cleaned['Coverage Status'].isin(valid_statuses)]
    if len(invalid_rows) > 0:
        print(f"\nRemoving {len(invalid_rows)} rows with invalid Coverage Status values:")
        for idx, row in invalid_rows.iterrows():
            status_preview = str(row['Coverage Status'])[:50] + "..." if len(str(row['Coverage Status'])) > 50 else str(row['Coverage Status'])
            print(f"  - {status_preview}")
    
    df_cleaned = df_cleaned[df_cleaned['Coverage Status'].isin(valid_statuses)]
    
    # Count Coverage Status categories
    print("-" * 60)
    print("COVERAGE STATUS DISTRIBUTION:")
    print("-" * 60)
    status_counts = df_cleaned['Coverage Status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print(f"\nTotal records: {len(df_cleaned)}")
    
    # Print final row count
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    
    print("-" * 60)
    print("PREPROCESSING SUMMARY")
    print("-" * 60)
    print(f"Rows before processing: {initial_count}")
    print(f"Rows removed (null Coverage Status): {removed_count}")
    print(f"Rows after processing: {final_count}")
    
    # Save to output path if provided
    if output_path:
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nCleaned data saved to: {output_path}")
    
    print("=" * 60)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Input and output file paths - Updated for 2026 data
    input_file = r"C:\Users\VH0000812\Desktop\Coverage\data\January_Acronym 2026_Copy(Consolidate).csv"
    output_file = r"C:\Users\VH0000812\Desktop\Coverage\data\January_Acronym_2026_cleaned.csv"
    
    # Clean the data
    cleaned_df = clean_coverage_data(input_file, output_file)
