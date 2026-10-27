from pathlib import Path
import duckdb

# Define paths
DATA_PATH = Path("data/raw/ad_impressions.csv")

# Check that the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

# Create an in-memory DuckDB connection
con = duckdb.connect(database=":memory:")

# Convert the path to an absolute path
csv_path = DATA_PATH.resolve().as_posix()

# Create a view over the CSV file
con.execute(f"""
    CREATE VIEW ad_events AS
    SELECT *
    FROM read_csv_auto('{csv_path}')
""")

print("DuckDB view created successfully!")

# Test the view
result = con.execute("""
    SELECT COUNT(*) AS total_rows
    FROM ad_events
""").fetchone()

print(f"Total rows loaded: {result[0]:,}")
