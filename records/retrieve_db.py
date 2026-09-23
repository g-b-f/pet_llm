from pathlib import Path

import psycopg2
import pandas as pd

records_dir = Path(__file__).parent.resolve()

tables = ["runs", "thought_events", "memory_clear_events", "malformed_json_events"]

with psycopg2.connect(
    database="postgres",
    user="postgres",
    password="password",
    host="localhost",
    port= "5432"
) as conn:
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(f"{records_dir}/{table}.csv", index=False)