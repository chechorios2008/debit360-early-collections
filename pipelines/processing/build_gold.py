import duckdb


DB_PATH = "database/analytics.duckdb"


def save_to_duckdb(df, table_namem, schema):

    con = duckdb.connect(DB_PATH)

    pdf = df.toPandas()  # noqa: F841

    con.execute(f"""
        CREATE OR REPLACE TABLE {schema}.{table_namem} 
        AS SELECT * FROM pdf
    """)

    con.close()