# pipelines/processing/extract.py
import duckdb

DB_PATH = "database/analytics.duckdb"


def read_table(spark, table_name: str, schema="raw"):
    con = duckdb.connect(DB_PATH)
    # Usamos arrow para una transferencia de datos mucho más veloz y ligera
    df_arrow = con.execute(f"SELECT * FROM {schema}.{table_name}").fetch_arrow_table()
    con.close()

    return spark.createDataFrame(df_arrow.to_pandas())