import duckdb
from pyspark.sql import SparkSession

DB_PATH = "database/analytics.duckdb"

def read_table(spack: SparkSession, table_name: str):
    
    con = duckdb.connect(DB_PATH)

    df_pd = con.execute(
        f"SELECT * FROM raw.{table_name}"
    ).fetchdf()

    con.close()

    df_spark = spack.createDataFrame(df_pd)

    return df_spark