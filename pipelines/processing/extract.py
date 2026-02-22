# pipelines/processing/extract.py
from pathlib import Path
import os
import duckdb

# Raíz del proyecto (op_cobro) basada en la ubicación de este archivo:
# .../op_cobro/pipelines/processing/extract.py  -> parents[2] = .../op_cobro
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permite override por variable de entorno si algún día lo necesitas:
# set DUCKDB_PATH=C:\ruta\custom\analytics.duckdb
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "analytics.duckdb"
DB_PATH = Path(os.getenv("DUCKDB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()


def read_table(spark, table_name: str, schema: str = "raw"):
    """
    Lee una tabla desde DuckDB y la retorna como Spark DataFrame.
    - spark: SparkSession
    - table_name: nombre de tabla (ej: "canales")
    - schema: esquema en DuckDB (ej: "raw", "silver", etc.)
    """

    # Validación clara para evitar errores confusos
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la base DuckDB en: {DB_PATH}\n"
            f"Tip: verifica que exista el archivo en {PROJECT_ROOT / 'database'}\n"
            f"CWD actual: {Path.cwd()}"
        )

    con = None
    try:
        # read_only=True evita modificaciones accidentales
        con = duckdb.connect(database=str(DB_PATH), read_only=True)

        # Arrow para transferencia veloz
        df_arrow = con.execute(
            f"SELECT * FROM {schema}.{table_name}"
        ).fetch_arrow_table()

    finally:
        if con is not None:
            con.close()

    return spark.createDataFrame(df_arrow.to_pandas())