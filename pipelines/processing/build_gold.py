from __future__ import annotations

from pathlib import Path
import os
import tempfile
import shutil
import duckdb

# Raíz del repo: .../op_cobro/pipelines/processing/build_gold.py -> parents[2] = .../op_cobro
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "analytics.duckdb"
DB_PATH = Path(os.getenv("DUCKDB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()

# staging para parquet (puedes reutilizar la misma lógica que extract.py)
STAGING_DIR = Path(os.getenv("STAGING_DIR", str(PROJECT_ROOT / "database" / "staging"))).resolve()
STAGING_DIR.mkdir(parents=True, exist_ok=True)


def _table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = ? AND table_name = ?
    LIMIT 1
    """
    return con.execute(q, [schema, table]).fetchone() is not None


def save_to_duckdb(df, table_name: str, schema: str, *, mode: str = "skip"):
    """
    Guarda un Spark DataFrame en DuckDB sin pasar por Pandas.
    mode:
      - "skip": si existe {schema}.{table_name}, NO hace nada
      - "overwrite": reemplaza la tabla
    """

    if df is None:
        raise ValueError(f"❌ save_to_duckdb recibió df=None para {schema}.{table_name}. Revisa create_features().")

    # Asegurar carpeta database/
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=str(DB_PATH), read_only=False)
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        if mode == "skip" and _table_exists(con, schema, table_name):
            print(f"⏭️  DuckDB: {schema}.{table_name} ya existe. (mode=skip) -> no se re-crea.")
            return

        # 1) Spark escribe parquet a staging (directorio)
        out_dir = STAGING_DIR / f"{schema}.{table_name}.parquet_write"
        # limpieza previa
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        df.write.mode("overwrite").parquet(str(out_dir))

        # 2) DuckDB crea/replace desde parquet
        if mode == "overwrite":
            con.execute(f"DROP TABLE IF EXISTS {schema}.{table_name}")

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} AS
            SELECT * FROM read_parquet('{out_dir.as_posix()}/*.parquet')
        """)

        # Si la tabla ya existía y queremos sobrescribir “en serio”
        if mode == "overwrite":
            # Re-crear asegurando replace (DROP + CREATE)
            con.execute(f"DROP TABLE IF EXISTS {schema}.{table_name}")
            con.execute(f"""
                CREATE TABLE {schema}.{table_name} AS
                SELECT * FROM read_parquet('{out_dir.as_posix()}/*.parquet')
            """)

        print(f"✅ Guardado DuckDB: {schema}.{table_name}")

    finally:
        con.close()