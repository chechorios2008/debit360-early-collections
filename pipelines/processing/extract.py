import shutil
import time
import duckdb
from pathlib import Path
import os
from pyspark.sql import SparkSession

# --- Configuración de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "analytics.duckdb"
DB_PATH = Path(os.getenv("DUCKDB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()

# Directorio de Staging para evitar cuellos de botella de RAM
STAGING_DIR = Path(
    os.getenv("STAGING_DIR", str(PROJECT_ROOT / "database" / "staging"))
).resolve()


def read_table(
    spark: SparkSession,
    table_name: str,
    schema: str = "raw",
    *,
    force_refresh: bool = False,
    limit: int | None = None
):
    """
    Lee una tabla de DuckDB exportándola a Parquet para que Spark la consuma eficientemente.
    Este método evita el error de memoria (OOM) al no pasar por Pandas.
    """

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"❌ No se encontró la base DuckDB en: {DB_PATH}\n"
            f"Verifica la carpeta {PROJECT_ROOT / 'database'}"
        )

    # Definimos la ruta de salida en el staging
    out_dir = STAGING_DIR / f"{schema}.{table_name}.parquet_dir"

    # ✅ Validación de integridad de Staging
    parquet_files = list(out_dir.glob("*.parquet")) if out_dir.exists() else []
    needs_export = force_refresh or (not out_dir.exists()) or (len(parquet_files) == 0)

    if needs_export:
        # Limpieza de staging previo si existe o está roto
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Construcción de la Query
        base_query = f"SELECT * FROM {schema}.{table_name}"
        if limit is not None:
            base_query += f" LIMIT {int(limit)}"

        t0 = time.time()
        print(f"🦆 Exportando DuckDB [{schema}.{table_name}] -> Parquet (Staging)...")

        try:
            # ✅ Uso de context manager para asegurar que la conexión se cierre siempre
            with duckdb.connect(database=str(DB_PATH), read_only=True) as con:
                out_file = out_dir / "part-00000.parquet"
                # Exportación nativa de DuckDB a Parquet (altísima velocidad)
                con.execute(
                    f"COPY ({base_query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');"
                )

            duration = round(time.time() - t0, 2)
            print(f"✅ Exportación exitosa en {duration}s")

        except Exception as e:
            # Si falla, no dejamos rastro de archivos corruptos
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"❌ Error durante la exportación de {table_name}: {str(e)}")
            raise

    # ⚡ Carga en Spark: Al ser Parquet, Spark lee solo los metadatos al inicio (Lazy Loading)
    print(f"⚡ Cargando en Spark desde Staging: {schema}.{table_name}")
    return spark.read.parquet(str(out_dir))