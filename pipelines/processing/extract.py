import shutil
import time
import duckdb
from pathlib import Path
import os



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "analytics.duckdb"
DB_PATH = Path(os.getenv("DUCKDB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()

STAGING_DIR = Path(
    os.getenv("STAGING_DIR", str(PROJECT_ROOT / "database" / "staging"))
).resolve()


def read_table(
    spark,
    table_name: str,
    schema: str = "raw",
    *,
    force_refresh: bool = False,
    limit: int | None = None
):
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la base DuckDB en: {DB_PATH}\n"
            f"Tip: verifica que exista el archivo en {PROJECT_ROOT / 'database'}\n"
            f"CWD actual: {Path.cwd()}"
        )

    out_dir = STAGING_DIR / f"{schema}.{table_name}.parquet_dir"

    # ✅ Detectar “staging roto”: carpeta existe pero sin archivos parquet
    parquet_files = list(out_dir.glob("*.parquet")) if out_dir.exists() else []

    needs_export = force_refresh or (not out_dir.exists()) or (len(parquet_files) == 0)

    if needs_export:
        # Si existe pero está vacío/roto, lo limpiamos
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        base_query = f"SELECT * FROM {schema}.{table_name}"
        if limit is not None:
            base_query += f" LIMIT {int(limit)}"

        t0 = time.time()
        print(f"🦆 Exportando DuckDB -> Parquet: {schema}.{table_name} ...")

        try:
            with duckdb.connect(database=str(DB_PATH), read_only=True) as con:
                out_file = out_dir / "part-00000.parquet"
                con.execute(
                    f"COPY ({base_query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');"
                )
            print(f"✅ Export Parquet listo en {round(time.time() - t0, 2)}s -> {out_dir}")

        except Exception:
            # ✅ Si falla el export, BORRAR staging para no dejar carpeta vacía
            shutil.rmtree(out_dir, ignore_errors=True)
            raise

    print(f"⚡ Leyendo Parquet con Spark: {out_dir}")
    return spark.read.parquet(str(out_dir))