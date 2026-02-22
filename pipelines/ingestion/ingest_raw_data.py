import duckdb
from pathlib import Path

# -----------------------------
# BASE DIR (raíz del repo)
# -----------------------------
# ingest_raw_data.py está en: <repo>/pipelines/ingestion/
# parents[2] = <repo>
BASE_DIR = Path(__file__).resolve().parents[2]

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DATABASE_PATH = BASE_DIR / "database" / "analytics.duckdb"
RAW_DATA_PATH = BASE_DIR / "data" / "raw_files"

FILES_MAPPING = {
    "canales.csv": "canales",
    "clientes.csv": "clientes",
    "excedente.csv": "excedentes",
    "gestiones.csv": "gestiones",
    "moras.csv": "moras",
    "tanque_movimiento.csv": "tanque_movimiento",
}

# -----------------------------
# CONEXIÓN DB
# -----------------------------
def get_connection():
    # Asegura que exista la carpeta /database
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DATABASE_PATH))

# -----------------------------
# CREAR SCHEMAS
# -----------------------------
def create_schemas(conn):
    schemas = ["raw", "silver", "gold", "ml"]
    for schema in schemas:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    print("✅ Schemas verificados/creados")

# -----------------------------
# INGESTIÓN CSV → RAW
# -----------------------------
def ingest_csv_files(conn):
    # Valida que exista la ruta de entrada
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe RAW_DATA_PATH: {RAW_DATA_PATH}")

    for file_name, table_name in FILES_MAPPING.items():
        file_path = RAW_DATA_PATH / file_name

        if not file_path.exists():
            print(f"⚠️ Archivo no encontrado: {file_path}")
            continue

        print(f"\n📥 Cargando {file_name} → raw.{table_name}")

        # Recomendación: pasar path como parámetro (evita problemas de escape en Windows)
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE raw.{table_name} AS
            SELECT *
            FROM read_csv_auto(?, HEADER=TRUE);
            """,
            [str(file_path)]
        )

        count = conn.execute(f"SELECT COUNT(*) FROM raw.{table_name}").fetchone()[0]
        print(f"✅ raw.{table_name} cargada con {count} registros")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("\n🚀 Iniciando ingestión RAW")
    print(f"📌 BASE_DIR: {BASE_DIR}")
    print(f"📌 DATABASE_PATH: {DATABASE_PATH}")
    print(f"📌 RAW_DATA_PATH: {RAW_DATA_PATH}")

    conn = get_connection()
    create_schemas(conn)
    ingest_csv_files(conn)
    conn.close()

    print("\n🎯 Ingestión finalizada correctamente")

if __name__ == "__main__":
    main()