# op_cobro/pipelines/prep/prepare_gold.py
import argparse
import sys
from pathlib import Path
import duckdb


DEFAULT_PROJECT_ROOT = Path(r"C:\Users\serrios\01_prueba_analitico_4")

def build_view_sql_from_parquet(parquet_glob:str, mode:str, id_column:str=None,
                                train_before:str=None, test_start:str=None, test_end:str=None) -> str:
    """
    Genera el SQL para crear la vista gold.dataset_final desde Parquet.
    - mode = 'hash'  -> split determinístico por hash del id_column
    - mode = 'temporal' -> split por rangos de fecha usando f_analisis
    """
    if mode == "hash":
        if not id_column:
            raise ValueError("Para split 'hash' debes indicar --id-column (ej: num_doc).")
        return f"""
        CREATE OR REPLACE VIEW gold.dataset_final AS
        WITH base AS (
            SELECT
                CAST(var_rta AS INTEGER) AS target_final,
                *
            FROM read_parquet('{parquet_glob}')
        )
        SELECT
            target_final,
            CASE
                WHEN MOD(ABS(HASH(COALESCE(CAST({id_column} AS VARCHAR), 'NA'))), 10) < 7 THEN 'TRAIN' -- 70%
                WHEN MOD(ABS(HASH(COALESCE(CAST({id_column} AS VARCHAR), 'NA'))), 10) < 9 THEN 'TEST'  -- 20%
                ELSE 'OOT'                                                                           -- 10%
            END AS split_group,
            base.* EXCLUDE (var_rta)
        FROM base;
        """
    elif mode == "temporal":
        for p in (train_before, test_start, test_end):
            if not p:
                raise ValueError("Para split 'temporal' debes indicar --train-before, --test-start y --test-end (YYYY-MM-DD).")
        return f"""
        CREATE OR REPLACE VIEW gold.dataset_final AS
        WITH base AS (
            SELECT
                CAST(var_rta AS INTEGER) AS target_final,
                CAST(f_analisis AS DATE) AS f_analisis_date,
                * EXCLUDE (var_rta)
            FROM read_parquet('{parquet_glob}')
        )
        SELECT
            target_final,
            CASE
                WHEN f_analisis_date <  DATE '{train_before}' THEN 'TRAIN'
                WHEN f_analisis_date >= DATE '{test_start}'   AND f_analisis_date < DATE '{test_end}' THEN 'TEST'
                ELSE 'OOT'
            END AS split_group,
            base.* EXCLUDE (f_analisis)
        FROM base;
        """
    else:
        raise ValueError("split-mode desconocido. Usa 'hash' o 'temporal'.")

def main():
    parser = argparse.ArgumentParser(description="Prepara gold.dataset_final en DuckDB a partir de Parquet (GOLD).")
    parser.add_argument("--project-root", type=str, default=str(DEFAULT_PROJECT_ROOT),
                        help="Ruta del proyecto base. Default: C:\\Users\\serrios\\01_prueba_analitico_4")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Ruta completa a analytics.duckdb. Default: <project_root>\\op_cobro\\database\\analytics.duckdb")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="Carpeta GOLD con part-*.parquet. Default: <project_root>\\op_cobro\\database\\staging\\gold.dataset_modelo.parquet_write")
    parser.add_argument("--split-mode", type=str, default="hash", choices=["hash", "temporal"],
                        help="Modo de split: 'hash' (determinístico) o 'temporal' (por fechas). Default: hash")
    parser.add_argument("--id-column", type=str, default="num_doc",
                        help="Columna ID para split hash. Default: num_doc")
    parser.add_argument("--train-before", type=str, default=None,
                        help="Fecha corte TRAIN (YYYY-MM-DD) si split temporal.")
    parser.add_argument("--test-start", type=str, default=None,
                        help="Fecha inicio TEST (YYYY-MM-DD) si split temporal.")
    parser.add_argument("--test-end", type=str, default=None,
                        help="Fecha fin   TEST (YYYY-MM-DD) si split temporal.")
    parser.add_argument("--materialize", action="store_true",
                        help="Si se indica, materializa gold.dataset_modelo como TABLA antes de crear la vista final (opcional).")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    # *** MUY IMPORTANTE ***
    # Para que coincida con train_model.py, usamos la base bajo op_cobro\database\
    db_path = Path(args.db_path) if args.db_path else (project_root / "op_cobro" / "database" / "analytics.duckdb")
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else (project_root / "op_cobro" / "database" / "staging" / "gold.dataset_modelo.parquet_write")

    print(f"📁 project_root: {project_root}")
    print(f"🗄️  db_path     : {db_path}")
    print(f"📦 parquet_dir : {parquet_dir}")

    # Validaciones básicas
    if not parquet_dir.exists():
        print(f"ERROR: No existe la carpeta de Parquet: {parquet_dir}", file=sys.stderr)
        sys.exit(2)
    parts = list(parquet_dir.glob("*.parquet"))
    if not parts:
        print(f"ERROR: No se encontraron .parquet en {parquet_dir}", file=sys.stderr)
        sys.exit(2)

    parquet_glob = (parquet_dir / "part-*.parquet").as_posix()

    # Conectar y crear schema
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    print("✅ Schema 'gold' OK.")

    # (Opcional) materializar gold.dataset_modelo como TABLA (performance estable)
    if args.materialize:
        print("🧱 Materializando gold.dataset_modelo como TABLA a partir de Parquet...")
        con.execute(f"""
            CREATE OR REPLACE TABLE gold.dataset_modelo AS
            SELECT * FROM read_parquet('{parquet_glob}');
        """)
        print("✅ Tabla gold.dataset_modelo creada/reemplazada.")

    # Crear vista gold.dataset_final con columnas requeridas
    print(f"🧩 Construyendo gold.dataset_final con split '{args.split_mode}' ...")
    view_sql = build_view_sql_from_parquet(
        parquet_glob=parquet_glob,
        mode=args.split_mode,
        id_column=args.id_column,
        train_before=args.train_before,
        test_start=args.test_start,
        test_end=args.test_end
    )
    con.execute(view_sql)
    print("✅ Vista gold.dataset_final creada/reemplazada.")

    # Validaciones
    cols = set(con.execute("DESCRIBE gold.dataset_final").fetchdf()["column_name"].str.lower())
    required = {"split_group", "target_final"}
    missing = required - cols
    if missing:
        print(f"ERROR: Faltan columnas requeridas en gold.dataset_final: {missing}", file=sys.stderr)
        sys.exit(2)

    # Conteos rápidos
    total = con.execute("SELECT COUNT(*) AS n FROM gold.dataset_final").fetchdf().iloc[0,0]
    print(f"📊 Registros en gold.dataset_final: {total:,}")

    print("🔎 Conteo por split_group:")
    print(con.execute("""
        SELECT split_group, COUNT(*) AS n
        FROM gold.dataset_final
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    print("🔎 Distribución de target_final:")
    print(con.execute("""
        SELECT target_final, COUNT(*) AS n
        FROM gold.dataset_final
        GROUP BY 1 ORDER BY 1
    """).fetchdf())

    con.close()
    print("🏁 Preparación GOLD finalizada con éxito.")

if __name__ == "__main__":
    main()