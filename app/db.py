from pathlib import Path
import os
import duckdb
import pandas as pd

def get_db_path() -> str:
    """
    Retorna la ruta al DuckDB.
    - Si existe variable de entorno DB_PATH, la usa.
    - Si no, asume estructura del repo: op_cobro/database/analytics.duckdb
    """
    env = os.getenv("DB_PATH")
    if env:
        return str(Path(env).resolve())

    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # ...\op_cobro
    db_path = (repo_root / "database" / "analytics.duckdb").resolve()
    return str(db_path)

def query_one(sql: str, params=None) -> dict | None:
    con = duckdb.connect(get_db_path(), read_only=True)
    try:
        df = con.execute(sql, params).df() if params else con.execute(sql).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    finally:
        con.close()

def query_df(sql: str, params=None) -> pd.DataFrame:
    con = duckdb.connect(get_db_path(), read_only=True)
    try:
        return con.execute(sql, params).df() if params else con.execute(sql).df()
    finally:
        con.close()

def exec_sql(sql: str) -> None:
    """
    Ejecuta SQL con permisos de escritura (para orquestación).
    """
    con = duckdb.connect(get_db_path(), read_only=False)
    try:
        con.execute(sql)
    finally:
        con.close()