import duckdb

DATABASE_PATH = "database/analytics.duckdb"

TABLES = [
    "clientes",
    "canales",
    "moras",
    "gestiones",
    "excedentes",
    "tanque_movimiento",
]

KEY_COLUMNS = ["num_doc", "obl17", "f_analisis"]


def connect():
    return duckdb.connect(DATABASE_PATH)


# -------------------------
# CHECK 1 - EXISTENCIA TABLAS
# -------------------------
def check_tables(conn):
    print("\n🔎 CHECK 1 — Tablas RAW")

    for table in TABLES:
        result = conn.execute(
            f"SELECT COUNT(*) FROM raw.{table}"
        ).fetchone()[0]

        print(f"✅ raw.{table}: {result} registros")


# -------------------------
# CHECK 2 - NULL KEYS
# -------------------------
def check_null_keys(conn):
    print("\n🔎 CHECK 2 — Null Keys")

    for table in TABLES:
        for col in KEY_COLUMNS:
            try:
                nulls = conn.execute(f"""
                    SELECT COUNT(*)
                    FROM raw.{table}
                    WHERE {col} IS NULL
                """).fetchone()[0]

                print(f"{table}.{col} NULLs: {nulls}")
            except Exception:
                # columna puede no existir en alguna tabla
                pass


# -------------------------
# CHECK 3 - DUPLICADOS
# -------------------------
def check_duplicates(conn):
    print("\n🔎 CHECK 3 — Duplicados unidad analítica")

    duplicates = conn.execute("""
        SELECT num_doc, obl17, f_analisis, COUNT(*) as n
        FROM raw.clientes
        GROUP BY 1,2,3
        HAVING COUNT(*) > 1
    """).fetchall()

    if duplicates:
        print(f"⚠️ Duplicados encontrados: {len(duplicates)}")
    else:
        print("✅ Sin duplicados en clientes")


# -------------------------
# CHECK 4 - INTEGRIDAD REFERENCIAL
# -------------------------
def check_referential_integrity(conn):
    print("\n🔎 CHECK 4 — Integridad referencial")

    orphan_records = conn.execute("""
        SELECT COUNT(*)
        FROM raw.moras m
        LEFT JOIN raw.clientes c
        ON m.obl17 = c.obl17
        WHERE c.obl17 IS NULL
    """).fetchone()[0]

    print(f"Registros huérfanos en moras: {orphan_records}")


# -------------------------
# MAIN
# -------------------------
def main():

    print("\n🚀 Ejecutando Data Quality Checks")

    conn = connect()

    check_tables(conn)
    check_null_keys(conn)
    check_duplicates(conn)
    check_referential_integrity(conn)

    conn.close()

    print("\n✅ Data Quality Checks finalizados")


if __name__ == "__main__":
    main()