"""
pipelines/processing/002_gold_pipeline.py
Orquestador para consolidar el Dataset GOLD desde SILVER_V2.
"""

import sys
from pathlib import Path
from pyspark.sql import functions as F

# raíz del repo: .../op_cobro
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.processing.spark_session import get_spark_session
from pipelines.processing.extract import read_table
from pipelines.processing.joins import build_base_dataframe
from pipelines.processing.build_gold import save_to_duckdb


SILVER_SCHEMA = "silver_v2"
GOLD_SCHEMA = "gold"
GOLD_TABLE = "dataset_modelo"

JOIN_KEYS = ["num_doc_key", "obl17_key", "f_analisis"]


def show_schema_and_keys(df, name, n=5):
    print(f"\n🧾 Schema {name}:")
    print(df.dtypes)
    print(f"🔑 Sample keys {name} (top {n}):")
    df.select(*JOIN_KEYS).limit(n).show(truncate=False)


def match_rate(left_df, right_df, keys, name):
    l = left_df.select(*keys).dropDuplicates()
    r = right_df.select(*keys).dropDuplicates()
    left_n = l.count()
    matched_n = l.join(r, on=keys, how="inner").count()
    pct = (matched_n / left_n) if left_n else 0
    print(f"🔎 Match {name}: matched_keys={matched_n} / left_keys={left_n} -> {pct:.2%}")


def main():
    spark = get_spark_session()
    print("📀 Iniciando consolidación de capa GOLD desde SILVER_V2...")

    # 1) LEER DESDE SILVER_V2
    canales = read_table(spark, "canales", schema=SILVER_SCHEMA)
    clientes = read_table(spark, "clientes", schema=SILVER_SCHEMA)
    moras = read_table(spark, "moras", schema=SILVER_SCHEMA)
    gestiones = read_table(spark, "gestiones", schema=SILVER_SCHEMA)
    excedentes = read_table(spark, "excedentes", schema=SILVER_SCHEMA)
    tanque = read_table(spark, "tanque_movimiento", schema=SILVER_SCHEMA)

    # 2) Validación rápida (ligera)
    show_schema_and_keys(clientes, f"{SILVER_SCHEMA}.clientes")
    show_schema_and_keys(canales, f"{SILVER_SCHEMA}.canales")

    print("\n📌 Validando match vs clientes (llaves canónicas)...")
    match_rate(clientes, canales, JOIN_KEYS, "clientes vs canales")
    match_rate(clientes, moras, JOIN_KEYS, "clientes vs moras")
    match_rate(clientes, gestiones, JOIN_KEYS, "clientes vs gestiones")
    match_rate(clientes, excedentes, JOIN_KEYS, "clientes vs excedentes")
    match_rate(clientes, tanque, JOIN_KEYS, "clientes vs tanque_movimiento")

    # 3) Construcción ABT GOLD (join)
    print("\n🔗 Construyendo ABT GOLD (join por *_key)...")
    base_gold = build_base_dataframe(clientes, moras, tanque, canales, gestiones, excedentes)

    # 4) Sanity sums (para confirmar que no todo quedó en 0)
    candidate_cols = [c for c in ["moras_avg_mora_3m", "avg_pago_debito_3m", "total_mnt_all_ult6"] if c in base_gold.columns]
    if candidate_cols:
        exprs = [F.sum(F.col(c)).alias(f"sum_{c}") for c in candidate_cols]
        print("\n🧪 Sanity sums:")
        base_gold.select(*exprs).show(truncate=False)

    # 5) Guardar GOLD
    print(f"💾 Guardando Dataset Final en {GOLD_SCHEMA}.{GOLD_TABLE} (overwrite)...")
    save_to_duckdb(base_gold, GOLD_TABLE, GOLD_SCHEMA, mode="overwrite")

    print("✅ Proceso GOLD finalizado con éxito.")
    spark.stop()


if __name__ == "__main__":
    main()