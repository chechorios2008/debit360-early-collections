"""
pipelines/processing/001_silver_v2_pipeline.py

Construye SILVER_V2 desde RAW:
- Normaliza llaves creando columnas canónicas: num_doc_key, obl17_key, f_analisis (date)
- Mantiene intacta la capa silver original (NO se borra nada)
- Optimizado para canales (compactación) + resto de tablas (clean_dataset)
"""

from __future__ import annotations

import sys
import gc
from pathlib import Path

from pyspark.sql import functions as F  # noqa: F401

# raíz del repo: .../op_cobro
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.processing.spark_session import get_spark_session  # noqa: E402
from pipelines.processing.extract import read_table  # noqa: E402
from pipelines.processing.clean import clean_dataset  # noqa: E402
from pipelines.processing.build_gold import save_to_duckdb  # noqa: E402
from pipelines.processing.features.canales_features import (  # noqa: E402
    build_canales_compacto,
    CanalesCompactConfig,
)
from pipelines.processing.key_normalization import add_key_columns  # noqa: E402


# -----------------------------
# Configuración
# -----------------------------
TARGET_SCHEMA = "silver_v2"

# Para primera corrida: "overwrite" (reconstruye todo)
# Para corridas siguientes: "skip" (si la tabla ya existe, no la recalcula)
MODE_SILVER_V2 = "overwrite"  # "overwrite" | "skip"


def _free_spark_memory(spark, *dfs):
    """Libera referencias y memoria Spark."""
    for d in dfs:
        try:
            del d
        except Exception:
            pass
    spark.catalog.clearCache()
    gc.collect()


def build_canales_silver_v2(spark):
    print("📦 [SILVER_V2] Procesando CANALES (compactación + llaves canónicas)...")

    raw_canales = read_table(spark, "canales", schema="raw")

    cfg = CanalesCompactConfig(
        windows=("ult6", "ult12"),
        stats_prefixes=("sum_trx_", "avg_trx_"),
        drop_smmlv=True,
    )

    # Compactación de +4k columnas -> ~60-70
    canales_compacto = build_canales_compacto(raw_canales, cfg)

    # Agregamos llaves canónicas (num_doc_key, obl17_key, f_analisis date)
    canales_compacto = add_key_columns(canales_compacto)

    # Guardamos en DuckDB como silver_v2.canales
    save_to_duckdb(canales_compacto, "canales", TARGET_SCHEMA, mode=MODE_SILVER_V2)

    _free_spark_memory(spark, raw_canales, canales_compacto)
    print("✅ [SILVER_V2] canales listo.")


def build_generic_silver_v2(spark, table_name: str):
    print(f"📦 [SILVER_V2] Procesando {table_name} (clean_dataset + llaves canónicas)...")

    raw_df = read_table(spark, table_name, schema="raw")
    silver_df = clean_dataset(raw_df)

    # Agregamos llaves canónicas
    silver_df = add_key_columns(silver_df)

    # Guardamos en DuckDB como silver_v2.<table_name>
    save_to_duckdb(silver_df, table_name, TARGET_SCHEMA, mode=MODE_SILVER_V2)

    _free_spark_memory(spark, raw_df, silver_df)
    print(f"✅ [SILVER_V2] {table_name} listo.")


def main():
    spark = get_spark_session()
    print("🚀 Iniciando construcción de SILVER_V2 (sin tocar SILVER actual)...")

    # 1) Canales (crítico por ancho)
    build_canales_silver_v2(spark)

    # 2) Resto de tablas
    for t in ["clientes", "moras", "gestiones", "excedentes", "tanque_movimiento"]:
        build_generic_silver_v2(spark, t)

    print("✨ SILVER_V2 construido con éxito.")
    spark.stop()


if __name__ == "__main__":
    main()