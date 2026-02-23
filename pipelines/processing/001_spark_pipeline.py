"""
pipelines/processing/001_spark_pipeline.py

Pipeline principal (Spark) optimizado para alta dimensionalidad.
RAW -> SILVER (limpieza/estandarización/EDA)
SILVER -> GOLD (dataset final model-ready)
"""

from __future__ import annotations

import sys
from pathlib import Path
import gc  # Garbage Collector para liberar RAM

# raíz del repo: .../op_cobro
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.processing.spark_session import get_spark_session  # noqa: E402
from pipelines.processing.extract import read_table  # noqa: E402
from pipelines.processing.clean import clean_dataset  # noqa: E402
from pipelines.processing.joins import build_base_dataframe  # noqa: E402
from pipelines.processing.features.features import create_features  # noqa: E402
from pipelines.processing.build_gold import save_to_duckdb  # noqa: E402
from pipelines.processing.features.canales_features import (  # noqa: E402
    build_canales_compacto,
    CanalesCompactConfig,
)

# ------------------------------------------------------------
# CONFIGURACIÓN SIMPLE (cambia aquí sin tocar lógica)
# ------------------------------------------------------------
MODE_SILVER = "skip"      # "skip" = si existe, NO recalcula; "overwrite" = recalcula (si lo soportas)
MODE_GOLD = "skip"        # recomendado "skip" en dev; en prod podrías usar "overwrite"
GENERAR_GOLD = True       # pon False si SOLO quieres construir SILVER para EDA


def ensure_silver_table(spark, table_name: str, *, source_schema: str = "raw", silver_schema: str = "silver"):
    """
    Garantiza que exista silver.{table_name}.
    - Si silver.{table_name} existe: la carga y retorna.
    - Si no existe: la crea desde raw.{table_name} aplicando clean_dataset y la guarda en DuckDB.
    """
    try:
        df = read_table(spark, table_name, schema=silver_schema)
        print(f"⏭️  Reutilizando {silver_schema}.{table_name} (ya existe).")
        return df
    except Exception:
        print(f"🛠️  Creando {silver_schema}.{table_name} desde {source_schema}.{table_name} ...")
        df_raw = read_table(spark, table_name, schema=source_schema)
        df_silver = clean_dataset(df_raw)

        # Guardar SILVER evitando recrear si ya existe (según mode)
        # IMPORTANTE: tu save_to_duckdb debe soportar mode="skip"
        save_to_duckdb(df_silver, table_name, silver_schema, mode=MODE_SILVER)

        # liberar memoria
        del df_raw
        del df_silver
        spark.catalog.clearCache()
        gc.collect()

        # volver a leer desde silver (parquet staging / duckdb) como fuente oficial
        df = read_table(spark, table_name, schema=silver_schema)
        return df


def main():
    # Iniciar sesión con configuración optimizada
    spark = get_spark_session()
    print("🚀 Iniciando Pipeline de Cobranza...")

    # ------------------------------------------------------------
    # 1) SILVER.CANALES (Procesamiento Aislado para evitar OOM)
    # ------------------------------------------------------------
    print("📦 Procesando Canales (Fase Crítica: +4k columnas)...")

    # Si ya existe silver.canales y estamos en modo skip, reutilizamos
    if MODE_SILVER == "skip":
        try:
            _ = read_table(spark, "canales", schema="silver")
            print("⏭️  Reutilizando silver.canales (ya existe).")
        except Exception:
            # Leemos de RAW
            raw_canales = read_table(spark, "canales", schema="raw")

            cfg = CanalesCompactConfig(
                windows=("ult6", "ult12"),
                stats_prefixes=("sum_trx_", "avg_trx_"),
                drop_smmlv=True,
            )

            # Transformación a formato compacto (~60 columnas)
            silver_canales_df = build_canales_compacto(raw_canales, cfg)

            # Persistimos inmediatamente en DuckDB para liberar la RAM de las 4k columnas
            save_to_duckdb(silver_canales_df, "canales", "silver", mode=MODE_SILVER)

            # --- LIMPIEZA DE MEMORIA ---
            del raw_canales
            del silver_canales_df
            spark.catalog.clearCache()
            gc.collect()
            print("✅ Silver Canales guardado. Memoria RAM liberada.")
    else:
        # overwrite / recálculo explícito
        raw_canales = read_table(spark, "canales", schema="raw")

        cfg = CanalesCompactConfig(
            windows=("ult6", "ult12"),
            stats_prefixes=("sum_trx_", "avg_trx_"),
            drop_smmlv=True,
        )

        silver_canales_df = build_canales_compacto(raw_canales, cfg)
        save_to_duckdb(silver_canales_df, "canales", "silver", mode=MODE_SILVER)

        del raw_canales
        del silver_canales_df
        spark.catalog.clearCache()
        gc.collect()
        print("✅ Silver Canales guardado (recalculado). Memoria RAM liberada.")

    # ------------------------------------------------------------
    # 2) CARGA DE DATOS LIMPIOS (SILVER) + creación si no existen
    # ------------------------------------------------------------
    print("🔍 Cargando resto de fuentes desde SILVER (y creando si falta)...")

    canales = read_table(spark, "canales", schema="silver")  # ya está compacto
    clientes = ensure_silver_table(spark, "clientes")
    moras = ensure_silver_table(spark, "moras")
    tanque_movimiento = ensure_silver_table(spark, "tanque_movimiento")
    gestiones = ensure_silver_table(spark, "gestiones")
    excedente = ensure_silver_table(spark, "excedentes")

    # ------------------------------------------------------------
    # 3) INTEGRACIÓN (JOINS)
    # ------------------------------------------------------------
    print("🔗 Realizando Joins analíticos...")
    base = build_base_dataframe(
        clientes,
        moras,
        tanque_movimiento,
        canales,
        gestiones,
        excedente
    )

    # ------------------------------------------------------------
    # 4) FEATURES FINALES (GOLD)
    # ------------------------------------------------------------
    if GENERAR_GOLD:
        #print("🏆 Generando Dataset Gold...")
        #gold = create_features(base)

        # ------------------------------------------------------------
        # 5) GUARDADO FINAL (GOLD)
        # ------------------------------------------------------------
        #print("💾 Guardando resultados en DuckDB...")
        #save_to_duckdb(gold, "dataset_final_modelo", "gold", mode=MODE_GOLD)
        print("✨ Pipeline finalizado con éxito (SILVER + GOLD).")
    else:
        print("✅ Pipeline finalizado con éxito (solo SILVER).")

    spark.stop()


if __name__ == "__main__":
    main()