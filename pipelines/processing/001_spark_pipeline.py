"""
pipelines/processing/001_spark_pipeline.py

Pipeline principal (Spark) optimizado para alta dimensionalidad.
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
from pipelines.processing.extract import read_table
from pipelines.processing.clean import clean_dataset
from pipelines.processing.joins import build_base_dataframe
from pipelines.processing.features.features import create_features
from pipelines.processing.build_gold import save_to_duckdb
from pipelines.processing.features.canales_features import build_canales_compacto, CanalesCompactConfig


def main():
    # Iniciar sesión con configuración optimizada (8GB+ RAM si es posible)
    spark = get_spark_session()
    print("🚀 Iniciando Pipeline de Cobranza...")

    # ------------------------------------------------------------
    # 1) SILVER.CANALES (Procesamiento Aislado para evitar OOM)
    # ------------------------------------------------------------
    print("📦 Procesando Canales (Fase Crítica: +4k columnas)...")
    
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
    save_to_duckdb(silver_canales_df, "canales", "silver")
    
    # --- LIMPIEZA DE MEMORIA ---
    # Eliminamos referencias y limpiamos caché de Spark
    del raw_canales
    del silver_canales_df
    spark.catalog.clearCache()
    gc.collect()
    print("✅ Silver Canales guardado. Memoria RAM liberada.")

    # ------------------------------------------------------------
    # 2) CARGA DE DATOS LIMPIOS (Usando la versión compacta)
    # ------------------------------------------------------------
    print("🔍 Cargando resto de fuentes y Canales pre-procesado...")
    
    # Ahora cargamos canales desde SILVER (ya es liviano)
    canales = read_table(spark, "canales", schema="silver")
    
    clientes = clean_dataset(read_table(spark, "clientes"))
    moras = clean_dataset(read_table(spark, "moras"))
    tanque_movimiento = clean_dataset(read_table(spark, "tanque_movimiento"))
    gestiones = clean_dataset(read_table(spark, "gestiones"))
    excedente = clean_dataset(read_table(spark, "excedente"))

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
    print("🏆 Generando Dataset Gold...")
    gold = create_features(base)

    # ------------------------------------------------------------
    # 5) GUARDADO FINAL
    # ------------------------------------------------------------
    print("💾 Guardando resultados en DuckDB...")
    save_to_duckdb(gold, "dataset_final_modelo", "gold")
    
    print("✨ Pipeline finalizado con éxito.")
    spark.stop()


if __name__ == "__main__":
    main()