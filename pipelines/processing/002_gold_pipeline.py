"""
pipelines/processing/002_gold_pipeline.py
Orquestador para consolidar el Dataset GOLD.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.processing.spark_session import get_spark_session
from pipelines.processing.extract import read_table
from pipelines.processing.joins import build_base_dataframe
from pipelines.processing.build_gold import save_to_duckdb

def main():
    spark = get_spark_session()
    print("📀 Iniciando consolidación de capa GOLD...")

    # 1. LEER DESDE SILVER (Aquí está el ahorro de tiempo)
    # Ya no leemos de RAW, usamos lo que ya procesaste
    canales = read_table(spark, "canales", schema="silver")
    clientes = read_table(spark, "clientes", schema="silver")
    moras = read_table(spark, "moras", schema="silver")
    gestiones = read_table(spark, "gestiones", schema="silver")
    excedentes = read_table(spark, "excedentes", schema="silver")
    tanque = read_table(spark, "tanque_movimiento", schema="silver")

    # 2. EJECUTAR JOINS (Con la lógica de estandarización que te di)
    print("🔗 Uniendo fuentes y limpiando llaves...")
    base_gold = build_base_dataframe(
        clientes, moras, tanque, canales, gestiones, excedentes
    )

    # 3. GUARDAR EL DATASET MAESTRO
    # Este es el archivo que usaremos para el modelo
    print("💾 Guardando Dataset Final en gold.dataset_modelo...")
    save_to_duckdb(base_gold, "dataset_modelo", "gold")

    print("✅ Proceso GOLD finalizado con éxito.")
    spark.stop()

if __name__ == "__main__":
    main()