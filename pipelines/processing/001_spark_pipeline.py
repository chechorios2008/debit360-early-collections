"""
pipelines/processing/001_spark_pipeline.py

Pipeline principal (Spark).

Nota:
Para que este script funcione aunque lo ejecutes desde cualquier carpeta,
agregamos la raíz del proyecto al sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    spark = get_spark_session()

    # ------------------------------------------------------------
    # 1) SILVER.CANALES (compacto)
    # ------------------------------------------------------------
    raw_canales = read_table(spark, "canales")  # debe traer raw.canales desde DuckDB
    cfg = CanalesCompactConfig(
        windows=("ult6", "ult12"),
        stats_prefixes=("sum_trx_", "avg_trx_"),
        drop_smmlv=True,
    )
    silver_canales = build_canales_compacto(raw_canales, cfg)

    # Guardar tabla confiable/modelable: silver.canales
    save_to_duckdb(silver_canales, "canales", "silver")

    # ------------------------------------------------------------
    # 2) EXTRACT + CLEAN (resto de tablas)
    # ------------------------------------------------------------
    clientes = clean_dataset(read_table(spark, "clientes"))
    moras = clean_dataset(read_table(spark, "moras"))
    tanque_movimiento = clean_dataset(read_table(spark, "tanque_movimiento"))
    gestiones = clean_dataset(read_table(spark, "gestiones"))
    excedente = clean_dataset(read_table(spark, "excedente"))

    # Para joins, usamos el DF silver en memoria (más eficiente)
    canales = silver_canales

    # ------------------------------------------------------------
    # 3) JOINS
    # ------------------------------------------------------------
    base = build_base_dataframe(clientes, moras, tanque_movimiento, canales, gestiones, excedente)

    # ------------------------------------------------------------
    # 4) FEATURES (GOLD)
    # ------------------------------------------------------------
    gold = create_features(base)

    # ------------------------------------------------------------
    # 5) SAVE
    save_to_duckdb(canales, "canales", "silver") # Guardamos canales procesado
    save_to_duckdb(gold, "dataset_final_modelo", "gold") # El dataset listo para ML

    spark.stop()


if __name__ == "__main__":
    main()