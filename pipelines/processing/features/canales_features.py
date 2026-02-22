"""
pipelines/processing/features/canales_features.py

Silver (compacto) para la fuente `canales`.

Objetivo
--------
Transformar `raw.canales` (+4k columnas) en un set compacto (~20–60) de señales
agregadas por familia de canal, con foco en estabilidad y valor predictivo.


Importante
----------
`raw.canales` típicamente ya viene a nivel de llave (num_doc, obl17, f_analisis).
Por eso aquí NO hacemos groupBy agregando filas; hacemos agregaciones row-wise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class CanalesCompactConfig:
    # Llaves
    keys: Sequence[str] = ("num_doc", "obl17", "f_analisis")

    # Ventanas y stats a considerar
    windows: Sequence[str] = ("ult6", "ult12")
    stats_prefixes: Sequence[str] = ("sum_trx_", "avg_trx_")

    # Métricas (tokens) dentro del naming
    metrics: Sequence[str] = ("mnt", "cnt")

    # Excluir columnas con token smmlv
    drop_smmlv: bool = True

    # Familias basadas en canales reales (ajusta si quieres)
    channel_groups: Dict[str, List[str]] = None  # type: ignore

    def __post_init__(self):
        if self.channel_groups is None:
            object.__setattr__(
                self,
                "channel_groups",
                {
                    # Canales digitales / remotos
                    "digital": [
                        "app_per", "app_pyme", "svp", "sve", "sv_pyme",
                        "pse_per", "pse_emp", "btn_bco", "bill_mvl",
                    ],
                    # Canales físicos (sucursal / ventanilla / grupo físico)
                    "fisico": ["suc_fis", "gr"],
                    # Efectivo / ATM
                    "efectivo": ["cajero"],
                    # Tarjeta / POS
                    "tarjeta": ["pos"],
                    # Otros (especiales)
                    "otros": ["cr_bcrio", "rec_e_v", "suc_tel"],
                },
            )


def _normalize_columns(df: DataFrame) -> DataFrame:
    """Normaliza nombres (lower, reemplaza espacios/puntos) para consistencia."""
    for c in df.columns:
        clean = c.strip().lower().replace(" ", "_").replace(".", "_")
        if clean != c:
            df = df.withColumnRenamed(c, clean)
    return df


def _is_valid_col(col: str, cfg: CanalesCompactConfig) -> bool:
    if col in cfg.keys:
        return False
    if cfg.drop_smmlv and "smmlv" in col.split("_"):
        return False
    return True


def _matches(col: str, cfg: CanalesCompactConfig, *, group_patterns: List[str], metric: str, window: str) -> bool:
    # Debe iniciar con prefijo de stat
    if not any(col.startswith(p) for p in cfg.stats_prefixes):
        return False
    # Debe terminar en ventana
    if not col.endswith(window):
        return False
    # Debe contener métrica como token
    if f"_{metric}_" not in col:
        return False
    # Debe contener patrón de canal
    if not any(pat in col for pat in group_patterns):
        return False
    return True


def _row_sum(cols: List[str]) -> F.Column:
    """Suma row-wise de múltiples columnas (NULL->0)."""
    return reduce(lambda a, b: a + b, [F.coalesce(F.col(c), F.lit(0)) for c in cols])


def _row_max(cols: List[str]) -> F.Column:
    """Máximo row-wise de múltiples columnas (NULL->0)."""
    return F.greatest(*[F.coalesce(F.col(c), F.lit(0)) for c in cols])


def build_canales_compacto(df_raw: DataFrame, cfg: CanalesCompactConfig | None = None) -> DataFrame:
    """Construye dataset compacto de canales para Silver."""
    cfg = cfg or CanalesCompactConfig()
    df = _normalize_columns(df_raw)

    # Validar llaves
    missing_keys = [k for k in cfg.keys if k not in df.columns]
    if missing_keys:
        raise ValueError(f"Faltan llaves en canales: {missing_keys}")

    valid_cols = [c for c in df.columns if _is_valid_col(c, cfg)]

    feature_exprs = []

    # Totales y máximos por familia/ventana/métrica
    for group_name, patterns in cfg.channel_groups.items():
        for window in cfg.windows:
            for metric in cfg.metrics:
                cols = [c for c in valid_cols if _matches(c, cfg, group_patterns=patterns, metric=metric, window=window)]
                if not cols:
                    continue
                feature_exprs.append(_row_sum(cols).alias(f"total_{metric}_{group_name}_{window}"))
                feature_exprs.append(_row_max(cols).alias(f"max_{metric}_{group_name}_{window}"))

    # Totales globales por ventana (si existen)
    for window in cfg.windows:
        for metric in cfg.metrics:
            total_name = f"total_{metric}_all_{window}"
            chosen = None

            for p in cfg.stats_prefixes:
                colname = f"{p}{metric}_total_{window}"
                if colname in df.columns and _is_valid_col(colname, cfg):
                    chosen = F.coalesce(F.col(colname), F.lit(0)).alias(total_name)
                    break

            if chosen is not None:
                feature_exprs.append(chosen)
            else:
                # fallback (si por alguna razón no existe total): suma de familias disponibles
                family_cols = [f"total_{metric}_{g}_{window}" for g in cfg.channel_groups.keys()]
                family_cols = [c for c in family_cols if c in [e._jc.toString().split(" AS ")[-1] for e in feature_exprs]]  # best-effort
                if family_cols:
                    feature_exprs.append(reduce(lambda a, b: a + b, [F.col(c) for c in family_cols]).alias(total_name))
                else:
                    feature_exprs.append(F.lit(0).alias(total_name))

    # Keys + features
    df_out = df.select(
        *[
            F.col(k).cast("string").alias(k) if k in ("num_doc", "obl17") else F.col(k)
            for k in cfg.keys
        ],
        *feature_exprs
    )

    # Ratios/shares por ventana
    for window in cfg.windows:
        total_all = F.col(f"total_mnt_all_{window}") + F.lit(1)

        # Share por familia (monto)
        for group_name in cfg.channel_groups.keys():
            col_total = f"total_mnt_{group_name}_{window}"
            if col_total in df_out.columns:
                df_out = df_out.withColumn(f"share_mnt_{group_name}_{window}", F.col(col_total) / total_all)

        # Digitalización (digital / (digital + físico))
        if f"total_mnt_digital_{window}" in df_out.columns and f"total_mnt_fisico_{window}" in df_out.columns:
            df_out = df_out.withColumn(
                f"index_digitalizacion_{window}",
                F.col(f"total_mnt_digital_{window}") /
                (F.col(f"total_mnt_digital_{window}") + F.col(f"total_mnt_fisico_{window}") + F.lit(1))
            )

        # Preferencia efectivo (efectivo / total)
        if f"total_mnt_efectivo_{window}" in df_out.columns:
            df_out = df_out.withColumn(
                f"preferencia_efectivo_{window}",
                F.col(f"total_mnt_efectivo_{window}") / total_all
            )

    # Tendencia ult6 vs ult12
    if "ult6" in cfg.windows and "ult12" in cfg.windows:
        for group_name in cfg.channel_groups.keys():
            c6 = f"total_mnt_{group_name}_ult6"
            c12 = f"total_mnt_{group_name}_ult12"
            if c6 in df_out.columns and c12 in df_out.columns:
                df_out = df_out.withColumn(f"trend_mnt_{group_name}_ult6_vs_ult12", F.col(c6) / (F.col(c12) + F.lit(1)))

    # NULL -> 0 en features
    feature_cols = [c for c in df_out.columns if c not in cfg.keys]
    df_out = df_out.fillna(0, subset=feature_cols)

    return df_out