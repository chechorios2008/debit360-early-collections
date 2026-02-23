# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ------------------- Config de Rutas (robusto) -------------------
DB_PATH_ENV = os.getenv("DB_PATH")
if DB_PATH_ENV:
    DB_PATH = Path(DB_PATH_ENV)
else:
    here = Path(__file__).resolve()
    # __file__ -> ...\op_cobro\pipelines\models\train_model.py
    repo_root = here.parents[3]            # ...\01_prueba_analitico_4
    DB_PATH = (repo_root / "op_cobro" / "database" / "analytics.duckdb").resolve()

print(f"🗄️  Usando DuckDB en: {DB_PATH}")
assert DB_PATH.exists(), f"No existe la base en {DB_PATH}. Corre prepare_gold.py primero."


# ----------------- Utilidades de saneamiento -----------------
def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte todas las columnas a numéricas (no numéricas -> NaN)."""
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sanitize_train_df(X: pd.DataFrame, clip_quantiles=(0.001, 0.999)):
    """
    Sanea TRAIN: numérico, reemplaza inf por NaN, elimina columnas all-NaN,
    clipping suave por percentiles para evitar valores extremos.
    Devuelve (X_clean, features_list, meta_dict).
    """
    X = _to_numeric_df(X)

    # Reemplazar inf
    n_inf = int(np.isinf(X.to_numpy()).sum())
    if n_inf > 0:
        X = X.replace([np.inf, -np.inf], np.nan)

    # Eliminar columnas completamente NaN (para que SimpleImputer no falle)
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # Clipping suave
    if clip_quantiles is not None:
        q_low, q_high = clip_quantiles
        for col in X.columns:
            s = X[col]
            ql = s.quantile(q_low)
            qh = s.quantile(q_high)
            if pd.notna(ql) and pd.notna(qh) and qh > ql:
                X[col] = s.clip(lower=ql, upper=qh)

    features = X.columns.tolist()
    meta = {
        "n_inf_replaced_train": n_inf,
        "dropped_all_nan_cols_train": all_nan_cols,
        "n_features_after_sanitize": len(features),
    }
    return X, features, meta


def sanitize_apply_df(X: pd.DataFrame, features_ref: list):
    """
    Sanea TEST/OOT: numérico, reemplaza inf por NaN, reindexa a las columnas de TRAIN.
    """
    X = _to_numeric_df(X)
    n_inf = int(np.isinf(X.to_numpy()).sum())
    if n_inf > 0:
        X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reindex(columns=features_ref)
    meta = {"n_inf_replaced": n_inf}
    return X, meta


# ----------------- Detección y remoción de leakage -----------------
LEAK_NAME_PATTERNS = [
    r"\btarget\b",
    r"target_", r"_target", r"targetfinal", r"target_final",
    r"\bvar_rta\b", r"varrta", r"_rta", r"rta_",
    r"\blabel\b", r"label_", r"_label",
    r"\brespuesta\b", r"respuesta_", r"_respuesta",
]
_leak_name_regex = re.compile("|".join(LEAK_NAME_PATTERNS), flags=re.IGNORECASE)

def drop_leak_by_name(columns: list) -> (list, list):
    """Quita columnas cuyo nombre sugiere fuga (regex arriba)."""
    unsafe = [c for c in columns if _leak_name_regex.search(c)]
    safe = [c for c in columns if c not in unsafe]
    return safe, unsafe


def detect_perfect_leakage(X: pd.DataFrame, y: pd.Series, max_check=5000) -> list:
    """
    Detecta columnas binarias que replican exactamente el target (o su complemento).
    Se usa una muestra si el dataset es grande.
    """
    # Muestra para acelerar si es muy grande
    if len(X) > max_check:
        idx = np.random.RandomState(42).choice(len(X), size=max_check, replace=False)
        Xs = X.iloc[idx]
        ys = y.iloc[idx]
    else:
        Xs, ys = X, y

    leaked = []
    # Sólo revisamos columnas binarias (0/1 o {0,1} ignorando NaN)
    for col in Xs.columns:
        col_values = Xs[col].dropna().unique()
        if len(col_values) <= 2 and set(col_values).issubset({0, 1}):
            # Igual al target
            try:
                eq = (Xs[col].fillna(-1).astype(int).values == ys.fillna(-1).astype(int).values).all()
                inv = ((1 - Xs[col].fillna(0).astype(int).values) == ys.fillna(-1).astype(int).values).all()
                if eq or inv:
                    leaked.append(col)
            except Exception:
                # Si algo raro pasa en el casteo, ignoramos la columna
                pass
    return leaked


def drop_high_corr_with_target(X: pd.DataFrame, y: pd.Series, thr: float = 0.999) -> list:
    """Marca columnas con correlación (Pearson) ~1 con el target."""
    # Convertimos a float para correlación (NaN se ignoran por pairwise)
    df = X.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    corr = df.corrwith(y, method="pearson")
    high = corr.index[correlation_mask := (corr.abs() >= thr)].tolist()
    return high


# ----------------- Entrenamiento -----------------
def train():
    # 1) Carga datos
    print("📂 Cargando datos desde DuckDB (Capa GOLD)...")
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold.dataset_final").df()
    con.close()

    # 2) Definición base de variables (exclusiones obvias)
    exclude_cols_exact = {
        'num_doc', 'obl17', 'f_analisis',
        'split_group', 'target_final', 'var_rta',
        'f_analisis_key', 'num_doc_key', 'obl17_key'
    }
    candidates = [c for c in df.columns if c not in exclude_cols_exact and c != 'target_final']

    # 3) Quitar por patrón de nombre (anti‑leak rápido)
    candidates_no_name_leak, dropped_by_name = drop_leak_by_name(candidates)

    target = 'target_final'
    print(f"📊 Dataset cargado: {df.shape[0]} registros y {len(candidates_no_name_leak)} features (candidatas tras anti‑leak por nombre).")

    # 4) Particiones
    for req in ['split_group', target]:
        if req not in df.columns:
            raise ValueError(f"Falta la columna requerida '{req}' en gold.dataset_final")

    train_df = df[df['split_group'] == 'TRAIN']
    test_df  = df[df['split_group'] == 'TEST']
    oot_df   = df[df['split_group'] == 'OOT']

    X_train_raw, y_train = train_df[candidates_no_name_leak], train_df[target].astype(int)
    X_test_raw,  y_test  = test_df[candidates_no_name_leak],  test_df[target].astype(int)
    X_oot_raw,   y_oot   = oot_df[candidates_no_name_leak],   oot_df[target].astype(int)

    # 5) Detectar fugas perfectas (por valores)
    leaked_perfect = detect_perfect_leakage(X_train_raw, y_train)
    # 6) (Opcional) Altísima correlación con target (quitar si aplica)
    leaked_corr = drop_high_corr_with_target(X_train_raw, y_train, thr=0.999)

    # Unir y remover duplicados
    to_drop = sorted(set(dropped_by_name) | set(leaked_perfect) | set(leaked_corr))
    if to_drop:
        print(f"⚠️  Removiendo {len(to_drop)} columnas con fuga (nombre/perfect/corr): {to_drop[:10]}{' ...' if len(to_drop)>10 else ''}")
        X_train_raw = X_train_raw.drop(columns=to_drop, errors="ignore")
        X_test_raw  = X_test_raw.drop(columns=to_drop, errors="ignore")
        X_oot_raw   = X_oot_raw.drop(columns=to_drop, errors="ignore")

    # 7) Saneamiento de features
    X_train, feat_used, meta_tr = sanitize_train_df(X_train_raw, clip_quantiles=(0.001, 0.999))
    X_test,  meta_te = sanitize_apply_df(X_test_raw,  feat_used)
    X_oot,   meta_oo = sanitize_apply_df(X_oot_raw,   feat_used)

    print(f"🧼 Saneamiento:")
    print(f"   - Infs reemplazados (TRAIN): {meta_tr['n_inf_replaced_train']}")
    print(f"   - Columnas all-NaN drop (TRAIN): {len(meta_tr['dropped_all_nan_cols_train'])} -> {meta_tr['dropped_all_nan_cols_train'][:10]}{' ...' if len(meta_tr['dropped_all_nan_cols_train'])>10 else ''}")
    print(f"   - Features finales: {len(feat_used)}")
    print(f"   - Infs reemplazados (TEST): {meta_te['n_inf_replaced']}, (OOT): {meta_oo['n_inf_replaced']}")

    # 8) MLflow + Entrenamiento
    mlflow.set_experiment("Cobranza_Debito_Recurrente_RF")

    rf_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_leaf': 20,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }

    with mlflow.start_run(run_name="RandomForest_Base_Model"):
        print("🌳 Entrenando Random Forest (Modelo Robusto Corporativo)...")

        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**rf_params))
        ])

        pipe.fit(X_train, y_train)

        # Evaluación
        y_pred_test = pipe.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_pred_test)

        y_pred_oot = pipe.predict_proba(X_oot)[:, 1]
        auc_oot = roc_auc_score(y_oot, y_pred_oot)

        # MLflow logging
        mlflow.log_param("gold_table_used", "gold.dataset_final")
        mlflow.log_params(rf_params)
        mlflow.log_metric("auc_test", auc_test)
        mlflow.log_metric("auc_oot", auc_oot)
        mlflow.log_metric("n_features_final", len(feat_used))
        mlflow.log_metric("n_inf_replaced_train", meta_tr["n_inf_replaced_train"])
        mlflow.log_metric("n_inf_replaced_test", meta_te["n_inf_replaced"])
        mlflow.log_metric("n_inf_replaced_oot", meta_oo["n_inf_replaced"])

        # Log de columnas removidas por fuga
        mlflow.log_text("\n".join(to_drop), "leakage_columns_removed.txt")

        # Signature + input example
        try:
            input_example = X_train.head(5)
            signature = infer_signature(X_train, pipe.predict_proba(X_train)[:, 1])
            mlflow.sklearn.log_model(pipe, "model_rf", signature=signature, input_example=input_example)
        except Exception as e:
            print(f"⚠️  No se pudo registrar signature/input_example: {e}")
            mlflow.sklearn.log_model(pipe, "model_rf")

        print("-" * 30)
        print(f"✅ Entrenamiento Finalizado.")
        print(f"📈 AUC en Test (Validación Interna): {auc_test:.4f}")
        print(f"📈 AUC en OOT (Estabilidad Temporal): {auc_oot:.4f}")
        print("-" * 30)

        # Importancias
        rf_model = pipe.named_steps["rf"]
        importances = pd.Series(rf_model.feature_importances_, index=feat_used)
        print("🔝 Top 5 Variables Predictoras (sin columnas con fuga):")
        print(importances.sort_values(ascending=False).head(5))


if __name__ == "__main__":
    train()