# -*- coding: utf-8 -*-
import os
import re
import argparse
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 0) CONFIG GLOBAL (rutas + MLflow tracking)
# ============================================================

# --- Repo root (siempre disponible) ---
here = Path(__file__).resolve()
# __file__ -> ...\op_cobro\pipelines\models\train_model.py
repo_root = here.parents[3]  # ...\01_prueba_analitico_4

# --- DuckDB path ---
DB_PATH_ENV = os.getenv("DB_PATH")
if DB_PATH_ENV:
    DB_PATH = Path(DB_PATH_ENV).resolve()
else:
    DB_PATH = (repo_root / "op_cobro" / "database" / "analytics.duckdb").resolve()

print(f"🗄️  Usando DuckDB en: {DB_PATH}")
assert DB_PATH.exists(), f"No existe la base en {DB_PATH}. Corre prepare_gold.py primero."

# --- MLflow tracking store (FIJO para evitar mlruns duplicados) ---
# Guarda todo SIEMPRE en: op_cobro/pipelines/models/mlruns
MLRUNS_DIR = (repo_root / "op_cobro" / "pipelines" / "models" / "mlruns").resolve()
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")


# ============================================================
# 1) UTILIDADES: saneamiento de features
# ============================================================

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


# ============================================================
# 2) ANTI-LEAKAGE: por nombre, fuga perfecta, correlación extrema
# ============================================================

LEAK_NAME_PATTERNS = [
    r"\btarget\b",
    r"target_", r"_target", r"targetfinal", r"target_final",
    r"\bvar_rta\b", r"varrta", r"_rta", r"rta_",
    r"\blabel\b", r"label_", r"_label",
    r"\brespuesta\b", r"respuesta_", r"_respuesta",
]
_leak_name_regex = re.compile("|".join(LEAK_NAME_PATTERNS), flags=re.IGNORECASE)


def drop_leak_by_name(columns: list):
    """Quita columnas cuyo nombre sugiere fuga (regex arriba)."""
    unsafe = [c for c in columns if _leak_name_regex.search(c)]
    safe = [c for c in columns if c not in unsafe]
    return safe, unsafe


def detect_perfect_leakage(X: pd.DataFrame, y: pd.Series, max_check=5000) -> list:
    """
    Detecta columnas binarias que replican exactamente el target (o su complemento).
    Usa muestra para acelerar si el dataset es grande.
    """
    if len(X) > max_check:
        idx = np.random.RandomState(42).choice(len(X), size=max_check, replace=False)
        Xs = X.iloc[idx]
        ys = y.iloc[idx]
    else:
        Xs, ys = X, y

    leaked = []
    for col in Xs.columns:
        col_values = Xs[col].dropna().unique()
        if len(col_values) <= 2 and set(col_values).issubset({0, 1}):
            try:
                eq = (Xs[col].fillna(-1).astype(int).values == ys.fillna(-1).astype(int).values).all()
                inv = ((1 - Xs[col].fillna(0).astype(int).values) == ys.fillna(-1).astype(int).values).all()
                if eq or inv:
                    leaked.append(col)
            except Exception:
                pass
    return leaked


def drop_high_corr_with_target(X: pd.DataFrame, y: pd.Series, thr: float = 0.999) -> list:
    """Marca columnas con correlación (Pearson) ~1 con el target."""
    df = X.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    corr = df.corrwith(y, method="pearson")
    high = corr.index[(corr.abs() >= thr)].tolist()
    return high


# ============================================================
# 3) MÉTRICAS: KS + métricas por umbral
# ============================================================

def ks_stat(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def metrics_at_threshold(y_true, y_score, thr=0.5) -> dict:
    y_pred = (y_score >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


# ============================================================
# 4) ARGS CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest model and log to MLflow")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--run_name", type=str, default="RandomForest_Base_Model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for precision/recall/f1 metrics")
    return parser.parse_args()


# ============================================================
# 5) ENTRENAMIENTO
# ============================================================

def train(args):
    print("📂 Cargando datos desde DuckDB (Capa GOLD)...")
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold.dataset_final").df()
    con.close()

    # Exclusiones obvias
    exclude_cols_exact = {
        "num_doc", "obl17", "f_analisis",
        "split_group", "target_final", "var_rta",
        "f_analisis_key", "num_doc_key", "obl17_key"
    }

    target = "target_final"
    candidates = [c for c in df.columns if c not in exclude_cols_exact and c != target]

    # Anti-leak por nombre
    candidates_no_name_leak, dropped_by_name = drop_leak_by_name(candidates)

    print(f"📊 Dataset cargado: {df.shape[0]} registros y {len(candidates_no_name_leak)} features (tras anti‑leak por nombre).")

    # Validaciones mínimas
    for req in ["split_group", target]:
        if req not in df.columns:
            raise ValueError(f"Falta la columna requerida '{req}' en gold.dataset_final")

    train_df = df[df["split_group"] == "TRAIN"]
    test_df  = df[df["split_group"] == "TEST"]
    oot_df   = df[df["split_group"] == "OOT"]

    if len(train_df) == 0 or len(test_df) == 0 or len(oot_df) == 0:
        raise ValueError("Algún split quedó vacío (TRAIN/TEST/OOT). Revisa split_group en gold.dataset_final.")

    X_train_raw, y_train = train_df[candidates_no_name_leak], train_df[target].astype(int)
    X_test_raw,  y_test  = test_df[candidates_no_name_leak],  test_df[target].astype(int)
    X_oot_raw,   y_oot   = oot_df[candidates_no_name_leak],   oot_df[target].astype(int)

    # Leakage por valores y correlación extrema
    leaked_perfect = detect_perfect_leakage(X_train_raw, y_train)
    leaked_corr = drop_high_corr_with_target(X_train_raw, y_train, thr=0.999)

    to_drop = sorted(set(dropped_by_name) | set(leaked_perfect) | set(leaked_corr))
    if to_drop:
        print(f"⚠️  Removiendo {len(to_drop)} columnas con fuga (nombre/perfect/corr): {to_drop[:10]}{' ...' if len(to_drop) > 10 else ''}")
        X_train_raw = X_train_raw.drop(columns=to_drop, errors="ignore")
        X_test_raw  = X_test_raw.drop(columns=to_drop, errors="ignore")
        X_oot_raw   = X_oot_raw.drop(columns=to_drop, errors="ignore")

    # Saneamiento
    X_train, feat_used, meta_tr = sanitize_train_df(X_train_raw, clip_quantiles=(0.001, 0.999))
    X_test,  meta_te = sanitize_apply_df(X_test_raw, feat_used)
    X_oot,   meta_oo = sanitize_apply_df(X_oot_raw, feat_used)

    print("🧼 Saneamiento:")
    print(f"   - Infs reemplazados (TRAIN): {meta_tr['n_inf_replaced_train']}")
    print(f"   - Columnas all-NaN drop (TRAIN): {len(meta_tr['dropped_all_nan_cols_train'])} -> {meta_tr['dropped_all_nan_cols_train'][:10]}{' ...' if len(meta_tr['dropped_all_nan_cols_train']) > 10 else ''}")
    print(f"   - Features finales: {len(feat_used)}")
    print(f"   - Infs reemplazados (TEST): {meta_te['n_inf_replaced']}, (OOT): {meta_oo['n_inf_replaced']}")

    # MLflow experiment
    mlflow.set_experiment("Cobranza_Debito_Recurrente_RF")

    rf_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name=args.run_name):
        print("🌳 Entrenando Random Forest (Modelo Robusto Corporativo)...")

        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**rf_params))
        ])

        pipe.fit(X_train, y_train)

        # Predicciones
        y_pred_test = pipe.predict_proba(X_test)[:, 1]
        y_pred_oot  = pipe.predict_proba(X_oot)[:, 1]

        # Métrica principal
        auc_test = roc_auc_score(y_test, y_pred_test)
        auc_oot  = roc_auc_score(y_oot, y_pred_oot)

        # Métricas complementarias
        ap_test = average_precision_score(y_test, y_pred_test)
        ap_oot  = average_precision_score(y_oot, y_pred_oot)

        ks_test = ks_stat(y_test, y_pred_test)
        ks_oot  = ks_stat(y_oot, y_pred_oot)

        m_test = metrics_at_threshold(y_test, y_pred_test, thr=args.threshold)
        m_oot  = metrics_at_threshold(y_oot, y_pred_oot, thr=args.threshold)

        # ------------------ LOGS MLFLOW ------------------
        # Params
        mlflow.log_param("gold_table_used", "gold.dataset_final")
        mlflow.log_params(rf_params)
        mlflow.log_param("threshold_metrics", args.threshold)

        # Tags (para lectura / sustentación)
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("objective", "debito_recurrente_only")
        mlflow.set_tag("split", "TRAIN/TEST/OOT from gold.dataset_final")

        # Métricas principales + complementarias
        mlflow.log_metric("auc_test", auc_test)
        mlflow.log_metric("auc_oot", auc_oot)

        mlflow.log_metric("ap_test", ap_test)
        mlflow.log_metric("ap_oot", ap_oot)

        mlflow.log_metric("ks_test", ks_test)
        mlflow.log_metric("ks_oot", ks_oot)

        mlflow.log_metric(f"precision_test_thr_{args.threshold}", m_test["precision"])
        mlflow.log_metric(f"recall_test_thr_{args.threshold}", m_test["recall"])
        mlflow.log_metric(f"f1_test_thr_{args.threshold}", m_test["f1"])

        mlflow.log_metric(f"precision_oot_thr_{args.threshold}", m_oot["precision"])
        mlflow.log_metric(f"recall_oot_thr_{args.threshold}", m_oot["recall"])
        mlflow.log_metric(f"f1_oot_thr_{args.threshold}", m_oot["f1"])

        # Metadatos de saneamiento
        mlflow.log_metric("n_features_final", len(feat_used))
        mlflow.log_metric("n_inf_replaced_train", meta_tr["n_inf_replaced_train"])
        mlflow.log_metric("n_inf_replaced_test", meta_te["n_inf_replaced"])
        mlflow.log_metric("n_inf_replaced_oot", meta_oo["n_inf_replaced"])

        # Artifacts útiles
        mlflow.log_text("\n".join(to_drop), "leakage_columns_removed.txt")
        mlflow.log_text("\n".join(feat_used), "features_used.txt")

        cm_txt = (
            f"TEST thr={args.threshold} tn={m_test['tn']} fp={m_test['fp']} fn={m_test['fn']} tp={m_test['tp']}\n"
            f"OOT  thr={args.threshold} tn={m_oot['tn']} fp={m_oot['fp']} fn={m_oot['fn']} tp={m_oot['tp']}\n"
        )
        mlflow.log_text(cm_txt, f"confusion_thr_{args.threshold}.txt")

        # Signature + input example (cast a float para evitar warning de ints con NaN)
        try:
            input_example = X_train.head(5).astype("float64")
            signature = infer_signature(input_example, pipe.predict_proba(input_example)[:, 1])
            mlflow.sklearn.log_model(pipe, "model_rf", signature=signature, input_example=input_example)
        except Exception as e:
            print(f"⚠️  No se pudo registrar signature/input_example: {e}")
            mlflow.sklearn.log_model(pipe, "model_rf")

        # Importancias (artifact)
        rf_model = pipe.named_steps["rf"]
        importances = pd.Series(rf_model.feature_importances_, index=feat_used).sort_values(ascending=False)
        mlflow.log_text(importances.head(30).to_string(), "top_features.txt")

        print("-" * 30)
        print("✅ Entrenamiento Finalizado.")
        print(f"📈 AUC Test: {auc_test:.4f}")
        print(f"📈 AUC OOT : {auc_oot:.4f}")
        print(f"📈 AP  OOT : {ap_oot:.4f} | KS OOT: {ks_oot:.4f}")
        print(f"🎯 thr={args.threshold} | Precision OOT: {m_oot['precision']:.4f} | Recall OOT: {m_oot['recall']:.4f} | F1 OOT: {m_oot['f1']:.4f}")
        print("-" * 30)
        print("🔝 Top 5 Variables Predictoras:")
        print(importances.head(5))


if __name__ == "__main__":
    args = parse_args()
    train(args)