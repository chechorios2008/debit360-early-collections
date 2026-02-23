from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ===== CONFIG =====
EXPERIMENT_NAME = "Cobranza_Debito_Recurrente_RF"
METRIC_NAME = "auc_oot"              # criterio principal
MODEL_ARTIFACT = "model_rf"          # como lo loggeaste en train_model.py
GOLD_TABLE = "gold.dataset_final"

OUT_SCHEMA = "model_results"
OUT_TABLE = "scores_best"

# Filtros para evitar runs viejos / raros (incluye el tuyo final con tags)
REQUIRED_TAGS = {
    "objective": "debito_recurrente_only",
    "model_family": "RandomForest",
}

# Guardrail: ignorar AUC perfectas que suelen indicar leakage
MAX_REASONABLE_AUC = 0.999

# ===== PATHS =====
here = Path(__file__).resolve()
repo_root = here.parents[3]  # ...\01_prueba_analitico_4

DB_PATH = (repo_root / "op_cobro" / "database" / "analytics.duckdb").resolve()
MLRUNS_DIR = (repo_root / "op_cobro" / "pipelines" / "models" / "mlruns").resolve()

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")

def sanitize_for_scoring(X: pd.DataFrame, clip_quantiles=(0.001, 0.999)) -> pd.DataFrame:
    """
    Sanea X para scoring:
    - convierte a numérico
    - reemplaza inf/-inf por NaN
    - capar valores extremadamente grandes (por seguridad)
    - clipping suave por percentiles (opcional)
    """
    X = X.copy()

    # Convertir todo a numérico (strings raros -> NaN)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Reemplazar inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # Capar valores absurdos (si algo quedó muy grande)
    # (si un valor supera float64 max, ya suele convertirse en inf, pero esto ayuda)
    max_float = np.finfo("float64").max
    X = X.mask(X > max_float, np.nan)
    X = X.mask(X < -max_float, np.nan)

    # Clipping suave
    if clip_quantiles is not None:
        q_low, q_high = clip_quantiles
        for c in X.columns:
            s = X[c]
            ql = s.quantile(q_low)
            qh = s.quantile(q_high)
            if pd.notna(ql) and pd.notna(qh) and qh > ql:
                X[c] = s.clip(lower=ql, upper=qh)

    return X


def pick_best_run(experiment_name: str, metric_name: str):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"No existe el experimento: {experiment_name}")

    client = MlflowClient()

    # Trae runs ordenados por AUC OOT desc
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=200
    )
    if not runs:
        raise RuntimeError("No hay runs en el experimento.")

    # Filtra runs: requiere tags y evita AUC=1.0
    for r in runs:
        m = r.data.metrics.get(metric_name, None)
        if m is None:
            continue
        if m >= MAX_REASONABLE_AUC:
            # ignore perfect / suspicious runs
            continue

        # check tags
        ok = True
        for k, v in REQUIRED_TAGS.items():
            if r.data.tags.get(k) != v:
                ok = False
                break
        if not ok:
            continue

        return r.info.run_id, float(m)

    # Si no encuentra con tags, fallback: el mejor run con AUC razonable
    for r in runs:
        m = r.data.metrics.get(metric_name, None)
        if m is None:
            continue
        if m >= MAX_REASONABLE_AUC:
            continue
        return r.info.run_id, float(m)

    raise RuntimeError(f"No encontré runs válidos con {metric_name} y filtros aplicados.")


def load_features_used(run_id: str) -> list:
    """
    Intenta descargar features_used.txt desde artifacts del run.
    Si no existe, devuelve None y usaremos fallback.
    """
    try:
        # Descarga el archivo al directorio temporal interno de MLflow
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="features_used.txt"
        )
        with open(local_path, "r", encoding="utf-8") as f:
            cols = [line.strip() for line in f.readlines() if line.strip()]
        return cols
    except Exception:
        return None


def main():
    print("🔎 Buscando mejor modelo (filtrando runs sospechosos)...")
    best_run_id, best_auc = pick_best_run(EXPERIMENT_NAME, METRIC_NAME)
    print(f"✅ Mejor run seleccionado: {best_run_id} con {METRIC_NAME}={best_auc:.6f}")

    print("📦 Cargando modelo ganador desde MLflow...")
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/{MODEL_ARTIFACT}")

    print("📂 Leyendo data GOLD desde DuckDB...")
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {GOLD_TABLE}").df()

    # Validar llaves
    required_cols = {"num_doc", "obl17", "split_group"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas requeridas en {GOLD_TABLE}: {missing}")

    # 1) Intentar usar EXACTAMENTE las features usadas en training
    feat_used = load_features_used(best_run_id)

    # 2) Fallback: si no existe features_used.txt, usamos exclusiones como antes
    if feat_used is None:
        print("⚠️ No encontré features_used.txt en artifacts del run. Usando fallback por exclusión.")
        exclude = {
            "num_doc", "obl17", "f_analisis",
            "split_group", "target_final", "var_rta",
            "f_analisis_key", "num_doc_key", "obl17_key"
        }
        feat_used = [c for c in df.columns if c not in exclude]

    # Alinear columnas exactamente
    X = df.reindex(columns=feat_used)

    print(f"🧼 Saneando X para scoring (evitar inf/valores enormes)...")
    X = sanitize_for_scoring(X, clip_quantiles=(0.001, 0.999))

    print(f"🧠 Scoring sobre {len(df):,} filas con {X.shape[1]} features...")
    scores = model.predict_proba(X)[:, 1]

    out = df[["num_doc", "obl17", "split_group"]].copy()
    out["score"] = scores
    out["run_id"] = best_run_id
    out[METRIC_NAME] = best_auc

    # Guardar en DuckDB
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {OUT_SCHEMA}")
    con.register("out_df", out)
    con.execute(f"""
        CREATE OR REPLACE TABLE {OUT_SCHEMA}.{OUT_TABLE} AS
        SELECT * FROM out_df
    """)

    # View útil para negocio: top 200 obligaciones con mayor probabilidad
    con.execute(f"""
        CREATE OR REPLACE VIEW {OUT_SCHEMA}.top_scores AS
        SELECT *
        FROM {OUT_SCHEMA}.{OUT_TABLE}
        ORDER BY score DESC
        LIMIT 200
    """)

    con.close()

    print(f"✅ Guardado en DuckDB: {OUT_SCHEMA}.{OUT_TABLE}")
    print("👀 Ejemplo (top 10):")
    print(out.sort_values("score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()