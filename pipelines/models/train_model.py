import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path


# --- Configuración de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "analytics.duckdb"

def train():
    # 1. Carga de datos desde la capa GOLD
    print("📂 Cargando datos desde DuckDB (Capa GOLD)...")
    con = duckdb.connect(str(DB_PATH))
    # Leemos el dataset consolidado
    df = con.execute("SELECT * FROM gold.dataset_final").df()
    con.close()

    # 2. Definición de Variables (Features y Target)
    # Excluimos metadatos, llaves y la columna de partición
    exclude_cols = [
        'num_doc', 'obl17', 'f_analisis', 
        'split_group', 'target_final', 'var_rta',
        'f_analisis_key' # Por si quedó en el join
    ]

    features = [c for c in df.columns if c not in exclude_cols]
    target = 'target_final'

    print(f"📊 Dataset cargado: {df.shape[0]} registros y {len(features)} features.")

    # 3. Particionamiento Temporal (Train / Test / OOT)
    # Respetamos la lógica Senior de evaluación fuera de tiempo
    train_df = df[df['split_group'] == 'TRAIN']
    test_df = df[df['split_group'] == 'TEST']
    oot_df = df[df['split_group'] == 'OOT']

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    X_oot, y_oot = oot_df[features], oot_df[target]

    # 4. Configuración de Experimento en MLflow
    mlflow.set_experiment("Cobranza_Debito_Recurrente_RF")

    with mlflow.start_run(run_name="RandomForest_Base_Model"):
        print("🌳 Entrenando Random Forest (Modelo Robusto Corporativo)...")

        # Parámetros balanceados para evitar sobreajuste (Overfitting)
        rf_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_leaf': 20,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced' # Maneja el desbalanceo del 10% de match
        }

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        # 5. Evaluación de Desempeño
        # Predicción de probabilidades para el cálculo de AUC
        y_pred_test = model.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_pred_test)

        y_pred_oot = model.predict_proba(X_oot)[:, 1]
        auc_oot = roc_auc_score(y_oot, y_pred_oot)

        # 6. Registro de Metadatos en MLflow
        mlflow.log_params(rf_params)
        mlflow.log_metric("auc_test", auc_test)
        mlflow.log_metric("auc_oot", auc_oot)

        # Guardamos el modelo para la Etapa 6 (FastAPI)
        mlflow.sklearn.log_model(model, "model_rf")

        print("-" * 30)
        print(f"✅ Entrenamiento Finalizado.")
        print(f"📈 AUC en Test (Validación Interna): {auc_test:.4f}")
        print(f"📈 AUC en OOT (Estabilidad Temporal): {auc_oot:.4f}")
        print("-" * 30)

        # Verificamos importancia de variables (Top 5)
        importances = pd.Series(model.feature_importances_, index=features)
        print("🔝 Top 5 Variables Predictoras:")
        print(importances.sort_values(ascending=False).head(5))

if __name__ == "__main__":
    train()