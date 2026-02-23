import duckdb
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import mlflow.xgboost
from pathlib import Path

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "analytics.duckdb"

def train():
    # 1. Cargar datos de GOLD
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold.dataset_final").df()
    con.close()

    # 2. Definir Features y Target
    # Excluimos IDs, fechas y la columna de partición
    exclude_cols = ['num_doc', 'obl17', 'f_analisis', 'split_group', 'target_final', 'var_rta']
    features = [c for c in df.columns if c not in exclude_cols]
    target = 'target_final'

    # 3. Split basado en el diseño Senior (OOT)
    train_df = df[df['split_group'] == 'TRAIN']
    test_df = df[df['split_group'] == 'TEST']
    oot_df = df[df['split_group'] == 'OOT']

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    X_oot, y_oot = oot_df[features], oot_df[target]

    # 4. MLflow Experiment
    mlflow.set_experiment("Cobranza_Debito_Recurrente")

    with mlflow.start_run():
        print("🤖 Entrenando XGBoost...")
        
        # Parámetros base para evitar overfitting con pocos datos
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 4,
            'seed': 42
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # 5. Evaluación
        y_pred_test = model.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_pred_test)
        
        y_pred_oot = model.predict_proba(X_oot)[:, 1]
        auc_oot = roc_auc_score(y_oot, y_pred_oot)

        # 6. Registro de métricas
        mlflow.log_params(params)
        mlflow.log_metric("auc_test", auc_test)
        mlflow.log_metric("auc_oot", auc_oot)
        mlflow.xgboost.log_model(model, "model")

        print(f"✅ Entrenamiento Completo.")
        print(f"📊 AUC Test: {auc_test:.4f}")
        print(f"📊 AUC OOT: {auc_oot:.4f}")

if __name__ == "__main__":
    train()