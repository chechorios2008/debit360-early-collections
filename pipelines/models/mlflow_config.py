import mlflow


def setup_mlflow():
    # Establecemos el nombre del experimento para que no se mezcle con otros
    mlflow.set_experiment("Cobranza_Debito_Recurrente")

    # Configuramos el tracking local
    # Esto asegura que los resultados se guarden en la carpeta mlruns
    print("📊 MLflow configurado para tracking local en ./mlruns")

if __name__ == "__main__":
    setup_mlflow()