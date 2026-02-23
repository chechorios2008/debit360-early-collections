🧠 Justificación del Modelo de Machine Learning: train_model.py

Este documento detalla los criterios técnicos y las decisiones de diseño adoptadas para la fase de modelado predictivo, asegurando la alineación con los requerimientos de la Gerencia de Inteligencia de Originación y Cobranza.

1. Estrategia de Particionamiento Temporal (Train/Test/OOT)
Para garantizar la robustez estadística y la estabilidad del modelo, se rechazó el uso de particionamiento aleatorio tradicional. En su lugar, se implementó un esquema de validación fuera de tiempo (Out-of-Time - OOT):

Segmento Train/Test (Histórico): Utilizado para el aprendizaje de patrones y ajuste de hiperparámetros.

Segmento OOT (Validación Externa): Correspondiente al mes más reciente de la data (f_analisis máximo, Septiembre 2025). Este conjunto de datos se mantuvo aislado durante todo el proceso de entrenamiento.

Justificación: En modelos de cobranza, el comportamiento de pago presenta estacionalidad y dependencia temporal. Evaluar el modelo con un mes "futuro" respecto al entrenamiento permite medir el Concept Drift y asegurar que la probabilidad estimada sea confiable para la operación actual.

2. Selección del Algoritmo: XGBoost Classifier
Se seleccionó XGBoost (Extreme Gradient Boosting) como motor de inferencia principal debido a sus capacidades superiores en entornos financieros:

Manejo Nativo de Sparsity: Dado que el proceso de integración de fuentes (Joins) resultó en un dataset con alta presencia de valores nulos o ceros (ausencia de gestiones o transacciones en ciertos canales), XGBoost gestiona estas "ramas vacías" de forma óptima sin requerir imputaciones artificiales que distorsionen la distribución original.

Captura de Interacciones Complejas: El modelo identifica relaciones no lineales entre variables (ej. el efecto combinado de ser un cliente digital y tener excedentes de pago), superando la capacidad de una regresión logística tradicional.

Control de Overfitting: Mediante parámetros de regularización (Gamma, Lambda), se garantiza que el modelo aprenda tendencias generales y no ruidos específicos del dataset de entrenamiento.

3. Trazabilidad y Gobierno con MLflow
Siguiendo buenas prácticas de MLOps, el script integra MLflow para la gestión del ciclo de vida del modelo:

Reproducibilidad: Registro de cada experimento, incluyendo hiperparámetros y versiones de los datos.

Métricas de Desempeño: Monitoreo centralizado del AUC (Area Under the Curve) tanto en el set de Test como en el OOT para detectar degradación de performance.

4. Definición de la Variable Respuesta (Target)
El target se construyó bajo una lógica de negocio operativa:

Target = 1: Obligaciones que presentan pagos exclusivamente por canal débito y cumplen con el umbral de recurrencia ≥ 40%.

Target = 0: Otros comportamientos de pago.

Objetivo de Negocio: Este enfoque permite al modelo identificar con precisión el segmento "auto-pagador", permitiendo al banco excluir estas obligaciones de las campañas de gestión intensiva, generando un ahorro directo en costos de cobranza.

5. Evaluación de Estabilidad
La métrica de éxito principal es el AUC. Se considera un modelo exitoso si la diferencia de AUC entre el set de Test y el OOT es mínima, validando que el modelo es capaz de generalizar y mantener su poder discriminatorio en el tiempo.

# Para la visualización del rendimiento en MLFLOW.
mlflow ui --port 5000