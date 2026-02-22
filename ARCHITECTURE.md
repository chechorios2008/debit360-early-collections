# Arquitectura – Capacidad Analítica Operativa para Optimización de Cobranza en Mora Temprana

---

## 1. Visión General

Este proyecto tiene como objetivo diseñar una capacidad analítica que permita optimizar la gestión de cobranza en clientes en mora temprana (1–30 días), mediante la identificación de obligaciones con alta probabilidad de pago exclusivamente por débito recurrente.

La solución busca transformar datos operativos en decisiones accionables que reduzcan costos de cobranza y mejoren la eficiencia operativa del banco.

---

## 2. Problema de Negocio

Actualmente, los procesos de cobranza asignan gestiones operativas sin diferenciar adecuadamente entre:

- Clientes que requieren intervención activa.
- Clientes que probablemente pagarán de forma automática.

Esto genera:

- Costos operativos innecesarios.
- Sobrecarga en canales de cobranza.
- Baja eficiencia en la asignación de recursos.

---

## 3. Objetivo Analítico

Construir un modelo de clasificación binaria que estime la probabilidad de pago por obligación exclusivamente mediante débito recurrente.

Resultado esperado:

> Permitir decisiones operativas de asignación o no asignación de gestión de cobranza.

---

## 4. Principios de Diseño

La arquitectura fue diseñada bajo los siguientes principios:

- Separación de capas (Layered Architecture)
- Reproducibilidad analítica
- Escalabilidad conceptual
- Trazabilidad del modelo
- Preparación para operación productiva

---

## 5. Arquitectura General

La solución se compone de cinco capas principales:

### 5.1 Data Storage Layer

Organización de datos siguiendo un enfoque tipo Lakehouse:

- **Raw Zone**: datos originales provenientes de archivos CSV.
- **Curated (Silver)**: datos integrados y limpiados.
- **Analytics (Gold)**: dataset listo para modelado.
- **Model Results**: resultados de scoring.
- **ML Metadata**: información de experimentos y métricas.

Implementado utilizando DuckDB para facilitar portabilidad y ejecución local.

---

### 5.2 Data Processing Layer

Procesamiento distribuido mediante Apache Spark:

- Limpieza y estandarización.
- Integración cliente–obligación.
- Construcción de variables analíticas.
- Feature Engineering.

---

### 5.3 ML & MLOps Layer

Incluye el ciclo analítico completo:

- Particionamiento Train / Test / Out-of-Time.
- Entrenamiento del modelo.
- Evaluación mediante AUC y métricas complementarias.
- Tracking de experimentos con MLflow.

MLflow permite garantizar trazabilidad y reproducibilidad del modelo.

---

### 5.4 Serving & Orchestration Layer

FastAPI actúa como **Analytical Control Plane**, permitiendo:

- Ejecutar pipelines ETL.
- Ejecutar entrenamiento del modelo.
- Exponer resultados analíticos mediante APIs.
- Integración con herramientas de automatización.

---

### 5.5 Business Consumption Layer

Los resultados son consumidos por:

- Power BI para análisis estratégico.
- n8n para automatización operativa.
- Equipos de cobranza para toma de decisiones.

---

## 6. Flujo Operacional

1. Ingesta de datos crudos.
2. Procesamiento y construcción de features.
3. Entrenamiento del modelo.
4. Evaluación y registro en MLflow.
5. Generación de probabilidades de pago.
6. Consumo por áreas operativas.

---

## 7. Decisión Operativa Generada

El modelo habilita la siguiente decisión:

- **Asignar gestión de cobranza**
- **No asignar gestión (cliente auto-recuperable)**

Esto permite optimizar la asignación de recursos y reducir costos operativos.

---

## 8. Escalabilidad y Evolución

La arquitectura permite evolucionar hacia:

- Integración con Data Lake corporativo.
- Scheduling automatizado.
- Monitoreo continuo de modelos.
- Despliegue en ambientes productivos.

---

## 9. Conclusión

La solución propuesta trasciende la construcción de un modelo predictivo, estableciendo una capacidad analítica operacionalizable alineada con principios modernos de DataOps y MLOps.

El enfoque garantiza que los resultados analíticos puedan integrarse directamente en procesos de negocio y generación de valor.