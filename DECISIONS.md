# Architectural Decisions

Este documento describe las principales decisiones técnicas y arquitectónicas adoptadas durante el diseño de la solución analítica.

El objetivo es garantizar transparencia, trazabilidad y alineación entre decisiones tecnológicas y necesidades de negocio.

---

## 1. Enfoque Arquitectónico

### Decisión
Adoptar una arquitectura por capas (Layered Architecture).

### Justificación
Permite separar responsabilidades entre:

- almacenamiento de datos
- procesamiento
- modelado
- orquestación
- consumo de negocio

### Beneficio
Facilita escalabilidad futura sin rediseñar la solución.

---

## 2. Motor de Almacenamiento Analítico

### Decisión
Utilizar un motor SQL embebido orientado a analítica (DuckDB).

### Justificación
- Permite ejecución local reproducible.
- Optimizado para workloads analíticos.
- Simplifica la portabilidad del proyecto.
- Mantiene compatibilidad conceptual con motores analíticos corporativos (Impala, Snowflake, BigQuery).

### Escalabilidad
La arquitectura permite reemplazar el motor sin modificar pipelines.

---

## 3. Procesamiento de Datos con Apache Spark

### Decisión
Utilizar Spark para transformación y feature engineering.

### Justificación
- Simula entorno Big Data corporativo.
- Permite procesamiento distribuido.
- Refuerza diseño escalable.

### Beneficio
El pipeline puede migrar fácilmente a clusters productivos.

---

## 4. Orquestación mediante FastAPI

### Decisión
Usar FastAPI como Analytical Orchestrator.

### Justificación
- Permite ejecutar pipelines mediante endpoints.
- Facilita integración con sistemas externos.
- Prepara la solución para automatización.

### Beneficio
Transforma procesos analíticos en servicios operativos.

---

## 5. Tracking de Modelos con MLflow

### Decisión
Incorporar MLflow Tracking.

### Justificación
- Registro de métricas y parámetros.
- Versionamiento de experimentos.
- Reproducibilidad del entrenamiento.

### Beneficio
Introduce prácticas iniciales de MLOps sin sobreingeniería.

---

## 6. Particionamiento Temporal

### Decisión
Implementar Train / Test / Out-of-Time.

### Justificación
Evitar leakage temporal y simular comportamiento real en producción.

### Beneficio
Evaluación más robusta del modelo.

---

## 7. Consumo de Resultados

### Decisión
Exponer resultados mediante:

- Power BI (análisis estratégico)
- n8n (automatización operativa)

### Justificación
Conectar directamente analítica con decisiones de negocio.

---

## 8. Principios Adoptados

- Reproducibilidad
- Escalabilidad conceptual
- Separación de responsabilidades
- Orientación a negocio
- Operacionalización analítica

---

## 9. Conclusión

Las decisiones adoptadas buscan demostrar cómo una solución analítica puede evolucionar desde un entorno experimental hacia una capacidad operacional dentro de una organización financiera.