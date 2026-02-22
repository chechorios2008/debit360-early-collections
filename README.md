# 📊 Capacidad Analítica Operativa para Optimización de Cobranza en Mora Temprana

Solución analítica end‑to‑end para identificar obligaciones con **alta probabilidad de pago exclusivamente por débito recurrente** (recurrencia ≥ 40%), optimizando la asignación de gestión de cobranza en mora temprana (1–30 días) y reduciendo costos operativos.

---

## 🎯 Objetivo del Proyecto

Diseñar una solución completa que permita:

- Integrar múltiples fuentes (nivel cliente–obligación).
- Construir un modelo de **clasificación binaria** (probabilidad de pago por débito recurrente).
- Definir un umbral operativo (recurrencia ≥ 40%) y criterios de decisión.
- Operacionalizar resultados (scoring + consumo por negocio) con trazabilidad.

**Resultado esperado:** identificar obligaciones que **NO requieren gestión intensiva** porque tienen alta probabilidad de pago automático por débito recurrente.

---

## 🧩 Problema de Negocio

La cobranza tradicional asigna recursos sin diferenciar entre:

- Obligaciones con riesgo real de no pago.
- Obligaciones que pagan de forma natural por **débito recurrente**.

Esto genera:

- Costos innecesarios de gestión.
- Menor eficiencia operativa.
- Uso subóptimo de canales y estrategias.

---

## 🏗️ Arquitectura de la Solución

Diseño bajo enfoque **Data + ML + Operacionalización** siguiendo buenas prácticas de ingeniería y MLOps.

📄 Ver detalle en: [`ARCHITECTURE.md`](./ARCHITEincipales:**
- **Data Storage Layer** → organización RAW / SILVER / GOLD
- **Data Processing Layer** → procesamiento con Apache Spark
- **ML & MLOps Layer** → entrenamiento + tracking con MLflow
- **Serving Layer** → exposición/ejecución vía FastAPI (batch/online)
- **Business Consumption Layer** → Power BI / automatización (n8n)

---

## 🔄 Flujo del Proceso (alto nivel)

1. Ingesta de datos crudos (CSV).
2. Limpieza, estandarización y enriquecimiento (Spark).
3. Feature Engineering.
4. Entrenamiento del modelo.
5. Evaluación (AUC + métricas operativas).
6. Tracking de experimentos (MLflow).
7. Scoring (probabilidades).
8. Consumo operativo + visualización.

---

## ⚙️ Stack Tecnológico

- **Procesamiento:** Apache Spark (PySpark)
- **Data:** Pandas / DuckDB / PyArrow
- **ML:** scikit‑learn
- **MLOps:** MLflow
- **Serving:** FastAPI + Uvicorn
- **Calidad:** Pytest / Ruff / Black / Isort / Mypy
- **Visualización:** Power BI (fuera de este repo) / notebooks

---

## ✅ Requisitos del Entorno

- **Python 3.11 (64‑bit)** recomendado (entorno objetivo local y base para despliegue).
- Windows/Linux compatibles (comandos abajo incluyen Windows).

---

## 📦 Dependencias y Reproducibilidad

Este repositorio usa **dos archivos**:

- `requirements.txt`: dependencias directas del proyecto (curadas).
- `requirements-lock.txt`: snapshot exacto del entorno (**generado con `pip freeze`**).

> Nota: `pip freeze` **reporta lo instalado** (incluye dependencias transitivas) y **no calcula un lockfile/solver result**, pero sirve como snapshot reproducible de entorno. [1](https://pip.pypa.io/en/stable/cli/pip_freeze/)

### ¿Cuál usar?
- Para instalar **rápido y flexible**: `requirements.txt`
- Para replicar el entorno **1:1 (reproducible)**: `requirements-lock.txt`

---

## 🚀 Ejecución Local (Windows)

### 1) Crear y activar entorno virtual (Python 3.11)

cd C:\Users\serrios\01_prueba_analitico_4\op_cobro
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version

## 🔎 Inspección rápida de la base DuckDB (schemas/tablas)
###
La base local se persiste en: `database/analytics.duckdb`.

### Ver esquemas (schemas) disponibles
DuckDB expone metadatos vía `information_schema`. Para listar los esquemas únicos:

#### bash
python -c "import duckdb; con=duckdb.connect('database/analytics.duckdb'); print(con.execute(\"SELECT DISTINCT schema_name FROM information_schema.schemata ORDER BY schema_name\").fetchdf()); con.close()"

## Data Quality Checks (RAW)

Este proyecto incluye un script de validación de calidad de datos para las tablas del esquema `raw` en DuckDB.

### ¿Qué valida?
- **Existencia de tablas** y conteo de registros.
- **Nulos en columnas clave** (`num_doc`, `obl17`, `f_analisis`) si existen.
- **Duplicados** en `raw.clientes` por unidad analítica `(num_doc, obl17, f_analisis)`.
- **Integridad referencial**: registros huérfanos en `raw.moras` respecto a `raw.clientes` (por `obl17`).

### Requisitos
- Python 3.10+ (recomendado)
- Entorno virtual activo (`.venv`)
- Dependencias instaladas