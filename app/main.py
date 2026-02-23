from fastapi import FastAPI, HTTPException
import subprocess
import sys
from pathlib import Path

from app.db import query_one, query_df, exec_sql

app = FastAPI(title="Orquestador Cobranza (Demo)", version="0.1")

# --- Config: dónde está el script de scoring ---
# Estructura: op_cobro/pipelines/models/score_best_model.py
HERE = Path(__file__).resolve()
REPO_OP_COBRO = HERE.parents[1]  # .../op_cobro
SCORE_SCRIPT = (REPO_OP_COBRO / "pipelines" / "models" / "score_best_model.py").resolve()

# --- SQL: reconstrucción tabla final (tu regla conservadora score==1.0) ---
SQL_REBUILD_DECISION = """
CREATE SCHEMA IF NOT EXISTS model_results;

CREATE OR REPLACE TABLE model_results.decision_final_v1 AS
SELECT
  s.*,
  CASE WHEN s.score = 1.0 THEN 0 ELSE 1 END AS gestion_cobro,
  CASE WHEN s.score = 1.0 THEN 'NO_ASIGNAR_GESTION' ELSE 'ASIGNAR_GESTION' END AS decision_label,
  'v1_score_eq_1' AS policy_version,
  1.0 AS threshold_used,
  CURRENT_TIMESTAMP AS created_at
FROM model_results.scores_best s;
"""

@app.get("/decision/detail/{num_doc}/{obl17}")
def get_decision_detail(num_doc: str, obl17: str):
    """
    Devuelve la decisión operativa final por (num_doc, obl17).
    """
    sql = """
        SELECT *
        FROM model_results.decision_final_v1
        WHERE CAST(num_doc AS VARCHAR) = ?
          AND CAST(obl17 AS VARCHAR) = ?
        LIMIT 1
    """
    row = query_one(sql, [num_doc, obl17])
    if row is None:
        raise HTTPException(status_code=404, detail="No existe esa obligación en decision_final_v1")
    return row

@app.post("/orchestrate")
def orchestrate():
    """
    Endpoint DEMO: FastAPI orquesta el refresco:
    1) Ejecuta scoring_best_model.py (recalcula scores_best con mejor modelo)
    2) Reconstruye decision_final_v1 (con regla score==1.0)
    Devuelve un resumen para evidencia de operación.
    """
    if not SCORE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"No encuentro score_best_model.py en {SCORE_SCRIPT}")

    # 1) Ejecutar scoring (puede tardar unos segundos/minutos)
    try:
        proc = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=600  # 10 min por si acaso
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout ejecutando score_best_model.py")

    if proc.returncode != 0:
        # devolvemos parte del stderr para diagnóstico rápido
        raise HTTPException(status_code=500, detail=f"Falló scoring script: {proc.stderr[-1500:]}")

    # 2) Reconstruir tabla final
    exec_sql(SQL_REBUILD_DECISION)

    # 3) Resumen para negocio/operación
    summary = query_df("""
        SELECT decision_label, COUNT(*) AS n
        FROM model_results.decision_final_v1
        GROUP BY decision_label
        ORDER BY n DESC
    """).to_dict(orient="records")

    meta = query_one("""
        SELECT run_id, auc_oot
        FROM model_results.decision_final_v1
        LIMIT 1
    """)

    return {
        "status": "ok",
        "message": "scores_best recalculado y decision_final_v1 reconstruida",
        "model_metadata": meta,
        "decision_counts": summary,
        "scoring_stdout_tail": proc.stdout[-1000:]  # evidencia de ejecución (últimas líneas)
    }

@app.get("/decision/by_obligation/{obl17}")
def get_decisions_by_obligation(obl17: str, limit: int = 200):
    """
    Devuelve registros filtrando SOLO por obl17.
    """
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit debe estar entre 1 y 5000")

    sql = """
        SELECT *
        FROM model_results.decision_final_v1
        WHERE CAST(obl17 AS VARCHAR) = ?
        LIMIT ?
    """
    df = query_df(sql, [obl17, limit])
    if df.empty:
        raise HTTPException(status_code=404, detail="No se encontraron registros para ese obl17")
    return df.to_dict(orient="records")


@app.get("/decision/by_client/{num_doc}")
def get_decisions_by_client(num_doc: str, limit: int = 2000):
    """
    Devuelve TODAS las obligaciones de un cliente (num_doc).
    """
    if limit < 1 or limit > 20000:
        raise HTTPException(status_code=400, detail="limit debe estar entre 1 y 20000")

    sql = """
        SELECT *
        FROM model_results.decision_final_v1
        WHERE CAST(num_doc AS VARCHAR) = ?
        ORDER BY score DESC
        LIMIT ?
    """
    df = query_df(sql, [num_doc, limit])
    if df.empty:
        raise HTTPException(status_code=404, detail="No se encontraron obligaciones para ese num_doc")
    return df.to_dict(orient="records")