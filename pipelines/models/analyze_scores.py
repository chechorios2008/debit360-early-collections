from pathlib import Path
import duckdb

here = Path(__file__).resolve()
repo_root = here.parents[3]
DB_PATH = (repo_root / "op_cobro" / "database" / "analytics.duckdb").resolve()

con = duckdb.connect(str(DB_PATH))

print("Total filas:")
print(con.execute("SELECT COUNT(*) AS n FROM model_results.scores_best").df())

print("\nResumen score:")
print(con.execute("""
    SELECT
      MIN(score) AS min_score,
      MAX(score) AS max_score,
      AVG(score) AS avg_score
    FROM model_results.scores_best
""").df())

print("\nConteo score=1.0 y score=0.0:")
print(con.execute("""
    SELECT
      SUM(CASE WHEN score = 1.0 THEN 1 ELSE 0 END) AS n_score_1,
      SUM(CASE WHEN score = 0.0 THEN 1 ELSE 0 END) AS n_score_0
    FROM model_results.scores_best
""").df())

print("\nPercentiles:")
print(con.execute("""
    SELECT
      quantile_cont(score, 0.50) AS p50,
      quantile_cont(score, 0.90) AS p90,
      quantile_cont(score, 0.95) AS p95,
      quantile_cont(score, 0.99) AS p99
    FROM model_results.scores_best
""").df())

print("\nTop 10 OOT:")
print(con.execute("""
    SELECT *
    FROM model_results.scores_best
    WHERE split_group='OOT'
    ORDER BY score DESC
    LIMIT 10
""").df())

con.close()