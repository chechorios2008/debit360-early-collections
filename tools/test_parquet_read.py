import sys
from pathlib import Path
from pipelines.processing.spark_session import get_spark_session


# ✅ Agrega la raíz del proyecto (op_cobro) al PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sube de tools/ a op_cobro/
sys.path.insert(0, str(PROJECT_ROOT))


spark = get_spark_session("test_parquet")

path = r"C:\Users\serrios\01_prueba_analitico_4\op_cobro\database\staging\raw.canales.parquet_dir"
df = spark.read.parquet(path)

print("✅ OK - columnas:", len(df.columns))
cols = df.columns[:5]
df.select(*cols).limit(3).show(truncate=False)

spark.stop()