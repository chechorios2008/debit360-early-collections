from pipelines.processing.spark_session import get_spark_session

spark = get_spark_session("check_hadoop_version")

print("Spark version:", spark.version)
jvm = spark._jvm
print("Hadoop version:", jvm.org.apache.hadoop.util.VersionInfo.getVersion())
print("Hadoop revision:", jvm.org.apache.hadoop.util.VersionInfo.getRevision())

spark.stop()