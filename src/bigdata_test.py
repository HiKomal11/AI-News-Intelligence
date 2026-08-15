from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("AI-News-Intelligence")
    .master("local[*]")
    .getOrCreate()
)

print("======================================")
print("        PYSPARK TEST SUCCESS")
print("======================================")

print("Spark Version:", spark.version)

df = spark.range(10)

print("\nSample Spark DataFrame:")
df.show()

print("Number of rows:", df.count())

spark.stop()