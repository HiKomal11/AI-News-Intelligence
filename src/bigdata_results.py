from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = str(BASE_DIR / "data" / "news_dataset.csv")
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("AI-News-Intelligence-Results")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# LOAD DATA
# ============================================================

df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("mode", "PERMISSIVE")
    .csv(DATA_FILE)
)

df = df.withColumn(
    "label",
    col("label").cast("int")
).filter(
    col("label").isNotNull()
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

label_df = (
    df.groupBy("label")
    .agg(count("*").alias("article_count"))
    .orderBy("label")
)

label_output = RESULTS_DIR / "bigdata_label_distribution.csv"

label_df.toPandas().to_csv(
    label_output,
    index=False
)


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

subject_df = (
    df.groupBy("subject")
    .agg(count("*").alias("article_count"))
    .orderBy(desc("article_count"))
)

subject_output = RESULTS_DIR / "bigdata_subject_distribution.csv"

subject_df.toPandas().to_csv(
    subject_output,
    index=False
)


# ============================================================
# SUBJECT + LABEL DISTRIBUTION
# ============================================================

subject_label_df = (
    df.groupBy("subject", "label")
    .agg(count("*").alias("article_count"))
    .orderBy(desc("article_count"))
)

subject_label_output = (
    RESULTS_DIR / "bigdata_subject_label_distribution.csv"
)

subject_label_df.toPandas().to_csv(
    subject_label_output,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

total = df.count()

fake = df.filter(
    col("label") == 0
).count()

real = df.filter(
    col("label") == 1
).count()


summary_file = RESULTS_DIR / "bigdata_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("AI NEWS INTELLIGENCE - BIG DATA SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Total articles: {total}\n")
    file.write(f"Fake articles: {fake}\n")
    file.write(f"Real articles: {real}\n\n")

    file.write("Spark Version: ")
    file.write(spark.version)
    file.write("\n")


print("\n======================================")
print("      BIG DATA RESULTS SAVED")
print("======================================")

print("Results directory:")
print(RESULTS_DIR)

print("\nCreated files:")

print(label_output)
print(subject_output)
print(subject_label_output)
print(summary_file)


spark.stop()

print("\nSpark session stopped.")