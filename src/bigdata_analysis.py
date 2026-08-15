from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    length,
    avg,
    desc
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = str(BASE_DIR / "data" / "news_dataset.csv")


# ============================================================
# START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("AI-News-Intelligence-BigData")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("\n======================================")
print("     AI NEWS INTELLIGENCE - BIG DATA")
print("======================================")

print("Spark Version:", spark.version)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading news dataset...")

df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("mode", "PERMISSIVE")
    .csv(DATA_FILE)
)


# ============================================================
# FIX COLUMN TYPES
# ============================================================

# Explicitly convert label to integer.
# Invalid values become NULL instead of crashing the job.
df = df.withColumn(
    "label",
    col("label").cast("int")
)


# Remove rows where label could not be converted
df = df.filter(
    col("label").isNotNull()
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("Columns:")
print(df.columns)

total_records = df.count()

print("\nTotal records:", total_records)


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print("\n========== SAMPLE RECORDS ==========")

df.select(
    "title",
    "subject",
    "date",
    "label"
).show(
    5,
    truncate=50
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n========== FAKE / REAL DISTRIBUTION ==========")

label_distribution = (
    df.groupBy("label")
    .agg(
        count("*").alias("article_count")
    )
    .orderBy("label")
)

label_distribution.show()


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n========== SUBJECT DISTRIBUTION ==========")

subject_distribution = (
    df.groupBy("subject")
    .agg(
        count("*").alias("article_count")
    )
    .orderBy(
        desc("article_count")
    )
)

subject_distribution.show(
    15,
    truncate=False
)


# ============================================================
# ARTICLE LENGTH
# ============================================================

print("\n========== ARTICLE LENGTH ==========")

length_df = df.withColumn(
    "text_length",
    length(col("text"))
)

length_statistics = length_df.select(
    avg("text_length").alias(
        "average_characters"
    )
)

length_statistics.show()


# ============================================================
# LONGEST ARTICLES
# ============================================================

print("\n========== LONGEST ARTICLES ==========")

length_df.select(
    "title",
    "text_length",
    "label"
).orderBy(
    desc("text_length")
).show(
    10,
    truncate=60
)


# ============================================================
# SUBJECT + LABEL ANALYSIS
# ============================================================

print("\n========== SUBJECT / LABEL ANALYSIS ==========")

subject_label = (
    df.groupBy(
        "subject",
        "label"
    )
    .agg(
        count("*").alias("article_count")
    )
    .orderBy(
        desc("article_count")
    )
)

subject_label.show(
    20,
    truncate=False
)


# ============================================================
# CACHE DATAFRAME
# ============================================================

print("\nCaching DataFrame...")

df.cache()

cached_records = df.count()

print(
    "Cached records:",
    cached_records
)


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("        BIG DATA ANALYSIS COMPLETE")
print("======================================")

print("Total valid articles:", total_records)

print(
    "Fake articles:",
    df.filter(
        col("label") == 0
    ).count()
)

print(
    "Real articles:",
    df.filter(
        col("label") == 1
    ).count()
)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()

print("\nSpark session stopped.")