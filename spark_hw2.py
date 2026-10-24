"""COVID-19: three queries using PySpark DataFrame API.

Run:
    spark-submit --master local[2] covid_solution_fixed.py covid-data.csv results

Requires: pyspark; the input file must be readable by Spark.
"""

import csv
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def save_small_csv(dataframe, filename):
    """Save small result sets as a single CSV without Spark's output directories."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    rows = dataframe.collect()
    with filename.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(dataframe.columns)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    print(f"Saved {len(rows)} rows -> {filename}", flush=True)
    for row in rows:
        print(row, flush=True)


def main():
    input_csv = Path(sys.argv[1] if len(sys.argv) > 1 else "covid-data.csv").resolve()
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "results").resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    spark = (
        SparkSession.builder
        .appName("COVID March 2021")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        print("Reading CSV...", flush=True)
        raw = spark.read.option("header", "true").option("mode", "FAILFAST").csv(str(input_csv))
        df = (
            raw.select(
                "iso_code", "location", "continent",
                F.to_date("date", "yyyy-MM-dd").alias("date"),
                F.col("total_cases").cast("double").alias("total_cases"),
                F.col("new_cases").cast("double").alias("new_cases"),
                F.col("population").cast("double").alias("population"),
            )
            .filter(F.col("continent").isNotNull())
            .filter(F.col("iso_code").isNotNull())
            .filter(F.col("date").between("2021-03-24", "2021-03-31"))
            .cache()
        )


        top15 = (
            df.filter((F.col("date") == F.lit("2021-03-31")) & (F.col("population") > 0) &
                      F.col("total_cases").isNotNull())
            .select(
                "iso_code", F.col("location").alias("country"),
                F.round(100.0 * F.col("total_cases") / F.col("population"), 4).alias("cases_percent"),
            )
            .orderBy(F.desc("cases_percent"), F.asc("iso_code"))
            .limit(15)
        )
        save_small_csv(top15, output_dir / "top15_tsk1.csv")

        peak_window = Window.partitionBy("iso_code").orderBy(F.desc("new_cases"), F.asc("date"))
        top10 = (
            df.filter(F.col("date").between("2021-03-25", "2021-03-31") & F.col("new_cases").isNotNull())
            .withColumn("rank", F.row_number().over(peak_window))
            .filter(F.col("rank") == 1)
            .select(F.col("date"), F.col("location").alias("country"), "new_cases")
            .orderBy(F.desc("new_cases"), F.asc("country"))
            .limit(10)
        )
        save_small_csv(top10, output_dir / "top10_tsk2.csv")

        russian_window = Window.orderBy("date")
        russia = (
            df.filter(F.col("iso_code") == "RUS")
            .withColumn("previous_date", F.lag("date").over(russian_window))
            .withColumn("yesterday_new_cases", F.lag("new_cases").over(russian_window))
            .filter(F.col("date").between("2021-03-25", "2021-03-31"))
            .filter(F.datediff("date", "previous_date") == 1)
            .select(
                "date", "yesterday_new_cases", F.col("new_cases").alias("today_new_cases"),
                (F.col("new_cases") - F.col("yesterday_new_cases")).alias("delta"),
            )
            .orderBy("date")
        )
        save_small_csv(russia, output_dir / "tsk3.csv")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
