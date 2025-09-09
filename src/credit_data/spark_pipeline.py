from __future__ import annotations

import os
from typing import Any

SparkSession: Any = None
Fspark: Any = None
try:
    from pyspark.sql import SparkSession as _SparkSession  # noqa: F401
    from pyspark.sql import functions as _F  # noqa: F401

    SparkSession = _SparkSession
    Fspark = _F
except Exception:  # pragma: no cover
    pass


def run_pipeline(input_dir: str, output_dir: str) -> None:
    if SparkSession is None:
        raise RuntimeError(
            "PySpark not installed. Install pyspark to run this pipeline."
        )
    spark = SparkSession.builder.appName("credit_data_pipeline").getOrCreate()
    try:
        df = spark.read.parquet(os.path.join(input_dir, "loan_monthly_*.parquet"))
        agg = (
            df.groupBy("asof_month")
            .agg(Fspark.sum("balance_ead").alias("portfolio_ead"))
            .orderBy("asof_month")
        )
        os.makedirs(output_dir, exist_ok=True)
        agg.write.mode("overwrite").parquet(
            os.path.join(output_dir, "portfolio_ead_by_month.parquet")
        )
    finally:
        spark.stop()
