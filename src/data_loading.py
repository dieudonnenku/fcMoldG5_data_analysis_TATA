"""Data loading and preprocessing for CC23 sensor data.

Handles boExpert/dtExpert file discovery, timestamp normalization,
unit conversions, and metadata joining.

NOTE: This module uses `spark` and `dbutils` which are available in the
Databricks notebook environment. Import this module from a notebook cell.
"""

import pandas as pd
from typing import List, Tuple

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, TimestampType, LongType, IntegerType, DoubleType, FloatType, NumericType,
)

from .config import AnalysisConfig, StrandConfig, METADATA_PATH


def add_plain_timestamp(df):
    """Ensure df has a 'plainTimeStamp' column of TimestampType."""
    schema = df.schema
    cols = df.columns

    if "plainTimeStamp" in cols:
        dt = schema["plainTimeStamp"].dataType
        if isinstance(dt, StringType):
            df = df.withColumn("plainTimeStamp", F.to_timestamp("plainTimeStamp"))
        return df

    if "dt_timestamp_utc" in cols:
        dt = schema["dt_timestamp_utc"].dataType
        if isinstance(dt, StringType):
            return df.withColumn("plainTimeStamp", F.to_timestamp("dt_timestamp_utc"))
        elif isinstance(dt, TimestampType):
            return df.withColumn("plainTimeStamp", F.col("dt_timestamp_utc"))
        elif isinstance(dt, (LongType, IntegerType, DoubleType, FloatType)):
            col = F.col("dt_timestamp_utc")
            return df.withColumn(
                "plainTimeStamp",
                F.from_unixtime(F.when(col > 1e12, col / 1000).otherwise(col)).cast("timestamp"),
            )

    if "TIMESTAMP" in cols:
        dt = schema["TIMESTAMP"].dataType
        if isinstance(dt, TimestampType):
            return df.withColumn("plainTimeStamp", F.col("TIMESTAMP"))
        elif isinstance(dt, (LongType, IntegerType, DoubleType, FloatType)):
            col = F.col("TIMESTAMP")
            return df.withColumn(
                "plainTimeStamp",
                F.from_unixtime(F.when(col > 1e12, col / 1000).otherwise(col)).cast("timestamp"),
            )

    return df


def get_expert_files(folder_path: str) -> Tuple[List[str], List[str]]:
    """Recursively find boExpert and dtExpert files in a DBFS folder."""
    # NOTE: requires dbutils in scope (Databricks runtime)
    from pyspark.dbutils import DBUtils  # noqa: F401
    import importlib, types

    # Get dbutils from the active Spark session
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        dbutils = DBUtils(spark)
    except Exception:
        # Fallback: assume global dbutils exists (notebook context)
        import builtins
        dbutils = getattr(builtins, "dbutils", None)
        if dbutils is None:
            raise RuntimeError("dbutils not available. Run from a Databricks notebook.")

    bo_files, dt_files = [], []

    def walk(path: str):
        for fi in dbutils.fs.ls(path):
            if fi.path.endswith("/"):
                walk(fi.path)
            elif "boExpert" in fi.name:
                bo_files.append(fi.path)
            elif "dtExpert" in fi.name:
                dt_files.append(fi.path)

    walk(folder_path)
    return bo_files, dt_files


def load_expert_files(file_paths: List[str], spark):
    """Load parquet/CSV files into a Spark DataFrame with plainTimeStamp."""
    if not file_paths:
        return None

    parquet_files = [f for f in file_paths if f.lower().endswith(".parquet")]
    csv_files = [f for f in file_paths if f.lower().endswith(".csv")]

    if parquet_files:
        df = spark.read.parquet(*parquet_files)
    elif csv_files:
        df = spark.read.option("header", True).option("inferSchema", True).csv(csv_files)
    else:
        return None

    return add_plain_timestamp(df)


def convert_units(df):
    """Convert raw sensor units to physical quantities."""
    conversions = {
        "castingSpeed": 60,         # m/s -> m/min
        "Mold Level": 1000,         # m -> mm
        "Mold Level Sensor Left": 1000,
        "Mold Level Sensor Right": 1000,
        "Argon Flow SEN": 60000,    # m^3/s -> L/min
        "Argon Flow Stopper": 60000,
    }
    for col, factor in conversions.items():
        if col in df.columns:
            df = df.withColumn(col, F.col(col) * factor)
    return df


class StrandDataLoader:
    """End-to-end data loading pipeline for a single strand.

    Usage (in notebook):
        from src.config import STRAND_CONFIGS, CONFIG
        from src.data_loading import StrandDataLoader

        loader = StrandDataLoader(STRAND_CONFIGS["23_6"], CONFIG, spark, dbutils)
        df_pandas = loader.load_and_process()
    """

    def __init__(self, strand_config: StrandConfig, config: AnalysisConfig, spark, dbutils):
        self.strand_config = strand_config
        self.config = config
        self.spark = spark
        self.dbutils = dbutils
        self._prefix = f"[{strand_config.strand_name}]"

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def load_expert_data(self):
        """Load boExpert and dtExpert from DBFS."""
        self._log("Loading expert files...")
        bo_files, dt_files = get_expert_files(self.strand_config.data_path)
        self._log(f"  Found {len(bo_files)} boExpert, {len(dt_files)} dtExpert files")

        df_bo = load_expert_files(bo_files, self.spark)
        df_dt = load_expert_files(dt_files, self.spark)
        return df_bo, df_dt

    def aggregate_boexpert(self, df_bo):
        """Aggregate boExpert from 2Hz to 1Hz (mean numerics, first categoricals)."""
        key_col = "plainTimeStamp"
        numeric_cols = [
            f.name for f in df_bo.schema.fields
            if isinstance(f.dataType, NumericType) and f.name != key_col
        ]
        non_numeric_cols = [
            f.name for f in df_bo.schema.fields
            if not isinstance(f.dataType, NumericType) and f.name != key_col
        ]
        agg_exprs = [F.avg(c).alias(c) for c in numeric_cols]
        agg_exprs += [F.first(c).alias(c) for c in non_numeric_cols]
        return df_bo.groupBy(key_col).agg(*agg_exprs)

    def join_expert_data(self, df_bo_agg, df_dt):
        """Inner join on plainTimeStamp."""
        df = df_bo_agg.alias("bo").join(df_dt.alias("dt"), on="plainTimeStamp", how="inner")
        return df.cache()

    def join_metadata(self, df_joined):
        """Range-join with casting metadata (filtered by strand ID)."""
        df_meta = self.spark.read.csv(METADATA_PATH, header=True, inferSchema=True, sep=";")
        strand_num = int(self.strand_config.strand_id.split("_")[1])

        df_meta = (
            df_meta
            .withColumn("ts_start", F.col("Datetime start first heat").cast("timestamp"))
            .withColumn("ts_end", F.col("Datetime start last heat").cast("timestamp"))
            .filter(F.col("Strand ID") == strand_num)
        )

        join_cond = (
            (df_joined["plainTimeStamp"] >= df_meta["ts_start"])
            & (df_joined["plainTimeStamp"] <= df_meta["ts_end"])
        )
        return df_joined.join(
            df_meta.select("ts_start", "ts_end", "Quality casting"),
            on=join_cond, how="left",
        ).drop("ts_start", "ts_end")

    def apply_filters(self, df):
        """Remove non-casting data."""
        return (
            df
            .filter(F.col("castingSpeed") >= self.config.min_casting_speed)
            .filter(F.col("SENImmersionDepth").between(self.config.sen_depth_min, self.config.sen_depth_max))
        )

    def to_pandas(self, df_spark) -> pd.DataFrame:
        """Convert to pandas, sort by time."""
        tcol = "PlainTimeStamp" if "PlainTimeStamp" in df_spark.columns else "plainTimeStamp"
        cols = [
            tcol, "castingSpeed", "moldWidth", "SENImmersionDepth",
            "Mold Level", "Mold Level Sensor Left", "Mold Level Sensor Right",
            "Argon Flow SEN", "Argon Flow Stopper", "Quality casting",
        ] + self.strand_config.embr_current_cols
        cols = [c for c in cols if c in df_spark.columns]

        df_pd = (
            df_spark.select(*cols)
            .dropna(subset=["castingSpeed", "moldWidth", "SENImmersionDepth"])
            .toPandas()
            .sort_values(tcol)
            .reset_index(drop=True)
        )
        if tcol == "PlainTimeStamp":
            df_pd = df_pd.rename(columns={"PlainTimeStamp": "plainTimeStamp"})
        return df_pd

    def load_and_process(self) -> pd.DataFrame:
        """Full pipeline: load -> aggregate -> join -> convert -> filter -> pandas."""
        df_bo, df_dt = self.load_expert_data()
        df_bo_agg = self.aggregate_boexpert(df_bo)
        df_joined = self.join_expert_data(df_bo_agg, df_dt)
        df_joined = convert_units(df_joined)
        df_joined = self.join_metadata(df_joined)
        df_filtered = self.apply_filters(df_joined)
        return self.to_pandas(df_filtered)
