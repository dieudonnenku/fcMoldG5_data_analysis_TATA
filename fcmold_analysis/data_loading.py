"""Data loading, timestamp normalisation, unit conversion and preprocessing.

All Spark-dependent helpers are contained here so that the rest of the
package stays framework-agnostic (Pandas / NumPy only).
"""

from typing import List, Tuple

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    StringType,
    TimestampType,
)

from fcmold_analysis.config import AnalysisConfig, StrandConfig, METADATA_PATH


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def add_plain_timestamp(df: SparkDataFrame) -> SparkDataFrame:
    """Ensure *df* has a ``plainTimeStamp`` column of ``TimestampType``."""
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
            df = df.withColumn("plainTimeStamp", F.to_timestamp("dt_timestamp_utc"))
        elif isinstance(dt, TimestampType):
            df = df.withColumn("plainTimeStamp", F.col("dt_timestamp_utc"))
        elif isinstance(dt, (LongType, IntegerType, DoubleType, FloatType)):
            col = F.col("dt_timestamp_utc")
            df = df.withColumn(
                "plainTimeStamp",
                F.from_unixtime(F.when(col > 1e12, col / 1000).otherwise(col)).cast("timestamp"),
            )
        return df

    if "TIMESTAMP" in cols:
        dt = schema["TIMESTAMP"].dataType
        if isinstance(dt, TimestampType):
            df = df.withColumn("plainTimeStamp", F.col("TIMESTAMP"))
        elif isinstance(dt, (LongType, IntegerType, DoubleType, FloatType)):
            col = F.col("TIMESTAMP")
            df = df.withColumn(
                "plainTimeStamp",
                F.from_unixtime(F.when(col > 1e12, col / 1000).otherwise(col)).cast("timestamp"),
            )
        return df

    return df


def get_expert_files(folder_path: str) -> Tuple[List[str], List[str]]:
    """Recursively find boExpert and dtExpert files in a DBFS folder.

    Returns ``(bo_expert_files, dt_expert_files)`` as ``dbfs:/...`` paths.
    Uses ``os.walk`` on ``/dbfs/`` mount to avoid ``dbutils`` dependency.
    """
    import os

    bo_expert_files: List[str] = []
    dt_expert_files: List[str] = []

    # Convert dbfs:/ path to local /dbfs/ mount
    if folder_path.startswith("dbfs:/"):
        local_root = "/dbfs/" + folder_path[len("dbfs:/"):]
    elif folder_path.startswith("/dbfs/"):
        local_root = folder_path
    else:
        local_root = folder_path

    for dirpath, _, filenames in os.walk(local_root):
        for fname in filenames:
            local_full = os.path.join(dirpath, fname)
            # Convert back to dbfs:/ path for Spark
            dbfs_path = "dbfs:/" + local_full[len("/dbfs/"):]
            if "boExpert" in fname:
                bo_expert_files.append(dbfs_path)
            elif "dtExpert" in fname:
                dt_expert_files.append(dbfs_path)

    return bo_expert_files, dt_expert_files


def load_expert_files(file_paths: List[str]) -> SparkDataFrame | None:
    """Load parquet / CSV files and add a ``plainTimeStamp`` column."""
    if not file_paths:
        return None

    parquet_files = [f for f in file_paths if f.lower().endswith(".parquet")]
    csv_files = [f for f in file_paths if f.lower().endswith(".csv")]

    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    if parquet_files:
        df = spark.read.parquet(*parquet_files)
    elif csv_files:
        df = spark.read.option("header", True).option("inferSchema", True).csv(csv_files)
    else:
        return None

    return add_plain_timestamp(df)


def convert_units(df: SparkDataFrame) -> SparkDataFrame:
    """Apply unit conversions in-place on a Spark DataFrame.

    * ``castingSpeed``: m/s → m/min  (×60)
    * ``Mold Level*``: m → mm  (×1000)
    * ``Argon Flow*``: m³/s → L/min  (×60 000)
    """
    conversions = {
        "castingSpeed": 60,
        "Mold Level": 1000,
        "Mold Level Sensor Left": 1000,
        "Mold Level Sensor Right": 1000,
        "Argon Flow SEN": 60000,
        "Argon Flow Stopper": 60000,
    }
    for col_name, factor in conversions.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name) * factor)
    return df


# ---------------------------------------------------------------------------
# StrandDataLoader
# ---------------------------------------------------------------------------

class StrandDataLoader:
    """Load, join, filter and return an analysis-ready Pandas DataFrame for one strand."""

    def __init__(self, strand_config: StrandConfig, analysis_config: AnalysisConfig):
        self.strand_config = strand_config
        self.analysis_config = analysis_config
        self._prefix = f"[{strand_config.strand_name}]"

        for dir_path in [strand_config.figures_dir, strand_config.reports_dir, strand_config.data_dir]:
            try:
                dbutils.fs.mkdirs(dir_path)  # noqa: F821 – Databricks global
            except Exception:
                pass

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    # -- individual steps ---------------------------------------------------

    def load_expert_data(self) -> Tuple[SparkDataFrame, SparkDataFrame]:
        self._log("Loading expert files…")
        bo_files, dt_files = get_expert_files(self.strand_config.data_path)
        self._log(f"  Found {len(bo_files)} boExpert, {len(dt_files)} dtExpert files")

        df_bo = load_expert_files(bo_files)
        df_dt = load_expert_files(dt_files)
        self._log(f"  boExpert: {df_bo.count():,} rows")
        self._log(f"  dtExpert: {df_dt.count():,} rows")
        return df_bo, df_dt

    def aggregate_boexpert(self, df_bo: SparkDataFrame) -> SparkDataFrame:
        self._log("Aggregating boExpert (2 Hz → 1 Hz)…")
        key = "plainTimeStamp"
        numeric = [
            f.name for f in df_bo.schema.fields if isinstance(f.dataType, NumericType) and f.name != key
        ]
        non_numeric = [
            f.name for f in df_bo.schema.fields if not isinstance(f.dataType, NumericType) and f.name != key
        ]
        agg_exprs = [F.avg(c).alias(c) for c in numeric] + [F.first(c).alias(c) for c in non_numeric]
        df_agg = df_bo.groupBy(key).agg(*agg_exprs)
        self._log(f"  Aggregated to {df_agg.count():,} rows")
        return df_agg

    def join_expert_data(self, df_bo_agg: SparkDataFrame, df_dt: SparkDataFrame) -> SparkDataFrame:
        self._log("Joining boExpert ⟷ dtExpert…")
        df_joined = df_bo_agg.alias("bo").join(df_dt.alias("dt"), on="plainTimeStamp", how="inner").cache()
        self._log(f"  Joined: {df_joined.count():,} rows")
        return df_joined

    def join_metadata(self, df_joined: SparkDataFrame) -> SparkDataFrame:
        self._log("Joining with metadata…")
        from pyspark.sql import SparkSession
        _spark = SparkSession.getActiveSession()
        df_meta = _spark.read.csv(METADATA_PATH, header=True, inferSchema=True, sep=";")  # noqa: F821

        df_joined = df_joined.withColumn("PlainTimeStamp", F.col("plainTimeStamp").cast("timestamp"))
        strand_num = int(self.strand_config.strand_id.split("_")[1])

        df_meta = (
            df_meta
            .withColumn("Datetime start first heat", F.col("Datetime start first heat").cast("timestamp"))
            .withColumn("Datetime start last heat", F.col("Datetime start last heat").cast("timestamp"))
            .filter(F.col("Strand ID") == strand_num)
        )
        self._log(f"  Metadata for Strand {strand_num}: {df_meta.count()} periods")

        df_meta_sel = F.broadcast(df_meta.select(
            F.col("Datetime start first heat").cast("timestamp").alias("_meta_start"),
            F.col("Datetime start last heat").cast("timestamp").alias("_meta_end"),
            "Quality casting",
        ))
        cond = (
            (df_joined["PlainTimeStamp"] >= F.col("_meta_start"))
            & (df_joined["PlainTimeStamp"] <= F.col("_meta_end"))
        )
        df_out = df_joined.join(df_meta_sel, on=cond, how="left").drop("_meta_start", "_meta_end")
        total = df_out.count()
        matched = df_out.filter(F.col("Quality casting").isNotNull()).count()
        self._log(f"  After join: {total:,} rows, {matched:,} matched ({100*matched/total:.1f}%)")
        return df_out

    def apply_filters(self, df: SparkDataFrame) -> SparkDataFrame:
        self._log("Applying quality filters…")
        cfg = self.analysis_config
        df_f = (
            df
            .filter(F.col("castingSpeed") >= cfg.min_casting_speed)
            .filter(F.col("SENImmersionDepth").between(cfg.sen_depth_min, cfg.sen_depth_max))
        )
        before = df.count()
        after = df_f.count()
        self._log(f"  {before:,} → {after:,} rows ({100*(before-after)/before:.1f}% removed)")
        return df_f

    def to_pandas(self, df_spark: SparkDataFrame) -> pd.DataFrame:
        self._log("Converting to Pandas…")
        time_col = "PlainTimeStamp" if "PlainTimeStamp" in df_spark.columns else "plainTimeStamp"

        # Feature-engineered columns (added by engineer_features)
        feat_cols = [
            "meniscus_bff_avg", "meniscus_bfl_avg", "meniscus_FF_LF_asymmetry",
            "meniscus_bff_range", "meniscus_bfl_range",
            "cheb_X1_bff", "cheb_X2_bff", "cheb_X3_bff", "cheb_X4_bff",
            "cheb_X1_bfl", "cheb_X2_bfl", "cheb_X3_bfl", "cheb_X4_bfl",
            "abs_cheb_X1_bff", "abs_cheb_X2_bff", "abs_cheb_X1_bfl", "abs_cheb_X2_bfl",
            "cheb_X1_FF_LF_diff", "cheb_X2_FF_LF_diff",
            "EMBR_DC_Bottom_LR_diff", "EMBR_AC_Master_LR_diff", "EMBR_DC_Master_LR_diff",
            "ML_LR_asymmetry", "ML_LR_abs_asymmetry",
        ]
        # Process context
        context_cols = ["SEN_type", "castingLength", "castMode", "superHeat", "tundishTemperature"]

        wanted = [
            time_col, "castingSpeed", "moldWidth", "SENImmersionDepth",
            "Mold Level", "Mold Level Sensor Left", "Mold Level Sensor Right",
            "Argon Flow SEN", "Argon Flow Stopper", "Quality casting",
        ] + self.strand_config.embr_current_cols + feat_cols + context_cols
        available = [c for c in wanted if c in df_spark.columns]

        df_pd = (
            df_spark
            .select(*available)
            .dropna(subset=["castingSpeed", "moldWidth", "SENImmersionDepth"])
            .toPandas()
            .sort_values(time_col)
            .reset_index(drop=True)
        )
        if time_col == "PlainTimeStamp":
            df_pd = df_pd.rename(columns={"PlainTimeStamp": "plainTimeStamp"})

        self._log(f"  {df_pd.shape[0]:,} rows, {df_pd.shape[1]} columns")
        return df_pd

    # -- full pipeline ------------------------------------------------------

    def load_and_process(self, return_spark: bool = False):
        """Run the complete load → join → convert → engineer → filter pipeline.

        Parameters
        ----------
        return_spark : bool
            If True, return ``(df_spark, df_pandas)`` where *df_spark* is the
            feature-engineered Spark DF before Pandas conversion.  This is
            useful for downstream Spark-native work (e.g. persisting to Delta).
        """
        from fcmold_analysis.feature_engineering import engineer_features

        self._log("Starting data-loading pipeline…")
        df_bo, df_dt = self.load_expert_data()
        df_bo_agg = self.aggregate_boexpert(df_bo)
        df_joined = self.join_expert_data(df_bo_agg, df_dt)
        df_meta = self.join_metadata(df_joined)
        self._log("Converting units…")
        df_conv = convert_units(df_meta)
        self._log("Engineering features…")
        df_feat = engineer_features(df_conv)
        df_filt = self.apply_filters(df_feat)
        df_pd = self.to_pandas(df_filt)
        self._log("Data-loading pipeline complete.")
        if return_spark:
            return df_filt, df_pd
        return df_pd
