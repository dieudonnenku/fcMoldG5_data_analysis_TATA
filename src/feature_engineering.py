"""Feature engineering for FC Mold G5 analysis.

Derives physics-based features from raw sensor data:
- Meniscus temperature aggregates and asymmetry (FBG sensors)
- Chebyshev polynomial coefficients (meniscus flatness)
- EMBR current left-right imbalance
- Mold level left-right asymmetry

All operations are Spark-native (SparkDataFrame in, SparkDataFrame out).
"""

import functools

from pyspark.sql import functions as F


def engineer_features(df):
    """Add 21 derived features to a Spark DataFrame containing raw sensor data.

    Parameters
    ----------
    df : SparkDataFrame
        Must contain meniscus FBG columns (meniscus_bff_c01–c20,
        meniscus_bfl_c01–c20), Chebyshev flatness columns
        (meniscusFlatness_bff_1–4, meniscusFlatness_bfl_1–4),
        EMBR current columns, and mold level sensor columns.
        Columns that are absent are silently skipped.

    Returns
    -------
    SparkDataFrame
        Original columns plus up to 21 derived features.
    """
    # =========================================================================
    # 1. Meniscus temperature — row-wise averages across 20 FBG sensors
    # =========================================================================
    bff_cols = [f"meniscus_bff_c{i:02d}" for i in range(1, 21)]
    bfl_cols = [f"meniscus_bfl_c{i:02d}" for i in range(1, 21)]

    bff_present = [c for c in bff_cols if c in df.columns]
    bfl_present = [c for c in bfl_cols if c in df.columns]

    def _row_mean(cols):
        """Null-safe row-wise mean over cols."""
        s = functools.reduce(lambda a, b: a + b,
                             [F.coalesce(F.col(c), F.lit(0)) for c in cols])
        cnt = functools.reduce(lambda a, b: a + b,
                               [F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in cols])
        return F.when(cnt > 0, s / cnt)

    out = df
    if bff_present:
        out = out.withColumn("meniscus_bff_avg", _row_mean(bff_present))
        out = out.withColumn(
            "meniscus_bff_range",
            F.greatest(*[F.col(c) for c in bff_present])
            - F.least(*[F.col(c) for c in bff_present]),
        )
    if bfl_present:
        out = out.withColumn("meniscus_bfl_avg", _row_mean(bfl_present))
        out = out.withColumn(
            "meniscus_bfl_range",
            F.greatest(*[F.col(c) for c in bfl_present])
            - F.least(*[F.col(c) for c in bfl_present]),
        )
    if bff_present and bfl_present:
        out = out.withColumn(
            "meniscus_FF_LF_asymmetry",
            F.col("meniscus_bff_avg") - F.col("meniscus_bfl_avg"),
        )

    # =========================================================================
    # 2. Chebyshev polynomial coefficients — alias + derived cross-face diffs
    # =========================================================================
    cheb_map = {
        "cheb_X1_bff": "meniscusFlatness_bff_1",
        "cheb_X2_bff": "meniscusFlatness_bff_2",
        "cheb_X3_bff": "meniscusFlatness_bff_3",
        "cheb_X4_bff": "meniscusFlatness_bff_4",
        "cheb_X1_bfl": "meniscusFlatness_bfl_1",
        "cheb_X2_bfl": "meniscusFlatness_bfl_2",
        "cheb_X3_bfl": "meniscusFlatness_bfl_3",
        "cheb_X4_bfl": "meniscusFlatness_bfl_4",
    }
    for alias, src in cheb_map.items():
        if src in out.columns:
            out = out.withColumn(alias, F.col(src))

    # Absolute values
    for base in ["cheb_X1_bff", "cheb_X2_bff", "cheb_X1_bfl", "cheb_X2_bfl"]:
        if base in out.columns:
            out = out.withColumn(f"abs_{base}", F.abs(F.col(base)))

    # Fixed-face vs loose-face differences
    for order in [1, 2]:
        bff_col = f"cheb_X{order}_bff"
        bfl_col = f"cheb_X{order}_bfl"
        if bff_col in out.columns and bfl_col in out.columns:
            out = out.withColumn(f"cheb_X{order}_FF_LF_diff", F.col(bff_col) - F.col(bfl_col))

    # =========================================================================
    # 3. EMBR current left-right imbalance
    # =========================================================================
    embr_pairs = [
        ("EMBR_DC_Bottom_LR_diff",  "EMBR Current DC Left Bottom",  "EMBR Current DC Right Bottom"),
        ("EMBR_AC_Master_LR_diff",  "EMBR Current AC Left Master",  "EMBR Current AC Right Master"),
        ("EMBR_DC_Master_LR_diff",  "EMBR Current DC Left Master",  "EMBR Current DC Right Master"),
    ]
    for name, left, right in embr_pairs:
        if left in out.columns and right in out.columns:
            out = out.withColumn(name, F.col(left) - F.col(right))

    # =========================================================================
    # 4. Mold level left-right asymmetry
    # =========================================================================
    ml_left = "Mold Level Sensor Left"
    ml_right = "Mold Level Sensor Right"
    if ml_left in out.columns and ml_right in out.columns:
        out = (
            out
            .withColumn("ML_LR_asymmetry", F.col(ml_left) - F.col(ml_right))
            .withColumn("ML_LR_abs_asymmetry", F.abs(F.col(ml_left) - F.col(ml_right)))
        )

    # =========================================================================
    # 5. Normalise timestamp column name to lowercase
    # =========================================================================
    if "PlainTimeStamp" in out.columns and "plainTimeStamp" not in out.columns:
        out = out.withColumnRenamed("PlainTimeStamp", "plainTimeStamp")

    return out
