"""End-to-end analysis pipeline orchestrator.

Combines data loading, sequence analysis, and results export
into a single callable pipeline.
"""

import traceback
from typing import Dict, Any

import pandas as pd

from .config import AnalysisConfig, StrandConfig, STRAND_CONFIGS
from .data_loading import StrandDataLoader
from .sequence_analysis import SequenceAnalyzer
from .export import ResultsExporter


class StrandAnalysisPipeline:
    """Complete analysis pipeline for a single strand.

    Usage:
        from src.config import STRAND_CONFIGS, CONFIG
        from src.pipeline import StrandAnalysisPipeline

        pipeline = StrandAnalysisPipeline(STRAND_CONFIGS["23_6"], CONFIG, spark, dbutils)
        results = pipeline.run()
    """

    def __init__(
        self,
        strand_config: StrandConfig,
        config: AnalysisConfig,
        spark,
        dbutils,
    ):
        self.strand_config = strand_config
        self.config = config
        self.spark = spark
        self.dbutils = dbutils
        self._prefix = f"[{strand_config.strand_name}]"

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def run(self, export_results: bool = True) -> Dict[str, Any]:
        """Execute the full pipeline. Returns dict with results or error info."""
        self._log("Starting pipeline...")
        try:
            # 1. Load data (includes feature engineering)
            loader = StrandDataLoader(
                self.strand_config, self.config, self.spark, self.dbutils
            )
            df_pandas = loader.load_and_process()
            self._log(f"Data loaded: {df_pandas.shape}")

            # 2. Sequence analysis
            analyzer = SequenceAnalyzer(self.strand_config, self.config)
            df_seq, normal_groups, abnormal_groups = analyzer.analyze(df_pandas)
            self._log(f"Sequences: {len(normal_groups)} normal, {len(abnormal_groups)} abnormal")

            # 3. Classify disturbance types
            df_seq = self._classify_disturbances(df_seq)

            # 4. Summary
            n_stable = (df_seq["MOLD_LEVEL_std [mm]"] <= self.config.ml_stability_threshold_mm).sum()
            pct_stable = 100 * n_stable / len(df_seq) if len(df_seq) > 0 else 0
            self._log(f"Stable sequences: {n_stable}/{len(df_seq)} ({pct_stable:.1f}%)")

            # 5. Export
            if export_results:
                exporter = ResultsExporter(self.strand_config)
                exporter.save_dataframe(df_seq, "sequence_statistics", "csv")
                exporter.save_dataframe(df_seq, "sequence_statistics", "parquet")
                exporter.save_summary_report(df_seq)

            self._log("Pipeline complete.")

            return {
                "success": True,
                "strand_name": self.strand_config.strand_name,
                "df_raw": df_pandas,
                "df_seq": df_seq,
                "normal_groups": normal_groups,
                "abnormal_groups": abnormal_groups,
            }

        except Exception as e:
            self._log(f"ERROR: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    @staticmethod
    def _classify_disturbances(df_seq: pd.DataFrame) -> pd.DataFrame:
        """Add human-readable disturbance_type column."""

        def _assign(row):
            flags = []
            if row.get("has_excursion_event", False):
                flags.append("Excursion")
            if row.get("has_slow_drift", False):
                flags.append("Slow drift")
            if row.get("has_transient_bump", False):
                flags.append("Transient bump")
            if row.get("has_high_variability", False):
                flags.append("High variability")
            return "Normal" if not flags else " + ".join(flags)

        df_seq = df_seq.copy()
        df_seq["disturbance_type"] = df_seq.apply(_assign, axis=1)
        return df_seq


def run_all_strands(
    spark,
    dbutils,
    config: AnalysisConfig = None,
    export_results: bool = True,
) -> Dict[str, Any]:
    """Convenience: run the pipeline for every strand defined in STRAND_CONFIGS.

    Usage:
        from src.pipeline import run_all_strands
        results = run_all_strands(spark, dbutils)
    """
    if config is None:
        from .config import CONFIG
        config = CONFIG

    all_results: Dict[str, Any] = {}
    for strand_id, strand_cfg in STRAND_CONFIGS.items():
        pipeline = StrandAnalysisPipeline(strand_cfg, config, spark, dbutils)
        all_results[strand_id] = pipeline.run(export_results=export_results)

    print(f"\n{'=' * 70}")
    print("MULTI-STRAND SUMMARY")
    print(f"{'=' * 70}\n")
    for sid, res in all_results.items():
        status = "SUCCESS" if res["success"] else "FAILED"
        print(f"  {status} \u2013 {res.get('strand_name', sid)}")
        if res["success"]:
            n = len(res["df_seq"])
            print(f"    Sequences: {n}")
        else:
            print(f"    Error: {res['error']}")
    return all_results
