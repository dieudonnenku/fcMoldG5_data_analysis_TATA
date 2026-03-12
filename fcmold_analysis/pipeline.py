"""End-to-end analysis pipeline for a single strand."""

import traceback
from typing import Any, Dict

from fcmold_analysis.config import AnalysisConfig, StrandConfig, STRAND_CONFIGS
from fcmold_analysis.data_loading import StrandDataLoader
from fcmold_analysis.sequence_analysis import SequenceAnalyzer
from fcmold_analysis.visualization import VisualizationFactory
from fcmold_analysis.export import ResultsExporter


class StrandAnalysisPipeline:
    """Orchestrate load → analyse → visualise → export for one strand.

    Usage::

        pipeline = StrandAnalysisPipeline(STRAND_CONFIGS["23_6"], CONFIG)
        results = pipeline.run()
    """

    def __init__(self, strand_config: StrandConfig, analysis_config: AnalysisConfig):
        self.strand_config = strand_config
        self.config = analysis_config
        self._prefix = f"[{strand_config.strand_name}]"
        self.loader = StrandDataLoader(strand_config, analysis_config)
        self.analyzer = SequenceAnalyzer(strand_config, analysis_config)
        self.visualizer = VisualizationFactory(strand_config, analysis_config)
        self.exporter = ResultsExporter(strand_config)

    def _log(self, msg: str):
        print(f"\n{self._prefix} {msg}")

    def run(self, generate_visualizations: bool = True) -> Dict[str, Any]:
        self._log("=" * 60)
        self._log(f"STARTING PIPELINE – {self.strand_config.strand_name}")
        self._log("=" * 60)

        results: Dict[str, Any] = {
            "strand_id": self.strand_config.strand_id,
            "strand_name": self.strand_config.strand_name,
            "success": False,
            "error": None,
        }

        try:
            # 1 — Data loading
            self._log("STEP 1: Data Loading & Preprocessing")
            df_pandas = self.loader.load_and_process()
            results["df_raw"] = df_pandas
            results["raw_row_count"] = len(df_pandas)

            # 2 — Sequence analysis
            self._log("STEP 2: Sequence Analysis & Disturbance Detection")
            df_fc_mold, df_seq, normal_groups, abnormal_groups = self.analyzer.analyze(df_pandas)
            results.update(
                df_fc_mold=df_fc_mold,
                df_seq=df_seq,
                normal_groups=normal_groups,
                abnormal_groups=abnormal_groups,
                sequence_count=len(df_seq),
            )

            # 3 — Export
            self._log("STEP 3: Exporting Results")
            self.exporter.save_dataframe(df_seq, "sequence_statistics", "csv")
            self.exporter.save_dataframe(df_seq, "sequence_statistics", "parquet")
            self.exporter.save_summary_report(df_seq)

            # 4 — Visualizations
            if generate_visualizations:
                self._log("STEP 4: Generating Visualizations")
                example_indices = {
                    "Normal": min(127, len(normal_groups) - 1) if normal_groups else None,
                }
                for label, substr in [("High variability", "High variability"), ("Excursion", "Excursion")]:
                    matches = df_seq[df_seq["disturbance_type"].str.contains(substr, na=False)]
                    if len(matches):
                        example_indices[label] = matches.index[0]
                example_indices = {k: v for k, v in example_indices.items() if v is not None}

                if example_indices:
                    self.visualizer.plot_disturbance_examples(df_fc_mold, df_seq, normal_groups, example_indices)
                self.visualizer.plot_correlation_scatter(df_seq, filter_normal_only=False)
                self.visualizer.plot_correlation_scatter(df_seq, filter_normal_only=True)
                self.visualizer.plot_correlation_heatmap(df_seq)
                self.visualizer.plot_parameter_correlations(df_seq)

            results["success"] = True
            self._log("=" * 60)
            self._log(f"COMPLETE – {self.strand_config.strand_name}")
            self._log("=" * 60)

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            self._log(f"ERROR: {e}")
            print(traceback.format_exc())

        return results


def run_all_strands(config: AnalysisConfig | None = None, visualize: bool = True) -> Dict[str, Any]:
    """Convenience: run the pipeline for every strand in ``STRAND_CONFIGS``."""
    if config is None:
        from fcmold_analysis.config import CONFIG
        config = CONFIG

    all_results: Dict[str, Any] = {}
    for strand_id, strand_cfg in STRAND_CONFIGS.items():
        pipeline = StrandAnalysisPipeline(strand_cfg, config)
        all_results[strand_id] = pipeline.run(generate_visualizations=visualize)

    # Summary
    print(f"\n{'=' * 70}")
    print("MULTI-STRAND SUMMARY")
    print(f"{'=' * 70}\n")
    for sid, res in all_results.items():
        status = "SUCCESS" if res["success"] else "FAILED"
        print(f"  {status} – {res['strand_name']}")
        if res["success"]:
            print(f"    Sequences: {res['sequence_count']}")
        else:
            print(f"    Error: {res['error']}")
    return all_results
