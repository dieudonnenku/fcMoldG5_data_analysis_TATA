"""Save analysis results as CSV / Parquet and generate summary reports."""

import pandas as pd

from fcmold_analysis.config import AnalysisConfig, StrandConfig, CONFIG


class ResultsExporter:
    """Write DataFrames and text reports to the strand's output directory."""

    def __init__(self, strand_config: StrandConfig):
        self.strand_config = strand_config
        self._prefix = f"[{strand_config.strand_name}]"
        for d in [strand_config.figures_dir, strand_config.reports_dir, strand_config.data_dir]:
            try:
                dbutils.fs.mkdirs(d)  # noqa: F821 – Databricks global
            except Exception:
                pass

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def save_dataframe(self, df: pd.DataFrame, base_name: str, fmt: str = "csv"):
        filename = self.strand_config.get_output_filename(base_name, fmt)
        local = f"{self.strand_config.data_dir}/{filename}".replace("/dbfs", "")
        if fmt == "csv":
            df.to_csv(local, index=False)
        elif fmt == "parquet":
            df.to_parquet(local, index=False)
        self._log(f"Saved {fmt.upper()}: {filename} ({len(df)} rows)")

    def save_summary_report(self, df_seq: pd.DataFrame):
        self._log("Generating summary report…")
        lines = [
            f"# {self.strand_config.strand_name} – Analysis Summary",
            f"Generated: {CONFIG.timestamp}",
            "",
            "=" * 70,
            "",
            "## Overall Statistics",
            f"Total sequences: {len(df_seq)}",
            "",
            "Disturbance distribution:",
        ]
        for dtype, cnt in df_seq["disturbance_type"].value_counts().items():
            lines.append(f"  - {dtype}: {cnt} ({100*cnt/len(df_seq):.1f}%)")

        thr = CONFIG.ml_stability_threshold_mm
        stable = (df_seq["MOLD_LEVEL_std [mm]"] <= thr).sum()
        lines += [
            "",
            "## Mold Level Stability",
            f"σ ≤ {thr} mm: {stable}/{len(df_seq)} ({100*stable/len(df_seq):.1f}%)",
            f"Mean σ: {df_seq['MOLD_LEVEL_std [mm]'].mean():.2f} mm",
            f"Median σ: {df_seq['MOLD_LEVEL_std [mm]'].median():.2f} mm",
            "",
            "## Process Parameters (mean ± std)",
            f"Casting Speed: {df_seq['CASTING_SPEED_avg [m/min]'].mean():.2f} ± {df_seq['CASTING_SPEED_avg [m/min]'].std():.2f} m/min",
            f"Mold Width: {df_seq['MOLD_WIDTH_avg [m]'].mean():.3f} ± {df_seq['MOLD_WIDTH_avg [m]'].std():.3f} m",
            f"SEN Depth: {df_seq['SEN_avg [mm]'].mean():.1f} ± {df_seq['SEN_avg [mm]'].std():.1f} mm",
        ]
        text = "\n".join(lines)

        filename = self.strand_config.get_output_filename("summary_report", "txt")
        local = f"{self.strand_config.reports_dir}/{filename}".replace("/dbfs", "")
        with open(local, "w", encoding="utf-8") as fh:
            fh.write(text)
        self._log(f"Saved report: {filename}")
        print(f"\n{text}\n")
