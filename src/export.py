"""Save analysis results as CSV / Parquet and generate summary reports."""

import os
import pandas as pd

from .config import StrandConfig, CONFIG


class ResultsExporter:
    """Write DataFrames and text reports to the strand's output directory.

    Usage:
        from src.export import ResultsExporter
        exporter = ResultsExporter(strand_config)
        exporter.save_dataframe(df_seq, "sequence_statistics", "csv")
        exporter.save_summary_report(df_seq)
    """

    def __init__(self, strand_config: StrandConfig):
        self.strand_config = strand_config
        self._prefix = f"[{strand_config.strand_name}]"
        for d in [strand_config.figures_dir, strand_config.reports_dir, strand_config.data_dir]:
            os.makedirs(d, exist_ok=True)

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def save_dataframe(self, df: pd.DataFrame, base_name: str, fmt: str = "csv"):
        """Save df to the strand data directory as CSV or Parquet."""
        filename = self.strand_config.get_output_filename(base_name, fmt)
        local = f"{self.strand_config.data_dir}/{filename}"
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if fmt == "csv":
            df.to_csv(local, index=False)
        elif fmt == "parquet":
            df.to_parquet(local, index=False)
        self._log(f"Saved {fmt.upper()}: {filename} ({len(df)} rows)")

    def save_summary_report(self, df_seq: pd.DataFrame):
        """Generate and save a plain-text summary report."""
        self._log("Generating summary report...")
        thr = CONFIG.ml_stability_threshold_mm
        stable = (df_seq["MOLD_LEVEL_std [mm]"] <= thr).sum()

        lines = [
            f"# {self.strand_config.strand_name} \u2013 Analysis Summary",
            f"Generated: {CONFIG.timestamp}",
            "",
            "=" * 70,
            "",
            "## Overall Statistics",
            f"Total sequences: {len(df_seq)}",
            "",
            "Disturbance distribution:",
        ]
        if "disturbance_type" in df_seq.columns:
            for dtype, cnt in df_seq["disturbance_type"].value_counts().items():
                lines.append(f"  - {dtype}: {cnt} ({100 * cnt / len(df_seq):.1f}%)")
        else:
            for col in ["has_excursion_event", "has_slow_drift", "has_transient_bump", "has_high_variability"]:
                if col in df_seq.columns:
                    cnt = df_seq[col].sum()
                    lines.append(f"  - {col}: {cnt} ({100 * cnt / len(df_seq):.1f}%)")

        lines += [
            "",
            "## Mold Level Stability",
            f"\u03c3 \u2264 {thr} mm: {stable}/{len(df_seq)} ({100 * stable / len(df_seq):.1f}%)",
            f"Mean \u03c3: {df_seq['MOLD_LEVEL_std [mm]'].mean():.2f} mm",
            f"Median \u03c3: {df_seq['MOLD_LEVEL_std [mm]'].median():.2f} mm",
            "",
            "## Process Parameters (mean \u00b1 std)",
        ]
        for col, label in [
            ("CASTING_SPEED_avg [m/min]", "Casting Speed [m/min]"),
            ("MOLD_WIDTH_avg [m]",        "Mold Width [m]"),
            ("SEN_avg [mm]",              "SEN Depth [mm]"),
        ]:
            if col in df_seq.columns:
                std_col = col.replace("avg", "std")
                std_val = df_seq[std_col].std() if std_col in df_seq.columns else 0
                lines.append(f"{label}: {df_seq[col].mean():.3f} \u00b1 {std_val:.3f}")

        text = "\n".join(lines)
        filename = self.strand_config.get_output_filename("summary_report", "txt")
        local = f"{self.strand_config.reports_dir}/{filename}"
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as fh:
            fh.write(text)
        self._log(f"Saved report: {filename}")
        print(f"\n{text}\n")
