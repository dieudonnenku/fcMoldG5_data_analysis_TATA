"""Configuration dataclasses and strand definitions for FC Mold G5 analysis."""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class AnalysisConfig:
    """Global analysis parameters for mold level stability detection."""

    # Sequence identification parameters
    window_size: int = 300  # samples (~6 min at 1Hz)
    vc_threshold: float = 0.1  # m/min - casting speed stability threshold
    curr_threshold: float = 50.0  # A - EMBR current stability threshold
    max_gap_seconds: int = 5  # seconds - max gap before new segment

    # Data filtering thresholds
    min_casting_speed: float = 0.5  # m/min
    sen_depth_min: float = 0.1  # m
    sen_depth_max: float = 0.26  # m

    # Disturbance detection parameters
    excursion_threshold_mm: float = 8.0  # mm deviation from baseline
    excursion_min_duration_s: float = 5.0  # seconds
    slow_drift_min_run_s: float = 60.0  # seconds
    slow_drift_min_amplitude_mm: float = 10.0  # mm
    transient_bump_k_amp: float = 8.0  # multiplier for noise sigma
    transient_bump_min_mm: float = 6.0  # mm absolute minimum
    high_var_ptp_threshold_mm: float = 10.0  # peak-to-peak range
    high_var_band_mm: float = 4.0  # ±mm around baseline
    high_var_fraction_threshold: float = 0.1  # 10% outside band

    # Mold level stability threshold
    ml_stability_threshold_mm: float = 2.0  # σ threshold for "stable"

    # Sampling for visualization
    viz_sample_fraction: float = 0.25  # 25% sample for large datasets

    # Output settings
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __repr__(self):
        return (
            f"AnalysisConfig(window={self.window_size}, "
            f"Vc_thr={self.vc_threshold}, "
            f"ML_thr={self.ml_stability_threshold_mm}mm)"
        )


# Module-level singleton — import once, use everywhere
CONFIG = AnalysisConfig()


@dataclass
class StrandConfig:
    """Configuration for a specific casting strand."""

    strand_id: str  # e.g., "23_5" or "23_6"
    strand_name: str  # e.g., "Strand 23-5"
    data_path: str  # DBFS path to strand data

    # EMBR current columns (strand-specific naming)
    embr_current_cols: List[str] = field(default_factory=list)

    # Casting speed column name
    vc_column: str = "castingSpeed"

    # Output directory
    output_base: str = "/dbfs/FileStore/TATAIjmulden_FCMoldG5"

    @property
    def output_dir(self) -> str:
        return f"{self.output_base}/strand_{self.strand_id}"

    @property
    def figures_dir(self) -> str:
        return f"{self.output_dir}/figures"

    @property
    def reports_dir(self) -> str:
        return f"{self.output_dir}/reports"

    @property
    def data_dir(self) -> str:
        return f"{self.output_dir}/processed_data"

    def get_output_filename(self, base_name: str, extension: str = "html") -> str:
        return f"{base_name}_strand{self.strand_id}_{CONFIG.timestamp}.{extension}"

    def __repr__(self):
        return f"StrandConfig({self.strand_name}, path={self.data_path})"


# ---------------------------------------------------------------------------
# Strand definitions
# ---------------------------------------------------------------------------
STRAND_CONFIGS = {
    "23_6": StrandConfig(
        strand_id="23_6",
        strand_name="Strand 23-6",
        data_path="dbfs:/FileStore/TATA_IJmuiden_CC23/data/strand_6",
        embr_current_cols=[
            "EMBR Current AC Left Master",
            "EMBR Current DC Left Master",
            "EMBR Current DC Left Bottom",
            "EMBR Current AC Right Master",
            "EMBR Current DC Right Master",
            "EMBR Current DC Right Bottom",
        ],
    ),
    "23_5": StrandConfig(
        strand_id="23_5",
        strand_name="Strand 23-5",
        data_path="dbfs:/FileStore/TATA_IJmuiden_CC23/data/strand_5",
        embr_current_cols=[
            "EMBR Current AC Left Master",
            "EMBR Current DC Left Master",
            "EMBR Current DC Left Bottom",
            "EMBR Current AC Right Master",
            "EMBR Current DC Right Master",
            "EMBR Current DC Right Bottom",
        ],
    ),
}

# Metadata path (shared across strands)
METADATA_PATH = "dbfs:/FileStore/TATA_IJmuiden_CC23/data/Castings_TSN_2025_April_May_merged.csv"
