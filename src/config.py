"""Configuration classes for FC Mold G5 analysis pipeline.

Contains all tunable parameters and strand-specific settings.
"""

from dataclasses import dataclass
from typing import List
from datetime import datetime


@dataclass
class AnalysisConfig:
    """Global analysis parameters for mold level stability detection."""

    # Sequence identification
    window_size: int = 300  # samples (~6 min at 1Hz)
    vc_threshold: float = 0.1  # m/min
    curr_threshold: float = 50.0  # A
    max_gap_seconds: int = 5  # max gap before new segment

    # Data filtering
    min_casting_speed: float = 0.5  # m/min
    sen_depth_min: float = 0.1  # m
    sen_depth_max: float = 0.26  # m

    # Disturbance detection
    excursion_threshold_mm: float = 8.0
    excursion_min_duration_s: float = 5.0
    slow_drift_min_run_s: float = 60.0
    slow_drift_min_amplitude_mm: float = 10.0
    transient_bump_k_amp: float = 8.0
    transient_bump_min_mm: float = 6.0
    high_var_ptp_threshold_mm: float = 10.0
    high_var_band_mm: float = 4.0
    high_var_fraction_threshold: float = 0.1

    # Mold level stability
    ml_stability_threshold_mm: float = 2.0  # sigma threshold for "stable"

    # Visualization
    viz_sample_fraction: float = 0.25

    # Output
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def __repr__(self):
        return (
            f"AnalysisConfig(window={self.window_size}, "
            f"Vc_thr={self.vc_threshold}, "
            f"ML_thr={self.ml_stability_threshold_mm}mm)"
        )


@dataclass
class StrandConfig:
    """Configuration for a specific casting strand."""

    strand_id: str  # e.g., "23_5" or "23_6"
    strand_name: str  # e.g., "Strand 23-5"
    data_path: str  # DBFS path to strand data
    embr_current_cols: List[str]
    vc_column: str = "castingSpeed"
    output_base: str = "/dbfs/FileStore/Results/FCMold/TATA_IJmuiden_CC23"

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
# Default instances
# ---------------------------------------------------------------------------
CONFIG = AnalysisConfig()

STRAND_CONFIGS = {
    "23_6": StrandConfig(
        strand_id="23_6",
        strand_name="Strand 23-6",
        data_path="dbfs:/FileStore/Data/FCMold/TATA_IJmuiden_CC23/data/strand_6",
        embr_current_cols=[
            "EMBR Current AC Left Master",
            "EMBR Current DC Left Master",
            "EMBR Current DC Left Bottom",
        ],
    ),
    "23_5": StrandConfig(
        strand_id="23_5",
        strand_name="Strand 23-5",
        data_path="dbfs:/FileStore/Data/FCMold/TATA_IJmuiden_CC23/data/strand_5",
        embr_current_cols=[
            "EMBR Current AC Left Master",
            "EMBR Current DC Left Master",
            "EMBR Current DC Left Bottom",
        ],
    ),
}

# ---------------------------------------------------------------------------
# Path constants - CHANGE THESE WHEN SETTING UP A NEW ENVIRONMENT
# ---------------------------------------------------------------------------
# WORKSPACE_ROOT: where notebooks and src/ live (editable project folder)
WORKSPACE_ROOT = "/Workspace/Users/dieudonne.nkulikiyimfura@se.abb.com/fcMoldG5_data_analysis_TATA"

# DBFS_DATA_BASE: where raw sensor data (boExpert/dtExpert parquet files) are stored
DBFS_DATA_BASE = "dbfs:/FileStore/Data/FCMold/TATA_IJmuiden_CC23/data"

# DBFS_OUTPUT_BASE: where generated figures, reports, and processed data go
DBFS_OUTPUT_BASE = "/dbfs/FileStore/Results/FCMold/TATA_IJmuiden_CC23"

# Metadata and reference files
METADATA_PATH = f"{DBFS_DATA_BASE}/Castings_TSN_2025_April_May_merged.csv"
GRADE_MAPPING_PATH = f"{DBFS_OUTPUT_BASE}/CastingGroups_ABB_April2026.xlsx"
