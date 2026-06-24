"""Sequence identification and statistical analysis.

Core algorithms for identifying stable casting sequences using
sliding-window analysis with temporal segmentation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from .config import AnalysisConfig, StrandConfig
from .disturbance_detection import (
    detect_excursion_event_robust,
    detect_slow_drift_event,
    detect_transient_bump_dynamic,
    detect_high_variability_event,
)


def custom_rounding(series, round_up: bool = True) -> list:
    """Round values with hysteresis to avoid jitter between adjacent values."""
    rounded_values = []
    for i in range(len(series)):
        if i > 0 and abs(series[i] - series[i - 1]) <= 0.01:
            rounded_value = rounded_values[-1]
        else:
            if round_up:
                rounded_value = round(series[i] * 100 + 0.5) / 100
            else:
                rounded_value = round(series[i] * 100 - 0.5) / 100
        rounded_values.append(rounded_value)
    return rounded_values


def segment_by_time_gaps(
    df: pd.DataFrame, tcol: str = "plainTimeStamp", max_gap_seconds: int = 5
) -> pd.DataFrame:
    """Split time-series into continuous segments based on time gaps > max_gap_seconds."""
    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol)

    time_diffs = df[tcol].diff().dt.total_seconds().fillna(0)
    df["segment_id"] = (time_diffs > max_gap_seconds).astype(int).cumsum()
    return df


def identify_sequences(
    df: pd.DataFrame,
    Vc_column: str,
    window_size: int,
    Vc_threshold: float,
    Curr_columns: List[str] = None,
    Curr_threshold: float = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Sliding window classifier: NORMAL windows skip by window_size, ABNORMAL skip by 1."""
    normal_groups = []
    abnormal_groups = []

    i = 0
    n = len(df)

    while i <= n - window_size:
        window_df = df.iloc[i : i + window_size]
        window_idx = window_df.index

        Vc_window = window_df[Vc_column]
        cond_vc = np.all(np.abs(Vc_window - Vc_window.iloc[0]) <= Vc_threshold)

        cond_curr = True
        if cond_vc and Curr_columns and Curr_threshold is not None:
            for col in Curr_columns:
                if col not in window_df.columns:
                    cond_curr = False
                    break
                if np.any(np.abs(window_df[col].diff().dropna()) >= Curr_threshold):
                    cond_curr = False
                    break

        if cond_vc and cond_curr:
            normal_groups.append(window_idx.tolist())
            i += window_size
        else:
            abnormal_groups.append(window_idx.tolist())
            i += 1

    return normal_groups, abnormal_groups


def identify_sequences_segmented(
    df: pd.DataFrame,
    Vc_column: str,
    window_size: int,
    Vc_threshold: float,
    Curr_columns: List[str] = None,
    Curr_threshold: float = None,
    tcol: str = "plainTimeStamp",
    max_gap_seconds: int = 5,
    min_segment_len: int = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Run identify_sequences within continuous time segments (prevents crossing gaps)."""
    if min_segment_len is None:
        min_segment_len = window_size

    df_seg = segment_by_time_gaps(df, tcol=tcol, max_gap_seconds=max_gap_seconds)

    normal_all, abnormal_all = [], []
    for _, seg_df in df_seg.groupby("segment_id"):
        if len(seg_df) < min_segment_len:
            continue
        n_g, a_g = identify_sequences(
            seg_df, Vc_column, window_size, Vc_threshold, Curr_columns, Curr_threshold
        )
        normal_all.extend(n_g)
        abnormal_all.extend(a_g)

    return normal_all, abnormal_all


class SequenceAnalyzer:
    """Encapsulates sequence identification, disturbance detection, and statistics.

    Usage:
        analyzer = SequenceAnalyzer(strand_config, analysis_config)
        df_seq, normal_groups, abnormal_groups = analyzer.analyze(df_pandas)
    """

    def __init__(self, strand_config: StrandConfig, analysis_config: AnalysisConfig):
        self.strand_config = strand_config
        self.config = analysis_config
        self._prefix = f"[{strand_config.strand_name}]"

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def filter_fc_mold_active(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows where all EMBR currents are non-zero (FC Mold ON)."""
        mask = pd.Series(True, index=df.index)
        for col in self.strand_config.embr_current_cols:
            if col in df.columns:
                mask &= df[col] != 0
        df_out = df[mask].reset_index(drop=True)
        self._log(f"FC Mold active: {len(df_out):,}/{len(df):,} rows")
        return df_out

    def identify_sequences(self, df: pd.DataFrame):
        """Identify normal/abnormal sequences via sliding window."""
        self._log("Identifying sequences...")
        df = df.copy()
        df[self.strand_config.vc_column] = custom_rounding(
            df[self.strand_config.vc_column], round_up=True
        )
        normal_groups, abnormal_groups = identify_sequences_segmented(
            df,
            Vc_column=self.strand_config.vc_column,
            window_size=self.config.window_size,
            Vc_threshold=self.config.vc_threshold,
            Curr_columns=self.strand_config.embr_current_cols,
            Curr_threshold=self.config.curr_threshold,
            tcol="plainTimeStamp",
            max_gap_seconds=self.config.max_gap_seconds,
            min_segment_len=self.config.window_size,
        )
        self._log(f"Found {len(normal_groups)} normal, {len(abnormal_groups)} abnormal sequences")
        return normal_groups, abnormal_groups

    def generate_statistics(self, df: pd.DataFrame, sequences: List[List[int]]) -> pd.DataFrame:
        """Compute per-sequence statistics with disturbance flags."""
        self._log("Computing sequence statistics...")
        results = []
        for idx, group in enumerate(sequences):
            if max(group) >= len(df):
                continue
            tmp = df.iloc[group]
            ml = tmp["Mold Level"].values

            quality = None
            if "Quality casting" in tmp.columns:
                qv = tmp["Quality casting"].dropna()
                if len(qv) > 0:
                    quality = qv.mode().iloc[0] if len(qv.mode()) > 0 else qv.iloc[0]

            stats = {
                "Seq_Name": f"seq_{idx:04d}",
                "Seq_time_Start": tmp["plainTimeStamp"].min(),
                "Seq_time_End": tmp["plainTimeStamp"].max(),
                "Seq_num_samples": len(tmp),
                "CASTING_SPEED_avg [m/min]": tmp["castingSpeed"].mean(),
                "MOLD_WIDTH_avg [m]": tmp["moldWidth"].mean(),
                "SEN_avg [mm]": tmp["SENImmersionDepth"].mean() * 1000,
                "MOLD_LEVEL_avg [mm]": tmp["Mold Level"].mean(),
                "MOLD_LEVEL_std [mm]": tmp["Mold Level"].std(),
                "ptp_mm": np.ptp(ml),
                "Quality casting": quality,
                "has_excursion_event": detect_excursion_event_robust(
                    ml, self.config.excursion_threshold_mm, self.config.excursion_min_duration_s
                ),
                "has_slow_drift": detect_slow_drift_event(
                    ml, self.config.slow_drift_min_run_s, self.config.slow_drift_min_amplitude_mm
                ),
                "has_transient_bump": detect_transient_bump_dynamic(
                    ml, self.config.transient_bump_k_amp, self.config.transient_bump_min_mm
                ),
                "has_high_variability": detect_high_variability_event(
                    ml, self.config.high_var_ptp_threshold_mm, self.config.high_var_band_mm
                ),
            }
            stats["has_disturbance"] = any([
                stats["has_excursion_event"],
                stats["has_slow_drift"],
                stats["has_transient_bump"],
                stats["has_high_variability"],
            ])
            results.append(stats)

        df_seq = pd.DataFrame(results)
        self._log(f"Generated stats for {len(df_seq)} sequences")
        return df_seq

    def analyze(self, df: pd.DataFrame):
        """Full analysis: filter -> identify -> statistics."""
        df_active = self.filter_fc_mold_active(df)
        normal_groups, abnormal_groups = self.identify_sequences(df_active)
        df_seq = self.generate_statistics(df_active, normal_groups)
        return df_seq, normal_groups, abnormal_groups
