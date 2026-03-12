"""Sequence identification, disturbance detection and statistical analysis.

Pure Pandas / NumPy — no Spark dependency.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from fcmold_analysis.config import AnalysisConfig, StrandConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def custom_rounding(series, round_up: bool = True):
    """Round values with hysteresis to suppress jitter."""
    rounded: list[float] = []
    for i, val in enumerate(series):
        if i > 0 and abs(val - series.iloc[i - 1] if hasattr(series, "iloc") else series[i - 1]) <= 0.01:
            rounded.append(rounded[-1])
        else:
            offset = 0.5 if round_up else -0.5
            rounded.append(round(val * 100 + offset) / 100)
    return rounded


def segment_by_time_gaps(df: pd.DataFrame, tcol: str = "plainTimeStamp", max_gap_seconds: int = 5) -> pd.DataFrame:
    """Add a ``segment_id`` column that increments at every gap > *max_gap_seconds*."""
    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol)
    diffs = df[tcol].diff().dt.total_seconds().fillna(0)
    df["segment_id"] = (diffs > max_gap_seconds).astype(int).cumsum()
    return df


# ---------------------------------------------------------------------------
# Sliding-window sequence identification
# ---------------------------------------------------------------------------

def identify_sequences(
    df: pd.DataFrame,
    Vc_column: str,
    window_size: int,
    Vc_threshold: float,
    Curr_columns: List[str] | None = None,
    Curr_threshold: float | None = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Classify windows as NORMAL (stable Vc + EMBR) or ABNORMAL.

    * Normal windows advance by ``window_size`` (non-overlapping).
    * Abnormal windows advance by 1 (overlapping search for next stable zone).
    """
    normal_groups: list[list[int]] = []
    abnormal_groups: list[list[int]] = []
    i, n = 0, len(df)

    while i <= n - window_size:
        wdf = df.iloc[i : i + window_size]
        widx = wdf.index.tolist()

        vc_win = wdf[Vc_column]
        vc_ok = bool(np.all(np.abs(vc_win - vc_win.iloc[0]) <= Vc_threshold))

        curr_ok = True
        if vc_ok and Curr_columns and Curr_threshold is not None:
            for col in Curr_columns:
                if col not in wdf.columns:
                    curr_ok = False
                    break
                if np.any(np.abs(wdf[col].diff().dropna()) >= Curr_threshold):
                    curr_ok = False
                    break

        if vc_ok and curr_ok:
            normal_groups.append(widx)
            i += window_size
        else:
            abnormal_groups.append(widx)
            i += 1

    return normal_groups, abnormal_groups


def identify_sequences_segmented(
    df: pd.DataFrame,
    Vc_column: str,
    window_size: int,
    Vc_threshold: float,
    Curr_columns: List[str] | None = None,
    Curr_threshold: float | None = None,
    tcol: str = "plainTimeStamp",
    max_gap_seconds: int = 5,
    min_segment_len: int | None = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Run :func:`identify_sequences` independently per continuous segment."""
    if min_segment_len is None:
        min_segment_len = window_size

    df_seg = segment_by_time_gaps(df, tcol=tcol, max_gap_seconds=max_gap_seconds)
    normal_all: list[list[int]] = []
    abnormal_all: list[list[int]] = []

    for _, seg_df in df_seg.groupby("segment_id"):
        if len(seg_df) < min_segment_len:
            continue
        ng, ag = identify_sequences(
            seg_df,
            Vc_column=Vc_column,
            window_size=window_size,
            Vc_threshold=Vc_threshold,
            Curr_columns=Curr_columns,
            Curr_threshold=Curr_threshold,
        )
        normal_all.extend(ng)
        abnormal_all.extend(ag)

    return normal_all, abnormal_all


# ---------------------------------------------------------------------------
# Disturbance detectors
# ---------------------------------------------------------------------------

def detect_excursion_event_robust(
    signal: np.ndarray,
    threshold_mm: float = 8.0,
    min_duration_s: float = 5.0,
    sampling_hz: float = 1.0,
) -> bool:
    """Large deviation (>*threshold_mm*) sustained for >*min_duration_s*."""
    baseline = np.median(signal)
    above = (np.abs(signal - baseline) > threshold_mm).astype(int)
    min_samples = int(min_duration_s * sampling_hz)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return any((e - s) >= min_samples for s, e in zip(starts, ends))


def detect_slow_drift_event(
    signal: np.ndarray,
    min_run_s: float = 60.0,
    min_amplitude_mm: float = 10.0,
    sampling_hz: float = 1.0,
) -> bool:
    """Sustained monotonic trend over *min_run_s* with peak-to-peak >= *min_amplitude_mm*."""
    min_samples = int(min_run_s * sampling_hz)
    if len(signal) < min_samples:
        return False

    diffs = np.diff(signal)

    def _longest_runs(positive: bool):
        runs: list[int] = []
        cur = 0
        for d in diffs:
            if (d > 0) == positive:
                cur += 1
            else:
                if cur >= min_samples:
                    runs.append(cur)
                cur = 0
        if cur >= min_samples:
            runs.append(cur)
        return runs

    for run_len in _longest_runs(True) + _longest_runs(False):
        if run_len >= min_samples and np.ptp(signal) >= min_amplitude_mm:
            return True
    return False


def detect_transient_bump_dynamic(
    signal: np.ndarray,
    k_amp: float = 8.0,
    min_amp_mm: float = 6.0,
    min_duration_s: float = 5.0,
    max_duration_s: float = 180.0,
    return_band_sigma: float = 2.5,
    sampling_hz: float = 1.0,
) -> bool:
    """Short-lived spike detected via dynamic (MAD-based) threshold."""
    win = int(20 * sampling_hz)
    baseline = median_filter(signal, size=win, mode="nearest")
    residuals = signal - baseline
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad
    threshold = max(k_amp * sigma, min_amp_mm)

    above = (np.abs(signal - baseline) > threshold).astype(int)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    lo, hi = int(min_duration_s * sampling_hz), int(max_duration_s * sampling_hz)
    for s, e in zip(starts, ends):
        dur = e - s
        if lo <= dur <= hi and e < len(signal):
            post = signal[e : min(e + int(30 * sampling_hz), len(signal))]
            if len(post) > 0 and np.abs(np.median(post) - np.median(baseline)) < return_band_sigma * sigma:
                return True
    return False


def detect_high_variability_event(
    signal: np.ndarray,
    ptp_threshold_mm: float = 10.0,
    band_mm: float = 4.0,
    fraction_threshold: float = 0.1,
) -> bool:
    """Excessive oscillation: PTP > threshold or too many points outside band."""
    if np.ptp(signal) > ptp_threshold_mm:
        return True
    baseline = np.median(signal)
    return float(np.mean(np.abs(signal - baseline) > band_mm)) > fraction_threshold


# ---------------------------------------------------------------------------
# SequenceAnalyzer
# ---------------------------------------------------------------------------

class SequenceAnalyzer:
    """Identify sequences, detect disturbances, compute statistics for one strand."""

    def __init__(self, strand_config: StrandConfig, analysis_config: AnalysisConfig):
        self.strand_config = strand_config
        self.cfg = analysis_config
        self._prefix = f"[{strand_config.strand_name}]"

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def filter_fc_mold_active(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for col in self.strand_config.embr_current_cols:
            if col in df.columns:
                mask &= df[col] != 0
        df_f = df[mask].reset_index(drop=True)
        self._log(f"FC Mold active: {len(df_f):,}/{len(df):,} ({100*len(df_f)/len(df):.1f}%)")
        return df_f

    def identify_sequences(self, df: pd.DataFrame) -> Tuple[List[List[int]], List[List[int]]]:
        self._log("Identifying sequences…")
        df = df.copy()
        df[self.strand_config.vc_column] = custom_rounding(df[self.strand_config.vc_column], round_up=True)
        ng, ag = identify_sequences_segmented(
            df,
            Vc_column=self.strand_config.vc_column,
            window_size=self.cfg.window_size,
            Vc_threshold=self.cfg.vc_threshold,
            Curr_columns=self.strand_config.embr_current_cols,
            Curr_threshold=self.cfg.curr_threshold,
            tcol="plainTimeStamp",
            max_gap_seconds=self.cfg.max_gap_seconds,
            min_segment_len=self.cfg.window_size,
        )
        self._log(f"  {len(ng)} normal, {len(ag)} abnormal sequences")
        return ng, ag

    def _detect_disturbances(self, ml_signal: np.ndarray) -> dict:
        c = self.cfg
        exc = detect_excursion_event_robust(ml_signal, c.excursion_threshold_mm, c.excursion_min_duration_s)
        drift = detect_slow_drift_event(ml_signal, c.slow_drift_min_run_s, c.slow_drift_min_amplitude_mm)
        bump = detect_transient_bump_dynamic(ml_signal, c.transient_bump_k_amp, c.transient_bump_min_mm)
        hvar = detect_high_variability_event(ml_signal, c.high_var_ptp_threshold_mm, c.high_var_band_mm, c.high_var_fraction_threshold)
        return {
            "has_excursion_event": exc,
            "has_slow_drift": drift,
            "has_transient_bump": bump,
            "has_high_variability": hvar,
            "has_disturbance": any([exc, drift, bump, hvar]),
        }

    def generate_statistics(self, df: pd.DataFrame, sequences: List[List[int]]) -> pd.DataFrame:
        self._log("Computing sequence statistics…")
        rows: list[dict] = []
        embr_cols = self.strand_config.embr_current_cols

        for idx, group in enumerate(sequences):
            if max(group) >= len(df):
                continue
            t = df.iloc[group]
            start, end = t["plainTimeStamp"].min(), t["plainTimeStamp"].max()
            dur_min = (end - start).total_seconds() / 60

            quality = None
            if "Quality casting" in t.columns:
                qv = t["Quality casting"].dropna()
                if len(qv):
                    modes = qv.mode()
                    quality = modes.iloc[0] if len(modes) else qv.iloc[0]

            ml = t["Mold Level"].values
            dist = self._detect_disturbances(ml)

            row = {
                "Seq_Name": f"seq_{idx:04d}",
                "Seq_time_Start": start,
                "Seq_time_End": end,
                "Seq_duration_min": dur_min,
                "Seq_num_samples": len(t),
                "CASTING_SPEED_avg [m/min]": t["castingSpeed"].mean(),
                "CASTING_SPEED_std [m/min]": t["castingSpeed"].std(),
                "MOLD_WIDTH_avg [m]": t["moldWidth"].mean(),
                "MOLD_WIDTH_std [m]": t["moldWidth"].std(),
                "SEN_avg [mm]": t["SENImmersionDepth"].mean() * 1000,
                "SEN_std [mm]": t["SENImmersionDepth"].std() * 1000,
                "MOLD_LEVEL_avg [mm]": t["Mold Level"].mean(),
                "MOLD_LEVEL_std [mm]": t["Mold Level"].std(),
                "MOLD_LEVEL_min [mm]": t["Mold Level"].min(),
                "MOLD_LEVEL_max [mm]": t["Mold Level"].max(),
                "ptp_mm": t["Mold Level"].max() - t["Mold Level"].min(),
                "ArFlow_avg [NL/min]": (
                    (t["Argon Flow SEN"].mean() + t["Argon Flow Stopper"].mean())
                    if "Argon Flow SEN" in t.columns
                    else np.nan
                ),
                "Quality casting": quality,
                **dist,
            }
            for col in embr_cols:
                if col in t.columns:
                    tag = col.replace(" ", "_").replace("EMBR_Current_", "")
                    row[f"{tag}_avg [A]"] = t[col].mean()
                    row[f"{tag}_std [A]"] = t[col].std()
            rows.append(row)

        df_seq = pd.DataFrame(rows)
        self._log(f"  {len(df_seq)} sequences statistically characterised")
        return df_seq

    @staticmethod
    def classify_disturbance(df_seq: pd.DataFrame) -> pd.DataFrame:
        def _label(r):
            flags = []
            if r.get("has_excursion_event"):
                flags.append("Excursion")
            if r.get("has_high_variability"):
                flags.append("High variability")
            if r.get("has_transient_bump"):
                flags.append("Transient bump")
            if r.get("has_slow_drift"):
                flags.append("Slow drift")
            return "Normal" if not flags else " + ".join(flags)

        df_seq["disturbance_type"] = df_seq.apply(_label, axis=1)
        return df_seq

    # -- full pipeline ------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[List[int]], List[List[int]]]:
        """Return ``(df_fc_mold, df_seq, normal_groups, abnormal_groups)``."""
        self._log("Starting sequence analysis…")
        df_fc = self.filter_fc_mold_active(df)
        ng, ag = self.identify_sequences(df_fc)
        df_seq = self.generate_statistics(df_fc, ng)
        df_seq = self.classify_disturbance(df_seq)
        self._log("Sequence analysis complete.")
        return df_fc, df_seq, ng, ag
