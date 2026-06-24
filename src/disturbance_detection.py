"""Disturbance detection functions for mold level signals.

Each function takes a 1-D signal array and returns True/False indicating
whether the disturbance pattern was detected.
"""

import numpy as np
from scipy.ndimage import median_filter


def detect_excursion_event_robust(
    signal: np.ndarray,
    threshold_mm: float = 8.0,
    min_duration_s: float = 5.0,
    sampling_hz: float = 1.0,
) -> bool:
    """Detect excursion: deviation > threshold_mm from baseline for > min_duration_s."""
    baseline = np.median(signal)
    deviation = np.abs(signal - baseline)
    min_samples = int(min_duration_s * sampling_hz)

    above_threshold = (deviation > threshold_mm).astype(int)
    diff = np.diff(np.concatenate(([0], above_threshold, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        if (end - start) >= min_samples:
            return True
    return False


def detect_slow_drift_event(
    signal: np.ndarray,
    min_run_s: float = 60.0,
    min_amplitude_mm: float = 10.0,
    sampling_hz: float = 1.0,
) -> bool:
    """Detect slow drift: sustained monotonic trend > min_run_s with amplitude > min_amplitude_mm."""
    min_samples = int(min_run_s * sampling_hz)
    if len(signal) < min_samples:
        return False

    diffs = np.diff(signal)

    def _find_long_runs(condition_array):
        runs = []
        current_run = 0
        for val in condition_array:
            if val:
                current_run += 1
            else:
                if current_run >= min_samples:
                    runs.append(current_run)
                current_run = 0
        if current_run >= min_samples:
            runs.append(current_run)
        return runs

    inc_runs = _find_long_runs(diffs > 0)
    dec_runs = _find_long_runs(diffs < 0)

    for _ in inc_runs + dec_runs:
        if np.ptp(signal) >= min_amplitude_mm:
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
    """Detect transient bumps using dynamic threshold from noise estimation (MAD)."""
    window_size = int(20 * sampling_hz)
    baseline = median_filter(signal, size=max(window_size, 5), mode="nearest")

    residuals = signal - baseline
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad

    threshold = max(k_amp * sigma, min_amp_mm)
    deviation = np.abs(signal - baseline)
    above_threshold = (deviation > threshold).astype(int)

    diff = np.diff(np.concatenate(([0], above_threshold, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    min_samples = int(min_duration_s * sampling_hz)
    max_samples = int(max_duration_s * sampling_hz)

    for start, end in zip(starts, ends):
        duration = end - start
        if min_samples <= duration <= max_samples:
            if end < len(signal):
                post_event = signal[end : min(end + int(30 * sampling_hz), len(signal))]
                if len(post_event) > 0:
                    post_baseline = np.median(post_event)
                    if np.abs(post_baseline - np.median(baseline)) < return_band_sigma * sigma:
                        return True
    return False


def detect_high_variability_event(
    signal: np.ndarray,
    ptp_threshold_mm: float = 10.0,
    band_mm: float = 4.0,
    fraction_threshold: float = 0.1,
) -> bool:
    """Detect high variability: excessive peak-to-peak or fraction outside band."""
    if np.ptp(signal) > ptp_threshold_mm:
        return True

    baseline = np.median(signal)
    outside_band = np.abs(signal - baseline) > band_mm
    if np.mean(outside_band) > fraction_threshold:
        return True

    return False
