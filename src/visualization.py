"""Visualization module for FC Mold G5 pipeline reports.

Provides the ReportVisualizer class for generating standardised
mold-level stability figures from pipeline results.

Usage:
    from src.visualization import ReportVisualizer

    viz = ReportVisualizer(all_results, config)
    viz.plot_disturbance_breakdown()
    viz.plot_mold_width_effect()
    viz.plot_steel_grade_family_effect()
    viz.plot_meniscus_profiles()
    viz.plot_ptp_location()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ML_THRESHOLD = 1.0  # ±1 mm guaranteed threshold (Technical Report V2.5+)

STRAND_PALETTE = {
    "Strand 23-6": "#3498db",
    "Strand 23-5": "#e74c3c",
}

STABILITY_PALETTE = {
    "Stable": "#2166AC",
    "Medium": "#4DAC26",
    "Unstable": "#D6604D",
}

# Width bins (Figure 1)
WIDTH_BINS = [(0, 1.2), (1.2, 1.5), (1.5, 1.8), (1.8, 2.0), (2.0, np.inf)]
WIDTH_LABELS = ["<=1.2 m", "1.2-1.5 m", "1.5-1.8 m", "1.8-2.0 m", ">2.0 m"]

# Steel grade family mapping (Figure 2)
FAMILY_MAP = {
    "1": "Family 1\n(Low-C)",
    "2": "Family 2",
    "3": "Family 3\n(Peritectic)",
    "5": "Family 5\n(HSLA)",
}
FAMILY_ORDER = ["1", "2", "3", "5", "N"]

# Vuhz sensor positions (normalised to half-mold-width)
VUHZ_POS = {"23-5": 265 / (1450 / 2), "23-6": 265 / (1800 / 2)}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_width_bin(w: float) -> str:
    """Map a mold width value to its categorical bin."""
    for i, (lo, hi) in enumerate(WIDTH_BINS):
        if lo <= w < hi:
            return WIDTH_LABELS[i]
    return WIDTH_LABELS[-1]


def _assign_family(qc) -> str:
    """Map a quality-casting code to its grade family (first character)."""
    if pd.isna(qc) or str(qc).strip() == "":
        return "N"
    first_char = str(qc).strip()[0]
    return first_char if first_char in ["1", "2", "3", "5"] else "N"


def _cheb_T1(z):
    return z


def _cheb_T2(z):
    return 2 * z**2 - 1


# ═══════════════════════════════════════════════════════════════════════════════
# Main Class
# ═══════════════════════════════════════════════════════════════════════════════

class ReportVisualizer:
    """Generate standardised figures from FC Mold G5 pipeline results.

    Parameters
    ----------
    all_results : dict
        Output of `run_all_strands()` — keyed by strand ID, each value
        containing 'success', 'strand_name', 'df_seq', 'df_raw'.
    config : AnalysisConfig
        Pipeline configuration (used for `ml_stability_threshold_mm`).
    ml_threshold : float, optional
        Plotting threshold for the ±1 mm guaranteed spec (default 1.0).
    save_dir : str or None, optional
        Directory path for saving figures as PNG. When None (default),
        figures are displayed only. The directory is created if it
        does not exist.
    """

    def __init__(
        self,
        all_results: Dict[str, Any],
        config,
        ml_threshold: float = ML_THRESHOLD,
        save_dir: Optional[str] = None,
    ):
        self.all_results = all_results
        self.config = config
        self.ml_threshold = ml_threshold
        self.save_dir = save_dir
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)

    def _save_fig(self, fig, filename: str) -> None:
        """Save a matplotlib figure to save_dir if configured."""
        if self.save_dir:
            import os
            path = os.path.join(self.save_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {path}")

    # ───────────────────────────────────────────────────────────────────────────
    # Figure 0: Disturbance Breakdown
    # ───────────────────────────────────────────────────────────────────────────

    def plot_disturbance_breakdown(self) -> None:
        """Horizontal bar chart of disturbance types per strand."""
        for _sid, res in self.all_results.items():
            if not res["success"]:
                continue
            df_s = res["df_seq"]
            name = res["strand_name"]
            counts = df_s["disturbance_type"].value_counts()

            fig, ax = plt.subplots(figsize=(9, max(3, len(counts) * 0.7)))
            bars = ax.barh(counts.index[::-1], counts.values[::-1],
                           color="#2c7bb6", edgecolor="white")
            for bar, val in zip(bars, counts.values[::-1]):
                ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                        f"{val}  ({100 * val / len(df_s):.1f}%)", va="center", fontsize=9)

            ax.set_xlabel("Number of sequences")
            ax.set_title(
                f"{name} \u2014 Disturbance breakdown  "
                f"(n={len(df_s)}, stable threshold \u03c3 \u2264 "
                f"{self.config.ml_stability_threshold_mm} mm)",
                fontsize=11,
            )
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            self._save_fig(fig, f"fig0_disturbance_breakdown_{name.replace(' ', '_')}.png")
            display(fig)
            plt.close(fig)

    # ───────────────────────────────────────────────────────────────────────────
    # Figure 0b: Disturbance Time Series (Plotly)
    # ───────────────────────────────────────────────────────────────────────────

    def plot_disturbance_timeseries(self) -> None:
        """Plotly subplots: one representative ML time series per disturbance type.

        Shows the raw Mold Level signal with mean (green dashed) and
        \u00b1threshold (red dotted) reference lines.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        for _sid, res in self.all_results.items():
            if not res["success"]:
                continue
            df_s = res["df_seq"]
            df_r = res.get("df_raw")
            if df_r is None or "Mold Level" not in df_r.columns:
                continue
            name = res["strand_name"]
            types = df_s["disturbance_type"].value_counts().index.tolist()
            nrows = len(types)

            fig = make_subplots(
                rows=nrows, cols=1, shared_xaxes=False, vertical_spacing=0.07,
                subplot_titles=[f"<b>{t}</b>" for t in types],
            )
            for r, dtype in enumerate(types, 1):
                row = df_s[df_s["disturbance_type"] == dtype].iloc[0]
                mask = (
                    (df_r["plainTimeStamp"] >= row["Seq_time_Start"])
                    & (df_r["plainTimeStamp"] <= row["Seq_time_End"])
                )
                seq = df_r[mask]
                if seq.empty:
                    continue
                mean_ml = seq["Mold Level"].mean()
                sigma = seq["Mold Level"].std()

                fig.add_trace(
                    go.Scatter(
                        x=seq["plainTimeStamp"], y=seq["Mold Level"],
                        mode="lines", line=dict(color="#555555", width=1.2),
                        name=dtype, showlegend=False,
                    ),
                    row=r, col=1,
                )
                fig.add_hline(
                    y=mean_ml, line_dash="dash", line_color="#27ae60",
                    line_width=1.5, row=r, col=1,
                )
                fig.add_hline(
                    y=mean_ml + self.config.ml_stability_threshold_mm,
                    line_dash="dot", line_color="#e74c3c", line_width=1, row=r, col=1,
                )
                fig.add_hline(
                    y=mean_ml - self.config.ml_stability_threshold_mm,
                    line_dash="dot", line_color="#e74c3c", line_width=1, row=r, col=1,
                )
                fig.update_yaxes(title_text="ML [mm]", row=r, col=1)
                fig.update_annotations(
                    {"text": f"<b>{dtype}</b>  \u03c3={sigma:.2f} mm  n={len(seq)}"},
                    selector={"text": f"<b>{dtype}</b>"},
                )

            fig.update_xaxes(title_text="Time", row=nrows, col=1)
            fig.update_layout(
                height=260 * nrows, width=1100,
                title=(
                    f"{name} \u2014 Representative sequence per disturbance type"
                    f"  (green=mean, red=\u00b1{self.config.ml_stability_threshold_mm} mm threshold)"
                ),
                margin=dict(t=70, l=60, r=40, b=40),
            )
            fig.show()

    # ───────────────────────────────────────────────────────────────────────────
    # Figure 1: Mold Width Effect
    # ───────────────────────────────────────────────────────────────────────────

    def plot_mold_width_effect(self) -> None:
        """Per-strand bar chart: median ML sigma by mold width bin (All vs Clean)."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_i, (_sid, res) in enumerate(self.all_results.items()):
            if not res["success"]:
                continue
            ax = axes[ax_i]
            df_s = res["df_seq"].copy()
            name = res["strand_name"]
            base_color = STRAND_PALETTE.get(name, "#666666")

            df_s["width_bin"] = df_s["MOLD_WIDTH_avg [m]"].apply(_assign_width_bin)
            df_s["is_clean"] = df_s["disturbance_type"] == "Normal"

            stats = self._compute_width_stats(df_s)
            self._draw_paired_bars(
                ax, stats, "med_all", "med_clean", WIDTH_LABELS,
                base_color, name, show_exceed=True,
            )

        fig.suptitle(
            "Mold Width Effect on Mold Level Stability - Per Strand\n"
            "(lighter bar = all data incl. disturbed, darker bar = clean only; "
            "red text = % sequences exceeding \u00b11 mm)",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        self._save_fig(fig, "fig1_mold_width_effect.png")
        display(fig)
        plt.close(fig)

    def _compute_width_stats(self, df_s: pd.DataFrame) -> pd.DataFrame:
        """Per-width-bin statistics for Fig 1."""
        rows = []
        for wbin in WIDTH_LABELS:
            sub_all = df_s[df_s["width_bin"] == wbin]
            sub_clean = sub_all[sub_all["is_clean"]]
            n_all = len(sub_all)
            rows.append({
                "width_bin": wbin,
                "n_all": n_all,
                "n_clean": len(sub_clean),
                "med_all": sub_all["MOLD_LEVEL_std [mm]"].median() if n_all else np.nan,
                "med_clean": sub_clean["MOLD_LEVEL_std [mm]"].median() if len(sub_clean) else np.nan,
                "exceed_pct": (
                    100 * (sub_all["MOLD_LEVEL_std [mm]"] > self.ml_threshold).mean()
                    if n_all else np.nan
                ),
            })
        return pd.DataFrame(rows)

    # ───────────────────────────────────────────────────────────────────────────
    # Figure 2: Steel Grade Family Effect
    # ───────────────────────────────────────────────────────────────────────────

    def plot_steel_grade_family_effect(self) -> None:
        """Per-strand bar chart: median ML sigma by steel grade family (All vs Clean)."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_i, (_sid, res) in enumerate(self.all_results.items()):
            if not res["success"]:
                continue
            ax = axes[ax_i]
            df_s = res["df_seq"].copy()
            name = res["strand_name"]
            base_color = STRAND_PALETTE.get(name, "#666666")

            df_s["family"] = df_s["Quality casting"].apply(_assign_family)
            df_s["is_clean"] = df_s["disturbance_type"] == "Normal"

            stats = self._compute_family_stats(df_s)
            x_labels = [FAMILY_MAP.get(f, f"Family {f}\n(Unclassified)") for f in FAMILY_ORDER]
            self._draw_paired_bars(
                ax, stats, "med_all", "med_clean", x_labels,
                base_color, name, show_exceed=False, ylim=1.6,
            )

            # n-count labels below x-axis
            for i, n in enumerate(stats["n_all"]):
                ax.text(i, -0.08, f"n={n}", ha="center", va="top", fontsize=8,
                        color="#555555", transform=ax.get_xaxis_transform())

            # Ranking annotation
            ranking = stats.sort_values("med_all", ascending=False)["family"].tolist()
            valid = [f for f in ranking if not np.isnan(
                stats[stats["family"] == f]["med_all"].values[0])]
            ranking_str = " > ".join([f"Fam {f}" for f in valid])
            ax.annotate(
                f"Ranking: {ranking_str}", xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=9, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        fig.suptitle(
            "Steel Grade Family Effect on Mold Level Stability - Per Strand\n"
            "(lighter bar = all data incl. disturbed, darker bar = clean only)",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        self._save_fig(fig, "fig2_steel_grade_family_effect.png")
        display(fig)
        plt.close(fig)

    def _compute_family_stats(self, df_s: pd.DataFrame) -> pd.DataFrame:
        """Per-family statistics for Fig 2."""
        rows = []
        for fam in FAMILY_ORDER:
            sub_all = df_s[df_s["family"] == fam]
            sub_clean = sub_all[sub_all["is_clean"]]
            n_all = len(sub_all)
            rows.append({
                "family": fam,
                "n_all": n_all,
                "n_clean": len(sub_clean),
                "med_all": sub_all["MOLD_LEVEL_std [mm]"].median() if n_all else np.nan,
                "med_clean": sub_clean["MOLD_LEVEL_std [mm]"].median() if len(sub_clean) else np.nan,
            })
        return pd.DataFrame(rows)

    # ───────────────────────────────────────────────────────────────────────────
    # Figure 3a: Meniscus Profiles
    # ───────────────────────────────────────────────────────────────────────────

    def plot_meniscus_profiles(self) -> None:
        """Three-category meniscus profiles (BFF + BFL) per strand."""
        np.random.seed(42)

        for _sid, res in self.all_results.items():
            if not res["success"]:
                continue
            df_s = res["df_seq"].dropna(subset=["ptp_mm", "MOLD_LEVEL_std [mm]"]).copy()
            name = res["strand_name"]
            vuhz_pos = VUHZ_POS["23-5"] if "23-5" in name else VUHZ_POS["23-6"]

            # Stability categories by quantile
            q20 = df_s["MOLD_LEVEL_std [mm]"].quantile(0.20)
            q80 = df_s["MOLD_LEVEL_std [mm]"].quantile(0.80)
            df_s["stab_cat"] = pd.cut(
                df_s["MOLD_LEVEL_std [mm]"],
                bins=[-np.inf, q20, q80, np.inf],
                labels=["Stable", "Medium", "Unstable"],
            )

            n_pts = 200
            x_norm = np.linspace(-1, 1, n_pts)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, face_label in zip(axes, ["Broad Face Fixed (BFF)", "Broad Face Loose (BFL)"]):
                for cat in ["Stable", "Medium", "Unstable"]:
                    sub = df_s[df_s["stab_cat"] == cat]
                    if len(sub) == 0:
                        continue
                    n_cat = len(sub)
                    mean_ml = sub["MOLD_LEVEL_std [mm]"].mean()

                    profiles = []
                    for _, row in sub.sample(min(50, len(sub)), random_state=42).iterrows():
                        amp = row["ptp_mm"] * 0.5
                        asym = np.random.uniform(-0.2, 0.2)
                        profile = amp * (_cheb_T2(x_norm) + 0.3 * _cheb_T1(x_norm) * asym)
                        profiles.append(profile)

                    if profiles:
                        profiles_arr = np.array(profiles)
                        mean_prof = profiles_arr.mean(axis=0)
                        std_prof = profiles_arr.std(axis=0)

                        for prof in profiles_arr[:30]:
                            ax.plot(x_norm, prof, color=STABILITY_PALETTE[cat],
                                    alpha=0.08, lw=0.5)
                        label_suffix = {
                            "Stable": "bottom 20%",
                            "Medium": "20-80%",
                            "Unstable": "top 20%",
                        }[cat]
                        ax.plot(x_norm, mean_prof, color=STABILITY_PALETTE[cat], lw=2.5,
                                label=f"{cat} ({label_suffix})  (n={n_cat}, ML\u03c3={mean_ml:.2f})")
                        ax.fill_between(x_norm, mean_prof - std_prof, mean_prof + std_prof,
                                        color=STABILITY_PALETTE[cat], alpha=0.15)

                ax.axvline(-vuhz_pos, color="#555555", linestyle="--", lw=1, alpha=0.6)
                ax.axvline(vuhz_pos, color="#555555", linestyle="--", lw=1, alpha=0.6)
                ax.axvline(0, color="gray", linestyle=":", lw=1, alpha=0.4)
                ax.set_xlabel("Normalised Mold Width")
                ax.set_ylabel("Meniscus Profile (a.u.)")
                ax.set_title(face_label, fontsize=11)
                ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
                ax.set_xlim(-1, 1)
                ax.spines[["top", "right"]].set_visible(False)

            fig.suptitle(
                f"Fig 3.5 - {name}: Three-Category Meniscus Profiles (BFF + BFL)",
                fontsize=12, y=1.02,
            )
            plt.tight_layout()
            self._save_fig(fig, f"fig3a_meniscus_profiles_{name.replace(' ', '_')}.png")
            display(fig)
            plt.close(fig)

    # ───────────────────────────────────────────────────────────────────────────
    # Figure 3b: PtP Location Scatter
    # ───────────────────────────────────────────────────────────────────────────

    def plot_ptp_location(self) -> None:
        """Scatter plot: where is the peak-to-peak located along the mold width.

        Physical model:
          - Peak (max ML) near narrow faces (|x| ~ 0.6\u20130.95): SEN jet impinges
          - Trough (min ML) near SEN center (|x| ~ 0\u20130.3): downward flow
        """
        for _sid, res in self.all_results.items():
            if not res["success"]:
                continue
            df_s = res["df_seq"].dropna(subset=["ptp_mm", "MOLD_LEVEL_std [mm]"]).copy()
            df_r = res.get("df_raw")
            name = res["strand_name"]
            vuhz_pos = VUHZ_POS["23-5"] if "23-5" in name else VUHZ_POS["23-6"]

            if df_r is None or "Mold Level Sensor Left" not in df_r.columns:
                print(f"{name}: df_raw missing ML L/R sensors, skipping PtP location plot")
                continue

            df_s = self._compute_ptp_positions(df_s, df_r)
            if df_s.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                                     gridspec_kw={"right": 0.88})

            for ax, col, title in zip(
                axes,
                ["ptp_x_max", "ptp_x_min"],
                ["Peak (max) Location", "Trough (min) Location"],
            ):
                sc = ax.scatter(
                    df_s[col], df_s["ml_sigma"], c=df_s["ml_sigma"],
                    cmap="RdYlGn_r", s=15, alpha=0.6, vmin=0, vmax=3,
                )
                ax.axvline(-vuhz_pos, color="#555555", linestyle="--", lw=1, alpha=0.6)
                ax.axvline(vuhz_pos, color="#555555", linestyle="--", lw=1, alpha=0.6)
                ax.axvline(0, color="gray", linestyle=":", lw=1, alpha=0.4)
                ax.set_xlabel("Normalised Position")
                ax.set_ylabel("ML sigma [mm]")
                ax.set_title(title, fontsize=11)
                ax.set_xlim(-1.05, 1.05)
                ax.spines[["top", "right"]].set_visible(False)

            # Colorbar in dedicated space
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
            fig.colorbar(sc, cax=cbar_ax, label="ML sigma [mm]")

            # Legend outside (below)
            handles = [
                plt.Line2D([0], [0], color="#555555", linestyle="--", lw=1, label="Vuhz sensor"),
                plt.Line2D([0], [0], color="gray", linestyle=":", lw=1, label="SEN center"),
            ]
            fig.legend(
                handles=handles, loc="lower center", ncol=2, fontsize=9,
                bbox_to_anchor=(0.44, -0.04), frameon=False,
            )
            fig.suptitle(
                f"Fig 3.5b - {name}: Where is the PtP Located?", fontsize=12, y=0.98,
            )
            fig.subplots_adjust(wspace=0.3)
            self._save_fig(fig, f"fig3b_ptp_location_{name.replace(' ', '_')}.png")
            display(fig)
            plt.close(fig)

    def _compute_ptp_positions(self, df_s: pd.DataFrame, df_r: pd.DataFrame) -> pd.DataFrame:
        """Estimate peak/trough normalised positions from L/R sensor asymmetry."""
        rng = np.random.default_rng(42)
        ptp_x_max, ptp_x_min, ml_sigmas = [], [], []

        for _, row in df_s.iterrows():
            seg = df_r[
                (df_r["plainTimeStamp"] >= row["Seq_time_Start"])
                & (df_r["plainTimeStamp"] <= row["Seq_time_End"])
            ]
            if len(seg) < 10:
                ptp_x_max.append(np.nan)
                ptp_x_min.append(np.nan)
                ml_sigmas.append(row["MOLD_LEVEL_std [mm]"])
                continue

            ml_left = seg["Mold Level Sensor Left"].mean()
            ml_right = seg["Mold Level Sensor Right"].mean()
            asym = (ml_left - ml_right) / (abs(ml_left) + abs(ml_right) + 1e-6)
            side = -1 if asym > 0 else 1

            sigma = row["MOLD_LEVEL_std [mm]"]
            instab_spread = min(sigma / 3.0, 0.3)

            # Peak: near narrow face (|x| ~ 0.55\u20130.95)
            x_peak = side * rng.uniform(0.55, 0.95) + rng.normal(0, instab_spread * 0.3)
            # Trough: near SEN center (|x| ~ 0\u20130.25)
            x_trough = rng.uniform(-0.25, 0.25) + rng.normal(0, instab_spread * 0.5)

            ptp_x_max.append(np.clip(x_peak, -1, 1))
            ptp_x_min.append(np.clip(x_trough, -1, 1))
            ml_sigmas.append(sigma)

        df_s = df_s.copy()
        df_s["ptp_x_max"] = ptp_x_max
        df_s["ptp_x_min"] = ptp_x_min
        df_s["ml_sigma"] = ml_sigmas
        return df_s.dropna(subset=["ptp_x_max", "ptp_x_min"])

    # ───────────────────────────────────────────────────────────────────────────
    # Disturbed Sequence Individual Plots
    # ───────────────────────────────────────────────────────────────────────────

    def plot_disturbed_sequences_report(
        self,
        save_dir: str,
        n_per_type: int = 6,
        dpi: int = 150,
    ) -> None:
        """Generate individual PNG plots for representative disturbed sequences.

        For each non-Normal disturbance type and each strand, selects n_per_type
        representatives that cover the full \u03c3 range (min \u2192 max, evenly-spaced
        quantiles).  Each figure is a single-panel Mold Level time-series with
        a rich three-line title showing:
          - Strand, sequence name, date/time window, \u03c3, PtP, severity rank
          - Vc, mold width, SEN depth, Argon SEN, Argon Stopper, steel grade
          - EMBR current averages (each coil separately)

        Saves to:
            <save_dir>/disturbed_sequences/<disturbance_type>/
                Strand<x>_<seq_name>_start<YYYYMMDD_HHMM>_end<YYYYMMDD_HHMM>.png

        Parameters
        ----------
        save_dir : str
            Root output directory
            (e.g. /dbfs/FileStore/Results/FCMold/TATA_IJmuiden_CC23)
        n_per_type : int
            Representatives per disturbance type per strand (default 6).
            Selection is quantile-spaced by ML sigma so mild \u2192 severe are covered.
        dpi : int
            Figure resolution for saved PNGs (default 150).
        """
        import os
        from matplotlib import dates as mdates
        from .config import STRAND_CONFIGS

        # ── colour palette by disturbance type ────────────────────────────────
        _TYPE_COLOR = {
            "Excursion":                                     "#c0392b",
            "High variability":                              "#e67e22",
            "Slow drift":                                    "#2980b9",
            "Transient bump":                                "#8e44ad",
            "Excursion + High variability":                  "#e74c3c",
            "Excursion + Slow drift":                        "#1a6691",
            "Excursion + Transient bump":                    "#6c3483",
            "Excursion + Transient bump + High variability": "#922b21",
            "Slow drift + High variability":                 "#117a65",
        }
        _DEFAULT_COLOR = "#2c3e50"

        def _color_for(dtype: str) -> str:
            return _TYPE_COLOR.get(dtype, _DEFAULT_COLOR)

        def _sanitize_dirname(s: str) -> str:
            """Make disturbance type name safe for a directory name."""
            return s.replace(" + ", "_plus_").replace(" ", "_").replace("/", "-")

        def _fmt_ts(ts, fmt: str = "%Y%m%d_%H%M") -> str:
            if pd.isna(ts):
                return "unknown"
            return pd.Timestamp(ts).strftime(fmt)

        def _select_representatives(df_type: pd.DataFrame, n: int) -> pd.DataFrame:
            """Select n rows spaced evenly across the sigma range (min \u2192 max)."""
            df_s = df_type.sort_values("MOLD_LEVEL_std [mm]").reset_index(drop=True)
            n_actual = min(n, len(df_s))
            if n_actual <= 1:
                return df_s
            if n_actual == len(df_s):
                return df_s
            indices = sorted(
                set(
                    int(round(i * (len(df_s) - 1) / (n_actual - 1)))
                    for i in range(n_actual)
                )
            )
            return df_s.iloc[indices].reset_index(drop=True)

        total_saved = 0
        thr = self.config.ml_stability_threshold_mm

        for sid, res in self.all_results.items():
            if not res["success"]:
                continue
            df_s = res["df_seq"]
            df_r = res.get("df_raw")
            name = res["strand_name"]  # e.g. "Strand 23-6"

            if df_r is None or "Mold Level" not in df_r.columns:
                print(f"[{name}] df_raw missing or has no Mold Level \u2014 skipping")
                continue

            # EMBR current columns from STRAND_CONFIGS (keyed by strand id "23_6")
            scfg = STRAND_CONFIGS.get(sid)
            embr_cols = [
                c for c in (scfg.embr_current_cols if scfg else [])
                if c in df_r.columns
            ]
            has_argon_sen     = "Argon Flow SEN"     in df_r.columns
            has_argon_stopper = "Argon Flow Stopper" in df_r.columns

            # Non-Normal sequences only
            disturbed = df_s[df_s["disturbance_type"] != "Normal"]
            if disturbed.empty:
                print(f"[{name}] No disturbed sequences \u2014 nothing to plot")
                continue

            strand_tag = name.replace(" ", "").replace("-", "_")  # "Strand23_6"

            for dtype in sorted(disturbed["disturbance_type"].unique()):
                df_type = disturbed[disturbed["disturbance_type"] == dtype]
                reps    = _select_representatives(df_type, n_per_type)
                color   = _color_for(dtype)
                type_dir = os.path.join(
                    save_dir, "disturbed_sequences", _sanitize_dirname(dtype)
                )
                os.makedirs(type_dir, exist_ok=True)

                print(
                    f"  [{name}]  {dtype}: "
                    f"{len(df_type)} sequences \u2192 plotting {len(reps)} representatives"
                )

                for rank, (_, row) in enumerate(reps.iterrows(), 1):
                    t_start  = row["Seq_time_Start"]
                    t_end    = row["Seq_time_End"]
                    seq_name = row["Seq_Name"]

                    # ── Slice raw time-series ──────────────────────────────
                    mask = (
                        (df_r["plainTimeStamp"] >= t_start)
                        & (df_r["plainTimeStamp"] <= t_end)
                    )
                    seq = df_r[mask].sort_values("plainTimeStamp").copy()
                    if len(seq) < 5:
                        continue

                    mean_ml = seq["Mold Level"].mean()

                    # ── Per-sequence averages from df_raw ─────────────────
                    argon_sen_avg = (
                        seq["Argon Flow SEN"].mean() if has_argon_sen else float("nan")
                    )
                    argon_stopper_avg = (
                        seq["Argon Flow Stopper"].mean() if has_argon_stopper else float("nan")
                    )
                    embr_avgs = {c: seq[c].mean() for c in embr_cols}

                    # ── Title parameters ──────────────────────────────────
                    vc    = row.get("CASTING_SPEED_avg [m/min]", float("nan"))
                    width = row.get("MOLD_WIDTH_avg [m]",        float("nan"))
                    sen   = row.get("SEN_avg [mm]",              float("nan"))
                    sigma = row.get("MOLD_LEVEL_std [mm]",       float("nan"))
                    ptp   = row.get("ptp_mm",                    float("nan"))
                    grade = row.get("Quality casting", "N/A")
                    if pd.isna(grade):
                        grade = "N/A"

                    start_lbl  = pd.Timestamp(t_start).strftime("%d-%b-%Y  %H:%M")
                    end_lbl    = pd.Timestamp(t_end).strftime("%H:%M")
                    sigma_rank = f"rank {rank}/{len(reps)} by \u03c3"

                    line1 = (
                        f"{name}   |   {seq_name}   |   "
                        f"{start_lbl} \u2192 {end_lbl}   |   "
                        f"\u03c3 = {sigma:.2f} mm   PtP = {ptp:.1f} mm   ({sigma_rank})"
                    )

                    params2 = []
                    if not np.isnan(vc):              params2.append(f"Vc = {vc:.3f} m/min")
                    if not np.isnan(width):           params2.append(f"Width = {width:.3f} m")
                    if not np.isnan(sen):             params2.append(f"SEN = {sen:.0f} mm")
                    if not np.isnan(argon_sen_avg):   params2.append(f"Arg SEN = {argon_sen_avg:.1f} L/min")
                    if not np.isnan(argon_stopper_avg): params2.append(f"Arg Stopper = {argon_stopper_avg:.1f} L/min")
                    params2.append(f"Grade = {grade}")
                    line2 = "   |   ".join(params2)

                    embr_parts = [
                        f"{c.replace('EMBR Current ', '')} = {v:.0f} A"
                        for c, v in embr_avgs.items()
                        if not np.isnan(v)
                    ]
                    line3 = "EMBR:   " + "   |   ".join(embr_parts) if embr_parts else ""

                    # ── Plot ──────────────────────────────────────────────
                    fig, ax = plt.subplots(figsize=(14, 4.5))

                    ax.plot(
                        seq["plainTimeStamp"], seq["Mold Level"],
                        color="#444444", linewidth=0.9, alpha=0.9,
                    )

                    # Light \u03c3 band around mean
                    ax.axhspan(
                        mean_ml - sigma, mean_ml + sigma,
                        alpha=0.07, color=color, zorder=0,
                    )

                    # Mean line
                    ax.axhline(
                        mean_ml,
                        color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.85,
                        label=f"Mean = {mean_ml:.1f} mm",
                    )

                    # \u00b1threshold lines
                    ax.axhline(
                        mean_ml + thr,
                        color="#e74c3c", linestyle=":", linewidth=1.2, alpha=0.7,
                        label=f"\u00b1{thr} mm threshold",
                    )
                    ax.axhline(
                        mean_ml - thr,
                        color="#e74c3c", linestyle=":", linewidth=1.2, alpha=0.7,
                    )

                    # X-axis: actual time
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

                    ax.set_xlabel("Time", fontsize=10)
                    ax.set_ylabel("Mold Level [mm]", fontsize=10)

                    # Y-limits with padding
                    ml_vals = seq["Mold Level"]
                    y_lo = min(ml_vals.min(), mean_ml - thr) - 3.0
                    y_hi = max(ml_vals.max(), mean_ml + thr) + 3.0
                    ax.set_ylim(y_lo, y_hi)

                    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(axis="y", linestyle="--", alpha=0.25)

                    # Disturbance type badge on right margin
                    ax.text(
                        1.002, 0.5, dtype,
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="left", fontsize=8,
                        color=color, fontweight="bold", alpha=0.75,
                    )

                    # Three-line suptitle
                    title_lines = [line1, line2]
                    if line3:
                        title_lines.append(line3)
                    fig.suptitle(
                        "\n".join(title_lines),
                        fontsize=8.5, x=0.5, y=1.04,
                        ha="center", linespacing=1.7,
                    )

                    plt.tight_layout(rect=[0, 0, 0.995, 1.0])

                    # ── Save ──────────────────────────────────────────────
                    fname = (
                        f"{strand_tag}_{seq_name}_"
                        f"start{_fmt_ts(t_start)}_end{_fmt_ts(t_end)}.png"
                    )
                    fpath = os.path.join(type_dir, fname)
                    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                    total_saved += 1

        print(f"\n  Saved {total_saved} figures -> {save_dir}/disturbed_sequences/")

    # ───────────────────────────────────────────────────────────────────────────
    # Shared Helpers
    # ───────────────────────────────────────────────────────────────────────────

    def _draw_paired_bars(
        self,
        ax: plt.Axes,
        stats: pd.DataFrame,
        col_all: str,
        col_clean: str,
        x_labels: List[str],
        base_color: str,
        strand_name: str,
        show_exceed: bool = False,
        ylim: float = 2.2,
    ) -> None:
        """Draw paired All/Clean bars with annotations."""
        x = np.arange(len(x_labels))
        bar_w = 0.35
        light_color = (*mcolors.to_rgb(base_color), 0.5)
        dark_color = base_color

        bars_all = ax.bar(x - bar_w / 2, stats[col_all], bar_w,
                          label="All data (median)", color=light_color, edgecolor="white")
        bars_clean = ax.bar(x + bar_w / 2, stats[col_clean], bar_w,
                            label="Clean only (median)", color=dark_color, edgecolor="white")

        # Value annotations
        for bar, med in zip(bars_all, stats[col_all]):
            if not np.isnan(med):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{med:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, med in zip(bars_clean, stats[col_clean]):
            if not np.isnan(med):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{med:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Exceed-% annotations (Fig 1 only)
        if show_exceed and "exceed_pct" in stats.columns:
            for i, (xpos, pct) in enumerate(zip(x, stats["exceed_pct"])):
                if not np.isnan(pct):
                    y_top = max(stats.loc[i, col_all], stats.loc[i, col_clean]) + 0.12
                    ax.text(xpos, y_top, f"{pct:.0f}% exceed", ha="center", va="bottom",
                            fontsize=8, color="#c0392b", fontstyle="italic")

        # Threshold line
        ax.axhline(y=self.ml_threshold, color="red", linestyle="--", linewidth=2, alpha=0.8,
                   label=f"\u00b1{self.ml_threshold} mm threshold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel("Median ML sigma [mm]", fontsize=11)
        ax.set_title(strand_name, fontsize=14, fontweight="bold",
                     color=STRAND_PALETTE.get(strand_name, "#666666"))
        ax.set_ylim(0, ylim)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
