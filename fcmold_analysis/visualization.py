"""Strand-specific visualization generation (Plotly + Matplotlib)."""

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from fcmold_analysis.config import AnalysisConfig, StrandConfig


class VisualizationFactory:
    """Generate strand-specific charts with automatic file naming."""

    def __init__(self, strand_config: StrandConfig, analysis_config: AnalysisConfig):
        self.strand_config = strand_config
        self.config = analysis_config
        self._prefix = f"[{strand_config.strand_name}]"
        self.ml_cmap = LinearSegmentedColormap.from_list(
            "ml_sigma_cmap",
            [
                (0.0, "#440053"),
                (0.3, "#404388"),
                (0.6, "#2a788e"),
                (0.8, "#21a784"),
                (0.95, "#78d151"),
                (1.0, "#fde624"),
            ],
            N=256,
        )

    def _log(self, msg: str):
        print(f"{self._prefix} {msg}")

    def save_figure(self, fig, base_name: str, fmt: str = "html"):
        filename = self.strand_config.get_output_filename(base_name, fmt)
        filepath = f"{self.strand_config.figures_dir}/{filename}"
        local = filepath.replace("/dbfs", "")
        if fmt == "html":
            fig.write_html(local)
        elif fmt == "png":
            fig.savefig(local, dpi=300, bbox_inches="tight")
        self._log(f"Saved: {filename}")
        return filepath

    # -- plots --------------------------------------------------------------

    def plot_disturbance_examples(
        self,
        df_fc_mold: pd.DataFrame,
        df_seq: pd.DataFrame,
        sequences: List[List[int]],
        example_indices: dict | None = None,
    ):
        self._log("Creating disturbance-type examples…")
        if example_indices is None:
            example_indices = (
                df_seq.groupby("disturbance_type").head(1).reset_index()["index"].to_dict()
            )
        types = list(example_indices.keys())
        nrows = len(types)

        fig = make_subplots(
            rows=nrows, cols=1, shared_xaxes=False, vertical_spacing=0.08,
            subplot_titles=[
                f"Disturbance type: {t}  |  Sequence index: {example_indices[t]}" for t in types
            ],
        )
        for r, dtype in enumerate(types, 1):
            idx = example_indices[dtype]
            raw = df_fc_mold.iloc[sequences[idx]].copy()
            mean_v = raw["Mold Level"].mean()
            fig.add_trace(
                go.Scatter(x=raw["plainTimeStamp"], y=raw["Mold Level"], mode="lines", name=dtype),
                row=r, col=1,
            )
            fig.add_hline(y=mean_v, line_dash="dash", line_color="green", row=r, col=1)
            fig.update_yaxes(title_text="Mold Level [mm]", row=r, col=1)

        fig.update_xaxes(title_text="Time", row=nrows, col=1)
        fig.update_layout(
            height=320 * nrows, width=1200,
            title=f"{self.strand_config.strand_name} – Representative sequences per disturbance type",
            showlegend=False,
        )
        self.save_figure(fig, "disturbance_type_examples", "html")
        fig.show()

    def plot_correlation_scatter(self, df_seq: pd.DataFrame, filter_normal_only: bool = False):
        self._log("Creating correlation scatter…")
        dfp = df_seq.copy()
        if filter_normal_only:
            dfp = dfp[dfp["disturbance_type"] == "Normal"]

        x = dfp["MOLD_WIDTH_avg [m]"].to_numpy()
        y = dfp["CASTING_SPEED_avg [m/min]"].to_numpy()
        ml_std = dfp["MOLD_LEVEL_std [mm]"].to_numpy()
        dtype = dfp["disturbance_type"].astype(str).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(ml_std)
        x, y, ml_std, dtype = x[mask], y[mask], ml_std[mask], dtype[mask]
        above = ml_std > self.config.ml_stability_threshold_mm

        order = ["Normal", "High variability", "Transient bump", "Slow drift", "Excursion"]
        present = [c for c in order if c in set(dtype)]
        present += [c for c in np.unique(dtype) if c not in present]
        colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = {cat: colors_cycle[i % len(colors_cycle)] for i, cat in enumerate(present)}
        pcols = np.array([cmap[c] for c in dtype])

        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.scatter(x, y, c=pcols, s=40, edgecolor="none", alpha=0.8)
        ax.scatter(x[above], y[above], c=pcols[above], s=70, edgecolor="black", linewidth=0.9, alpha=0.95)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap[c], markersize=8, label=c)
            for c in present
        ]
        ax.legend(handles=handles, title="Disturbance type", loc="best")
        ax.set_xlabel("Mold Width avg [m]")
        ax.set_ylabel("Casting Speed avg [m/min]")
        suffix = " (Normal only)" if filter_normal_only else ""
        ax.set_title(f"{self.strand_config.strand_name} – Mold Width vs Casting Speed{suffix}")
        tag = "correlation_scatter_normal" if filter_normal_only else "correlation_scatter_all"
        self.save_figure(fig, tag, "png")
        plt.show()

    def plot_correlation_heatmap(self, df_seq: pd.DataFrame):
        self._log("Creating correlation heatmap (Normal only)…")
        dfp = df_seq[df_seq["disturbance_type"] == "Normal"].copy()
        x = dfp["MOLD_WIDTH_avg [m]"].to_numpy()
        y = dfp["CASTING_SPEED_avg [m/min]"].to_numpy()
        ml_std = dfp["MOLD_LEVEL_std [mm]"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(ml_std)
        x, y, ml_std = x[mask], y[mask], ml_std[mask]
        above = ml_std > self.config.ml_stability_threshold_mm

        norm = TwoSlopeNorm(vmin=0, vcenter=self.config.ml_stability_threshold_mm, vmax=np.nanmax(ml_std))
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        sc = ax.scatter(x, y, c=ml_std, cmap=self.ml_cmap, norm=norm, s=40, edgecolor="none", alpha=0.85)
        ax.scatter(x[above], y[above], c=ml_std[above], cmap=self.ml_cmap, norm=norm, s=70, edgecolor="black", linewidth=0.9, alpha=0.95)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("MOLD_LEVEL_std [mm]")
        cbar.ax.axhline(self.config.ml_stability_threshold_mm, color="black", linewidth=1)
        ax.set_xlabel("Mold Width avg [m]")
        ax.set_ylabel("Casting Speed avg [m/min]")
        ax.set_title(f"{self.strand_config.strand_name} – Normal sequences (colour = σ)")
        self.save_figure(fig, "correlation_heatmap_normal", "png")
        plt.show()

    def plot_parameter_correlations(self, df_seq: pd.DataFrame):
        self._log("Creating parameter-correlation plots…")
        df = df_seq[df_seq["disturbance_type"] == "Normal"].copy()
        if "Quality_casting" not in df.columns and "Quality casting" in df.columns:
            df["Quality_casting"] = df["Quality casting"]

        fig = make_subplots(
            rows=2, cols=2, shared_xaxes=False, shared_yaxes=False,
            vertical_spacing=0.12, horizontal_spacing=0.10,
            subplot_titles=[
                "Vc vs Mold Level σ", "Mold Width vs Mold Level σ",
                "Argon Flow vs Mold Level σ", "Quality Casting vs Mold Level σ",
            ],
        )
        panels = [
            (1, 1, "CASTING_SPEED_avg [m/min]", False),
            (1, 2, "MOLD_WIDTH_avg [m]", False),
            (2, 1, "ArFlow_avg [NL/min]", False),
            (2, 2, "Quality_casting", True),
        ]
        for r, c, xcol, is_cat in panels:
            if xcol not in df.columns:
                continue
            x_plot = df[xcol].astype(str) if is_cat else df[xcol].round(1)
            fig.add_trace(
                go.Box(x=x_plot, y=df["MOLD_LEVEL_std [mm]"], marker_color="#bdb76b", opacity=0.45, showlegend=False),
                row=r, col=c,
            )
            fig.add_trace(
                go.Scatter(x=x_plot, y=df["MOLD_LEVEL_std [mm]"], mode="markers", marker=dict(color="#bdb76b", size=7, opacity=0.8), showlegend=False),
                row=r, col=c,
            )
            fig.update_yaxes(title_text="Mold Level σ [mm]", row=r, col=c)

        fig.update_layout(
            height=900, width=1200,
            title=f"{self.strand_config.strand_name} – Normal process: parameter correlations",
            boxmode="overlay", showlegend=False,
        )
        self.save_figure(fig, "parameter_correlations", "html")
        fig.show()
