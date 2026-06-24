# FC Mold G5 - TATA IJmuiden CC23 Analysis

Mold level stability analysis for continuous casting machine CC23, Strands 5 and 6.

## Project Structure

```
TATAIjmulden_FCMoldG5/
├── README.md                      ← You are here
├── run_pipeline.ipynb             ← Path A: production pipeline (imports src/, end-to-end)
├── explore_step_by_step.ipynb     ← Path B: clean step-by-step exploration (25 cells)
├── generate_onboarding_ppt.ipynb  ← Generates onboarding PowerPoint (19 slides)
├── test_pipeline.ipynb            ← Validation and smoke tests
├── EDA_data_grouping.ipynb        ← Archive: original EDA (157 cells, reference only)
│
├── src/                           ← Production Python modules (single source of truth)
│   ├── __init__.py
│   ├── config.py                  ← ALL paths, thresholds, strand configs
│   ├── data_loading.py            ← File discovery, Spark loading, unit conversion
│   ├── sequence_analysis.py       ← Sliding window, SequenceAnalyzer class
│   ├── disturbance_detection.py   ← Excursion, drift, bump, variability detectors
│   ├── feature_engineering.py     ← Spark-derived features (FBG, Chebyshev, asymmetry)
│   ├── export.py                  ← ResultsExporter (CSV, Parquet, text summary)
│   ├── visualization.py           ← ReportVisualizer (all figures, optional PNG export)
│   └── pipeline.py                ← StrandAnalysisPipeline orchestrator
│
├── figures/                       ← Generated plots (HTML, PNG, PPTX)
└── reports/                       ← Exported CSVs, Word docs, summaries
```

## Two Execution Paths

| | Path A (`run_pipeline`) | Path B (`explore_step_by_step`) |
|--|------------------------|----------------------------------|
| Cells | ~6 total | 25 total |
| Speed | ~5 min end-to-end | Run section by section |
| Use case | Production, CI/CD, batch runs | Debugging, learning, prototyping |
| Code source | Imports from `src/` | Imports from `src/` (same code) |
| Output | `all_results` dict | Individual DataFrames + inline plots |

**Rule:** Both paths import from `src/` — fix once, works everywhere.

## Quick Start (Path A)

1. Open `run_pipeline` notebook
2. Edit **Run Settings** in cell 3 (strand, export, table display)
3. Run All — pipeline executes, figures display + save to DBFS
4. Results are in `all_results` dict with `df_seq`, `df_raw` per strand

### Run Settings (cell 3)

```python
STRAND            = "both"   # "both" | "23_6" | "23_5"
EXPORT_RESULTS    = True     # Save CSV + Parquet + figures to DBFS
SHOW_STABLE_TABLE = True     # Display filtered stable-sequence table
```

No widgets — just edit the variables and Run All.

## How to Run Path B (Step-by-Step Exploration)

Open `explore_step_by_step.ipynb` — a clean 25-cell notebook.

### Prerequisites

- Cluster: DBR 16.x ML, Standard_D32a_v4 (32 GB RAM minimum)
- The `src/` folder must exist and be populated

### Steps

| Cell | Title | What it does |
|------|-------|--------------|
| 2 | Install Dependencies | `%pip install mpl-scatter-density astropy` |
| 3 | Setup: Imports from src/ | Loads all shared modules, prints CONFIG |
| 4 | Select Strand | Set `STRAND_ID = "23_6"` (or `"23_5"`) |
| 6–10 | Data Loading | List files, load parquet, aggregate 2Hz→1Hz, join, convert units |
| 12–14 | Metadata Join | Attach quality labels, check coverage, convert to Pandas |
| 16 | FC Mold Filtering | Keep only EMBR-active rows |
| 18–19 | Sequence ID | Sliding window → stable sequences → per-sequence stats |
| 21 | Disturbance Detection | Classify: Excursion, Drift, Bump, High Variability, Normal |
| 23–25 | Results & Viz | Histograms, scatter correlations, summary table |

### Key Variables After Running

| Variable | Type | Description |
|----------|------|-------------|
| `df` | pd.DataFrame | Full cleaned data (all rows, FC Mold on + off) |
| `df_fc` | pd.DataFrame | FC Mold active subset (EMBR currents ≠ 0) |
| `df_seq` | pd.DataFrame | Per-sequence statistics (1168 sequences for S6) |
| `df_clean` | pd.DataFrame | Clean (Normal) sequences only (921 for S6) |
| `normal_groups` | list[list[int]] | Index arrays for each stable sequence |

### Tips for New Users

- Change `STRAND_ID` in cell 4 to switch between Strand 5 and 6
- You can run any section independently after cells 2–3 (install + imports)
- Thresholds come from `CONFIG` — change in `src/config.py` to affect both paths
- If `ModuleNotFoundError` after pip install, restart Python and re-run cell 3
- Use `display(df_seq.head())` to inspect DataFrames at any point

## Key Concepts

| Term | Meaning |
|------|---------|
| Stable sequence | 6-min window where casting speed varies < 0.1 m/min AND EMBR currents are steady |
| Mold level sigma | Standard deviation of mold level within a sequence (target: < ±1 mm) |
| FC Mold active | EMBR currents all non-zero (electromagnetic braking is ON) |
| Disturbance | Excursion, slow drift, transient bump, or high variability in mold level signal |

## Data Sources

| Source | Frequency | Contents |
|--------|-----------|----------|
| boExpert | 2 Hz | FBG temperatures, casting parameters |
| dtExpert | 1 Hz | EMBR currents, casting parameters |
| Metadata CSV | per casting | Quality labels, start/end times |

**DBFS paths:**
- Strand 5: `dbfs:/FileStore/TATA_IJmuiden_CC23/data/strand_5`
- Strand 6: `dbfs:/FileStore/TATA_IJmuiden_CC23/data/strand_6`
- Metadata: `dbfs:/FileStore/TATA_IJmuiden_CC23/data/Castings_TSN_2025_April_May_merged.csv`
- Grade mapping: `dbfs:/FileStore/TATAIjmulden_FCMoldG5/CastingGroups_ABB_April2026.xlsx`

## Configuration

All thresholds are defined in `src/config.py` → `AnalysisConfig`. Key values:

- `window_size = 300` (6 min at 1 Hz)
- `vc_threshold = 0.1` m/min
- `curr_threshold = 50` A
- `ml_stability_threshold_mm = 2.0` (sigma)
- `excursion_threshold_mm = 8.0`

## Related Assets

- **Technical Report:** `/Repos/.../fcMoldG5_data_analysis_TATA/FC Mold G5 Technical Report CC23`
- **Reusable package:** `src/` (9 modules — the single source of truth)
- **Grade classification:** 19 TATA casting groups from `CastingGroups_ABB_April2026.xlsx`
- **Onboarding PPT:** Run `generate_onboarding_ppt` → saves to `figures/onboarding_FC_Mold_G5.pptx`

## Time Coverage

April–May 2025 (CC23 Strands 5 and 6)

## What to Change (New Environment Setup)

If you are setting up this project in a **new workspace** or under a **different user**, update these locations:

### 1. `src/config.py` — Path Constants

| Variable | Current Value | What it is |
|----------|--------------|------------|
| `WORKSPACE_ROOT` | `/Workspace/Users/dieudonne.nkulikiyimfura@se.abb.com/TATAIjmulden_FCMoldG5` | Your project folder |
| `DBFS_DATA_BASE` | `dbfs:/FileStore/TATA_IJmuiden_CC23/data` | Raw parquet/CSV sensor data |
| `DBFS_OUTPUT_BASE` | `/dbfs/FileStore/TATAIjmulden_FCMoldG5` | Generated outputs (HTML, CSV, PNG) |
| `METADATA_PATH` | Auto-derived from `DBFS_DATA_BASE` | Casting metadata CSV |
| `GRADE_MAPPING_PATH` | Under `DBFS_OUTPUT_BASE` | Steel grade → casting group Excel |

### 2. `src/config.py` — Strand Data Paths

```python
STRAND_CONFIGS["23_6"].data_path = "dbfs:/FileStore/TATA_IJmuiden_CC23/data/strand_6"
STRAND_CONFIGS["23_5"].data_path = "dbfs:/FileStore/TATA_IJmuiden_CC23/data/strand_5"
```

Change these if your data is in a different DBFS location or Unity Catalog volume.

### 3. `explore_step_by_step` — `project_root`

```python
project_root = "/Workspace/Users/<YOUR_EMAIL>/TATAIjmulden_FCMoldG5"
```

Update in cell 3 of `explore_step_by_step`.

> **Note:** `run_pipeline` uses `os.getcwd()` automatically — no manual path needed
> as long as you run it from within the repo.

### 4. `src/config.py` — Analysis Thresholds (lines 12-38)

These are **domain parameters** — only change if the process changes:
- `ml_stability_threshold_mm = 2.0` — mold level sigma threshold (ask domain expert)
- `window_size = 300` — sliding window duration (6 min at 1 Hz)
- `excursion_threshold_mm = 8.0` — what counts as a large deviation

### 5. Adding a New Strand

1. Add a new entry in `STRAND_CONFIGS` dict in `src/config.py`
2. Upload the strand data to a DBFS folder
3. Set `data_path` to the new folder
4. Set `embr_current_cols` to match the column names in that strand's data

### Cluster Requirements

- **Runtime:** DBR 16.x ML (includes scipy, numpy, pandas)
- **Node type:** Standard_D32a_v4 or similar (32 GB RAM minimum)
- **Extra libraries:** `mpl-scatter-density`, `astropy` (installed by cell 2 of Path B)
- **For PPT generation:** `python-pptx` (installed by `generate_onboarding_ppt`)

## New Workspace Setup

After cloning this repo in a new workspace:

1. Update `src/config.py` → `WORKSPACE_ROOT` (your email)
2. Update `src/config.py` → `DBFS_DATA_BASE` (if data is in a different location)
3. Upload sensor data to DBFS (see Data Sources section above for expected paths)
4. Run `test_pipeline` → all assertions should pass
5. Run `run_pipeline` → verify end-to-end execution

> **Note:** `run_pipeline` and `test_pipeline` use `os.getcwd()` to resolve `src/` —
> no manual path configuration needed as long as you run from the repo root.
