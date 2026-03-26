"""FC Mold G5 - Scalable Multi-Strand Analysis Pipeline for TATA IJmuiden CC23."""

from fcmold_analysis.config import AnalysisConfig, StrandConfig, STRAND_CONFIGS, METADATA_PATH
from fcmold_analysis.feature_engineering import engineer_features
from fcmold_analysis.pipeline import StrandAnalysisPipeline
