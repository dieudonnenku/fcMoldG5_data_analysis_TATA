%md
# 🏭 FC Mold G5 - Scalable Multi-Strand Analysis Pipeline
## TATA IJmuiden CC23 Continuous Casting

### 🎯 Project Objectives
* **Identify stable casting sequences** across multiple strands
* **Remove sensor artifacts** through intelligent filtering
* **Relate mold level stability** to process parameters
* **Enable scalable analysis** for CC23 strands (23_5 and 23_6)

### 📊 Data Sources
* **dtExpert**: 1Hz frequency - EMBR currents & casting parameters
* **boExpert**: 2Hz frequency - FBG temperature & casting parameters
* **Metadata**: Casting quality records with temporal boundaries

### 🔧 Architecture
This notebook implements a **modular, class-based pipeline** that:
* Supports **multi-strand analysis** with parameterized configuration
* Uses **reusable components** (DataLoader, SequenceAnalyzer, Visualizer)
* Generates **strand-specific outputs** with consistent naming
* Enables **parallel processing** and easy extension to new strands

---
**Last Updated**: 2026-02-09

