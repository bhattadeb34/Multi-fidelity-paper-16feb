
# Multi-fidelity Proton Affinity Prediction

This repository contains the publication-quality codebase for the analysis of Proton Affinity (PA) using a multi-fidelity approach combining PM7 semi-empirical calculations and machine learning.

## Repository Structure

- `data/`: Contains dataset CSVs and pre-computed features.
  - `Dataset.csv`: Main dataset with SMILES and Experimental PA.
  - `FINAL_PM7_ALL_neutral_cleaned.csv`: PM7 neutral state data.
  - `FINAL_PM7_ALL_protonated_cleaned.csv`: PM7 protonated state data.
  - `mordred_features_all.csv`: Pre-computed Mordred descriptors.
- `src/`: Core Python modules for the analysis pipeline.
  - `config.py`: Configuration settings (paths, constants).
  - `data.py`: Data loading and preprocessing.
  - `features.py`: Feature engineering (Mordred, MACCS, Morgan, RDKit, PM7, 3D).
  - `selection.py`: Feature selection logic.
  - `models.py`: Model definitions and cross-validation framework.
  - `plotting.py`: Visualization functions.
  - `utils.py`: Utility functions.
  - `bottchscore.py`: Böttcher complexity score implementation.
- `scripts/`: Executable scripts.
  - `run_analysis.py`: Main orchestration script to run the full analysis and generate figures.
  - `calculate_complexity.py`: Script to calculate dataset complexity scores.
  - `calculate_mordred.py`: Script to generate Mordred descriptors (computationally expensive).
- `notebooks/`: Jupyter notebooks for exploratory analysis and prototyping.
- `figures/`: Output directory for generated figures.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   python scripts/run_analysis.py
   ```
   This will load data, perform feature engineering/selection, run cross-validation, and generate figures in the `figures/` directory (created automatically).

3. **Auxiliary Scripts**:
   - To calculate complexity scores: `python scripts/calculate_complexity.py`
   - To re-generate Mordred descriptors: `python scripts/calculate_mordred.py`

## Features

- **Comprehensive Feature Set**: Combines 1D (MACCS, Morgan), 2D (RDKit, Mordred), and Quantum (PM7) features.
- **Automated Selection**: Variance thresholding, correlation filtering, and recursive feature elimination.
- **Robust Evaluation**: 5-fold cross-validation with "1-std rule" for parsimonious model selection.
- **Visualization**: Extensive plotting including SHAP analysis, error distribution, and learning curves.
