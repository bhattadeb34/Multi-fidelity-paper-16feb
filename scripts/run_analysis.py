
#!/usr/bin/env python3
"""
run_analysis.py
===============
Main orchestration script for the Proton Affinity Analysis Pipeline.
Reproduces Method V3 (Publication Quality).
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import OUT_DIR
from src.utils import safe_float
from src.data import load_data
from src.features import build_feature_matrix
from src.selection import trim_features
from src.models import get_all_models, cross_val_with_metrics, get_best_model_name
from src.plotting import (
    plot_feature_composition, plot_data_exploration, plot_model_parity_grid,
    plot_model_benchmark, plot_learning_curve, plot_extrapolation,
    plot_interpolation, plot_shap_analysis, plot_feature_importance,
    plot_error_analysis, plot_error_distribution, plot_best_worst,
    save_summary
)

def main():
    print('=' * 60)
    print('COMPREHENSIVE PA ANALYSIS - PUBLICATION PIPELINE')
    print('=' * 60)


    import argparse
    parser = argparse.ArgumentParser(description='Run Comprehensive Proton Affinity Analysis Pipeline')
    parser.add_argument('--recompute', action='store_true', help='Force recomputation of features (ignore cache)')
    args = parser.parse_args()

    # 1. Load Data
    df, d_exp, y_exp, y_pm7 = load_data()
    y_delta = y_exp - y_pm7
    smiles = df['can'].values

    # 2. Build Features
    # Note: compute_3d=True computes descriptors on the fly (slow). 
    # Set to False if not needed or reuse cache if implemented.
    X_all, names_all, sources_all, initial_counts = build_feature_matrix(
        df, d_exp, compute_3d=True, recompute=args.recompute
    )

    # 3. Feature Selection
    X_sel, names_sel, sources_sel, selection_results = trim_features(X_all, names_all, sources_all, y_delta)
    from collections import Counter
    selected_counts = dict(Counter(sources_sel))

    # 4. Model Training & CV
    models = get_all_models()
    print('\n' + '=' * 60)
    print('CROSS-VALIDATION')
    print('=' * 60)
    
    # Pass y_pm7 as baseline for reconstruction of absolute PA
    results, cv_predictions = cross_val_with_metrics(models, X_sel, y_delta, y_pm7)

    # 5. Best Model Identification
    best_model_name = get_best_model_name(results)
    best_model = models[best_model_name]
    best_preds_data = cv_predictions[best_model_name]
    best_pred_pa = best_preds_data['test_pred'] # This is absolute PA (delta + pm7)
    
    # Re-calculate error based on best_pred_pa
    errors = np.abs(y_exp - best_pred_pa)

    print(f'\nBest Model: {best_model_name} (Test MAE: {results[best_model_name]["test_mae_mean"]:.3f})')

    # 6. Figures & Output
    print('\n' + '=' * 60)
    print('GENERATING FIGURES')
    print('=' * 60)

    plot_feature_composition(initial_counts, selected_counts, selection_results)
    plot_data_exploration(df, y_exp, y_pm7, names_sel, sources_sel)
    plot_model_parity_grid(cv_predictions, results, len(y_exp))
    plot_model_benchmark(results, y_exp, y_pm7)
    plot_learning_curve(best_model, X_sel, y_delta, y_pm7)
    plot_extrapolation(X_sel, y_exp, y_pm7, y_delta, best_model)
    plot_interpolation(X_sel, y_exp, y_pm7, y_delta, best_model)
    
    save_summary(results, y_exp, y_pm7, names_sel, sources_sel, best_model_name)

    # Save Best Predictions
    pd.DataFrame({
        'smiles': smiles,
        'y_exp': y_exp, 
        'y_pm7': y_pm7, 
        'y_pred_final': best_pred_pa,
        'error': errors
    }).to_csv(f'{OUT_DIR}/cv_predictions_best.csv', index=False)
    print(f'  Saved CV Predictions to {OUT_DIR}/cv_predictions_best.csv')

    # Advanced Analysis
    plot_shap_analysis(best_model, best_model_name, X_sel, y_delta, names_sel, smiles, y_exp, best_pred_pa, models, results)
    plot_feature_importance(best_model, best_model_name, X_sel, y_delta, names_sel, sources_sel)
    plot_error_analysis(smiles, errors, y_exp)
    plot_error_distribution(y_exp, best_pred_pa, y_pm7)
    plot_best_worst(smiles, errors, y_exp, best_pred_pa, y_pm7, best_model_name)

    print('\n' + '=' * 60)
    print('COMPLETE')
    print(f'Outputs: {OUT_DIR}/')
    print('=' * 60)

if __name__ == '__main__':
    main()
