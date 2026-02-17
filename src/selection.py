
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from .config import RANDOM_STATE

def trim_features(X, names, sources, y, corr_thresh=0.95):
    """
    Data-driven feature selection with automated optimal count determination.

    Pipeline:
    1. Variance filter: Remove near-constant features
    2. Correlation filter: Remove highly correlated features
    3. Importance ranking: Use LassoCV (or RF fallback) to rank features
    4. CV-based selection: Find optimal number via cross-validation

    Returns: X_final, names_final, sources_final, selection_results (for plotting)
    """
    print('\nRunning feature selection...')

    # Step 1: Variance filter
    sel_var = VarianceThreshold(threshold=1e-4)
    X_var = sel_var.fit_transform(X)
    mask = sel_var.get_support()
    names_var = [n for i, n in enumerate(names) if mask[i]]
    sources_var = [s for i, s in enumerate(sources) if mask[i]]
    print(f'  Variance filter: {len(names)} -> {len(names_var)}')

    # Step 2: Correlation filter
    df_X = pd.DataFrame(X_var, columns=names_var)
    corr = df_X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set(c for c in upper.columns if any(upper[c] > corr_thresh))

    keep_mask = [n not in to_drop for n in names_var]
    X_uncorr = df_X.loc[:, keep_mask].values
    names_uncorr = [n for n, k in zip(names_var, keep_mask) if k]
    sources_uncorr = [s for s, k in zip(sources_var, keep_mask) if k]
    print(f'  Correlation filter: {len(names_var)} -> {len(names_uncorr)}')

    # Step 3: Rank features by importance
    print('  Ranking features by importance...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_uncorr)
    lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=2000).fit(X_scaled, y)
    importances = np.abs(lasso.coef_)

    if np.sum(importances > 0) < min(50, len(importances)):
        print('    Lasso too sparse, using RF importance...')
        rf = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=RANDOM_STATE)
        rf.fit(X_uncorr, y)
        importances = rf.feature_importances_

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]

    # Step 4: Find optimal number of features via CV
    print('  Finding optimal feature count via CV...')

    # Test different numbers of features
    max_test = min(len(sorted_idx), 600)
    n_features_to_test = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600]
    n_features_to_test = [n for n in n_features_to_test if n <= max_test]

    selection_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for n_feat in n_features_to_test:
        top_idx = sorted_idx[:n_feat]
        X_subset = X_uncorr[:, top_idx]

        maes = []
        for train_idx, test_idx in kf.split(X_subset):
            model = Ridge(alpha=1.0)
            model.fit(X_subset[train_idx], y[train_idx])
            pred = model.predict(X_subset[test_idx])
            maes.append(mean_absolute_error(y[test_idx], pred))

        selection_results.append({
            'n_features': n_feat,
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes)
        })
        print(f'    n={n_feat:4d}: MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}')

    # Find optimal: best MAE or within 1-std (prefer simpler)
    best_result = min(selection_results, key=lambda x: x['mae_mean'])
    best_mae = best_result['mae_mean']
    best_std = best_result['mae_std']

    # 1-std rule: choose smallest n_features within 1 std of best
    threshold = best_mae + best_std
    for res in selection_results:
        if res['mae_mean'] <= threshold:
            optimal_n = res['n_features']
            break
    else:
        optimal_n = best_result['n_features']

    print(f'  Optimal features (1-std rule): {optimal_n} (Best MAE: {best_mae:.4f} at n={best_result["n_features"]})')

    # Select optimal number of features
    final_idx = sorted_idx[:optimal_n]
    X_final = X_uncorr[:, final_idx]
    names_final = [names_uncorr[i] for i in final_idx]
    sources_final = [sources_uncorr[i] for i in final_idx]

    print(f'  Final selection: {len(names_uncorr)} -> {len(names_final)}')

    return X_final, names_final, sources_final, selection_results
