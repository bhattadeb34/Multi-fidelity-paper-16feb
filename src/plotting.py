
import os
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, Descriptors
from .config import OUT_DIR, RANDOM_STATE

# Set style globally or in init? Using matplotlib config from main usually best.
# But we can set defaults here too.

def plot_feature_composition(initial_counts, selected_counts, selection_results):
    """Plot feature composition: initial vs selected counts by source + CV selection curve."""
    print('Generating feature composition plot...')

    all_sources = ['Mordred', 'MACCS', 'Morgan', 'RDKit', 'PM7', '3D']
    sources = [s for s in all_sources if s in initial_counts or s in selected_counts]

    initial = [initial_counts.get(s, 0) for s in sources]
    selected = [selected_counts.get(s, 0) for s in sources]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A
    ax1 = axes[0, 0]
    x = np.arange(len(sources))
    width = 0.35
    bars1 = ax1.bar(x - width/2, initial, width, label='Initial', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, selected, width, label='Selected', color='#e74c3c', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sources, rotation=45, ha='right')
    ax1.set_ylabel('Feature Count')
    ax1.set_title(f'A. Feature Counts by Source\n(Total: {sum(initial)} → {sum(selected)})')
    ax1.legend()
    ax1.bar_label(bars1, fmt='%d', fontsize=8)
    ax1.bar_label(bars2, fmt='%d', fontsize=8)

    # Panel B
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    bottom_init = 0
    bottom_sel = 0
    for i, (src, col) in enumerate(zip(sources, colors)):
        ax2.bar('Initial', initial[i], bottom=bottom_init, color=col, edgecolor='black', label=src)
        ax2.bar('Selected', selected[i], bottom=bottom_sel, color=col, edgecolor='black')
        bottom_init += initial[i]
        bottom_sel += selected[i]
    ax2.set_ylabel('Feature Count')
    ax2.set_title('B. Feature Composition')
    ax2.legend(loc='upper right', fontsize=9)

    # Panel C
    ax3 = axes[1, 0]
    retention = [100 * selected[i] / initial[i] if initial[i] > 0 else 0 for i in range(len(sources))]
    bars3 = ax3.bar(sources, retention, color=colors, edgecolor='black')
    ax3.set_ylabel('Retention Rate (%)')
    ax3.set_title('C. Feature Retention After Selection')
    ax3.set_ylim(0, max(retention) * 1.15)
    ax3.bar_label(bars3, fmt='%.1f%%', fontsize=9)
    ax3.tick_params(axis='x', rotation=45)

    # Panel D
    ax4 = axes[1, 1]
    n_feats = [r['n_features'] for r in selection_results]
    maes = [r['mae_mean'] for r in selection_results]
    stds = [r['mae_std'] for r in selection_results]

    ax4.errorbar(n_feats, maes, yerr=stds, marker='o', capsize=4, color='#2c3e50', linewidth=2, markersize=6)

    best_idx = np.argmin(maes)
    best_mae = maes[best_idx]
    best_std = stds[best_idx]
    threshold = best_mae + best_std

    optimal_idx = 0
    for i, mae in enumerate(maes):
        if mae <= threshold:
            optimal_idx = i
            break

    optimal_n = n_feats[optimal_idx]
    ax4.axhline(threshold, color='gray', linestyle='--', alpha=0.7, label=f'1-std threshold: {threshold:.4f}')
    ax4.scatter([n_feats[best_idx]], [maes[best_idx]], color='blue', s=150, zorder=5, marker='*', label=f'Best MAE: {best_mae:.4f} @ n={n_feats[best_idx]}')
    ax4.scatter([optimal_n], [maes[optimal_idx]], color='red', s=150, zorder=5, marker='D', edgecolor='black', label=f'Selected (1-std): n={optimal_n}')

    ax4.set_xlabel('Number of Features')
    ax4.set_ylabel('CV MAE (kcal/mol)')
    ax4.set_title('D. Automated Feature Selection\n(5-Fold CV with Ridge)')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Feature Engineering Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '00_feature_composition.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 00_feature_composition.png')


def plot_data_exploration(df, y_exp, y_pm7, feature_names, feature_sources):
    """Data exploration plots."""
    print('Generating data exploration...')

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. PA Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(y_exp, bins=40, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(y_exp.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {y_exp.mean():.1f}')
    ax1.set_xlabel('Proton Affinity (kcal/mol)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'A. PA Distribution (N={len(y_exp)})')
    ax1.legend()

    # 2. Feature pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    source_counts = Counter(feature_sources)
    labels = list(source_counts.keys())
    sizes = list(source_counts.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'B. Feature Distribution (Total: {len(feature_names)})')

    # 3. PM7 vs Experimental
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_exp, y_pm7, alpha=0.5, s=20, c='#e74c3c')
    lims = [min(y_exp.min(), y_pm7.min()), max(y_exp.max(), y_pm7.max())]
    ax3.plot(lims, lims, 'k--')
    ax3.text(0.05, 0.95, f'MAE={mean_absolute_error(y_exp, y_pm7):.2f}\nR²={r2_score(y_exp, y_pm7):.3f}',
             transform=ax3.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.set_xlabel('Experimental PA')
    ax3.set_ylabel('PM7 PA')
    ax3.set_title('C. PM7 Baseline')

    # 4. Feature source bar
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(labels, sizes, color=colors)
    ax4.bar_label(bars, fmt='%d')
    ax4.set_xlabel('Feature Source')
    ax4.set_ylabel('Count')
    ax4.set_title('D. Features per Source')
    ax4.tick_params(axis='x', rotation=45)

    # 5. Delta (correction) distribution
    ax5 = fig.add_subplot(gs[1, 1])
    delta = y_exp - y_pm7
    ax5.hist(delta, bins=40, alpha=0.7, color='#27ae60', edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', lw=2)
    ax5.set_xlabel('Correction = Exp - PM7 (kcal/mol)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('E. Correction Distribution')

    # 6. Molecular weight vs PA
    smiles = df['can'].values
    mol_weights = [Descriptors.ExactMolWt(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else 200 for s in smiles]
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(mol_weights, y_exp, c=y_pm7 - y_exp, cmap='coolwarm', alpha=0.6, s=20)
    ax6.set_xlabel('Molecular Weight')
    ax6.set_ylabel('Experimental PA')
    ax6.set_title('F. PA vs Mol Weight')

    plt.suptitle('Data Exploration', fontsize=16)
    plt.savefig(os.path.join(OUT_DIR, '01_data_exploration.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 01_data_exploration.png')


def plot_model_parity_grid(cv_predictions, results, n_total):
    """Parity plots."""
    print('Generating model parity grid...')

    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_mae_mean'])
    n_models = len(sorted_models)
    cols = 3
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(sorted_models):
        ax = axes[idx]
        preds = cv_predictions[name]

        ax.scatter(preds['test_true'], preds['test_pred'], alpha=0.4, s=15, c='#3498db')

        lims = [min(preds['test_true'].min(), preds['test_pred'].min()) - 5,
                max(preds['test_true'].max(), preds['test_pred'].max()) + 5]
        ax.plot(lims, lims, 'r--', lw=1.5)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Experimental PA (kcal/mol)')
        ax.set_ylabel('Predicted PA (kcal/mol)')
        ax.set_title(f'{name}\nMAE={res["test_mae_mean"]:.2f}±{res["test_mae_std"]:.2f}, R²={res["test_r2_mean"]:.3f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'5-Fold Cross-Validation Parity Plots (N={n_total})\nEach point = one molecule\'s out-of-sample prediction',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '02_model_parity_grid.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 02_model_parity_grid.png')


def plot_model_benchmark(results, y_exp, y_pm7):
    """Bar chart with train/test MAE."""
    print('Generating model benchmark...')

    pm7_mae = mean_absolute_error(y_exp, y_pm7)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_mae_mean'])

    names = [n for n, _ in sorted_models] + ['PM7']
    train_maes = [r['train_mae_mean'] for _, r in sorted_models] + [pm7_mae]
    test_maes = [r['test_mae_mean'] for _, r in sorted_models] + [pm7_mae]
    test_stds = [r['test_mae_std'] for _, r in sorted_models] + [0]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, train_maes, width, label='Train MAE', color='#3498db', alpha=0.8)
    bars = ax.bar(x + width/2, test_maes, width, yerr=test_stds, label='Test MAE', color='#e74c3c', capsize=3)

    ax.set_ylabel('MAE (kcal/mol)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_title('Model Benchmark: 5-Fold CV (Train vs Test MAE)')

    for bar, mae in zip(bars, test_maes):
        ax.annotate(f'{mae:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '03_model_benchmark.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 03_model_benchmark.png')


def plot_learning_curve(best_model, X, y_delta, y_base):
    """Learning curve."""
    print('Generating learning curve...')

    fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    X_train, X_test, y_tr, y_te, yb_tr, yb_te = train_test_split(
        X, y_delta, y_base, test_size=0.2, random_state=RANDOM_STATE
    )

    train_sizes = (fractions * len(X_train)).astype(int)
    test_maes, test_stds = [], []

    for n in train_sizes:
        mae_runs = []
        for seed in range(3):
            idx = np.random.RandomState(RANDOM_STATE + seed).choice(len(X_train), n, replace=False)
            model = copy.deepcopy(best_model)
            model.fit(X_train[idx], y_tr[idx])
            pred = model.predict(X_test) + yb_te
            mae_runs.append(mean_absolute_error(y_te + yb_te, pred))
        test_maes.append(np.mean(mae_runs))
        test_stds.append(np.std(mae_runs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(train_sizes, test_maes, yerr=test_stds, marker='o', capsize=4, color='#e74c3c', linewidth=2)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Test Set MAE (kcal/mol)')
    ax.set_title('Learning Curve')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels([str(n) for n in train_sizes], rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '04_learning_curve.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 04_learning_curve.png')


def plot_extrapolation(X, y_exp, y_pm7, y_delta, best_model):
    """Extrapolation analysis."""
    print('Running extrapolation analysis...')

    percentages = [10, 20, 30, 40, 50, 60]
    results_list = []

    for pct in percentages:
        lower_pct = (100 - pct) / 2
        upper_pct = 100 - lower_pct
        lower_bound = np.percentile(y_exp, lower_pct)
        upper_bound = np.percentile(y_exp, upper_pct)

        train_mask = (y_exp >= lower_bound) & (y_exp <= upper_bound)
        test_mask = ~train_mask

        if train_mask.sum() < 20 or test_mask.sum() < 20:
            continue

        model = copy.deepcopy(best_model)
        model.fit(X[train_mask], y_delta[train_mask])

        train_pred = model.predict(X[train_mask]) + y_pm7[train_mask]
        train_true = y_delta[train_mask] + y_pm7[train_mask]
        test_pred = model.predict(X[test_mask]) + y_pm7[test_mask]
        test_true = y_delta[test_mask] + y_pm7[test_mask]

        results_list.append({
            'pct': pct, 'train_true': train_true, 'train_pred': train_pred,
            'test_true': test_true, 'test_pred': test_pred,
            'train_mae': mean_absolute_error(train_true, train_pred),
            'test_mae': mean_absolute_error(test_true, test_pred),
            'train_r2': r2_score(train_true, train_pred),
            'test_r2': r2_score(test_true, test_pred),
            'n_train': train_mask.sum(), 'n_test': test_mask.sum(),
            'range': (lower_bound, upper_bound),
        })

    n = len(results_list)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 7*rows), squeeze=False)

    all_vals = np.concatenate([np.concatenate([r['train_true'], r['test_true']]) for r in results_list])
    lims = [np.percentile(all_vals, 1) - 5, np.percentile(all_vals, 99) + 5]

    for i, res in enumerate(results_list):
        ax = axes[i // cols, i % cols]
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("right", size="15%", pad=0.1)

        ax.scatter(res['train_true'], res['train_pred'], alpha=0.5, c='blue', marker='o', label=f'Train (N={res["n_train"]})')
        ax.scatter(res['test_true'], res['test_pred'], alpha=0.5, c='red', marker='^', label=f'Test (N={res["n_test"]})')
        ax.plot(lims, lims, 'k--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Actual PA')
        ax.set_ylabel('Predicted PA')
        ax.set_title(f'{res["pct"]}% Middle for Training\n[{res["range"][0]:.0f}, {res["range"][1]:.0f}]')
        ax.text(0.05, 0.95, f'Train: MAE={res["train_mae"]:.2f}, R²={res["train_r2"]:.3f}',
                transform=ax.transAxes, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.82, f'Test: MAE={res["test_mae"]:.2f}, R²={res["test_r2"]:.3f}',
                transform=ax.transAxes, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(loc='lower right', fontsize=9)

        ax_hist.hist(res['train_true'], bins=15, orientation='horizontal', alpha=0.5, color='blue', range=lims)
        ax_hist.hist(res['test_true'], bins=15, orientation='horizontal', alpha=0.5, color='red', range=lims)
        ax_hist.set_ylim(lims)
        ax_hist.set_xlabel('Count')

    for i in range(len(results_list), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.suptitle('EXTRAPOLATION: Train on Middle PA, Test on Extremes', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '05_extrapolation.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 05_extrapolation.png')


def plot_interpolation(X, y_exp, y_pm7, y_delta, best_model):
    """Interpolation analysis."""
    print('Running interpolation analysis...')

    extreme_pcts = [10, 15, 20, 25, 30, 35, 40]
    results_list = []

    for pct in extreme_pcts:
        lower_bound = np.percentile(y_exp, pct)
        upper_bound = np.percentile(y_exp, 100 - pct)

        train_mask = (y_exp <= lower_bound) | (y_exp >= upper_bound)
        test_mask = ~train_mask

        if train_mask.sum() < 20 or test_mask.sum() < 20:
            continue

        model = copy.deepcopy(best_model)
        model.fit(X[train_mask], y_delta[train_mask])

        train_pred = model.predict(X[train_mask]) + y_pm7[train_mask]
        train_true = y_delta[train_mask] + y_pm7[train_mask]
        test_pred = model.predict(X[test_mask]) + y_pm7[test_mask]
        test_true = y_delta[test_mask] + y_pm7[test_mask]

        results_list.append({
            'pct': pct, 'train_true': train_true, 'train_pred': train_pred,
            'test_true': test_true, 'test_pred': test_pred,
            'train_mae': mean_absolute_error(train_true, train_pred),
            'test_mae': mean_absolute_error(test_true, test_pred),
            'train_r2': r2_score(train_true, train_pred),
            'test_r2': r2_score(test_true, test_pred),
            'n_train': train_mask.sum(), 'n_test': test_mask.sum(),
            'test_range': (lower_bound, upper_bound),
        })

    n = len(results_list)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 7*rows), squeeze=False)

    all_vals = np.concatenate([np.concatenate([r['train_true'], r['test_true']]) for r in results_list])
    lims = [np.percentile(all_vals, 1) - 5, np.percentile(all_vals, 99) + 5]

    for i, res in enumerate(results_list):
        ax = axes[i // cols, i % cols]
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("right", size="15%", pad=0.1)

        ax.scatter(res['train_true'], res['train_pred'], alpha=0.5, c='blue', marker='o', label=f'Train/Extremes (N={res["n_train"]})')
        ax.scatter(res['test_true'], res['test_pred'], alpha=0.5, c='red', marker='^', label=f'Test/Middle (N={res["n_test"]})')
        ax.plot(lims, lims, 'k--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Actual PA')
        ax.set_ylabel('Predicted PA')
        ax.set_title(f'{res["pct"]}-{100-res["pct"]}% Extremes for Training\nTest: [{res["test_range"][0]:.0f}, {res["test_range"][1]:.0f}]')
        ax.text(0.05, 0.95, f'Train: MAE={res["train_mae"]:.2f}, R²={res["train_r2"]:.3f}',
                transform=ax.transAxes, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.82, f'Test: MAE={res["test_mae"]:.2f}, R²={res["test_r2"]:.3f}',
                transform=ax.transAxes, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(loc='lower right', fontsize=9)

        ax_hist.hist(res['train_true'], bins=15, orientation='horizontal', alpha=0.5, color='blue', range=lims)
        ax_hist.hist(res['test_true'], bins=15, orientation='horizontal', alpha=0.5, color='red', range=lims)
        ax_hist.set_ylim(lims)
        ax_hist.set_xlabel('Count')

    for i in range(len(results_list), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.suptitle('INTERPOLATION: Train on Extremes, Test on Middle', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '06_interpolation.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 06_interpolation.png')


def plot_shap_analysis(best_model, best_model_name, X, y_delta, feature_names, smiles, y_exp, best_pred, models, results):
    """SHAP analysis."""
    try:
        import shap
    except ImportError:
        print('  SHAP not available')
        return

    print(f'Computing SHAP for best model ({best_model_name})...')

    model = copy.deepcopy(best_model)
    model.fit(X, y_delta)

    n_samples = min(300, X.shape[0])
    idx = np.random.RandomState(RANDOM_STATE).choice(X.shape[0], n_samples, replace=False)
    X_sample = X[idx]

    linear_models = ['Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge', 'LinearRegression']
    tree_models = ['XGBoost', 'RandomForest', 'GradientBoosting', 'DecisionTree']

    try:
        if best_model_name in linear_models:
            print(f'  Using LinearExplainer for {best_model_name}')
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        elif best_model_name in tree_models:
            print(f'  Using TreeExplainer for {best_model_name}')
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            print(f'  Using KernelExplainer for {best_model_name}')
            kmeans_summary = shap.kmeans(X_sample, 10)
            explainer = shap.KernelExplainer(model.predict, kmeans_summary)
            shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f'  SHAP Explainer failed: {e}')
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=25)
    plt.title(f'Global Feature Importance (SHAP)\nModel: {best_model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '07_shap_global.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 07_shap_global.png')

    # Local SHAP
    errors = np.abs(y_exp - best_pred)
    sorted_idx = np.argsort(errors)
    best_idx = sorted_idx[:2]
    worst_idx = sorted_idx[-2:]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig, wspace=0.4, hspace=0.4)

    cases = [(best_idx[0], 'Best #1'), (best_idx[1], 'Best #2'),
             (worst_idx[0], 'Worst #1'), (worst_idx[1], 'Worst #2')]

    for i, (mol_idx, title) in enumerate(cases):
        try:
            row = i // 2
            col_start = (i % 2) * 3

            ax_mol = fig.add_subplot(gs[row, col_start])
            mol = Chem.MolFromSmiles(str(smiles[mol_idx]))
            if mol:
                img = Draw.MolToImage(mol, size=(180, 180))
                ax_mol.imshow(img)
            ax_mol.axis('off')
            ax_mol.set_title(f'{title}\nExp={y_exp[mol_idx]:.1f}, Pred={best_pred[mol_idx]:.1f}\nErr={errors[mol_idx]:.2f}', fontsize=10)

            ax_shap = fig.add_subplot(gs[row, col_start+1:col_start+3])
            
            instance_X = X[mol_idx:mol_idx+1]
            # Handle list vs array return
            if best_model_name in tree_models or best_model_name in linear_models:
                 shap_vals_inst = explainer.shap_values(instance_X)
                 if isinstance(shap_vals_inst, list): shap_vals_inst = shap_vals_inst[0]
                 if len(shap_vals_inst.shape) > 1: shap_vals_inst = shap_vals_inst[0]
            else:
                 shap_vals_inst = explainer.shap_values(instance_X)
                 if len(shap_vals_inst.shape) > 1: shap_vals_inst = shap_vals_inst[0]

            top_idx = np.argsort(np.abs(shap_vals_inst))[-10:][::-1]
            colors = ['#e74c3c' if shap_vals_inst[j] < 0 else '#27ae60' for j in top_idx]
            
            ax_shap.barh(range(len(top_idx)), shap_vals_inst[top_idx], color=colors, height=0.6)
            ax_shap.set_yticks(range(len(top_idx)))
            ax_shap.set_yticklabels([feature_names[j][:18] for j in top_idx], fontsize=8)
            ax_shap.invert_yaxis()
            ax_shap.axvline(0, color='black', lw=0.5)
            ax_shap.set_xlabel('SHAP Value')
        except Exception as e:
            print(f"Skipping SHAP local plot for {title}: {e}")

    plt.suptitle(f'Local SHAP Analysis ({best_model_name}): Best vs Worst Predictions', fontsize=14, y=0.98)
    plt.savefig(os.path.join(OUT_DIR, '08_shap_local.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 08_shap_local.png')


def plot_feature_importance(best_model, best_model_name, X, y_delta, feature_names, feature_sources):
    """Feature importance analysis."""
    print(f'Analyzing feature importance ({best_model_name})...')

    model = copy.deepcopy(best_model)
    model.fit(X, y_delta)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X, y_delta)
        importances = rf.feature_importances_

    top_idx = np.argsort(importances)[-25:][::-1]

    # RFE
    max_features = X.shape[1]
    n_features_list = list(range(max_features, 49, -50))
    for low_n in [25, 10]:
        if low_n not in n_features_list:
            n_features_list.append(low_n)
    n_features_list = sorted(list(set(n_features_list)), reverse=True)
    rfe_results = []
    for n in n_features_list:
        if n > X.shape[1]: continue
        rfe = RFE(RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE), n_features_to_select=n, step=20)
        rfe.fit(X, y_delta)
        scores = cross_val_score(model, X[:, rfe.support_], y_delta, cv=5, scoring='neg_mean_absolute_error')
        rfe_results.append({'n': n, 'mae': -scores.mean(), 'std': scores.std()})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    colors = [plt.cm.tab10(list(set(feature_sources)).index(feature_sources[i]) % 10) for i in top_idx]
    ax1.barh(range(len(top_idx)), importances[top_idx], color=colors)
    ax1.set_yticks(range(len(top_idx)))
    ax1.set_yticklabels([feature_names[i][:25] for i in top_idx], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance')
    ax1.set_title(f'Top 25 Features ({best_model_name})')

    ax2 = axes[1]
    ax2.errorbar([r['n'] for r in rfe_results], [r['mae'] for r in rfe_results],
                 yerr=[r['std'] for r in rfe_results], marker='o', capsize=4)
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('CV MAE')
    ax2.set_title('Recursive Feature Elimination')
    ax2.invert_xaxis()

    plt.suptitle(f'Feature Importance Analysis ({best_model_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '09_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 09_feature_importance.png')


def plot_error_analysis(smiles, errors, y_exp):
    """Error vs complexity."""
    print('Analyzing error vs complexity...')

    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    n_atoms = [m.GetNumAtoms() if m else 0 for m in mols]
    n_rings = [rdMolDescriptors.CalcNumRings(m) if m else 0 for m in mols]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.scatter(n_atoms, errors, c=y_exp, cmap='viridis', alpha=0.5, s=20)
    ax1.set_xlabel('Number of Atoms')
    ax1.set_ylabel('|Error| (kcal/mol)')
    ax1.set_title('Error vs Molecular Size')

    ax2 = axes[1]
    df = pd.DataFrame({'rings': n_rings, 'error': errors})
    agg = df.groupby('rings')['error'].agg(['mean', 'std', 'count']).reset_index()
    agg = agg[agg['count'] >= 5]
    ax2.errorbar(agg['rings'], agg['mean'], yerr=agg['std'], marker='o', capsize=4)
    ax2.set_xlabel('Number of Rings')
    ax2.set_ylabel('Mean |Error|')
    ax2.set_title('Error vs Ring Count')

    plt.suptitle('Error Analysis (5-Fold CV Out-of-Sample Predictions)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '10_error_analysis.png'), bbox_inches='tight')
    plt.close()
    print('  Saved: 10_error_analysis.png')


def plot_error_distribution(y_exp, pred, y_pm7):
    """Structure error distribution."""
    print('Generating error distribution comparison...')
    
    err_ml = pred - y_exp
    err_pm7 = y_pm7 - y_exp
    
    mae_ml = mean_absolute_error(y_exp, pred)
    mae_pm7 = mean_absolute_error(y_exp, y_pm7)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(err_pm7, color='gray', label=f'PM7 Baseline (MAE={mae_pm7:.2f})', kde=True, ax=ax, alpha=0.3, stat='density', bins=30)
    sns.histplot(err_ml, color='#3498db', label=f'Best Model (MAE={mae_ml:.2f})', kde=True, ax=ax, alpha=0.6, stat='density', bins=30)
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Error (Predicted - Actual) [kcal/mol]')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution Comparison: PM7 vs Best ML Model')
    ax.legend(fontsize=11)
    ax.set_xlim(-25, 25)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '12_error_distribution_comparison.png'), dpi=300)
    plt.close()
    print(f'  Saved: 12_error_distribution_comparison.png')


def plot_best_worst(smiles, errors, y_exp, y_pred, y_pm7, best_model_name):
    """Best and worst molecules."""
    print('Rendering best/worst molecules...')

    ml_correction = y_pred - y_pm7

    df = pd.DataFrame({
        'smiles': smiles,
        'error': errors,
        'y_true': y_exp,
        'y_pred': y_pred,
        'y_pm7': y_pm7,
        'ml_corr': ml_correction
    })
    df = df.sort_values('error')

    for label, subset in [('best', df.head(9)), ('worst', df.tail(9))]:
        mols = [Chem.MolFromSmiles(str(s)) for s in subset['smiles']]
        legends = [
            f'Actual: {t:.1f}\nPred: {p:.1f}\n(PM7:{pm7:.1f} + ML:{ml:+.1f})'
            for t, p, pm7, ml in zip(subset['y_true'], subset['y_pred'], subset['y_pm7'], subset['ml_corr'])
        ]
        img = Draw.MolsToGridImage(mols, molsPerRow=3, legends=legends, subImgSize=(300, 300))
        img.save(os.path.join(OUT_DIR, f'11_molecules_{label}.png'))

    print(f'  Saved: 11_molecules_best/worst.png (using {best_model_name})')


def save_summary(results, y_exp, y_pm7, feature_names, feature_sources, best_model_name):
    """Save summary files."""
    pm7_mae = mean_absolute_error(y_exp, y_pm7)

    rows = []
    for name, res in results.items():
        rows.append({
            'model': name, 'train_mae': res['train_mae_mean'], 'test_mae': res['test_mae_mean'],
            'test_mae_std': res['test_mae_std'], 'test_r2': res['test_r2_mean'],
        })
    rows.append({'model': 'PM7_Baseline', 'train_mae': pm7_mae, 'test_mae': pm7_mae, 'test_mae_std': 0, 'test_r2': r2_score(y_exp, y_pm7)})

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'summary_metrics.csv'), index=False)
    pd.Series(Counter(feature_sources)).to_csv(os.path.join(OUT_DIR, 'summary_feature_sources.csv'))
    pd.DataFrame({'feature': feature_names}).to_csv(os.path.join(OUT_DIR, 'summary_features.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'summary_best_model.txt'), 'w') as f:
        f.write(f'Best Model: {best_model_name}\n')

    print(f'  Saved summary files')
