
import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from .config import RANDOM_STATE

def get_all_models():
    """Return dictionary of all regression models."""
    return {
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE),
        'BayesianRidge': BayesianRidge(),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=4, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.9, n_jobs=4, random_state=RANDOM_STATE),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_STATE),
    }

def cross_val_with_metrics(models, X, y_delta, y_base, n_splits=5):
    """5-fold CV returning per-molecule predictions."""
    print('\nRunning cross-validation...')

    results = {}
    cv_predictions = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f'  {name}...')

        train_maes, test_maes = [], []
        train_r2s, test_r2s = [], []

        # Store per-molecule predictions
        test_true_all = np.zeros(len(y_delta))
        test_pred_all = np.zeros(len(y_delta))

        for train_idx, test_idx in kf.split(X):
            model_copy = copy.deepcopy(model)
            model_copy.fit(X[train_idx], y_delta[train_idx])

            # Train metrics
            train_pred = model_copy.predict(X[train_idx]) + y_base[train_idx]
            train_true = y_delta[train_idx] + y_base[train_idx]
            train_maes.append(mean_absolute_error(train_true, train_pred))
            train_r2s.append(r2_score(train_true, train_pred))

            # Test metrics
            test_pred = model_copy.predict(X[test_idx]) + y_base[test_idx]
            test_true = y_delta[test_idx] + y_base[test_idx]
            test_maes.append(mean_absolute_error(test_true, test_pred))
            test_r2s.append(r2_score(test_true, test_pred))

            # Store per-molecule
            test_true_all[test_idx] = test_true
            test_pred_all[test_idx] = test_pred

        results[name] = {
            'train_mae_mean': np.mean(train_maes), 'train_mae_std': np.std(train_maes),
            'test_mae_mean': np.mean(test_maes), 'test_mae_std': np.std(test_maes),
            'train_r2_mean': np.mean(train_r2s), 'test_r2_mean': np.mean(test_r2s),
        }
        cv_predictions[name] = {'test_true': test_true_all, 'test_pred': test_pred_all}

        print(f'    Test MAE: {results[name]["test_mae_mean"]:.3f}±{results[name]["test_mae_std"]:.3f}')

    return results, cv_predictions

def get_best_model_name(results):
    """Identify best model based on MAE."""
    return min(results.items(), key=lambda x: x[1]['test_mae_mean'])[0]
