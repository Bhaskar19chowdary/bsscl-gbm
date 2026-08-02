import optuna
import warnings
import numpy as np
import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_tuning_all():
    print(f"\n{'='*70}\n  PHASE 3: FAIRNESS VIA HYPERPARAMETER OPTIMIZATION (ALL 4 MODELS)\n{'='*70}")
    print("Downloading REAL Dataset: Adult Income (48,000 rows) from OpenML...")
    
    # Adult dataset
    dataset = openml.datasets.get_dataset(1590)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    
    # Preprocess
    for col in X.columns:
        if X[col].dtype == 'category' or X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(y)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective_bsscl(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = HybridHistGBMNumbaV2(**params, verbose=False)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', n_jobs=4)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=4)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    def objective_cb(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = cb.CatBoostClassifier(**params, verbose=0, thread_count=4)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        return np.mean(scores)

    models = [
        ("BSSCL-GBM", objective_bsscl),
        ("XGBoost", objective_xgb),
        ("LightGBM", objective_lgb),
        ("CatBoost", objective_cb)
    ]
    
    print("\nRunning 20 trials per model to ensure it finishes relatively fast...")
    for name, obj_func in models:
        print(f"\n> Tuning {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(obj_func, n_trials=20)
        print(f"  Best Accuracy: {study.best_value:.4f}")
        print(f"  Best Params: {study.best_params}")

if __name__ == "__main__":
    run_optuna_tuning_all()
