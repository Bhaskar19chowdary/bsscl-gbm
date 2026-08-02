import os
import time
import psutil
import warnings
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=100, depth=6, verbose=0)
    }

def run_extreme_scale():
    print(f"\n{'='*70}\n  PHASE 3: EXTREME SCALE BENCHMARK (11M Rows)\n{'='*70}")
    print("WARNING: This script will download ~2.5GB of data and may take several hours to run.")
    print("It evaluates the massive Higgs Boson dataset (11,000,000 rows).")
    
    print("\n[ Downloading 11 Million Row Higgs Dataset from OpenML... ]")
    # OpenML ID 23512 is the 11M row Higgs dataset
    try:
        dataset = openml.datasets.get_dataset(23512)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    # Prepare Data
    print("  > Processing data...")
    X = X.to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print(f"  Shape: {X_train.shape} training rows")

    for model_name, model in get_models().items():
        print(f"\n  > Training {model_name}...")
        start = time.perf_counter()
        
        try:
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            
            print(f"    Acc: {acc:.3f} | F1: {f1:.3f} | Train Time: {train_time:.2f}s")
        except Exception as e:
            print(f"    Failed: {e}")

if __name__ == "__main__":
    import sys
    # Require explicit flag to run this massive test
    if '--confirm' not in sys.argv:
        print("Error: This script requires the '--confirm' flag to run because it downloads 11M rows.")
        sys.exit(1)
    run_extreme_scale()
