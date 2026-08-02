import time
import numpy as np
import warnings
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=10, max_depth=4),
        "XGBoost": xgb.XGBClassifier(n_estimators=10, max_depth=4, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=10, max_depth=4, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=10, depth=4, verbose=0)
    }

def run_data_quality_suite(fast_mode=False):
    print_header("SUITE B: Data Quality Stress Testing (Side-by-Side)")
    
    # Base dataset
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_test, y_test = make_classification(n_samples=1000, n_features=20, random_state=43)
    
    # 1. Missing Value Testing
    print("\n--- 6. Missing Value Testing ---")
    missing_rates = [0.1, 0.3, 0.5, 0.9] if not fast_mode else [0.1, 0.5]
    for rate in missing_rates:
        X_miss = X.copy()
        mask = np.random.rand(*X.shape) < rate
        X_miss[mask] = np.nan
        print(f"\n[ Missing {rate*100:2.0f}% ]")
        
        for name, model in get_models().items():
            try:
                model.fit(X_miss, y)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                print(f"  {name:<12s} Test Acc: {acc:.3f}")
            except Exception as e:
                print(f"  {name:<12s} FAILED: {str(e)[:50]}")

    # 2. Label Noise Testing
    print("\n--- 7. Label Noise Testing ---")
    noise_rates = [0.01, 0.10, 0.30] if not fast_mode else [0.10]
    for rate in noise_rates:
        y_noise = y.copy()
        mask = np.random.rand(len(y)) < rate
        y_noise[mask] = 1 - y_noise[mask] # flip binary labels
        
        print(f"\n[ Label Noise {rate*100:2.0f}% ]")
        for name, model in get_models().items():
            model.fit(X, y_noise)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"  {name:<12s} Clean Test Acc: {acc:.3f}")

    # 3. Outlier Testing
    print("\n--- 8. Extreme Outlier Testing ---")
    outliers = [1e12, -1e12]
    for out in outliers:
        X_out = X.copy()
        X_out[0:10, 0:5] = out
        
        print(f"\n[ Outlier: {out:>6.1e} ]")
        for name, model in get_models().items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_out, y)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                print(f"  {name:<12s} Clean Test Acc: {acc:.3f}")
            except Exception as e:
                print(f"  {name:<12s} FAILED: {str(e)[:50]}")

    print("\n✅ Suite B (Data Quality) Completed")

if __name__ == "__main__":
    run_data_quality_suite(fast_mode=True)
