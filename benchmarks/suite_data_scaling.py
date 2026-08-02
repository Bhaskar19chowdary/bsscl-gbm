import time
import numpy as np
from sklearn.datasets import make_classification
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

def run_data_scaling_suite(fast_mode=False):
    print_header("SUITE A: Dataset Benchmark Categories (Side-by-Side)")
    
    # 1. Dataset Size Scaling
    sizes = [500, 10000, 100000]
    if not fast_mode:
        sizes.extend([1000000, 5000000])
        
    print("\n--- 1. Dataset Size Scaling ---")
    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=20, random_state=42)
        print(f"\n[ Size: {n:>9,d} rows ]")
        for name, model in get_models().items():
            start = time.perf_counter()
            model.fit(X, y)
            duration = time.perf_counter() - start
            print(f"  {name:<12s} Time: {duration:6.3f}s")
            
    # 2. Feature Dimension Scaling
    print("\n--- 2. Feature Dimension Scaling ---")
    dims = [10, 100, 500]
    if not fast_mode:
        dims.extend([2000, 5000])
        
    for d in dims:
        X, y = make_classification(n_samples=5000, n_features=d, n_informative=d//2, random_state=42)
        print(f"\n[ Features: {d:>5d} ]")
        for name, model in get_models().items():
            start = time.perf_counter()
            model.fit(X, y)
            duration = time.perf_counter() - start
            print(f"  {name:<12s} Time: {duration:6.3f}s")

    # 3. Distribution Testing
    print("\n--- 3. Distribution Testing ---")
    distributions = {
        'Normal': np.random.randn(10000, 20),
        'Uniform': np.random.rand(10000, 20),
        'Exponential': np.random.exponential(scale=1.0, size=(10000, 20))
    }
    y_dist = np.random.randint(0, 2, 10000)
    
    for dist_name, X_dist in distributions.items():
        print(f"\n[ Distribution: {dist_name} ]")
        for name, model in get_models().items():
            start = time.perf_counter()
            model.fit(X_dist, y_dist)
            duration = time.perf_counter() - start
            print(f"  {name:<12s} Time: {duration:6.3f}s")

    print("\n✅ Suite A (Scaling) Completed")

if __name__ == "__main__":
    run_data_scaling_suite(fast_mode=True)
