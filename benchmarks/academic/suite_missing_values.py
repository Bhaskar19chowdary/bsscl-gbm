import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def inject_missing_values(X, fraction):
    """Randomly injects np.nan into the dataset at the given fraction."""
    if fraction == 0:
        return X.copy()
    X_miss = X.copy()
    mask = np.random.rand(*X_miss.shape) < fraction
    X_miss[mask] = np.nan
    return X_miss

def run_missing_value_benchmark():
    print(f"\n{'='*70}\n  PHASE 4: NATIVE MISSING VALUE HANDLING STRESS TEST\n{'='*70}")
    print("Injecting missing data at 0%, 10%, 30%, 50%, and 70% to test algorithm robustness.\n")
    
    X, y = make_classification(n_samples=20000, n_features=20, n_informative=15, random_state=42)
    
    missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    results = []
    
    for rate in missing_rates:
        print(f"\n[ Dataset missing rate: {rate*100:.0f}% NaN ]")
        
        X_dirty = inject_missing_values(X, rate)
        X_train, X_test, y_train, y_test = train_test_split(X_dirty, y, test_size=0.2, random_state=42)
        
        for name, model in get_models().items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            acc = accuracy_score(y_test, model.predict(X_test))
            print(f"  {name:10s} -> Acc: {acc:.4f} | Time: {train_time:.3f}s")
            
            results.append({
                "NaN Rate": f"{rate*100:.0f}%",
                "Model": name,
                "Accuracy": acc,
                "Train Time": train_time
            })
            
    df = pd.DataFrame(results)
    
    print("\n--- Missing Value Robustness Table ---")
    
    # Pivot for clean display
    pivot_df = df.pivot(index="NaN Rate", columns="Model", values="Accuracy")
    print(pivot_df.to_markdown())
    
    with open("missing_values_results.md", "w") as f:
        f.write("# Phase 4: Missing Value Robustness\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n")
    print("✅ Saved missing values table to 'missing_values_results.md'")

if __name__ == "__main__":
    run_missing_value_benchmark()
