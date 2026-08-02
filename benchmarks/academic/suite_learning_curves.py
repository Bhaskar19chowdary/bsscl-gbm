import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

def run_learning_curves():
    print(f"\n{'='*70}\n  PHASE 4: LEARNING CURVE ANALYSIS\n{'='*70}")
    print("Measuring F1 Score and Training Time across different data fractions.\n")
    
    # Generate a large 100k row dataset
    X, y = make_classification(n_samples=100000, n_features=20, n_informative=10, random_state=42)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    results = {name: {'f1': [], 'time': []} for name in get_models().keys()}
    
    for frac in fractions:
        if frac < 1.0:
            # We just slice the arrays since train_test_split already shuffled
            n_samples = int(len(X_train_full) * frac)
            X_train = X_train_full[:n_samples]
            y_train = y_train_full[:n_samples]
        else:
            X_train = X_train_full
            y_train = y_train_full
            
        print(f"\n[ Training on {frac*100:.0f}% of data ({len(X_train)} rows) ]")
        
        for name, model in get_models().items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            t = time.perf_counter() - start
            
            f1 = f1_score(y_test, model.predict(X_test))
            
            results[name]['f1'].append(f1)
            results[name]['time'].append(t)
            print(f"  {name:10s} -> F1: {f1:.4f} | Time: {t:.3f}s")
            
    # Plotting F1 Score Curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, metrics in results.items():
        plt.plot(fractions, metrics['f1'], marker='o', label=name, linewidth=2 if 'BSSCL' in name else 1.5)
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve (Data Efficiency)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Plotting Time Curve
    plt.subplot(1, 2, 2)
    for name, metrics in results.items():
        plt.plot(fractions, metrics['time'], marker='x', label=name, linewidth=2 if 'BSSCL' in name else 1.5)
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("Training Time (seconds)")
    plt.title("Computational Scaling")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("\n✅ Saved learning curves graph to 'learning_curves.png'")

if __name__ == "__main__":
    run_learning_curves()
