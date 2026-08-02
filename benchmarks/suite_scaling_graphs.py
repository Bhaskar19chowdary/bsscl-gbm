import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=100, depth=6, verbose=0)
    }

def run_scaling_benchmark():
    print(f"\n{'='*70}\n  PHASE 3: SCALING GRAPHS (Training Time vs Rows)\n{'='*70}")
    
    # We will test scaling from 10k to 5 Million rows
    row_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    n_features = 20
    
    results = {name: [] for name in get_models().keys()}
    
    for n_rows in row_counts:
        print(f"\n[ Generating Dataset: {n_rows:,} rows ]")
        X, y = make_classification(n_samples=n_rows, n_features=n_features, random_state=42)
        
        for model_name, model in get_models().items():
            print(f"  > Training {model_name}...")
            start = time.perf_counter()
            model.fit(X, y)
            train_time = time.perf_counter() - start
            results[model_name].append(train_time)
            print(f"    Time: {train_time:.2f}s")
            
    # Plotting
    plt.figure(figsize=(10, 6))
    for model_name, times in results.items():
        if "BSSCL" in model_name:
            plt.plot(row_counts, times, label=model_name, linewidth=3, marker='o')
        else:
            plt.plot(row_counts, times, label=model_name, linestyle='--', marker='x')
            
    plt.xlabel('Number of Rows', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Algorithm Scalability: Training Time vs Dataset Size', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('scaling_graph.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved scaling graph to 'scaling_graph.png'")
    
    df = pd.DataFrame(results, index=row_counts)
    df.index.name = 'Rows'
    df.to_csv('scaling_data.csv')
    print("✅ Saved raw data to 'scaling_data.csv'")

if __name__ == "__main__":
    run_scaling_benchmark()
