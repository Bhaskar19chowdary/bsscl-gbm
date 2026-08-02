import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from bsscl_gbm import HybridHistGBMNumbaV2

warnings.filterwarnings('ignore')

def get_ablation_models():
    return {
        "1. Full BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "2. W/O Hybrid Binning": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6, binning_strategy='standard'),
        "3. W/O Balanced Sqrt": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6, class_weight=None),
        "4. W/O Leaf-Wise Growth": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6, grow_policy='level')
    }

def run_extended_ablation():
    print(f"\n{'='*70}\n  PHASE 4: EXTENDED ABLATION STUDY\n{'='*70}")
    print("Testing individual algorithmic contributions mathematically.\n")
    
    datasets = {
        "Normal (Breast Cancer)": load_breast_cancer(return_X_y=True),
        "Imbalanced (99:1)": make_classification(n_samples=20000, n_features=20, weights=[0.99], random_state=42),
        "Large Synth (50k)": make_classification(n_samples=50000, n_features=30, random_state=42)
    }

    results = []

    for ds_name, (X, y) in datasets.items():
        print(f"\n[ Dataset: {ds_name} ]")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for model_name, model in get_ablation_models().items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            f1 = f1_score(y_test, model.predict(X_test))
            print(f"  {model_name:25s} -> F1: {f1:.4f} | Time: {train_time:.3f}s")
            
            results.append({
                "Dataset": ds_name,
                "Ablation": model_name,
                "F1 Score": f1,
                "Train Time (s)": train_time
            })

    print("\nSaving Extended Ablation Table...")
    df = pd.DataFrame(results)
    pivot_f1 = df.pivot(index="Dataset", columns="Ablation", values="F1 Score")
    
    with open("extended_ablation_results.md", "w") as f:
        f.write("# Phase 4: Extended Ablation Study (F1 Scores)\n\n")
        f.write(pivot_f1.to_markdown())
        f.write("\n")
    print("✅ Saved to extended_ablation_results.md")

if __name__ == "__main__":
    run_extended_ablation()
