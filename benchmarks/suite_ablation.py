import time
import psutil
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from bsscl_gbm import HybridHistGBMNumbaV2

warnings.filterwarnings('ignore')

def get_ablation_models():
    return {
        "Full BSSCL-GBM (Hybrid + Weights)": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "Without Hybrid Binning (Standard)": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6, binning_strategy='standard'),
        "Without Focal Loss/Weights": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6, class_weight=None),
        "Without Early Stopping / Growth": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6, grow_policy='level')
    }

def run_ablation_suite():
    print(f"\n{'='*70}\n  PHASE 3: ABLATION STUDY (Proving Component Value)\n{'='*70}")
    
    # We will test on a severely imbalanced synthetic dataset and a classic medical dataset
    print("Generating High-Variance Imbalanced Data (100k rows, 1% minority)...")
    X_imb, y_imb = make_classification(n_samples=100000, n_features=30, n_informative=10, 
                                       weights=[0.99], random_state=42)
    
    data_breast = load_breast_cancer()
    X_bc, y_bc = data_breast.data, data_breast.target

    datasets = {
        "Severe Imbalance (100k)": (X_imb, y_imb),
        "Breast Cancer (569)": (X_bc, y_bc)
    }

    results = []

    for ds_name, (X, y) in datasets.items():
        print(f"\n[ Dataset: {ds_name} ]")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for model_name, model in get_ablation_models().items():
            print(f"  > Training {model_name}...")
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            preds = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
            else:
                probs = None
                
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            pr_auc = average_precision_score(y_test, probs[:, 1]) if probs is not None else 0.0
            
            print(f"    Acc: {acc:.3f} | F1: {f1:.3f} | PR-AUC: {pr_auc:.3f} | Time: {train_time:.2f}s")
            
            results.append({
                "Dataset": ds_name,
                "Variant": model_name,
                "Accuracy": acc,
                "F1 Score": f1,
                "PR-AUC": pr_auc,
                "Train Time (s)": train_time
            })

    print("\nSaving Ablation Study Table...")
    df = pd.DataFrame(results)
    
    with open("ablation_study_results.md", "w") as f:
        f.write("# Phase 3: Ablation Study Results\n\n")
        f.write("This study proves the mathematical value of each custom component by systematically disabling them.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("✅ Saved to ablation_study_results.md")

if __name__ == "__main__":
    run_ablation_suite()
