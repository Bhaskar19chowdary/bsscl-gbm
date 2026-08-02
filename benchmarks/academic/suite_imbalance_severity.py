import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, recall_score, f1_score
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1, scale_pos_weight=1.0), # Baseline without explicit weighting
        "CatBoost": cb.CatBoostClassifier(n_estimators=100, depth=6, verbose=0, auto_class_weights=None)
    }

def run_imbalance_severity():
    print(f"\n{'='*70}\n  PHASE 4: IMBALANCE SEVERITY EXPERIMENT\n{'='*70}")
    print("Testing the mathematical efficacy of BSSCL's native `balanced_sqrt` class weights.\n")
    
    ratios = [
        (0.5, 0.5, "50:50"),
        (0.9, 0.1, "90:10"),
        (0.99, 0.01, "99:1"),
        (0.999, 0.001, "99.9:0.1")
    ]
    
    results = []
    
    for w0, w1, ratio_name in ratios:
        print(f"\n[ Class Ratio: {ratio_name} ]")
        
        # 50,000 samples to ensure minority class has enough data even at 0.1%
        X, y = make_classification(n_samples=50000, n_features=20, n_informative=10, 
                                   weights=[w0], random_state=42)
                                   
        # Ensure we have at least 1 minority class
        if np.sum(y) == 0:
            y[-1] = 1
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for name, model in get_models().items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            preds = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
                pr_auc = average_precision_score(y_test, probs[:, 1])
            else:
                pr_auc = 0.0
                
            f1 = f1_score(y_test, preds)
            rec = recall_score(y_test, preds)
            
            print(f"  {name:10s} -> F1: {f1:.4f} | Recall: {rec:.4f} | PR-AUC: {pr_auc:.4f}")
            
            results.append({
                "Ratio": ratio_name,
                "Model": name,
                "F1 Score": f1,
                "Recall": rec,
                "PR-AUC": pr_auc
            })
            
    df = pd.DataFrame(results)
    
    # Pivot for clean display
    pivot_df = df.pivot(index="Ratio", columns="Model", values="F1 Score")
    
    print("\n--- F1 Score by Imbalance Ratio Table ---")
    print(pivot_df.to_markdown())
    
    with open("imbalance_severity_results.md", "w") as f:
        f.write("# Phase 4: Imbalance Severity (F1 Scores)\n\n")
        f.write(pivot_df.to_markdown())
        f.write("\n")
    print("✅ Saved imbalance table to 'imbalance_severity_results.md'")

if __name__ == "__main__":
    run_imbalance_severity()
