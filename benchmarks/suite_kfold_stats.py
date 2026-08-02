import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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

def run_kfold_stats():
    print(f"\n{'='*70}\n  PHASE 3: STATISTICAL SIGNIFICANCE (5-Fold CV)\n{'='*70}")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    final_results = []
    
    for model_name, model in get_models().items():
        print(f"\n  > Evaluating {model_name}...")
        acc_scores = []
        f1_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # CatBoost fails gracefully if re-instantiation isn't clean sometimes, 
            # so we re-instantiate
            models_dict = get_models()
            fold_model = models_dict[model_name]
            
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_test)
            
            acc_scores.append(accuracy_score(y_test, preds))
            f1_scores.append(f1_score(y_test, preds))
            
        mean_acc = np.mean(acc_scores)
        std_acc = np.std(acc_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"    Acc: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"    F1:  {mean_f1:.3f} ± {std_f1:.3f}")
        
        final_results.append({
            "Model": model_name,
            "Accuracy": f"{mean_acc:.3f} ± {std_acc:.3f}",
            "F1 Score": f"{mean_f1:.3f} ± {std_f1:.3f}"
        })
        
    print("\nSaving Statistical Study Table...")
    df = pd.DataFrame(final_results)
    
    with open("kfold_stats_results.md", "w") as f:
        f.write("# Phase 3: Statistical Significance Results (5-Fold CV)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("✅ Saved to kfold_stats_results.md")

if __name__ == "__main__":
    run_kfold_stats()
