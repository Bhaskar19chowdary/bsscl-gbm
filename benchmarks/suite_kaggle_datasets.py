import time
import warnings
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
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

def run_kaggle_datasets():
    print(f"\n{'='*70}\n  PHASE 3: MORE REAL-WORLD KAGGLE DATASETS\n{'='*70}")
    
    # Rossmann is Regression, BSSCL is Classification, so we test the classification ones:
    datasets = {
        "Amazon Employee Access": 4135,
        "Santander Customer Satisfaction": 42728, # or 41147
        "Kaggle Credit Card Fraud": 42733,
    }
    
    results = []

    for name, openml_id in datasets.items():
        print(f"\n[ Fetching Dataset: {name} (OpenML ID: {openml_id}) ]")
        try:
            dataset = openml.datasets.get_dataset(openml_id)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
            
            # Preprocess
            for col in X.columns:
                if X[col].dtype == 'category' or X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                    
            X = X.to_numpy(dtype=np.float32)
            y = LabelEncoder().fit_transform(y)
            
            # Handle NaN values for CatBoost which can be finicky without explicit preprocessing
            X = np.nan_to_num(X, nan=0.0)
            
            print(f"  Shape: {X.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            for model_name, model in get_models().items():
                print(f"  > Training {model_name}...")
                start = time.perf_counter()
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start
                
                preds = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)
                    if probs.shape[1] > 1:
                        roc_auc = roc_auc_score(y_test, probs[:, 1])
                    else:
                        roc_auc = 0.0
                else:
                    roc_auc = 0.0
                    
                f1 = f1_score(y_test, preds, average='macro')
                
                print(f"    F1 (Macro): {f1:.3f} | ROC-AUC: {roc_auc:.3f} | Time: {train_time:.2f}s")
                
                results.append({
                    "Dataset": name,
                    "Model": model_name,
                    "F1 (Macro)": round(f1, 3),
                    "ROC-AUC": round(roc_auc, 3),
                    "Train Time (s)": round(train_time, 2)
                })
        except Exception as e:
            print(f"  Failed to evaluate dataset: {e}")
            
    print("\nSaving Kaggle Datasets Results Table...")
    df = pd.DataFrame(results)
    
    with open("kaggle_results.md", "w") as f:
        f.write("# Phase 3: Kaggle Real-World Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("✅ Saved to kaggle_results.md")

if __name__ == "__main__":
    run_kaggle_datasets()
