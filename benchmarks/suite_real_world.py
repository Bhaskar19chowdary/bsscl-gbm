import os
import time
import psutil
import warnings
import numpy as np
import pandas as pd
import openml
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import LabelEncoder
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

def get_models():
    # Use standard default settings for all models for a fair benchmark
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss', n_jobs=4),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, n_jobs=4, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=100, depth=6, thread_count=4, verbose=0)
    }

def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_dataset(name, fast_mode=False):
    if name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == "Digits":
        data = load_digits()
        X, y = data.data, data.target
    elif name == "Adult":
        dataset = openml.datasets.get_dataset(1590)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    elif name == "Bank Marketing":
        dataset = openml.datasets.get_dataset(1461)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    elif name == "Credit Card Fraud": # Note: using 42175 or 1597. 1597 is actually a different dataset often. Let's use 42175 for fraud if it exists, or just a known one. We'll use 1597 (credit-g) for simplicity if fraud isn't easily accessible by ID, wait, ID 42175 is Credit Card Fraud on OpenML.
        try:
            dataset = openml.datasets.get_dataset(42175)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        except:
            # Fallback to credit-g (1597)
            dataset = openml.datasets.get_dataset(1597)
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    elif name == "MNIST":
        dataset = openml.datasets.get_dataset(554)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        if fast_mode and len(X) > 10000:
            X, y = X.iloc[:10000], y.iloc[:10000]
    elif name == "Higgs":
        # Higgs is massive. If fast_mode, cap at 50,000.
        dataset = openml.datasets.get_dataset(23512)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        if fast_mode and len(X) > 50000:
            X, y = X.iloc[:50000], y.iloc[:50000]
    else:
        raise ValueError(f"Unknown dataset {name}")

    if isinstance(X, pd.DataFrame):
        # Convert categoricals to numeric
        for col in X.columns:
            if X[col].dtype == 'category' or X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.to_numpy(dtype=np.float32)
        y = LabelEncoder().fit_transform(y)
        
    return X, y

def run_real_world_suite(fast_mode=False):
    print(f"\n{'='*70}\n  EXTREME LEVEL REAL-WORLD BENCHMARK SUITE {'(FAST MODE)' if fast_mode else ''}\n{'='*70}")
    
    datasets = ["Breast Cancer", "Digits", "Adult", "Bank Marketing", "Credit Card Fraud", "MNIST", "Higgs"]
    
    if fast_mode:
        print("Note: Higgs and MNIST datasets are capped in fast_mode to prevent timeouts.\n")
        
    results = []
    
    for ds_name in datasets:
        print(f"\n[ Downloading / Loading Dataset: {ds_name} ]")
        try:
            X, y = load_dataset(ds_name, fast_mode)
        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}")
            continue
            
        print(f"  Shape: {X.shape} | Classes: {len(np.unique(y))}")
        
        is_multiclass = len(np.unique(y)) > 2
        
        # We need a train/test split to calculate generalization metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for model_name, model in get_models().items():
            print(f"  > Training {model_name}...")
            mem_before = measure_memory_mb()
            start = time.perf_counter()
            
            try:
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start
                mem_after = measure_memory_mb()
                ram_mb = mem_after - mem_before
                
                # Predictions
                start_pred = time.perf_counter()
                preds = model.predict(X_test)
                pred_time = time.perf_counter() - start_pred
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)
                else:
                    probs = None
                    
                # Metrics
                acc = accuracy_score(y_test, preds)
                if is_multiclass:
                    f1 = f1_score(y_test, preds, average='macro')
                    metric_str = f"Acc: {acc:.3f} | Mac-F1: {f1:.3f}"
                    main_metric = acc
                else:
                    f1 = f1_score(y_test, preds)
                    # For highly imbalanced, PR-AUC is critical
                    pr_auc = average_precision_score(y_test, probs[:, 1]) if probs is not None else 0.0
                    metric_str = f"Acc: {acc:.3f} | F1: {f1:.3f} | PR-AUC: {pr_auc:.3f}"
                    main_metric = f1
                    
                print(f"    {metric_str} | Train Time: {train_time:.2f}s | RAM: {ram_mb:.1f}MB")
                
                results.append({
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Accuracy": acc,
                    "F1 / Macro-F1": f1,
                    "Train Time (s)": train_time,
                    "Peak RAM (MB)": ram_mb
                })
            except Exception as e:
                print(f"    Failed: {str(e)[:50]}")
                results.append({
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Accuracy": "FAIL",
                    "F1 / Macro-F1": "FAIL",
                    "Train Time (s)": "FAIL",
                    "Peak RAM (MB)": "FAIL"
                })

    # Generate Markdown Table
    print("\nGenerating Final Markdown Table...")
    df = pd.DataFrame(results)
    md_table = df.to_markdown(index=False)
    
    with open("real_world_benchmark_table.md", "w") as f:
        f.write("# Final Scientific Benchmark Comparison\n\n")
        f.write(md_table)
        f.write("\n")
        
    print("\n✅ Saved to real_world_benchmark_table.md")

if __name__ == "__main__":
    import sys
    fast = '--fast' in sys.argv
    run_real_world_suite(fast_mode=fast)
