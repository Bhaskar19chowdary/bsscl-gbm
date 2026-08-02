import time
import os
import psutil
import threading
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

# Global variable to store peak memory
peak_memory = 0
stop_monitor = False

def monitor_memory():
    global peak_memory, stop_monitor
    process = psutil.Process(os.getpid())
    peak_memory = process.memory_info().rss
    while not stop_monitor:
        current_mem = process.memory_info().rss
        if current_mem > peak_memory:
            peak_memory = current_mem
        time.sleep(0.01) # Sample every 10ms

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=100, depth=6, verbose=0)
    }

def run_memory_test():
    global peak_memory, stop_monitor
    
    print(f"\n{'='*70}\n  PHASE 3: MEMORY MEASUREMENT METHODOLOGY\n{'='*70}")
    print("Methodology: Measuring Peak RSS (Resident Set Size) via background thread polling every 10ms.")
    print("This ensures C++ allocations from XGBoost/LightGBM/CatBoost are captured alongside Python.")
    
    print("\nGenerating 1,000,000 row dataset...")
    X, y = make_classification(n_samples=1000000, n_features=20, random_state=42)
    
    results = []
    
    for name, model in get_models().items():
        print(f"\n> Profiling {name}...")
        
        # Warmup and clear
        stop_monitor = False
        peak_memory = 0
        
        # Start background memory monitor
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        # Train
        start_time = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - start_time
        
        # Stop monitor
        stop_monitor = True
        monitor_thread.join()
        
        # Calculate memory footprint (Peak - Base)
        # Convert bytes to MB
        peak_mb = peak_memory / (1024 * 1024)
        
        print(f"  Train Time: {train_time:.2f}s")
        print(f"  Peak RSS:   {peak_mb:.2f} MB")
        
        results.append({
            "Model": name,
            "Train Time (s)": round(train_time, 2),
            "Peak RSS (MB)": round(peak_mb, 2)
        })
        
    df = pd.DataFrame(results)
    print("\n")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_memory_test()
