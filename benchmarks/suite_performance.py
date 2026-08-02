import time
import psutil
import os
import numpy as np
from sklearn.datasets import make_classification
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_models(n_jobs=1):
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=20, max_depth=5),
        "XGBoost": xgb.XGBClassifier(n_estimators=20, max_depth=5, use_label_encoder=False, eval_metric='logloss', n_jobs=n_jobs),
        "LightGBM": lgb.LGBMClassifier(n_estimators=20, max_depth=5, n_jobs=n_jobs, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=20, depth=5, thread_count=n_jobs, verbose=0)
    }

def run_performance_suite(fast_mode=False):
    print_header("SUITE E: Performance & Speed Evaluation (Side-by-Side)")
    
    n_samples = 50000 if fast_mode else 500000
    X, y = make_classification(n_samples=n_samples, n_features=30, random_state=42)
    
    # 1. Training Speed & Memory
    print("\n--- 18/20. Training Speed & Memory ---")
    print(f"[ Dataset: {n_samples} rows, 30 cols ]")
    
    # We instantiate models and measure fit time + RAM
    for name, model in get_models(n_jobs=-1).items():
        mem_before = measure_memory_mb()
        
        start = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - start
        
        mem_after = measure_memory_mb()
        print(f"  {name:<12s} Train Time: {train_time:6.3f}s | RAM Spiked: {mem_after - mem_before:6.1f} MB")

    # 2. Prediction Latency
    print("\n--- 19. Prediction Latency ---")
    batch_sizes = [1, 100, 10000]
    
    # Pre-train models for prediction tests
    trained_models = {}
    for name, model in get_models(n_jobs=1).items():
        model.fit(X, y)
        trained_models[name] = model

    for b in batch_sizes:
        print(f"\n[ Latency ({b:>5d} samples) ]")
        X_batch = X[:b]
        for name, model in trained_models.items():
            start = time.perf_counter()
            _ = model.predict(X_batch)
            duration = time.perf_counter() - start
            throughput = b / duration if duration > 0 else 0
            print(f"  {name:<12s} Time: {duration*1000:7.2f} ms | Throughput: {throughput:9.0f} iter/sec")

    # 3. CPU Scaling
    print("\n--- 21. CPU Thread Scaling (Prediction) ---")
    threads = [1, 2, 4] if fast_mode else [1, 2, 4, 8]
    
    for t in threads:
        print(f"\n[ Workers: {t} ]")
        
        # BSSCL-GBM uses ProcessPoolExecutor explicitly in predict_batch_parallel
        model_bsscl = HybridHistGBMNumbaV2(n_estimators=20, max_depth=5)
        model_bsscl.fit(X, y)
        start = time.perf_counter()
        _ = model_bsscl.predict_batch_parallel(X, n_workers=t, batch_size=len(X)//t)
        duration = time.perf_counter() - start
        print(f"  BSSCL-GBM    Batch Parallel Time: {duration:6.3f}s")
        
        # XGBoost natively supports n_jobs
        model_xgb = xgb.XGBClassifier(n_estimators=20, max_depth=5, n_jobs=t, use_label_encoder=False, eval_metric='logloss')
        model_xgb.fit(X, y)
        start = time.perf_counter()
        _ = model_xgb.predict(X)
        duration = time.perf_counter() - start
        print(f"  XGBoost      Native Parallel Time: {duration:6.3f}s")

    print("\n✅ Suite E (Performance) Completed")

if __name__ == "__main__":
    run_performance_suite(fast_mode=True)
