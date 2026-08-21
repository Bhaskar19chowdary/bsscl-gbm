import numpy as np
import pytest
import time
import os
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_memory_mb():
    if not HAS_PSUTIL:
        return 0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

@pytest.mark.stress
def test_memory_leak():
    """Run training repeatedly and monitor RAM usage."""
    if not HAS_PSUTIL:
        pytest.skip("psutil required for memory leak test")
        
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    # Warmup
    model = HybridHistGBMNumbaV1_0_1(n_estimators=1, max_depth=2, random_state=42)
    model.fit(X, y)
    model.predict(X)
    
    start_mem = get_memory_mb()
    
    for _ in range(10000):
        model = HybridHistGBMNumbaV1_0_1(n_estimators=1, max_depth=2)
        model.fit(X, y)
        model.predict(X)
        
    end_mem = get_memory_mb()
    mem_diff = end_mem - start_mem
    
    # Allow a small buffer (e.g., 50MB) for Python's natural garbage collection overhead
    assert mem_diff < 50.0, f"Memory leak detected: grew by {mem_diff:.2f} MB"

@pytest.mark.stress
def test_long_duration():
    """Run the library continuously for a massive number of iterations."""
    X = np.random.RandomState(42).randn(10, 5)
    y = np.random.RandomState(42).randint(0, 3, size=10)
    
    for _ in range(50000):
        model = HybridHistGBMNumbaV1_0_1(n_estimators=1, max_depth=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

def test_performance_regression():
    """Performance regression testing to prevent future code from slowing down the library."""
    X = np.random.RandomState(42).randn(5000, 20)
    y = np.random.RandomState(42).randint(0, 3, size=5000)
    
    # Warmup Numba JIT first so compilation time isn't counted
    warmup_model = HybridHistGBMNumbaV1_0_1(n_estimators=1, max_depth=2)
    warmup_model.fit(X[:100], y[:100])
    
    start_time = time.perf_counter()
    
    model = HybridHistGBMNumbaV1_0_1(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)
    model.predict(X)
    
    duration = time.perf_counter() - start_time
    
    # Baseline for this dataset on a modern CPU is usually < 0.5s. 
    # We set a generous upper bound of 1.5s to prevent massive regressions 
    # without flaking on slower CI runners.
    baseline_limit = 1.5 
    assert duration <= baseline_limit, f"Performance regression! Took {duration:.2f}s, expected <= {baseline_limit}s"
