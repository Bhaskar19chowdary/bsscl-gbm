import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor
from bsscl_gbm import HybridHistGBMNumbaV2

def concurrent_predict(m, X_data):
    return m.predict(X_data)

def test_multiprocessing_inference():
    """Test that a trained model can safely predict concurrently across multiple processes."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    # Train ONCE on the main process
    model = HybridHistGBMNumbaV2(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X, y)
    
    baseline = model.predict(X)
    
    import multiprocessing as mp
    # Concurrent processes simulating gunicorn workers
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = [executor.submit(concurrent_predict, model, X) for _ in range(4)]
        results = [f.result() for f in futures]
        
    for i in range(4):
        np.testing.assert_array_equal(baseline, results[i])
