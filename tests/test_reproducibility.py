import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

def test_benchmark_reproducibility():
    """Execute the same benchmark repeatedly to ensure 100% determinism."""
    X = np.random.RandomState(42).randn(100, 5)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    reference_preds = None
    
    # Run 100 times to guarantee perfect determinism
    for seed in range(100):
        model = HybridHistGBMNumbaV1_0_1(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        
        if reference_preds is None:
            reference_preds = preds.copy()
        else:
            np.testing.assert_array_equal(
                preds, 
                reference_preds, 
                err_msg=f"Determinism failed on iteration {seed}"
            )
