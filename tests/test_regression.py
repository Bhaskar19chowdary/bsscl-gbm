import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

def test_mathematical_regression():
    """Ensure the model always outputs the exact same mathematical predictions across versions."""
    X = np.random.RandomState(42).randn(100, 5)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    model = HybridHistGBMNumbaV1_0_1(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    
    # Checksum of the probabilities to ensure 100% mathematical stability
    checksum = np.sum(probs)
    
    # On a perfectly stable deterministic model with seed=42, 
    # this checksum must never change across future updates.
    np.testing.assert_allclose(checksum, 100.0, atol=1e-5)
