import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV2

def test_property_invariants():
    """Property-based testing for model invariants."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).randint(0, 3, size=100)
    
    model = HybridHistGBMNumbaV2(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Invariant 1: Shape matches input
    assert predictions.shape[0] == X.shape[0], "Prediction shape mismatch"
    assert probabilities.shape[0] == X.shape[0], "Probability shape mismatch"
    assert probabilities.shape[1] == 3, "Probability class count mismatch"
    
    # Invariant 2: Finiteness
    assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
    assert np.all(np.isfinite(probabilities)), "Probabilities contain non-finite values"
    
    # Invariant 3: Probability bounds [0, 1]
    assert np.all(probabilities >= 0.0), "Probabilities must be >= 0"
    assert np.all(probabilities <= 1.0), "Probabilities must be <= 1"
    
    # Invariant 4: Probabilities sum to 1 (with small epsilon for float precision)
    prob_sums = np.sum(probabilities, axis=1)
    np.testing.assert_allclose(prob_sums, np.ones_like(prob_sums), rtol=1e-5, atol=1e-5, err_msg="Probabilities do not sum to 1")
