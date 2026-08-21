import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

def test_fault_invalid_hyperparams():
    """Test that model rejects invalid hyperparameters gracefully."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    with pytest.raises(ValueError):
        # Invalid depth
        model = HybridHistGBMNumbaV1_0_1(max_depth=-5)
        model.fit(X, y)

    with pytest.raises(ValueError):
        # Invalid learning rate
        model = HybridHistGBMNumbaV1_0_1(learning_rate=-0.1)
        model.fit(X, y)

def test_fault_corrupted_data():
    """Test that model rejects mismatched X and y."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 99) # 99 instead of 100
    
    model = HybridHistGBMNumbaV1_0_1()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_fault_unfitted_predict():
    """Test that model rejects predictions if not fitted."""
    X = np.random.randn(100, 5)
    model = HybridHistGBMNumbaV1_0_1()
    
    with pytest.raises(ValueError):
        model.predict(X)
