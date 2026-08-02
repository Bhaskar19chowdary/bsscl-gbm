import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV2

def test_fuzz_invalid_types():
    """Test that the model gracefully rejects or handles invalid types."""
    model = HybridHistGBMNumbaV2(n_estimators=1, max_depth=2)
    
    # 1. Strings
    X_str = np.array([["a", "b"], ["c", "d"]])
    y_str = np.array([0, 1])
    with pytest.raises((ValueError, TypeError)):
        model.fit(X_str, y_str)

    # 2. Object arrays (Numpy might cast None to NaN and succeed!)
    X_obj = np.empty((10, 2), dtype=object)
    y_obj = np.zeros(10)
    try:
        model.fit(X_obj, y_obj)
        model.predict(X_obj)
    except (ValueError, TypeError):
        pass # Also acceptable

    # 3. Empty arrays
    X_empty = np.array([[]])
    y_empty = np.array([])
    with pytest.raises((ValueError, IndexError)):
        model.fit(X_empty, y_empty)

    # 4. Sparse matrices (scipy)
    try:
        from scipy.sparse import csr_matrix
        X_sparse = csr_matrix([[1, 2], [3, 4]])
        y_sparse = np.array([0, 1])
        with pytest.raises((ValueError, TypeError)):
            # Our current implementation requires dense numpy arrays
            model.fit(X_sparse, y_sparse)
    except ImportError:
        pass

def test_fuzz_extreme_values():
    """Test NaN, Inf, and extreme values."""
    model = HybridHistGBMNumbaV2(n_estimators=1, max_depth=2)
    
    # NaN and Inf are supported by the binning logic or rejected gracefully
    X = np.random.randn(100, 10)
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    X[2, 2] = -np.inf
    y = np.random.randint(0, 2, 100)
    
    # This should not segfault. It might raise ValueError if Inf is not supported,
    # or it might train successfully. We just ensure it doesn't crash Python.
    try:
        model.fit(X, y)
        model.predict(X)
    except ValueError:
        pass  # Numpy/Sklearn check_array might reject Inf

@pytest.mark.stress
def test_fuzz_massive_loop():
    """Run 100,000 random fuzzing iterations (Highest Priority)."""
    # 100,000 is massive and will take ~3-5 minutes, marked as stress
    for i in range(100000):
        # Use small dimensions to focus on fuzzing overhead, not pure computation time
        X = np.random.randn(10, 5) 
        y = np.random.randint(0, 3, 10)
        
        # Inject random NaNs
        if i % 2 == 0:
            X[0, 0] = np.nan
            
        model = HybridHistGBMNumbaV2(n_estimators=1, max_depth=2)
        model.fit(X, y)
        preds = model.predict(X)
        
        assert preds.shape == (10,)
