import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV2

def test_numerical_precision_float32():
    """Test model behavior on float32 arrays."""
    X = np.random.RandomState(42).randn(100, 5).astype(np.float32)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    model = HybridHistGBMNumbaV2(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    
    assert preds.shape == (100,)
    assert np.all(np.isfinite(preds))

def test_numerical_precision_float16():
    """Test model behavior on float16 arrays. Numpy casts them internally, we just ensure it doesn't crash."""
    X = np.random.RandomState(42).randn(100, 5).astype(np.float16)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    model = HybridHistGBMNumbaV2(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    
    assert np.all(np.isfinite(preds))

def test_numerical_precision_int_types():
    """Test model behavior on integer inputs (which are valid categorical/ordinal features)."""
    X_int32 = np.random.RandomState(42).randint(-1000, 1000, size=(100, 5), dtype=np.int32)
    y = np.random.RandomState(42).randint(0, 2, size=100)
    
    model = HybridHistGBMNumbaV2(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X_int32, y)
    preds_int32 = model.predict(X_int32)
    
    X_int64 = X_int32.astype(np.int64)
    model_int64 = HybridHistGBMNumbaV2(n_estimators=2, max_depth=2, random_state=42)
    model_int64.fit(X_int64, y)
    preds_int64 = model_int64.predict(X_int64)
    
    # Assert perfectly identical behavior between int32 and int64 inputs
    np.testing.assert_array_almost_equal(preds_int32, preds_int64)

def test_overflow_protection():
    """Test behavior with extreme float limits to prevent overflow crashes."""
    X = np.array([
        [1e308, 1e-308],
        [-1e308, -1e-308],
        [0.0, 0.0]
    ])
    y = np.array([0, 1, 0])
    
    model = HybridHistGBMNumbaV2(n_estimators=2, max_depth=2)
    
    try:
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))
    except (ValueError, OverflowError):
        # Gracefully failing on numpy overflow is acceptable
        pass
