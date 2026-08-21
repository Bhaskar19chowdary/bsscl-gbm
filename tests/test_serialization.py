import numpy as np
import pytest
import pickle
import joblib
import os
import tempfile
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

def test_serialization_integrity():
    """Serialization integrity testing for Pickle and Joblib."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).randint(0, 3, size=100)
    
    model = HybridHistGBMNumbaV1_0_1(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)
    
    original_preds = model.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Pickle
        pkl_path = os.path.join(tmpdir, "model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
            
        with open(pkl_path, "rb") as f:
            loaded_pkl = pickle.load(f)
            
        np.testing.assert_array_equal(original_preds, loaded_pkl.predict(X))
        
        # Test 2: Joblib
        jl_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(model, jl_path)
        
        loaded_jl = joblib.load(jl_path)
        np.testing.assert_array_equal(original_preds, loaded_jl.predict(X))
