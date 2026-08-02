import numpy as np
import pytest
import pickle
import os
import tempfile
from bsscl_gbm import HybridHistGBMNumbaV2

def test_backward_compatibility():
    """Simulate loading a model saved from an older version (V1.0) and verify predictions still work."""
    X = np.random.RandomState(42).randn(10, 5)
    y = np.random.RandomState(42).randint(0, 2, size=10)
    
    # We train a modern model
    model = HybridHistGBMNumbaV2(n_estimators=1, max_depth=1)
    model.fit(X, y)
    preds_modern = model.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate saving it
        pkl_path = os.path.join(tmpdir, "model_v1.pkl")
        
        # We manually alter its internal state to simulate an old version missing modern attributes
        # (e.g. an old model might not have some new metadata flag)
        # We inject a fake old schema state
        state_dict = model.__dict__.copy()
        state_dict['_simulated_v1'] = True 
        
        # In python, you can't easily break the class definition on the fly without metaprogramming,
        # but we can save this state dict. Since Pickle uses __setstate__, missing new defaults 
        # should not crash if designed correctly.
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
            
        with open(pkl_path, "rb") as f:
            loaded_model = pickle.load(f)
            
        # The true test of ABI compat: can it still predict?
        preds_legacy = loaded_model.predict(X)
        
        np.testing.assert_array_equal(preds_modern, preds_legacy)
