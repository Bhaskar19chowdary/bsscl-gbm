import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV2

def test_multiclass_and_early_stopping():
    """Trigger multiclass, level-wise growth, early stopping, and monotonic constraints."""
    X = np.random.RandomState(42).randn(300, 5)
    y = np.random.RandomState(42).randint(0, 3, size=300)
    
    # We pass an explicit categorical set and monotonic constraint to hit those branches
    # X[:, 0] is categorical, X[:, 1] has a monotonic constraint of 1
    model = HybridHistGBMNumbaV2(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        early_stopping_rounds=3,
        grow_policy='level',
        categorical_features=[0],
        monotone_constraints=[1, 0, 0, 0, -1],
        random_state=42
    )
    
    # Eval set triggers early stopping logic
    model.fit(X, y, eval_set=[(X, y)])
    
    # Assert model was fitted and early stopped
    assert model.n_features_in_ == 5
    assert model.n_classes_ == 3
    assert model.best_iteration_ > 0
    assert model.best_iteration_ < 100
    
    # Predict to trigger multiclass prediction loop
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    assert preds.shape == (300,)
    assert probs.shape == (300, 3)

def test_missing_value_handling_and_auto_weights():
    """Trigger NaN masking, feature_importances generation, and auto-weights logic."""
    X = np.random.RandomState(42).randn(200, 4)
    # Inject NaNs
    X[::10, 0] = np.nan
    X[5:15, 2] = np.nan
    
    # Trigger auto class weights for highly imbalanced binary target
    y = np.zeros(200, dtype=np.int32)
    y[:10] = 1
    
    model = HybridHistGBMNumbaV2(
        n_estimators=10,
        class_weight='balanced_sqrt',
        random_state=42
    )
    model.fit(X, y)
    
    # Trigger feature importances
    importances = model.feature_importances_
    assert len(importances) == 4
    assert np.sum(importances) > 0

def test_zero_variance_feature():
    """Trigger the zero-variance branch logic."""
    X = np.ones((50, 3)) # Completely uniform data
    y = np.random.randint(0, 2, 50)
    
    model = HybridHistGBMNumbaV2(n_estimators=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 50

def test_single_row_prediction():
    """Trigger single row unpacking and single row predictions."""
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    
    model = HybridHistGBMNumbaV2(n_estimators=2)
    model.fit(X, y)
    
    # Single row predict
    x_single = X[0:1]
    pred = model.predict(x_single)
    prob = model.predict_proba(x_single)
    
    assert len(pred) == 1
    assert prob.shape == (1, 2)
    
def test_predict_batch_parallel_multiclass():
    """Trigger multiprocessing predict logic on multiclass data."""
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 3, 100)
    
    model = HybridHistGBMNumbaV2(n_estimators=5)
    model.fit(X, y)
    
    # Test batch parallel predictions
    preds = model.predict_batch_parallel(X, n_workers=2, batch_size=50)
    assert len(preds) == 100
