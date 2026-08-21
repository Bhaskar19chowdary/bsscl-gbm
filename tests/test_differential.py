import numpy as np
import pytest
from bsscl_gbm import HybridHistGBMNumbaV1_0_1
from sklearn.ensemble import HistGradientBoostingClassifier

def test_differential_sklearn():
    """Compare predictions against scikit-learn's HistGradientBoostingClassifier."""
    X = np.random.RandomState(42).randn(500, 10)
    y = np.random.RandomState(42).randint(0, 2, size=500)
    
    # Train scikit-learn
    sk_model = HistGradientBoostingClassifier(
        max_iter=10, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42
    )
    sk_model.fit(X, y)
    sk_preds = sk_model.predict_proba(X)[:, 1]
    
    # Train bsscl-gbm
    our_model = HybridHistGBMNumbaV1_0_1(
        n_estimators=10, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42
    )
    our_model.fit(X, y)
    our_preds = our_model.predict_proba(X)[:, 1]
    
    # We don't expect exactly identical probabilities due to different binning 
    # and leaf logic, but they should be highly correlated and have similar accuracy.
    sk_acc = (sk_model.predict(X) == y).mean()
    our_acc = (our_model.predict(X) == y).mean()
    
    # Both should learn the dataset (accuracy > 0.6)
    assert sk_acc > 0.6
    assert our_acc > 0.6
    
    # Accuracies should be within 5% of each other
    assert abs(sk_acc - our_acc) < 0.05
