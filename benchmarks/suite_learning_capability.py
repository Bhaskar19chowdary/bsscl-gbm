import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import roc_auc_score, accuracy_score
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def generate_xor(n_samples):
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y

def get_models(is_imbalanced=False):
    models = {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=50, max_depth=4),
        "XGBoost": xgb.XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=50, depth=4, verbose=0)
    }
    
    if is_imbalanced:
        # Turn on auto-class weights where applicable
        models["BSSCL-GBM"] = HybridHistGBMNumbaV2(n_estimators=20, max_depth=4, class_weight='balanced_sqrt')
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=20, max_depth=4, class_weight='balanced', verbose=-1)
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=20, max_depth=4, scale_pos_weight=99, use_label_encoder=False, eval_metric='logloss')
        models["CatBoost"] = cb.CatBoostClassifier(n_estimators=20, depth=4, auto_class_weights='Balanced', verbose=0)
        
    return models

def run_learning_capability_suite(fast_mode=False):
    print_header("SUITE C: Learning Capability Testing (Side-by-Side)")
    
    # 1. XOR Problem
    print("\n--- 11. XOR Problem (Nonlinear Interaction) ---")
    X, y = generate_xor(5000)
    for name, model in get_models().items():
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"  {name:<12s} XOR Accuracy: {acc:.3f}")
    
    # 2. Spiral / Moons
    print("\n--- 11. Complex Boundaries (Moons) ---")
    X, y = make_moons(n_samples=5000, noise=0.2, random_state=42)
    for name, model in get_models().items():
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"  {name:<12s} Moons Accuracy: {acc:.3f}")

    # 3. Class Imbalance
    print("\n--- 12. Severe Class Imbalance ---")
    imbalances = [0.1, 0.01, 0.001] if not fast_mode else [0.01]
    for imb in imbalances:
        print(f"\n[ Imbalance {imb*100:.1f}% positive class ]")
        weights = [1.0 - imb, imb]
        X, y = make_classification(n_samples=10000, n_features=10, weights=weights, random_state=42)
        
        for name, model in get_models(is_imbalanced=True).items():
            model.fit(X, y)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
            else:
                probs = model.predict(X)
            auc = roc_auc_score(y, probs)
            print(f"  {name:<12s} ROC-AUC: {auc:.3f}")

    print("\n✅ Suite C (Learning Capability) Completed")

if __name__ == "__main__":
    run_learning_capability_suite(fast_mode=True)
