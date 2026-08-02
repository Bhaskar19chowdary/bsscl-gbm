import numpy as np
import shap
from sklearn.datasets import make_classification
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=10, max_depth=3),
        "XGBoost": xgb.XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=10, max_depth=3, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=10, depth=3, verbose=0)
    }

def run_enterprise_suite(fast_mode=False):
    print_header("SUITE L/I: Advanced Enterprise & Explainability (Side-by-Side)")
    
    # 1. Concept Drift Testing
    print("\n--- 43. Concept Drift Testing ---")
    # Train on distribution A
    X_old, y_old = make_classification(n_samples=5000, n_features=10, random_state=42)
    # Test on distribution B (shifted features)
    X_new = X_old + 2.0
    y_new = y_old
    
    for name, model in get_models().items():
        model.fit(X_old, y_old)
        
        preds_old = model.predict(X_old)
        preds_new = model.predict(X_new)
        
        acc_old = np.mean(preds_old == y_old)
        acc_new = np.mean(preds_new == y_new)
        print(f"\n[ {name} ]")
        print(f"  Accuracy on original data: {acc_old:.3f}")
        print(f"  Accuracy on drifted data:  {acc_new:.3f}")
        print(f"  Drift Degradation:         {(acc_old - acc_new)*100:.1f}% drop")

    # 2. Explainability (SHAP vs Native Importances)
    print("\n--- 29. Explainability (SHAP Validation) ---")
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, random_state=1)
    
    for name, model in get_models().items():
        model.fit(X, y)
        
        # Native Importances
        if name == "CatBoost":
            native_importances = model.get_feature_importance()
        else:
            native_importances = model.feature_importances_
            
        top_feature_native = np.argmax(native_importances)
        
        # SHAP Importances
        # For BSSCL-GBM we use KernelExplainer. For others we use TreeExplainer
        if name == "BSSCL-GBM":
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X, 10))
            shap_values = explainer.shap_values(X[:100])
            mean_shap = np.abs(shap_values[:, :, 1]).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:100])
            if isinstance(shap_values, list) and len(shap_values) == 2:
                mean_shap = np.abs(shap_values[1]).mean(axis=0)
            elif len(shap_values.shape) == 3: # e.g. LightGBM multiclass output shape
                mean_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
        top_feature_shap = np.argmax(mean_shap)
        
        print(f"\n[ {name} ]")
        print(f"  Top Feature (Native): Feature {top_feature_native}")
        print(f"  Top Feature (SHAP):   Feature {top_feature_shap}")
        if top_feature_native == top_feature_shap:
            print("  ✅ SHAP and Native feature importances align.")
        else:
            print("  ⚠️ Warning: SHAP and Native importances diverge.")

    print("\n✅ Suite L/I (Enterprise) Completed")

if __name__ == "__main__":
    run_enterprise_suite(fast_mode=True)
