import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon, friedmanchisquare
from bsscl_gbm import HybridHistGBMNumbaV2
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

def get_models():
    return {
        "BSSCL-GBM": HybridHistGBMNumbaV2(n_estimators=100, max_depth=6),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(n_estimators=100, depth=6, verbose=0)
    }

def get_datasets():
    X_bc, y_bc = load_breast_cancer(return_X_y=True)
    
    X1, y1 = make_classification(n_samples=2000, n_features=20, random_state=1)
    X2, y2 = make_classification(n_samples=5000, n_features=30, n_informative=15, random_state=2)
    X3, y3 = make_classification(n_samples=10000, n_features=10, random_state=3)
    X4, y4 = make_classification(n_samples=15000, n_features=50, n_informative=20, random_state=4)
    
    return {
        "Breast Cancer (Real)": (X_bc, y_bc),
        "Synth_2k_20f": (X1, y1),
        "Synth_5k_30f": (X2, y2),
        "Synth_10k_10f": (X3, y3),
        "Synth_15k_50f": (X4, y4)
    }

def run_statistical_tests():
    print(f"\n{'='*70}\n  PHASE 4: FORMAL STATISTICAL SIGNIFICANCE TESTS\n{'='*70}")
    print("Running Friedman Test (Ranking) and Wilcoxon Signed-Rank Test across 5 datasets.\n")
    
    datasets = get_datasets()
    model_names = list(get_models().keys())
    
    # Store accuracy scores: dict[model] = [score_ds1, score_ds2, ...]
    scores = {name: [] for name in model_names}
    
    from sklearn.model_selection import train_test_split
    
    for ds_name, (X, y) in datasets.items():
        print(f"Evaluating Dataset: {ds_name}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for name, model in get_models().items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            scores[name].append(acc)
            
    print("\n--- Raw Accuracy Scores ---")
    df_scores = pd.DataFrame(scores, index=datasets.keys())
    print(df_scores.to_markdown())
    
    print("\n--- Friedman Test (Overall Ranking) ---")
    # Friedman test evaluates if there's a statistically significant difference in ranks
    stat, p_value = friedmanchisquare(scores["BSSCL-GBM"], scores["XGBoost"], scores["LightGBM"], scores["CatBoost"])
    print(f"Friedman chi-square statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    if p_value < 0.05:
        print("=> Result: Significant difference found among the models (p < 0.05).")
    else:
        print("=> Result: No significant difference found among the models (p >= 0.05).")
        
    print("\n--- Wilcoxon Signed-Rank Tests (Pairwise) ---")
    # Wilcoxon evaluates BSSCL vs each baseline specifically
    for baseline in ["XGBoost", "LightGBM", "CatBoost"]:
        diff = np.array(scores["BSSCL-GBM"]) - np.array(scores[baseline])
        if np.all(diff == 0):
             print(f"BSSCL-GBM vs {baseline}: Models are identical on these datasets.")
             continue
             
        stat, p_value = wilcoxon(scores["BSSCL-GBM"], scores[baseline], zero_method='zsplit')
        print(f"BSSCL-GBM vs {baseline}:")
        print(f"  Wilcoxon statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  => Result: BSSCL-GBM is significantly different (p < 0.05).")
        else:
            print("  => Result: No statistically significant difference (p >= 0.05).")

if __name__ == "__main__":
    run_statistical_tests()
