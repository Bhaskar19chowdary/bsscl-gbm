import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from bsscl_gbm import HybridHistGBMNumbaV1_0_1
import time
import os

def run_real_world_dataset(name, data_id, task="classification"):
    print(f"\n=============================================")
    print(f" 🌍 REAL-WORLD TEST: {name} (OpenML ID: {data_id})")
    print(f"=============================================")
    
    try:
        # Fetch data
        print(f"Downloading dataset...")
        data = fetch_openml(data_id=data_id, as_frame=False, parser='auto')
        X, y = data.data, data.target
        
        # Preprocessing
        # Convert string categories to numerical
        if y.dtype == object or isinstance(y[0], str):
            unique_classes = np.unique(y)
            class_map = {c: i for i, c in enumerate(unique_classes)}
            y = np.array([class_map[c] for c in y])
            
        # Clean categorical columns in X
        from sklearn.preprocessing import OrdinalEncoder
        if X.dtype == object:
            # We must encode strings to numbers
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X = enc.fit_transform(X)
            
        # Clean NaNs in X
        X = np.asarray(X, dtype=np.float64)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Dataset Shape: {X.shape}")
        print(f"Training...")
        
        start = time.perf_counter()
        model = HybridHistGBMNumbaV1_0_1(
            n_estimators=50, 
            max_depth=5,
            learning_rate=0.1,
            early_stopping_rounds=10,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        duration = time.perf_counter() - start
        
        preds = model.predict(X_test)
        acc = (preds == y_test).mean()
        
        print(f"✅ Success! Training took {duration:.2f}s")
        print(f"🎯 Test Accuracy: {acc:.4f}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("WARNING: This downloads hundreds of megabytes of real-world datasets from OpenML.")
    print("If you are on a metered connection, press Ctrl+C now.\n")
    
    # 1. Medical (Breast Cancer)
    run_real_world_dataset("Medical - Breast Cancer", data_id=13)
    
    # 2. Fraud / Imbalanced (Credit-g)
    run_real_world_dataset("Imbalanced - Credit Risk", data_id=31)
    
    # 3. Cybersecurity Data (Spambase)
    run_real_world_dataset("Cybersecurity - Spambase", data_id=44)
    
    # 4. Large Dataset (Adult Income)
    run_real_world_dataset("Large Data - Adult Income", data_id=1590)
    
    # 5. Image Data (MNIST 784 - heavily downsampled for speed)
    print("\n--- Note: MNIST is very large, this may take a moment to fetch ---")
    run_real_world_dataset("Image Data - MNIST (subset)", data_id=554, task="classification")
    
    # 6. Rare Classes (Sick)
    run_real_world_dataset("Rare Classes - Sick", data_id=38, task="classification")
    
    print("\n✅ All real-world tests completed.")
