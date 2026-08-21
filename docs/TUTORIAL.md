# bsscl-gbm Tutorial

Welcome to `bsscl-gbm`! This tutorial covers everything from basic binary classification to advanced features like categorical handling, multi-class modeling, and monotonic constraints.

## 1. Basic Binary Classification

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

# 1. Generate Dummy Data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Initialize Model
model = HybridHistGBMNumbaV1_0_1(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1
)

# 3. Train
model.fit(X_train, y_train)

# 4. Predict
preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")
```

## 2. Early Stopping
If you pass an `eval_set`, the model will automatically monitor the validation loss. Use `early_stopping_rounds` to stop training if the validation loss doesn't improve.

```python
model = HybridHistGBMNumbaV1_0_1(
    n_estimators=1000, 
    early_stopping_rounds=10
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

print(f"Best iteration: {model.best_iteration_}")
```

## 3. Categorical Features (No OHE required!)
You don't need to One-Hot Encode your categorical data. Simply pass the feature indices using `categorical_features`. The model will natively find the optimal splits!

```python
# Suppose column 0 and column 3 are categorical
model = HybridHistGBMNumbaV1_0_1(categorical_features=[0, 3])
model.fit(X_train, y_train)
```

## 4. Multi-class Classification
The model automatically detects multi-class targets (when `y` has > 2 unique values) and trains a One-vs-Rest ensemble natively.

```python
y_multiclass = np.random.randint(0, 5, 10000) # 5 classes
model = HybridHistGBMNumbaV1_0_1(n_estimators=50)
model.fit(X_train, y_multiclass)

# predict_proba returns a matrix of shape (n_samples, n_classes)
probs = model.predict_proba(X_test)
```

## 5. Monotonic Constraints
If you know that a specific feature should always have a positive (or negative) effect on the target, you can constrain the model.

```python
# 1 = Increasing, -1 = Decreasing, 0 = Unconstrained
# Length must equal the number of features (e.g., 5 features)
constraints = [1, 0, -1, 0, 0] 

model = HybridHistGBMNumbaV1_0_1(monotone_constraints=constraints)
model.fit(X_train, y_train)
```

## 6. Auto-Handling Missing Values (NaN)
`bsscl-gbm` natively supports `np.nan` values. Just leave them in your `X` matrix, and the algorithm will automatically learn the best direction to send missing values at every tree node.

```python
X_train[0, 5] = np.nan # Inject a missing value
model.fit(X_train, y_train) # Works perfectly!
```
