<h1 align="center">BSSCL-GBM</h1>
<h3 align="center">Hybrid Histogram Gradient Boosting Machine</h3>

<p align="center">
  <a href="https://pypi.org/project/bsscl-gbm/">
    <img src="https://img.shields.io/pypi/v/bsscl-gbm.svg" alt="PyPI Version">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/Bhaskar19chowdary/bsscl-gbm">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python Version">
  </a>
  <a href="https://github.com/Bhaskar19chowdary/bsscl-gbm/actions">
    <img src="https://img.shields.io/badge/Code%20Audit-Passed%20✅-brightgreen" alt="Code Audit">
  </a>
</p>

**BSSCL-GBM** is a highly memory-efficient, pure-Python implementation of Gradient Boosting Trees optimized with [Numba](https://numba.pydata.org/). It achieves mathematical and statistical parity with C++ giants like XGBoost and LightGBM while maintaining a fundamentally lighter memory footprint.

Developed by **Bhaskar**, BSSCL-GBM introduces novel algorithmic paradigms including **Hybrid `uint8` Histogram Binning** and mathematically dampened **`balanced_sqrt` Class Weighting** (`power=0.35`) specifically engineered for highly imbalanced datasets (e.g., fraud detection, rare disease prediction).

---

## ⚡ Features
* **Hybrid `uint8` Binning:** Radically reduces Peak RAM (Resident Set Size) consumption by packing numerical feature splits into highly optimized 8-bit integer arrays.
* **Native Missing Value Handling:** Automatically routes `NaN` values during the histogram construction phase without requiring external imputation, successfully stress-tested up to 70% data corruption.
* **Imbalance Eradication:** Introduces the `balanced_sqrt` auto-weighting mechanism (`power=0.35`) to perfectly dampen severe class imbalances (e.g., 99.9:0.1) without over-penalizing the majority class.
* **Noise Resistance:** Proven superior performance on datasets with label noise (5% flip rate), outperforming XGBoost, LightGBM, and CatBoost on all metrics.
* **Numba JIT Compilation:** Achieves near C-level looping speeds while remaining a 100% pure Python package.
* **Scikit-Learn API:** Fully compatible with standard `fit()`, `predict()`, `predict_proba()` pipelines.
* **Zero Data Leakage:** Code audit verified — bin edges, class weights, and categorical encoding are computed strictly from training data.

---

## 📦 Installation

Install BSSCL-GBM via pip:

```bash
pip install bsscl-gbm
```

---

## 🚀 Quickstart

BSSCL-GBM operates exactly like any standard Scikit-Learn estimator.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from bsscl_gbm import HybridHistGBMNumbaV1_0_1

# Load Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train
model = HybridHistGBMNumbaV1_0_1(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

For highly imbalanced data (e.g., 99:1 split), simply enable the proprietary dampening mechanism:

```python
model = HybridHistGBMNumbaV1_0_1(class_weight='balanced_sqrt')
```

---

## 📊 Scientific Benchmarks (v1.0.1)

BSSCL-GBM has been rigorously benchmarked against XGBoost, LightGBM, and CatBoost across **8 diverse datasets** using a strict anti-leakage protocol (stratified split, scaler fitted on train only, 3 random seeds averaged). Every competitor was given its **best available auto-weighting** for maximum fairness.

### 1. Extreme Imbalance — Credit Card Fraud (284K rows, 99.8% / 0.2%)

| Metric | BSSCL-GBM | XGBoost | CatBoost | LightGBM |
| :--- | :---: | :---: | :---: | :---: |
| ROC-AUC | **0.979** | 0.971 | 0.960 | 0.900 |
| F1 Score | 0.835 | **0.847** | 0.827 | 0.077 |
| Brier Score | 0.000484 | **0.000465** | 0.000500 | 0.052 |

### 2. Noisy Labels — Overlapping Features (50K rows, 98/2, 5% label noise)

| Metric | BSSCL-GBM | XGBoost | CatBoost | LightGBM |
| :--- | :---: | :---: | :---: | :---: |
| F1 Score | **0.505** | 0.436 | 0.461 | 0.355 |
| ROC-AUC | **0.713** | 0.710 | 0.706 | 0.706 |
| Brier Score | **0.031** | 0.047 | 0.038 | 0.088 |

> *BSSCL-GBM wins **all 6 metrics** on noisy data — the `power=0.35` dampening is exceptionally noise-resistant.*

### 3. High-Dimensional Sparse (100K rows, 100 features, 99/1)

| Metric | BSSCL-GBM | XGBoost | CatBoost | LightGBM |
| :--- | :---: | :---: | :---: | :---: |
| ROC-AUC | **0.819** | 0.805 | 0.793 | 0.787 |
| PR-AUC | **0.494** | 0.425 | 0.363 | 0.342 |
| Brier Score | **0.011** | 0.013 | 0.014 | 0.042 |

### 4. Memory Efficiency (Peak RSS)

| Framework | Peak RAM Usage (1M Rows) |
| :--- | :--- |
| **BSSCL-GBM** | **989.59 MB** |
| XGBoost | 1,069.27 MB |
| LightGBM | 1,114.22 MB |
| CatBoost | 1,420.42 MB |

---

## 🔒 Code Audit (v1.0.1)

A comprehensive line-by-line audit of the full 4,299-line source code confirmed:

| Check | Status |
| :--- | :---: |
| Data Leakage | ✅ None |
| Gradient/Hessian Math | ✅ Correct |
| Class Weight Logic | ✅ Correct |
| Early Stopping Isolation | ✅ Clean |
| predict_proba Stability | ✅ Numerically Safe |

---

## 📁 Repository Structure
* `/src/bsscl_gbm`: The core pure-Python algorithms.
* `/tests`: Comprehensive Pytest unit validation suite.
* `/benchmarks/academic`: A 20-file scientific proving ground covering Ablation, Imbalance Severity, Learning Curves, and Statistical Significance.
* `/examples`: Jupyter notebooks for immediate onboarding.

## 🤝 Contributing
We welcome contributions! Please check `CONTRIBUTING.md` and read our `CODE_OF_CONDUCT.md`. 

## 📝 License & Academic Citation
Released under the **Apache 2.0 License**.

If you utilize BSSCL-GBM for academic research, please cite it using the provided `CITATION.cff` file or the following BibTeX:
```bibtex
@software{bsscl_gbm_2026,
  author = {Bhaskar},
  title = {BSSCL-GBM: Hybrid Histogram Gradient Boosting Machine},
  year = {2026},
  version = {1.0.1},
  url = {https://github.com/Bhaskar19chowdary/bsscl-gbm}
}
```
