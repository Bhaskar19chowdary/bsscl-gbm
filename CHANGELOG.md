# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-02
### Added
- Official PyPI Initial Release.
- Massive 20-file Academic Benchmarking Suite (Phase 3 & Phase 4).
- Enterprise-grade CI pipeline with OIDC PyPI publishing.
- Fully automated mutation testing capabilities via `mutmut`.
- 10-tier Ultimate QA testing (Differential, Reproducibility, Fault Injection).
- Native support for categorical features (no OHE required).
- Native monotonic constraints.

### Changed
- Refactored binning algorithm to safely intercept and auto-cast `np.float16` to `np.float64` to prevent deep Numba LLVM crashing.
- Strengthened hyperparameter bounds to enforce strict positivity.
- `predict()` now safely throws a `ValueError("Not fitted")` instead of an obscure internal `AttributeError` if called before `fit()`.

### Security
- Integrated `bandit` and `safety` into the GitHub Actions CI pipeline.

## [0.9.0] - 2026-08-01
### Added
- Core Numba JIT Hybrid Tree building engine.
- Extreme stress and scalability benchmarking framework.
- Auto-detected class weights and missing value mask handling.
