# Contributing to bsscl-gbm

First off, thank you for considering contributing to `bsscl-gbm`! It's people like you that make this model faster and better for everyone.

## 1. Local Development Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/your-username/bsscl-gbm.git
   cd bsscl-gbm
   ```
3. **Install dependencies**:
   ```bash
   pip install -e .
   pip install pytest pytest-cov ruff mypy bandit safety build twine
   ```

## 2. Writing Code
We enforce rigorous code quality standards.
- All core computational loops **must** be written as Numba `@njit` functions.
- Avoid passing Python objects into Numba contexts.
- Use `np.float64` for all internal data structures to maintain precision.

## 3. Running Tests
Before submitting a Pull Request, you **must** ensure all tests pass.
```bash
# Run the fast suite (takes ~10 seconds)
pytest tests/ -m "not stress"

# Run the stress suite (Memory Leaks, Fuzzing - takes ~2 minutes)
pytest tests/ -m stress

# Run Static Analysis
ruff check src/

# Run Security Scans
bandit -r src/
```

## 4. Submitting a Pull Request
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Commit your changes: `git commit -m 'Add some feature'`
3. Push to the branch: `git push origin feature/your-feature-name`
4. Open a Pull Request on GitHub.

Ensure your PR describes *why* the change is necessary, and links to any relevant Issues. All PRs require approval from a core maintainer before merging.
