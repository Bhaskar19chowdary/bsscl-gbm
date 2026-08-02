# Release Governance & Strategy

This document outlines the standard operating procedure (SOP) for releasing new versions of `bsscl-gbm`.

## 1. Semantic Versioning (SemVer)
We strictly adhere to [Semantic Versioning 2.0.0](https://semver.org/).
- **MAJOR (`X.0.0`)**: Breaking API changes (e.g., modifying `fit()` arguments in a non-backward-compatible way).
- **MINOR (`0.Y.0`)**: New features that are backward-compatible (e.g., adding a new loss function).
- **PATCH (`0.0.Z`)**: Backward-compatible bug fixes (e.g., fixing a JIT compilation error).

## 2. Automated Release Workflow
Our release process is fully automated via GitHub Actions to eliminate human error during deployment.

### How to Trigger a Release:
1. Ensure `CHANGELOG.md` is updated with the new version notes.
2. Update the `version` field in `pyproject.toml`.
3. Create a Git tag matching the version:
   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   ```
4. The `.github/workflows/publish.yml` action will automatically trigger. It will:
   - Build the `.tar.gz` (SDist) and `.whl` (Wheel).
   - Publish securely to PyPI using OIDC Trusted Publishing (no API tokens required).
   - Generate a GitHub Release with automated notes.

## 3. PyPI Trusted Publishing Setup
To allow GitHub Actions to publish to PyPI without storing sensitive passwords:
1. Log in to [PyPI](https://pypi.org/manage/account/).
2. Navigate to **Publishers**.
3. Add a new **GitHub publisher**.
4. Set the Owner, Repository name, and workflow filename (`publish.yml`).

## 4. Rollback Procedure
If a critical bug is discovered in production immediately after release:
1. **Yank the Release on PyPI:** Navigate to the release page on PyPI and click "Yank". This prevents new users from downloading it while keeping it available for anyone who pinned it.
2. **Patch Immediately:** Create a new branch, fix the bug, and write a targeted PyTest to prevent recurrence.
3. **Bump Patch Version:** Increment the `PATCH` version (e.g., `v1.0.1` -> `v1.0.2`).
4. **Release:** Push the new tag to trigger the deployment.
