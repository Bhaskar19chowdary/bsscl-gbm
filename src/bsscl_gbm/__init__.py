"""
bsscl-gbm: A lightweight, Pure-Python Gradient Boosting Machine with JIT compilation.
"""

from .estimator import HybridHistGBMNumbaV2
from .gpu_engine import BSSCL_GBM_GPU

__version__ = "1.0.1"
__all__ = ["HybridHistGBMNumbaV2", "BSSCL_GBM_GPU"]
