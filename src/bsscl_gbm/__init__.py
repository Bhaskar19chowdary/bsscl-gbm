"""
bsscl-gbm: A lightweight, Pure-Python Gradient Boosting Machine with JIT compilation.
"""

from .estimator_v1_0_0 import HybridHistGBMNumbaV2
from .estimator_v1_0_1 import HybridHistGBMNumbaV1_0_1

__version__ = "1.0.1"
__all__ = ["HybridHistGBMNumbaV1_0_1", "HybridHistGBMNumbaV2"]
