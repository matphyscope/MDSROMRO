"""NumPy version-compatibility shims.

``np.trapezoid`` is the NumPy 2.0+ name; older 1.x ships it as ``np.trapz``.
Import ``trapezoid`` from here so the package runs on both.
"""
from __future__ import annotations
import numpy as np

# np.trapezoid (NumPy >= 2.0) vs np.trapz (NumPy 1.x)
trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:
    trapezoid = np.trapz
