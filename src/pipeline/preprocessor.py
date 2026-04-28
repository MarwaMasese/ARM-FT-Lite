"""
Image preprocessing module.

Provides two implementations of the same preprocessing contract so that
before/after performance can be measured cleanly:

- ``preprocess_image_slow``  – baseline with nested Python loops
- ``preprocess_image_fast``  – optimised with vectorised NumPy operations
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MobileNet v1 expects 224×224 RGB, normalised to [-1, 1]
TARGET_SIZE: Tuple[int, int] = (224, 224)
_NORM_MEAN = 127.5
_NORM_SCALE = 127.5


# ---------------------------------------------------------------------------
# Baseline implementation (slow – nested Python loops)
# ---------------------------------------------------------------------------

def preprocess_image_slow(image: Image.Image) -> np.ndarray:
    """
    Baseline preprocessing with nested Python loops.

    This implementation is intentionally unoptimised so that profiling with
    Arm Performix can highlight it as the dominant hotspot.

    Parameters
    ----------
    image:
        Raw PIL image (any mode, any size).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, 224, 224, 3)`` normalised to ``[-1, 1]``.
    """
    logger.debug("preprocess_image_slow: resizing to %s", TARGET_SIZE)
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    image = image.convert("RGB")

    width, height = TARGET_SIZE
    output = np.empty((height, width, 3), dtype=np.float32)

    # Nested loops – very slow, but intentionally left as the baseline
    pixels = image.load()
    for row in range(height):
        for col in range(width):
            r, g, b = pixels[col, row]
            output[row, col, 0] = _normalize_pixel(r)
            output[row, col, 1] = _normalize_pixel(g)
            output[row, col, 2] = _normalize_pixel(b)

    return output[np.newaxis, ...]  # add batch dimension


def _normalize_pixel(value: int) -> float:
    """Normalise a single uint8 pixel value to ``[-1, 1]``."""
    return (float(value) - _NORM_MEAN) / _NORM_SCALE


# ---------------------------------------------------------------------------
# Optimised implementation (fast – vectorised NumPy)
# ---------------------------------------------------------------------------

def preprocess_image_fast(image: Image.Image) -> np.ndarray:
    """
    Optimised preprocessing using vectorised NumPy operations.

    Replaces the three nested ``for`` loops with a single array expression,
    allowing NumPy to delegate to ARM SIMD instructions (NEON / SVE) under
    the hood. Also avoids repeated RGB conversions and reduces memory
    allocations by operating in-place.

    Parameters
    ----------
    image:
        Raw PIL image (any mode, any size).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, 224, 224, 3)`` normalised to ``[-1, 1]``.
    """
    logger.debug("preprocess_image_fast: resizing to %s", TARGET_SIZE)

    # Convert to RGB exactly once before resize (avoids a second conversion)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(TARGET_SIZE, Image.LANCZOS)

    # Vectorised normalisation – a single NumPy broadcast expression
    img_array = np.asarray(image, dtype=np.float32)  # (224, 224, 3)
    img_array = (img_array - _NORM_MEAN) / _NORM_SCALE

    return img_array[np.newaxis, ...]  # add batch dimension
