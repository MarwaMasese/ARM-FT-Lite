"""
Unit tests for the preprocessor module.

These tests run on any platform (including x86 dev machines) and do NOT
require tflite_runtime or a real model file.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pipeline.preprocessor import (
    TARGET_SIZE,
    preprocess_image_fast,
    preprocess_image_slow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_rgb_image() -> Image.Image:
    """Return a solid-colour 320×240 RGB image."""
    return Image.new("RGB", (320, 240), color=(128, 64, 32))


@pytest.fixture()
def sample_rgba_image() -> Image.Image:
    """Return a 512×512 RGBA image (mode conversion must be handled)."""
    return Image.new("RGBA", (512, 512), color=(200, 100, 50, 255))


# ---------------------------------------------------------------------------
# Shape & dtype tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_slow_shape(self, sample_rgb_image):
        out = preprocess_image_slow(sample_rgb_image)
        assert out.shape == (1, *TARGET_SIZE, 3)

    def test_fast_shape(self, sample_rgb_image):
        out = preprocess_image_fast(sample_rgb_image)
        assert out.shape == (1, *TARGET_SIZE, 3)

    def test_slow_dtype(self, sample_rgb_image):
        out = preprocess_image_slow(sample_rgb_image)
        assert out.dtype == np.float32

    def test_fast_dtype(self, sample_rgb_image):
        out = preprocess_image_fast(sample_rgb_image)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Normalisation range tests
# ---------------------------------------------------------------------------

class TestNormalisationRange:
    def test_slow_range(self, sample_rgb_image):
        out = preprocess_image_slow(sample_rgb_image)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_fast_range(self, sample_rgb_image):
        out = preprocess_image_fast(sample_rgb_image)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_black_pixel_normalises_to_minus_one(self):
        black = Image.new("RGB", (224, 224), color=(0, 0, 0))
        out = preprocess_image_fast(black)
        np.testing.assert_allclose(out, -1.0, atol=1e-5)

    def test_white_pixel_normalises_to_plus_one(self):
        white = Image.new("RGB", (224, 224), color=(255, 255, 255))
        out = preprocess_image_fast(white)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Equivalence between slow and fast
# ---------------------------------------------------------------------------

class TestEquivalence:
    def test_slow_and_fast_agree(self, sample_rgb_image):
        slow_out = preprocess_image_slow(sample_rgb_image)
        fast_out = preprocess_image_fast(sample_rgb_image)
        np.testing.assert_allclose(slow_out, fast_out, atol=1e-4)

    def test_rgba_mode_conversion(self, sample_rgba_image):
        """Both implementations should handle non-RGB input without error."""
        slow_out = preprocess_image_slow(sample_rgba_image)
        fast_out = preprocess_image_fast(sample_rgba_image)
        assert slow_out.shape == (1, *TARGET_SIZE, 3)
        assert fast_out.shape == (1, *TARGET_SIZE, 3)
        np.testing.assert_allclose(slow_out, fast_out, atol=1e-4)
