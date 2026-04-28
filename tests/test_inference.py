"""
Unit tests for the TFLiteClassifier.

A real model file is not required: we mock the interpreter so that the
classifier logic can be tested on any platform.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_interpreter(output_scores: list[float] | None = None):
    """Return a mock that satisfies the tflite Interpreter contract."""
    if output_scores is None:
        output_scores = [0.1, 0.5, 0.3, 0.05, 0.05]

    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": 0, "shape": np.array([1, 224, 224, 3])}]
    interp.get_output_details.return_value = [{"index": 1}]
    interp.get_tensor.return_value = np.array(output_scores, dtype=np.float32).reshape(1, -1)
    return interp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTFLiteClassifierGetTopK:
    """Test get_top_k without requiring a real model file."""

    def test_top_k_length(self, tmp_path):
        # Create a dummy model file so the path check passes
        model_file = tmp_path / "model.tflite"
        model_file.write_bytes(b"\x00" * 16)

        mock_interp = _make_mock_interpreter([0.1, 0.6, 0.2, 0.05, 0.05])

        with patch(
            "pipeline.inference.TFLiteClassifier._load_interpreter",
            return_value=mock_interp,
        ):
            from pipeline.inference import TFLiteClassifier
            clf = TFLiteClassifier(model_path=model_file, num_threads=1)

        output = np.array([0.1, 0.6, 0.2, 0.05, 0.05], dtype=np.float32)
        results = clf.get_top_k(output, k=3)
        assert len(results) == 3

    def test_top_k_ordering(self, tmp_path):
        model_file = tmp_path / "model.tflite"
        model_file.write_bytes(b"\x00" * 16)

        mock_interp = _make_mock_interpreter()

        with patch(
            "pipeline.inference.TFLiteClassifier._load_interpreter",
            return_value=mock_interp,
        ):
            from pipeline.inference import TFLiteClassifier
            clf = TFLiteClassifier(model_path=model_file, num_threads=1)

        scores = [0.05, 0.60, 0.20, 0.10, 0.05]
        output = np.array(scores, dtype=np.float32)
        results = clf.get_top_k(output, k=2)
        assert results[0]["score"] >= results[1]["score"]

    def test_model_not_found_raises(self):
        from pipeline.inference import TFLiteClassifier
        with pytest.raises(FileNotFoundError):
            TFLiteClassifier(model_path="/nonexistent/model.tflite")
