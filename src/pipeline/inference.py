"""
TFLite inference module.

Wraps tflite-runtime's ``Interpreter`` so that the rest of the pipeline does
not need to import tflite directly.  Falls back gracefully when
``tflite_runtime`` is not installed (useful for unit-testing on x86 dev
machines).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class TFLiteClassifier:
    """
    Thin wrapper around a TFLite INT8 / float32 image-classification model.

    Parameters
    ----------
    model_path:
        Path to the ``.tflite`` model file.
    num_threads:
        Number of CPU threads to hand to the interpreter (default: ``os.cpu_count()``).
    """

    def __init__(self, model_path: str | Path, num_threads: int | None = None) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        self._num_threads = num_threads or os.cpu_count() or 2

        self._interpreter = self._load_interpreter()
        self._interpreter.allocate_tensors()

        self._input_details: List[Dict] = self._interpreter.get_input_details()
        self._output_details: List[Dict] = self._interpreter.get_output_details()

        logger.info(
            "Loaded model %s  threads=%d  input_shape=%s",
            self._model_path.name,
            self._num_threads,
            self._input_details[0]["shape"].tolist(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on a pre-processed image tensor.

        Parameters
        ----------
        input_tensor:
            Float32 array of shape ``(1, H, W, C)`` matching the model's
            expected input.  For MobileNet v1 that is ``(1, 224, 224, 3)``
            normalised to ``[-1, 1]``.

        Returns
        -------
        np.ndarray
            Raw logits / probabilities from the model output tensor.
        """
        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_tensor
        )
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        return output

    def get_top_k(
        self, output: np.ndarray, k: int = 5, labels: List[str] | None = None
    ) -> List[Dict]:
        """
        Return the top-*k* predictions from a raw output tensor.

        Parameters
        ----------
        output:
            Raw output from :py:meth:`predict`.
        k:
            Number of top results to return.
        labels:
            Optional list of class-name strings indexed by class id.

        Returns
        -------
        list of dict
            Each entry has keys ``class_id``, ``score``, and optionally
            ``label``.
        """
        scores = output.flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            entry: Dict = {"class_id": int(idx), "score": float(scores[idx])}
            if labels and idx < len(labels):
                entry["label"] = labels[idx]
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_interpreter(self):  # type: ignore[return]
        """
        Try to import ``tflite_runtime`` first (production ARM build),
        then fall back to ``tensorflow.lite`` for development on x86.
        """
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            logger.debug("Using tflite_runtime backend")
            return Interpreter(
                model_path=str(self._model_path),
                num_threads=self._num_threads,
            )
        except ImportError:
            pass

        try:
            import tensorflow as tf  # type: ignore

            logger.debug("Using tensorflow.lite backend (dev fallback)")
            return tf.lite.Interpreter(
                model_path=str(self._model_path),
                num_threads=self._num_threads,
            )
        except ImportError as exc:
            raise ImportError(
                "Neither tflite_runtime nor tensorflow is installed. "
                "Install tflite-runtime on ARM or tensorflow on x86."
            ) from exc
