"""
End-to-end inference pipeline.

Wires together the downloader, preprocessor, and TFLite classifier into a
single callable that can be exercised in benchmarks and production code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .downloader import download_image
from .inference import TFLiteClassifier
from .preprocessor import preprocess_image_fast, preprocess_image_slow

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Holds per-image timings and classification output."""

    url: str
    download_ms: float
    preprocess_ms: float
    inference_ms: float
    total_ms: float
    predictions: List[dict] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Images per second based on total wall-clock time."""
        return 1_000.0 / self.total_ms if self.total_ms > 0 else 0.0


class ClassificationPipeline:
    """
    High-level pipeline that:

    1. Downloads an image from a URL.
    2. Preprocesses it (slow baseline **or** fast vectorised path).
    3. Runs TFLite inference.
    4. Returns timings and top-k predictions.

    Parameters
    ----------
    model_path:
        Path to the ``.tflite`` model file.
    use_fast_preprocessing:
        ``True`` (default) uses the vectorised NumPy preprocessor;
        ``False`` uses the nested-loop baseline for benchmarking.
    num_threads:
        TFLite interpreter thread count.
    top_k:
        Number of top predictions to return per image.
    labels:
        Optional list of class label strings.
    """

    def __init__(
        self,
        model_path: str | Path,
        use_fast_preprocessing: bool = True,
        num_threads: Optional[int] = None,
        top_k: int = 5,
        labels: Optional[List[str]] = None,
    ) -> None:
        self._classifier = TFLiteClassifier(
            model_path=model_path, num_threads=num_threads
        )
        self._use_fast = use_fast_preprocessing
        self._top_k = top_k
        self._labels = labels

        preprocess_name = "fast (vectorised)" if use_fast_preprocessing else "slow (loops)"
        logger.info("Pipeline ready  preprocessing=%s", preprocess_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, url: str) -> InferenceResult:
        """
        Execute the complete pipeline for a single image URL.

        Parameters
        ----------
        url:
            Publicly accessible image URL.

        Returns
        -------
        InferenceResult
            Timing breakdown and top-k predictions.
        """
        t0 = time.perf_counter()

        # Stage 1 – download
        image = download_image(url)
        t1 = time.perf_counter()

        # Stage 2 – preprocess
        if self._use_fast:
            tensor = preprocess_image_fast(image)
        else:
            tensor = preprocess_image_slow(image)
        t2 = time.perf_counter()

        # Stage 3 – inference
        raw_output = self._classifier.predict(tensor)
        predictions = self._classifier.get_top_k(
            raw_output, k=self._top_k, labels=self._labels
        )
        t3 = time.perf_counter()

        result = InferenceResult(
            url=url,
            download_ms=(t1 - t0) * 1_000,
            preprocess_ms=(t2 - t1) * 1_000,
            inference_ms=(t3 - t2) * 1_000,
            total_ms=(t3 - t0) * 1_000,
            predictions=predictions,
        )

        logger.debug(
            "url=%s  download=%.1fms  preprocess=%.1fms  inference=%.1fms  total=%.1fms",
            url,
            result.download_ms,
            result.preprocess_ms,
            result.inference_ms,
            result.total_ms,
        )

        return result

    def run_batch(self, urls: List[str]) -> List[InferenceResult]:
        """
        Run the pipeline sequentially over a list of image URLs.

        Parameters
        ----------
        urls:
            List of image URLs to process.

        Returns
        -------
        list of InferenceResult
        """
        results = []
        for i, url in enumerate(urls, start=1):
            logger.info("Processing image %d/%d", i, len(urls))
            results.append(self.run(url))
        return results
