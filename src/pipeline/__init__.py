"""
Pipeline package.

Exports the public surface of the inference pipeline so callers only need::

    from pipeline import ClassificationPipeline, InferenceResult
"""

from .downloader import download_image
from .inference import TFLiteClassifier
from .pipeline import ClassificationPipeline, InferenceResult
from .preprocessor import preprocess_image_fast, preprocess_image_slow

__all__ = [
    "download_image",
    "TFLiteClassifier",
    "ClassificationPipeline",
    "InferenceResult",
    "preprocess_image_fast",
    "preprocess_image_slow",
]
