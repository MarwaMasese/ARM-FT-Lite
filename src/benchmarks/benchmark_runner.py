"""
Benchmark runner.

Compares slow (baseline) vs. fast (optimised) preprocessing on a fixed set of
image URLs.  Reports per-stage timings, aggregate throughput, and a summary
table so results can be pasted directly into reports.

Usage
-----
.. code-block:: bash

    python -m benchmarks.benchmark_runner \\
        --model models/mobilenet_v1_1.0_224_quant.tflite \\
        --images 100 \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is on sys.path when executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline import ClassificationPipeline, InferenceResult
from utils.image_urls import load_image_urls
from utils.logger import setup_logging
from utils.report import print_benchmark_report, print_hotspot_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark slow vs. fast preprocessing on AWS Graviton / ARM"
    )
    parser.add_argument(
        "--model",
        default="models/mobilenet_v1_1.0_224_quant.tflite",
        help="Path to the TFLite model file",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=100,
        help="Number of images to process (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to return per image (default: 5)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="TFLite interpreter threads (default: cpu_count)",
    )
    parser.add_argument(
        "--urls-file",
        default="data/image_urls.txt",
        help="File with one image URL per line (default: data/image_urls.txt)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def run_benchmark(
    pipeline: ClassificationPipeline,
    urls: List[str],
    label: str,
) -> List[InferenceResult]:
    logger.info("Starting benchmark: %s  images=%d", label, len(urls))
    t_start = time.perf_counter()
    results = pipeline.run_batch(urls)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "Finished benchmark: %s  total_wall=%.2fs  throughput=%.3f img/s",
        label,
        elapsed,
        len(results) / elapsed,
    )
    return results


def aggregate(results: List[InferenceResult], stage: str) -> dict:
    """Return mean / stdev / min / max for a timing stage (ms)."""
    values = [getattr(r, f"{stage}_ms") for r in results]
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(debug=args.verbose)

    model_path = Path(args.model)
    if not model_path.is_file():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)

    urls = load_image_urls(args.urls_file, limit=args.images)
    if not urls:
        logger.error("No image URLs loaded from %s", args.urls_file)
        sys.exit(1)

    logger.info("Loaded %d image URLs", len(urls))

    # ---- Slow baseline ----
    slow_pipeline = ClassificationPipeline(
        model_path=model_path,
        use_fast_preprocessing=False,
        num_threads=args.threads,
        top_k=args.top_k,
    )
    slow_results = run_benchmark(slow_pipeline, urls, label="BASELINE (slow loops)")

    # ---- Fast optimised ----
    fast_pipeline = ClassificationPipeline(
        model_path=model_path,
        use_fast_preprocessing=True,
        num_threads=args.threads,
        top_k=args.top_k,
    )
    fast_results = run_benchmark(fast_pipeline, urls, label="OPTIMISED (vectorised)")

    # ---- Report ----
    print_benchmark_report(slow_results, fast_results)
    print_hotspot_table(slow_results, fast_results)


if __name__ == "__main__":
    main()
