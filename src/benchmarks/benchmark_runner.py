"""
APX-Ready Benchmark Runner
===========================

Designed to produce a genuine CPU workload that Arm Performix (APX) can
profile with its **CPU Cycle Hotspots** recipe on AWS Graviton.

Key design decisions
--------------------
* ``--duration`` runs a continuous inference loop for N seconds, not a fixed
  image count.  This gives APX's PMU sampler enough wall-clock time to
  accumulate statistically meaningful per-function counts.
* ``--warmup`` runs a silent pre-roll before measurements begin, ensuring JIT
  caches and branch predictors are in steady state.
* ``--mode slow|fast|compare`` lets you run each pass independently so APX
  sees a clean, single-purpose profile per run.
* Pre-loads all images into RAM before starting the timed loop so that network
  I/O does not pollute CPU-cycle attribution.

Recommended APX workflow (on Graviton)
---------------------------------------
1. Start APX, connect to the EC2 instance via SSH.
2. Open CPU Cycle Hotspots recipe, set duration to 60 s.
3. In a separate terminal, start the slow-path run:

       python -m benchmarks.benchmark_runner \
           --model models/mobilenet_v1_1.0_224_quant.tflite \
           --mode slow \
           --duration 60 \
           --warmup 5

4. Trigger APX profiling immediately after the warmup message appears.
5. Repeat with ``--mode fast`` for the after profile.

Usage (local smoke test without a model)
-----------------------------------------
    python -m benchmarks.benchmark_runner --mock --mode compare --duration 10
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Literal
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline import ClassificationPipeline, InferenceResult
from utils.image_urls import load_image_urls
from utils.logger import setup_logging

logger = logging.getLogger(__name__)

Mode = Literal["slow", "fast", "compare"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="APX-ready benchmark: sustained TFLite workload for Arm Performix profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="models/mobilenet_v1_1.0_224_quant.tflite",
        help="Path to .tflite model (default: models/mobilenet_v1_1.0_224_quant.tflite)",
    )
    p.add_argument(
        "--mode",
        choices=["slow", "fast", "compare"],
        default="compare",
        help="slow=baseline loops, fast=vectorised NumPy, compare=both (default)",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Seconds to run the inference loop (default: 60). Use >=60 for APX.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Seconds of warmup before measurements begin (default: 5).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="TFLite interpreter threads (default: os.cpu_count())",
    )
    p.add_argument(
        "--urls-file",
        default="data/image_urls.txt",
        help="File with one image URL per line (default: data/image_urls.txt)",
    )
    p.add_argument(
        "--preload",
        type=int,
        default=20,
        help="Images to pre-download into RAM (default: 20). Removes network I/O from profiles.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use a mock TFLite interpreter. No model file required. Local smoke test only.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k predictions per image (default: 5)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image pre-loading
# ---------------------------------------------------------------------------

def preload_images(urls: List[str], count: int) -> List[Image.Image]:
    """Download images into RAM. Cycles URLs if count > len(urls)."""
    from pipeline.downloader import download_image

    images: List[Image.Image] = []
    cycle_urls = (urls * ((count // max(len(urls), 1)) + 1))[:count]

    logger.info("Pre-loading %d images into RAM...", count)
    for i, url in enumerate(cycle_urls, 1):
        try:
            images.append(download_image(url))
            logger.debug("  [%d/%d] %s", i, count, url)
        except Exception as exc:
            logger.warning("  Skipping %s: %s", url, exc)

    if not images:
        raise RuntimeError("No images could be downloaded. Check internet connection.")

    logger.info("Pre-loaded %d image(s)", len(images))
    return images


# ---------------------------------------------------------------------------
# Pipeline extension: bypass downloader for timed loops
# ---------------------------------------------------------------------------

def _patch_run_on_pil(pipeline: ClassificationPipeline) -> None:
    """
    Attach a _run_on_pil(image) method to pipeline that skips the
    download stage. This keeps timed loops pure CPU: preprocess + inference.
    """
    import time as _t
    from pipeline.preprocessor import preprocess_image_fast, preprocess_image_slow

    use_fast = pipeline._use_fast
    clf = pipeline._classifier
    top_k = pipeline._top_k
    labels = pipeline._labels

    def _run_on_pil(image: Image.Image) -> InferenceResult:
        t0 = _t.perf_counter()
        tensor = preprocess_image_fast(image) if use_fast else preprocess_image_slow(image)
        t1 = _t.perf_counter()
        raw = clf.predict(tensor)
        preds = clf.get_top_k(raw, k=top_k, labels=labels)
        t2 = _t.perf_counter()
        return InferenceResult(
            url="preloaded",
            download_ms=0.0,
            preprocess_ms=(t1 - t0) * 1_000,
            inference_ms=(t2 - t1) * 1_000,
            total_ms=(t2 - t0) * 1_000,
            predictions=preds,
        )

    pipeline._run_on_pil = _run_on_pil  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Timed loop
# ---------------------------------------------------------------------------

def run_timed_loop(
    pipeline: ClassificationPipeline,
    images: List[Image.Image],
    duration_s: int,
    warmup_s: int,
    label: str,
) -> List[InferenceResult]:
    """
    Run inference continuously for duration_s seconds after warmup_s seconds
    of silent warmup.  Cycles through images in RAM — no network I/O.
    """
    cycle = images * 1000
    idx = 0

    if warmup_s > 0:
        print(f"\n  [{label}] Warming up for {warmup_s}s  (do NOT start APX yet) ...", flush=True)
        end = time.perf_counter() + warmup_s
        while time.perf_counter() < end:
            pipeline._run_on_pil(cycle[idx % len(cycle)])
            idx += 1
        print(f"  [{label}] Warmup done  >>>  START APX 'CPU Cycle Hotspots' now  <<<", flush=True)

    print(f"  [{label}] Measuring for {duration_s}s ...", flush=True)
    results: List[InferenceResult] = []
    t_start = time.perf_counter()
    end = t_start + duration_s

    while time.perf_counter() < end:
        results.append(pipeline._run_on_pil(cycle[idx % len(cycle)]))
        idx += 1

    elapsed = time.perf_counter() - t_start
    tput = len(results) / elapsed
    print(
        f"  [{label}] Complete — {len(results)} inferences in {elapsed:.1f}s  ({tput:.3f} img/s)",
        flush=True,
    )
    return results


# ---------------------------------------------------------------------------
# Mock interpreter
# ---------------------------------------------------------------------------

def _make_mock_interpreter(*args, **kwargs):
    """CPU-bound mock: uses real NumPy ops so PMU counters register activity."""
    rng = np.random.default_rng(seed=42)
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": 0, "shape": np.array([1, 224, 224, 3])}]
    interp.get_output_details.return_value = [{"index": 1}]

    def _cpu_invoke():
        data = rng.random((224, 224, 3), dtype=np.float32)
        _ = np.einsum("ijk,ijk->", data, data)  # scalar product — forces SIMD

    interp.invoke.side_effect = _cpu_invoke
    interp.get_tensor.return_value = rng.random((1, 1001), dtype=np.float32)
    return interp


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _tput(r: List[InferenceResult]) -> float:
    total_s = sum(x.total_ms for x in r) / 1_000
    return len(r) / total_s if total_s else 0.0


def _pct(stage: str, r: List[InferenceResult]) -> float:
    sm = sum(getattr(x, f"{stage}_ms") for x in r)
    tm = sum(x.total_ms for x in r)
    return 100.0 * sm / tm if tm else 0.0


def _bar(pct: float, w: int = 20) -> str:
    f = round(pct / 100 * w)
    return "█" * f + "░" * (w - f)


def _print_report(
    slow: List[InferenceResult] | None,
    fast: List[InferenceResult] | None,
    duration_s: int,
) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          ARM FT Lite  —  APX-Ready Benchmark Results               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    row = "  {:<36} {:>12}  {:>12}  {:>10}"
    div = "  " + "─" * 74

    if slow and fast:
        st, ft = _tput(slow), _tput(fast)
        sl = statistics.mean(r.total_ms for r in slow)
        fl = statistics.mean(r.total_ms for r in fast)
        sp = statistics.mean(r.preprocess_ms for r in slow)
        fp = statistics.mean(r.preprocess_ms for r in fast)
        si = statistics.mean(r.inference_ms for r in slow)
        fi = statistics.mean(r.inference_ms for r in fast)

        print(row.format("Metric", "Baseline", "Optimised", "Delta"))
        print(div)
        print(row.format("Inferences completed", len(slow), len(fast), ""))
        print(row.format("Throughput (img/s)", f"{st:.3f}", f"{ft:.3f}", f"{100*(ft-st)/st:+.1f}%"))
        print(row.format("Avg latency (ms)", f"{sl:.1f}", f"{fl:.1f}", f"{100*(fl-sl)/sl:+.1f}%"))
        print(row.format("Avg preprocess (ms)", f"{sp:.1f}", f"{fp:.1f}", f"{100*(fp-sp)/sp:+.1f}%"))
        print(row.format("Avg inference (ms)", f"{si:.1f}", f"{fi:.1f}", f"{100*(fi-si)/si:+.1f}%"))
        print()
        print("  APX CPU Hotspot Breakdown")
        print(div)
        print(f"  {'Stage':<26} {'Baseline':>9}  {'bar':<22}  {'Optimised':>9}  bar")
        print(div)
        for stage, lbl in [("preprocess", "preprocess_image"), ("inference", "run_inference (TFLite)")]:
            b, o = _pct(stage, slow), _pct(stage, fast)
            print(f"  {lbl:<26} {b:>8.1f}%  {_bar(b):<22}  {o:>8.1f}%  {_bar(o)}")

    else:
        results = slow or fast
        assert results
        mode_label = "BASELINE (nested loops)" if slow else "OPTIMISED (vectorised NumPy)"
        t = _tput(results)
        lat = statistics.mean(r.total_ms for r in results)
        print(f"  Mode          : {mode_label}")
        print(f"  Duration      : {duration_s}s")
        print(f"  Inferences    : {len(results)}")
        print(f"  Throughput    : {t:.3f} img/s")
        print(f"  Avg latency   : {lat:.1f} ms")
        for stage, lbl in [("preprocess", "preprocess_image"), ("inference", "run_inference (TFLite)")]:
            pct = _pct(stage, results)
            print(f"  {lbl:<22}: {pct:5.1f}%  {_bar(pct)}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(debug=args.verbose)

    model_path = Path(args.model)

    # ── Model validation ──────────────────────────────────────────────────────
    if args.mock:
        logger.warning("MOCK mode: not suitable for APX profiling.")
        mock_patches = [
            patch("pipeline.inference.TFLiteClassifier._load_interpreter",
                  side_effect=_make_mock_interpreter),
            patch.object(Path, "is_file", return_value=True),
        ]
    else:
        if not model_path.is_file():
            logger.error(
                "Model not found: %s\n"
                "  Download: wget https://storage.googleapis.com/download.tensorflow.org"
                "/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz\n"
                "  Then:     tar -xzf mobilenet_v1_1.0_224_quant.tgz && mv *.tflite models/\n"
                "  Or use:   --mock for a local smoke test.",
                model_path,
            )
            sys.exit(1)
        mock_patches = []

    # ── Pre-load images ───────────────────────────────────────────────────────
    urls = load_image_urls(args.urls_file, limit=args.preload)

    if args.mock:
        rng = np.random.default_rng(42)
        images = [
            Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), mode="RGB")
            for _ in range(len(urls))
        ]
        logger.info("Generated %d synthetic images", len(images))
    else:
        images = preload_images(urls, count=args.preload)

    print(
        f"\nARM FT Lite  —  APX-Ready Benchmark\n"
        f"  Model    : {args.model}\n"
        f"  Mode     : {args.mode}\n"
        f"  Duration : {args.duration}s  (+{args.warmup}s warmup)\n"
        f"  Preloaded: {len(images)} images (cycling in RAM, no network I/O)\n"
        f"  Threads  : {args.threads or os.cpu_count()} TFLite threads\n"
        f"  Mock     : {'YES (no real TFLite kernels)' if args.mock else 'NO (real TFLite)'}\n"
    )

    slow_results: List[InferenceResult] | None = None
    fast_results: List[InferenceResult] | None = None

    def _build_and_run(use_fast: bool, label: str) -> List[InferenceResult]:
        pipe = ClassificationPipeline(
            model_path=model_path,
            use_fast_preprocessing=use_fast,
            num_threads=args.threads,
            top_k=args.top_k,
        )
        _patch_run_on_pil(pipe)
        return run_timed_loop(pipe, images, args.duration, args.warmup, label)

    if mock_patches:
        with mock_patches[0], mock_patches[1]:
            if args.mode in ("slow", "compare"):
                slow_results = _build_and_run(False, "BASELINE")
            if args.mode in ("fast", "compare"):
                fast_results = _build_and_run(True, "OPTIMISED")
    else:
        if args.mode in ("slow", "compare"):
            slow_results = _build_and_run(False, "BASELINE")
        if args.mode in ("fast", "compare"):
            fast_results = _build_and_run(True, "OPTIMISED")

    _print_report(slow_results, fast_results, args.duration)

    if not args.mock:
        print(
            "  APX tip: Run --mode slow and --mode fast as separate invocations.\n"
            "           Start 'CPU Cycle Hotspots' recipe AFTER the warmup message.\n"
        )


if __name__ == "__main__":
    main()
