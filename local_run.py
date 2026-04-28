"""
local_run.py — end-to-end smoke test for the ARM FT Lite pipeline.

Designed to run on any platform (Windows x86, macOS, Linux) without a real
TFLite model file.  It exercises every stage of the pipeline using:
  • Real HTTP image downloads from public Wikimedia URLs
  • Real preprocessing (both slow and fast paths)
  • A mock TFLite interpreter that returns random scores

This gives you full wall-clock timings, the APX-style hotspot table, and
proof that all pipeline wiring is correct — before you deploy to Graviton.

Usage
-----
    python local_run.py              # 5 images (quick)
    python local_run.py --images 20  # more images
    python local_run.py --no-download # skip real downloads, use synthetic data
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# ── Make src/ importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.pipeline import ClassificationPipeline, InferenceResult
from utils.logger import setup_logging

logger = logging.getLogger(__name__)

# ── Public-domain test images (Wikimedia Commons) ─────────────────────────────
TEST_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/320px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/320px-Collage_of_Nine_Dogs.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/320px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/320px-Image_created_with_a_mobile_phone.png",
]


# ── Mock TFLite interpreter ───────────────────────────────────────────────────

def _make_mock_interpreter(*args, **kwargs):
    """
    Returns a mock that satisfies the tflite Interpreter contract.
    Simulates ~20ms inference latency (realistic for MobileNet v1 on CPU).
    """
    rng = np.random.default_rng(seed=42)

    interp = MagicMock()
    interp.get_input_details.return_value = [
        {"index": 0, "shape": np.array([1, 224, 224, 3])}
    ]
    interp.get_output_details.return_value = [{"index": 1}]

    def _fake_invoke():
        # Simulate ~20 ms inference time
        time.sleep(0.020)

    interp.invoke.side_effect = _fake_invoke
    interp.get_tensor.return_value = rng.random((1, 1000), dtype=np.float32)
    return interp


# ── Synthetic image (no network) ─────────────────────────────────────────────

def _make_synthetic_urls(n: int) -> List[str]:
    """Return dummy 'file://' URLs — download_image will be monkeypatched."""
    return [f"synthetic://image_{i}.jpg" for i in range(n)]


def _synthetic_download(url: str) -> Image.Image:
    """Return a random RGB image instead of making an HTTP request."""
    rng = np.random.default_rng(seed=hash(url) % 2**31)
    arr = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _throughput(results: List[InferenceResult]) -> float:
    total_s = sum(r.total_ms for r in results) / 1_000
    return len(results) / total_s if total_s > 0 else 0.0


def _pct(stage: str, results: List[InferenceResult]) -> float:
    stage_ms = [getattr(r, f"{stage}_ms") for r in results]
    total_ms = [r.total_ms for r in results]
    return 100.0 * sum(stage_ms) / sum(total_ms) if sum(total_ms) > 0 else 0.0


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _print_report(
    slow: List[InferenceResult],
    fast: List[InferenceResult],
) -> None:
    slow_tput = _throughput(slow)
    fast_tput = _throughput(fast)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    ARM FT Lite  —  Local Smoke Test Results                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    row = "  {:<34} {:>12}  {:>12}  {:>10}"
    div = "  " + "─" * 72

    print(row.format("Metric", "Baseline", "Optimised", "Δ"))
    print(div)

    def fmt(label, bv, ov, delta):
        print(row.format(label, bv, ov, delta))

    lat_slow = statistics.mean(r.total_ms for r in slow)
    lat_fast = statistics.mean(r.total_ms for r in fast)

    fmt(
        "Throughput (img/s)",
        f"{slow_tput:.3f}",
        f"{fast_tput:.3f}",
        f"{100*(fast_tput-slow_tput)/slow_tput:+.1f}%",
    )
    fmt(
        "Avg total latency (ms)",
        f"{lat_slow:.1f}",
        f"{lat_fast:.1f}",
        f"{100*(lat_fast-lat_slow)/lat_slow:+.1f}%",
    )
    fmt(
        "Avg preprocess (ms)",
        f"{statistics.mean(r.preprocess_ms for r in slow):.1f}",
        f"{statistics.mean(r.preprocess_ms for r in fast):.1f}",
        "",
    )
    fmt(
        "Avg inference (ms)",
        f"{statistics.mean(r.inference_ms for r in slow):.1f}",
        f"{statistics.mean(r.inference_ms for r in fast):.1f}",
        "",
    )
    fmt(
        "Avg download (ms)",
        f"{statistics.mean(r.download_ms for r in slow):.1f}",
        f"{statistics.mean(r.download_ms for r in fast):.1f}",
        "",
    )
    print()

    print("APX-Style CPU Hotspot Breakdown")
    print("─" * 72)
    print(f"  {'Stage':<26} {'Baseline':>9}  {'bar':<22}  {'Optimised':>9}  bar")
    print("─" * 72)
    for stage, label in [
        ("download", "download_image"),
        ("preprocess", "preprocess_image"),
        ("inference", "run_inference (TFLite)"),
    ]:
        b = _pct(stage, slow)
        o = _pct(stage, fast)
        print(f"  {label:<26} {b:>8.1f}%  {_bar(b):<22}  {o:>8.1f}%  {_bar(o)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARM FT Lite — local smoke test")
    p.add_argument("--images", type=int, default=5, help="Number of images (default: 5)")
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip real HTTP downloads; use synthetic random images",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(debug=args.verbose)

    # ── Build URL list ────────────────────────────────────────────────────────
    if args.no_download:
        urls = _make_synthetic_urls(args.images)
    else:
        # Repeat built-in URLs to reach requested count
        urls = []
        while len(urls) < args.images:
            urls.extend(TEST_URLS)
        urls = urls[: args.images]

    print(f"\nARM FT Lite — Local Smoke Test")
    print(f"Mode    : {'synthetic (no network)' if args.no_download else 'real HTTP downloads'}")
    print(f"Images  : {len(urls)}")
    print(f"Backend : mock TFLite interpreter (~20 ms/inference)")
    print()

    # ── Patch TFLite interpreter + optionally downloader ─────────────────────
    dummy_model = Path("models/mobilenet_v1_1.0_224_quant.tflite")

    def _fake_get(url, **kw):
        from io import BytesIO

        img = _synthetic_download(url)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = MagicMock()
        resp.content = buf.read()
        resp.raise_for_status = MagicMock()
        return resp

    common_patches = [
        patch(
            "pipeline.inference.TFLiteClassifier._load_interpreter",
            side_effect=_make_mock_interpreter,
        ),
        patch.object(Path, "is_file", return_value=True),
    ]
    if args.no_download:
        common_patches.append(
            patch("pipeline.downloader.requests.get", side_effect=_fake_get)
        )

    with common_patches[0], common_patches[1]:
        if args.no_download:
            with common_patches[2]:
                slow_pipe = ClassificationPipeline(
                    model_path=dummy_model,
                    use_fast_preprocessing=False,
                    num_threads=os.cpu_count(),
                )
                print(f"[1/2] Running BASELINE (nested Python loops) on {len(urls)} image(s)...")
                t0 = time.perf_counter()
                slow_results = slow_pipe.run_batch(urls)
                print(f"      Done in {time.perf_counter()-t0:.2f}s\n")

            with common_patches[2]:
                fast_pipe = ClassificationPipeline(
                    model_path=dummy_model,
                    use_fast_preprocessing=True,
                    num_threads=os.cpu_count(),
                )
                print(f"[2/2] Running OPTIMISED (vectorised NumPy) on {len(urls)} image(s)...")
                t0 = time.perf_counter()
                fast_results = fast_pipe.run_batch(urls)
                print(f"      Done in {time.perf_counter()-t0:.2f}s")
        else:
            slow_pipe = ClassificationPipeline(
                model_path=dummy_model,
                use_fast_preprocessing=False,
                num_threads=os.cpu_count(),
            )
            print(f"[1/2] Running BASELINE (nested Python loops) on {len(urls)} image(s)...")
            t0 = time.perf_counter()
            slow_results = slow_pipe.run_batch(urls)
            print(f"      Done in {time.perf_counter()-t0:.2f}s\n")

            fast_pipe = ClassificationPipeline(
                model_path=dummy_model,
                use_fast_preprocessing=True,
                num_threads=os.cpu_count(),
            )
            print(f"[2/2] Running OPTIMISED (vectorised NumPy) on {len(urls)} image(s)...")
            t0 = time.perf_counter()
            fast_results = fast_pipe.run_batch(urls)
            print(f"      Done in {time.perf_counter()-t0:.2f}s")

    _print_report(slow_results, fast_results)

    # ── Per-image breakdown ───────────────────────────────────────────────────
    print("Per-image breakdown (optimised path):")
    print(f"  {'#':<4} {'download':>10}  {'preprocess':>12}  {'inference':>10}  {'total':>8}  {'img/s':>6}")
    print("  " + "─" * 58)
    for i, r in enumerate(fast_results, 1):
        print(
            f"  {i:<4} {r.download_ms:>9.1f}ms  {r.preprocess_ms:>11.1f}ms"
            f"  {r.inference_ms:>9.1f}ms  {r.total_ms:>7.1f}ms  {r.throughput:>5.2f}"
        )
    print()
    print("All pipeline stages exercised successfully.")
    print("Ready to deploy to AWS Graviton.")


if __name__ == "__main__":
    main()
