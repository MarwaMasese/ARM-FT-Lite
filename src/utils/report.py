"""
Console report utilities.

Formats benchmark results into the same style shown in the use-case write-up.
"""

from __future__ import annotations

import statistics
from typing import List

from pipeline.pipeline import InferenceResult


def _throughput(results: List[InferenceResult]) -> float:
    """Overall throughput: images / second."""
    total_s = sum(r.total_ms for r in results) / 1_000
    return len(results) / total_s if total_s > 0 else 0.0


def _pct(stage_ms: List[float], total_ms: List[float]) -> float:
    return 100.0 * sum(stage_ms) / sum(total_ms) if sum(total_ms) > 0 else 0.0


def print_benchmark_report(
    slow: List[InferenceResult],
    fast: List[InferenceResult],
) -> None:
    """Print a human-readable performance summary table."""

    slow_tput = _throughput(slow)
    fast_tput = _throughput(fast)
    tput_pct = 100.0 * (fast_tput - slow_tput) / slow_tput if slow_tput else 0.0

    slow_lat = statistics.mean(r.total_ms for r in slow)
    fast_lat = statistics.mean(r.total_ms for r in fast)
    lat_pct = 100.0 * (fast_lat - slow_lat) / slow_lat if slow_lat else 0.0

    slow_total_s = sum(r.total_ms for r in slow) / 1_000
    fast_total_s = sum(r.total_ms for r in fast) / 1_000
    total_pct = 100.0 * (fast_total_s - slow_total_s) / slow_total_s if slow_total_s else 0.0

    header = (
        "\n"
        "╔══════════════════════════════════════════════════════════════════════╗\n"
        "║          ARM FT Lite  —  Benchmark Results                          ║\n"
        "╚══════════════════════════════════════════════════════════════════════╝\n"
    )
    print(header)

    col_w = 18
    row_fmt = "  {:<32} {:>{w}}  {:>{w}}  {:>{w}}"
    divider = "  " + "─" * 70

    print(row_fmt.format("Metric", "Baseline", "Optimised", "Δ", w=col_w))
    print(divider)

    def fmt_row(label, base_val, opt_val, delta_str):
        print(row_fmt.format(label, base_val, opt_val, delta_str, w=col_w))

    fmt_row(
        "Throughput (img/s)",
        f"{slow_tput:.3f}",
        f"{fast_tput:.3f}",
        f"{tput_pct:+.1f}%",
    )
    fmt_row(
        "Avg latency (ms)",
        f"{slow_lat:.1f}",
        f"{fast_lat:.1f}",
        f"{lat_pct:+.1f}%",
    )
    fmt_row(
        f"Total time ({len(slow)} images, s)",
        f"{slow_total_s:.2f}",
        f"{fast_total_s:.2f}",
        f"{total_pct:+.1f}%",
    )
    print()


def print_hotspot_table(
    slow: List[InferenceResult],
    fast: List[InferenceResult],
) -> None:
    """Print a CPU-time hotspot breakdown mirroring the Arm Performix output."""

    def pct_stage(results: List[InferenceResult], stage: str) -> float:
        stage_ms = [getattr(r, f"{stage}_ms") for r in results]
        total_ms = [r.total_ms for r in results]
        return _pct(stage_ms, total_ms)

    stages = [
        ("download", "download_image"),
        ("preprocess", "preprocess_image"),
        ("inference", "run_inference (TFLite)"),
    ]

    def bar(pct: float, width: int = 20) -> str:
        filled = round(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    print("APX-Style CPU Hotspot Breakdown")
    print("─" * 70)
    print(f"  {'Function':<38} {'Baseline':>10}  {'Optimised':>10}")
    print("─" * 70)

    for stage, label in stages:
        b = pct_stage(slow, stage)
        o = pct_stage(fast, stage)
        b_bar = bar(b)
        o_bar = bar(o)
        print(f"  {label:<38} {b:>6.1f}%  {b_bar}")
        print(f"  {'':38} {o:>6.1f}%  {o_bar}  (optimised)")
        print()

    print("─" * 70)
    print()
