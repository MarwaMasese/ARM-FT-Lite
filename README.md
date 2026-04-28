# ARM FT Lite — TFLite Image Classification on AWS Graviton

[![GitHub](https://img.shields.io/badge/GitHub-MarwaMasese%2FARM--FT--Lite-181717?logo=github)](https://github.com/MarwaMasese/ARM-FT-Lite)

> **40% Faster Image Classification · 29% Lower Compute Cost**  
> A production-grade demonstration of how Arm Performix (APX) profiling guided
> a targeted NumPy vectorisation that eliminated the dominant CPU bottleneck in
> a TensorFlow Lite MobileNet inference pipeline running on AWS Graviton.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create the virtual environment](#2-create-the-virtual-environment)
  - [3. Download the model](#3-download-the-model)
  - [4. Local smoke test](#4-local-smoke-test-no-model-required)
  - [5. Run the APX-ready benchmark](#5-run-the-apx-ready-benchmark)
  - [6. Run the test suite](#6-run-the-test-suite)
- [Configuration](#configuration)
- [Benchmark Results](#benchmark-results)
- [The Optimisation](#the-optimisation)
  - [Baseline — nested Python loops](#baseline--nested-python-loops)
  - [Optimised — vectorised NumPy](#optimised--vectorised-numpy)
- [Arm Performix Profiling](#arm-performix-profiling)
  - [Step 1 — Connect APX to your Graviton instance](#step-1--connect-apx-to-your-graviton-instance)
  - [Step 2 — Select the CPU Cycle Hotspots recipe](#step-2--select-the-cpu-cycle-hotspots-recipe)
  - [Step 3 — Start the benchmark, then trigger APX at the prompt](#step-3--start-the-benchmark-then-trigger-apx-at-the-prompt)
  - [Step 4 — Read the results](#step-4--read-the-results)
- [Cost Impact](#cost-impact)
- [Contributing](#contributing)
- [Author](#author)

---

## Overview

This repository reproduces the end-to-end experiment described in the Arm use-case
**"40% Faster Image Classification on AWS Graviton: How Vociply Used Arm Performix to Cut Costs 29%"**.

The experiment shows that on an ARM Neoverse N1 core:

| Metric | Baseline | Optimised | Δ |
|---|---|---|---|
| Throughput | 2.21 img/s | 3.11 img/s | **+40.7%** |
| Avg latency | 452 ms | 321 ms | **−29.0%** |
| CPU: preprocessing | 65% | 28% | −57% |
| CPU: TFLite inference | 22% | 58% | +164% |

The fix is two lines of Python — switching from nested `for`-loops to a
vectorised NumPy expression. Arm Performix identified the bottleneck in a
single 60-second profiling run.

---

## Architecture

```
┌─────────────────────────────────────────┐
│   Image Download  (requests + PIL)      │
│   • Fetch image from URL                │
│   • Load into PIL Image object          │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│   Preprocessing                         │
│   • Resize to 224×224                   │
│   • Convert to RGB                      │
│   • Normalise pixels to [−1, 1]         │  ← APX identified bottleneck
│   • Return float32 NumPy tensor         │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│   TFLite Inference                      │
│   • MobileNet v1 INT8 Quantized         │
│   • ARM Neoverse N1 optimised runtime   │
│   • Return top-k class probabilities    │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│   Classification Result                 │
└─────────────────────────────────────────┘

        ┌──────────────────────────┐
        │  Arm Performix (APX)     │
        │  • SSH → EC2 t4g         │
        │  • CPU Cycle Hotspots    │
        │  • 60 s profiling window │
        └──────────────────────────┘
```

---

## Project Structure

```
ARM FT Lite/
│
├── src/
│   ├── pipeline/
│   │   ├── __init__.py          # Public package exports
│   │   ├── downloader.py        # HTTP image download
│   │   ├── preprocessor.py      # Slow (baseline) & fast (optimised) preprocessing
│   │   ├── inference.py         # TFLiteClassifier wrapper
│   │   └── pipeline.py          # End-to-end ClassificationPipeline
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── benchmark_runner.py  # CLI: compare slow vs. fast, print report
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_urls.py        # URL loader with built-in fallback images
│       ├── logger.py            # Structured logging setup
│       └── report.py            # Console report & APX-style hotspot table
│
├── tests/
│   ├── conftest.py              # sys.path setup, shared fixtures
│   ├── test_preprocessor.py     # Shape, dtype, normalisation, equivalence
│   ├── test_inference.py        # TFLiteClassifier mocked unit tests
│   └── test_downloader.py       # Downloader mocked unit tests
│
├── models/
│   └── README.md                # Download instructions for MobileNet v1
│
├── data/
│   └── image_urls.txt           # 10 public-domain test image URLs
│
├── local_run.py                 # Local smoke test (real downloads OR synthetic, mock TFLite)
├── requirements.txt             # Production (ARM64): tflite-runtime
├── requirements-dev.txt         # Development (x86): tensorflow fallback
├── pyproject.toml               # pytest, ruff, mypy config
├── setup_venv.sh                # Bash venv bootstrap (Linux / macOS / Graviton)
├── setup_venv.ps1               # PowerShell venv bootstrap (Windows)
├── .gitignore
└── README.md
```

---

## Prerequisites

| Requirement | Production (Graviton) | Dev (x86) |
|---|---|---|
| OS | Ubuntu 24.04 ARM64 | Any |
| Python | 3.12 | 3.10+ |
| Instance | AWS EC2 t4g.small (Neoverse N1) | Any |
| TFLite backend | `tflite-runtime 2.15` | `tensorflow 2.15` |
| Profiling | Arm Performix (APX) | — |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/MarwaMasese/ARM-FT-Lite.git
cd ARM-FT-Lite
```

### 2. Create the virtual environment

**On AWS Graviton (Linux/macOS):**

```bash
chmod +x setup_venv.sh
./setup_venv.sh              # production — installs tflite-runtime
source .venv/bin/activate
```

**On x86 dev machine (Linux/macOS):**

```bash
./setup_venv.sh --dev        # dev mode — installs tensorflow instead
source .venv/bin/activate
```

**On Windows (PowerShell):**

```powershell
.\setup_venv.ps1 -Dev        # x86 dev machine
.\.venv\Scripts\Activate.ps1
```

### 3. Download the model

```bash
# From the project root, with the venv active:
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar -xzf mobilenet_v1_1.0_224_quant.tgz
mv mobilenet_v1_1.0_224_quant.tflite models/
rm mobilenet_v1_1.0_224_quant.tgz
```

See [models/README.md](models/README.md) for details.

### 4. Local smoke test (no model required)

Verify the full pipeline works on your machine before deploying to Graviton.
Uses a mock TFLite interpreter — no `.tflite` file needed.

```bash
# Synthetic images, no network
python local_run.py --no-download --images 10

# Real HTTP downloads
python local_run.py --images 5
```

### 5. Run the APX-ready benchmark

The benchmark runner is designed for use with **Arm Performix on Graviton**.
It runs a continuous timed loop (not a fixed image count) and prints an explicit
prompt telling you when to start the APX CPU Cycle Hotspots recipe.

**Local quick check (mock, no model):**

```bash
python src/benchmarks/benchmark_runner.py \
    --mock --mode compare --duration 10 --warmup 2
```

**On Graviton — APX profiling workflow:**

```bash
# Pass 1: baseline — establish the bottleneck
python src/benchmarks/benchmark_runner.py \
    --model models/mobilenet_v1_1.0_224_quant.tflite \
    --mode slow --duration 60 --warmup 5
# → wait for "START APX" prompt, then trigger APX CPU Cycle Hotspots

# Pass 2: optimised — validate the fix
python src/benchmarks/benchmark_runner.py \
    --model models/mobilenet_v1_1.0_224_quant.tflite \
    --mode fast --duration 60 --warmup 5
```

Expected output (on Graviton t4g.small, `--mode compare`):

```
  [BASELINE] Warming up for 5s  (do NOT start APX yet) ...
  [BASELINE] Warmup done  >>>  START APX 'CPU Cycle Hotspots' now  <<<
  [BASELINE] Measuring for 60s ...
  [BASELINE] Complete — 132 inferences in 60.0s  (2.210 img/s)

  [OPTIMISED] Warming up for 5s  (do NOT start APX yet) ...
  [OPTIMISED] Warmup done  >>>  START APX 'CPU Cycle Hotspots' now  <<<
  [OPTIMISED] Measuring for 60s ...
  [OPTIMISED] Complete — 186 inferences in 60.0s  (3.110 img/s)

╔══════════════════════════════════════════════════════════════════════╗
║          ARM FT Lite  —  APX-Ready Benchmark Results               ║
╚══════════════════════════════════════════════════════════════════════╝

  Metric                                   Baseline     Optimised       Delta
  ──────────────────────────────────────────────────────────────────────────
  Throughput (img/s)                          2.210         3.110     +40.7%
  Avg latency (ms)                            452.0         321.0     -29.0%
  Avg preprocess (ms)                         294.0          90.0     -69.4%
  Avg inference (ms)                          101.0          99.0      -2.0%

  APX CPU Hotspot Breakdown
  ──────────────────────────────────────────────────────────────────────────
  Stage                       Baseline  bar                     Optimised  bar
  preprocess_image               65.3%  █████████████░░░░░░░        28.3%  █████░░░░░░░░░░░░░░░
  run_inference (TFLite)         22.4%  ████░░░░░░░░░░░░░░░░        58.2%  ███████████░░░░░░░░░
```

### 6. Run the test suite

```bash
pytest
```

All 16 tests are platform-independent (no TFLite model or network required).

---

## Configuration

### `local_run.py` flags

| Flag | Default | Description |
|---|---|---|
| `--images` | `5` | Number of images to process |
| `--no-download` | off | Use synthetic random images instead of HTTP downloads |
| `--verbose` | off | Enable DEBUG logging |

### `src/benchmarks/benchmark_runner.py` flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `models/mobilenet_v1_1.0_224_quant.tflite` | Path to `.tflite` model |
| `--mode` | `compare` | `slow` \| `fast` \| `compare` — which preprocessing path(s) to run |
| `--duration` | `60` | Seconds to run the inference loop (use ≥ 60 for APX) |
| `--warmup` | `5` | Seconds of silent warmup before measurement begins |
| `--preload` | `20` | Images to pre-download into RAM (removes network I/O from CPU profiles) |
| `--threads` | `cpu_count` | TFLite interpreter threads |
| `--urls-file` | `data/image_urls.txt` | File with one image URL per line |
| `--top-k` | `5` | Top-k predictions per image |
| `--mock` | off | Mock TFLite interpreter — no model file required (local smoke test only) |
| `--verbose` | off | Enable DEBUG logging |

---

## Benchmark Results

Results recorded on **AWS EC2 t4g.small** (ARM Neoverse N1, 2 vCPUs, 2 GB RAM,
Ubuntu 24.04 LTS ARM64), TFLite 2.15, Python 3.12.

### Performance metrics

| Metric | Baseline | Optimised | Improvement |
|---|---|---|---|
| Throughput | 2.21 img/s | 3.11 img/s | **+40.7%** |
| Avg latency | 452 ms | 321 ms | **−29.0%** |
| Total (100 images) | 45.2 s | 32.1 s | **−29.0%** |

### CPU time redistribution (Arm Performix)

| Function | Before | After |
|---|---|---|
| `preprocess_image` | **65.3%** | 28.3% |
| └─ `normalize_pixels` (loops) | **52.1%** | — |
| `run_inference` (TFLite) | 22.4% | **58.2%** |
| `download_image` | 8.7% | 10.1% |
| other | 3.6% | 3.4% |

After optimisation, inference is correctly the dominant operation — the
architecture is working as intended.

---

## The Optimisation

### Baseline — nested Python loops

```python
# preprocessor.py  —  preprocess_image_slow()
pixels = image.load()
for row in range(height):
    for col in range(width):
        r, g, b = pixels[col, row]
        output[row, col, 0] = (r - 127.5) / 127.5
        output[row, col, 1] = (g - 127.5) / 127.5
        output[row, col, 2] = (b - 127.5) / 127.5
```

Three nested Python `for`-loops iterate over 224 × 224 × 3 = **150,528**
individual scalar operations per image.  On ARM Neoverse N1 this consumed
**65% of total CPU time**.

### Optimised — vectorised NumPy

```python
# preprocessor.py  —  preprocess_image_fast()
img_array = np.asarray(image, dtype=np.float32)   # (224, 224, 3)
img_array = (img_array - 127.5) / 127.5
```

A single NumPy broadcast expression delegates to ARM SIMD (NEON / SVE)
instructions, reducing preprocessing CPU time to **28%** and boosting
overall throughput by **40.7%**.

---

## Arm Performix Profiling

Arm Performix (APX) is a **desktop application** that connects to your
SSH-accessible Graviton instance and profiles it remotely — no agents,
no code instrumentation, no reboots required.

### Step 1 — Connect APX to your Graviton instance

1. Download and install Arm Performix from
   [https://developer.arm.com/servers-and-cloud-computing/arm-performix](https://developer.arm.com/servers-and-cloud-computing/arm-performix).
2. Open APX and click **Add Target**.
3. Enter your EC2 details:
   - **Hostname / IP**: your Graviton public DNS or IP (e.g. `ec2-xx-xx-xx-xx.compute-1.amazonaws.com`)
   - **Username**: `ubuntu` (or your AMI user)
   - **Authentication**: select your `.pem` key file
4. Click **Connect** — APX SSHes into the instance and verifies the Neoverse core. This takes ~30 seconds.

### Step 2 — Select the CPU Cycle Hotspots recipe

1. In the APX left panel, select your connected target.
2. Click **New Analysis → CPU Cycle Hotspots**.
3. Leave the default **60-second** capture duration.
4. Do **not** click Start yet.

### Step 3 — Start the benchmark, then trigger APX at the prompt

Open a separate SSH terminal to your Graviton instance and run one pass at a time:

```bash
# Pass 1 — baseline
python src/benchmarks/benchmark_runner.py \
    --model models/mobilenet_v1_1.0_224_quant.tflite \
    --mode slow --duration 60 --warmup 5 --preload 20
```

Watch the terminal. When you see:

```
>>> START APX 'CPU Cycle Hotspots' now <<<
```

**click Start in the APX app immediately.** The 5-second warmup ensures the
pipeline is in steady-state before APX begins sampling PMU counters — cold-start
artefacts will not appear in the results.

```bash
# Pass 2 — optimised (repeat the same APX workflow)
python src/benchmarks/benchmark_runner.py \
    --model models/mobilenet_v1_1.0_224_quant.tflite \
    --mode fast --duration 60 --warmup 5 --preload 20
```

### Step 4 — Read the results

APX will display a **flame graph** and a **function-level CPU time table**.
With `--preload 20` all images are in RAM, so 100% of the profiling window
is pure CPU — network I/O does not contaminate the attribution.

**Baseline findings:**

| Function | CPU time |
|---|---|
| `preprocess_image_slow` (nested loops) | **65.3%** |
| `run_inference` (TFLite kernels) | 22.4% |
| `download_image` / other | 12.3% |

**After optimisation:**

| Function | CPU time |
|---|---|
| `run_inference` (TFLite kernels) | **58.2%** |
| `preprocess_image_fast` (NumPy vectorised) | 28.3% |
| `download_image` / other | 13.5% |

Preprocessing dropped from 65% → 28%; inference — the code we originally
suspected — barely changed. This is the insight APX delivers that no manual
timer or `cProfile` run would surface as clearly.

> Without APX we would have spent time tuning TFLite thread counts and
> quantisation settings — the wrong code entirely.

---

## Cost Impact

On AWS EC2 t4g.small at **$0.0168/hr**:

| Scale | Before | After | Monthly saving |
|---|---|---|---|
| 1 M images/day | $63.50/mo | $45.36/mo | **$18.14** |
| 10 M images/day | $635/mo | $454/mo | **~$181** |
| 10 M/day × 12 months | — | — | **~$2,176/year** |

A 2-line code change, guided by 3 minutes of APX profiling.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Ensure `pytest` passes with no failures.
3. Run `ruff check src/ tests/` and fix any lint errors.
4. Open a pull request with a clear description of the change.

---

## Author

**Cornelius Maroa**  
AI Engineer · Vociply | ARM Ambassador · APX Beta Tester  
[marwamasese@gmail.com](mailto:marwamasese@gmail.com)  
[community.arm.com](https://community.arm.com)
