# ARM FT Lite — TFLite Image Classification on AWS Graviton

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
  - [4. Run the benchmark](#4-run-the-benchmark)
  - [5. Run the test suite](#5-run-the-test-suite)
- [Configuration](#configuration)
- [Benchmark Results](#benchmark-results)
- [The Optimisation](#the-optimisation)
  - [Baseline — nested Python loops](#baseline--nested-python-loops)
  - [Optimised — vectorised NumPy](#optimised--vectorised-numpy)
- [Arm Performix Profiling](#arm-performix-profiling)
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
git clone https://github.com/<your-org>/arm-ft-lite.git
cd arm-ft-lite
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

### 4. Run the benchmark

```bash
# Compare slow baseline vs. fast optimised on 100 images
python -m benchmarks.benchmark_runner \
    --model models/mobilenet_v1_1.0_224_quant.tflite \
    --images 100 \
    --top-k 5

# Optional flags
#   --threads 2     override TFLite thread count (default: cpu_count)
#   --urls-file data/image_urls.txt
#   --verbose       enable DEBUG logging
```

Expected output (on Graviton t4g.small):

```
╔══════════════════════════════════════════════════════════════════════╗
║          ARM FT Lite  —  Benchmark Results                          ║
╚══════════════════════════════════════════════════════════════════════╝

  Metric                                    Baseline     Optimised     Δ
  ──────────────────────────────────────────────────────────────────────
  Throughput (img/s)                           2.210         3.110  +40.7%
  Avg latency (ms)                             452.0         321.0  -29.0%
  Total time (100 images, s)                   45.20         32.10  -29.0%

APX-Style CPU Hotspot Breakdown
──────────────────────────────────────────────────────────────────────
  Function                               Baseline   Optimised
──────────────────────────────────────────────────────────────────────
  download_image                            8.7%  ██░░░░░░░░░░░░░░░░░░
                                           10.1%  ██░░░░░░░░░░░░░░░░░░  (optimised)

  preprocess_image                         65.3%  █████████████░░░░░░░
                                           28.3%  █████░░░░░░░░░░░░░░░  (optimised)

  run_inference (TFLite)                   22.4%  ████░░░░░░░░░░░░░░░░
                                           58.2%  ███████████░░░░░░░░░  (optimised)
```

### 5. Run the test suite

```bash
pytest
```

All tests are platform-independent (no TFLite model required).

---

## Configuration

All benchmark options are exposed as CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `models/mobilenet_v1_1.0_224_quant.tflite` | Path to `.tflite` model |
| `--images` | `100` | Number of images to process |
| `--top-k` | `5` | Top-k predictions per image |
| `--threads` | `cpu_count` | TFLite interpreter threads |
| `--urls-file` | `data/image_urls.txt` | Image URL list |
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

Arm Performix (APX) was the key tool that made this optimisation possible:

1. **Setup** — APX connected to the EC2 t4g instance via SSH in ~3 minutes,
   with zero code instrumentation.
2. **CPU Cycle Hotspots recipe** — a 60-second profiling window on a
   100-image workload provided function-level CPU time breakdown.
3. **Finding** — `preprocess_image_slow` consumed 65.3% of CPU time;
   `normalize_pixels` alone was 52.1%.  TFLite inference (our initial
   suspect) was only 22.4%.
4. **Validation** — re-profiling after the fix confirmed preprocessing
   dropped from 65% → 28%, and inference rose from 22% → 58%.

> Without ARM-specific profiling we would have spent time tuning TFLite
> thread counts and quantisation settings — the wrong code entirely.

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
