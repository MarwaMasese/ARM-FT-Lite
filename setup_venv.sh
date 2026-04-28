#!/usr/bin/env bash
# setup_venv.sh
# ─────────────────────────────────────────────────────────────────────────────
# Creates and provisions a Python virtual environment for this project.
#
# Usage (on AWS Graviton / ARM64 Ubuntu):
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#
# Usage (on x86 dev machine):
#   ./setup_venv.sh --dev
#
# The --dev flag installs requirements-dev.txt (tensorflow instead of
# tflite-runtime) so the project can be developed without Graviton hardware.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
DEV_MODE=false

# ── Parse flags ──────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --dev) DEV_MODE=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Create venv ───────────────────────────────────────────────────────────────
echo "Creating virtual environment in ${VENV_DIR}/ …"
"$PYTHON" -m venv "$VENV_DIR"

# ── Activate ─────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Upgrade pip & wheel ───────────────────────────────────────────────────────
echo "Upgrading pip …"
pip install --quiet --upgrade pip wheel

# ── Install dependencies ──────────────────────────────────────────────────────
if [ "$DEV_MODE" = true ]; then
  echo "Installing development dependencies (requirements-dev.txt) …"
  pip install --quiet -r requirements-dev.txt
else
  echo "Installing production dependencies (requirements.txt) …"
  pip install --quiet -r requirements.txt
fi

echo ""
echo "✓ Virtual environment ready.  Activate with:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "Then run the benchmark:"
echo "    python -m benchmarks.benchmark_runner --help"
