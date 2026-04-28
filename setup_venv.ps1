# setup_venv.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Windows PowerShell equivalent of setup_venv.sh.
# Run from the project root:
#
#   .\setup_venv.ps1          # production (tflite-runtime)
#   .\setup_venv.ps1 -Dev     # dev mode   (tensorflow)
# ─────────────────────────────────────────────────────────────────────────────

param(
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$VenvDir = ".venv"
$Python  = "python"

Write-Host "Creating virtual environment in $VenvDir/ ..."
& $Python -m venv $VenvDir

# Activate
& "$VenvDir\Scripts\Activate.ps1"

Write-Host "Upgrading pip ..."
pip install --quiet --upgrade pip wheel

if ($Dev) {
    Write-Host "Installing development dependencies (requirements-dev.txt) ..."
    pip install --quiet -r requirements-dev.txt
} else {
    Write-Host "Installing production dependencies (requirements.txt) ..."
    pip install --quiet -r requirements.txt
}

Write-Host ""
Write-Host "Virtual environment ready.  Activate with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run the benchmark:"
Write-Host "    python -m benchmarks.benchmark_runner --help"
