<#
.SYNOPSIS
    Precompute sidecars for new DEMON fixtures and upload them to HuggingFace.

.DESCRIPTION
    Monday-side counterpart to the Mac prep work. Assumes:
      - You're on a Windows machine with a CUDA GPU.
      - DEMON is cloned and `uv sync` has been run.
      - The branch with new entries in KNOWN_FIXTURES is checked out.
      - The raw WAVs have already been uploaded to daydreamlive/demon-fixtures.
      - huggingface-cli is logged in.

    Pipeline per run:
      1. git pull (so the registry has the new fixture names)
      2. Run precompute_fixture_sidecars.py with --only <name> for each fixture.
         This auto-downloads each WAV from HF, then writes
         <name>.sidecar.json + <name>.sidecar.safetensors to
         out\fixture_sidecars\.
      3. Upload each generated sidecar pair back to the HF dataset.
      4. Print a summary.

.PARAMETER Fixtures
    The fixture WAV filenames to process (must match KNOWN_FIXTURES entries).

.EXAMPLE
    .\scripts\calibration\prepare_new_fixtures.ps1 -Fixtures @(
        "funk_loop_60s_anm.wav",
        "jazz_loop_60s_dnm.wav"
    )

.NOTES
    Run from the DEMON repo root.
    If a precompute step fails, the script stops; rerun after fixing
    the issue (the precompute is idempotent - it'll skip what already exists).
#>
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Fixtures
)

$ErrorActionPreference = "Stop"

function Write-Step($message) {
    Write-Host ""
    Write-Host $message -ForegroundColor Cyan
}

# --- preflight ---------------------------------------------------------
Write-Step "DEMON fixture sidecar pipeline"
Write-Host "Fixtures to process: $($Fixtures.Count)"
foreach ($f in $Fixtures) { Write-Host "  - $f" }

if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "acestep")) {
    Write-Host "ERROR: run this from the DEMON repo root (where pyproject.toml lives)." -ForegroundColor Red
    exit 1
}

# huggingface-cli auth check
try {
    $hfUser = (& huggingface-cli whoami 2>&1) -join " "
    if ($LASTEXITCODE -ne 0) { throw "not logged in" }
    Write-Host "HF user: $hfUser"
} catch {
    Write-Host "ERROR: huggingface-cli not logged in." -ForegroundColor Red
    Write-Host "  Run: huggingface-cli login" -ForegroundColor Red
    exit 1
}

# --- step 1: pull -------------------------------------------------------
Write-Step "[1/3] git pull (so KNOWN_FIXTURES has the new entries)"
& git pull
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git pull failed." -ForegroundColor Red
    exit 1
}

# --- step 2: precompute -------------------------------------------------
Write-Step "[2/3] Precomputing sidecars (model load is one-time, ~30s)"
$onlyArgs = @()
foreach ($f in $Fixtures) {
    $onlyArgs += "--only"
    $onlyArgs += $f
}
& uv run python -m scripts.calibration.precompute_fixture_sidecars @onlyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: precompute failed. Sidecars NOT uploaded." -ForegroundColor Red
    exit 1
}

# --- step 3: upload sidecars -------------------------------------------
Write-Step "[3/3] Uploading sidecars to daydreamlive/demon-fixtures"
$sidecarDir = Join-Path (Get-Location) "out\fixture_sidecars"
$failed = @()

foreach ($f in $Fixtures) {
    $jsonPath = Join-Path $sidecarDir "$f.sidecar.json"
    $sfPath = Join-Path $sidecarDir "$f.sidecar.safetensors"

    if (-not (Test-Path $jsonPath)) {
        Write-Host "  MISSING $f.sidecar.json at $jsonPath" -ForegroundColor Red
        $failed += $f
        continue
    }
    if (-not (Test-Path $sfPath)) {
        Write-Host "  MISSING $f.sidecar.safetensors at $sfPath" -ForegroundColor Red
        $failed += $f
        continue
    }

    Write-Host "  $f.sidecar.json"
    & huggingface-cli upload daydreamlive/demon-fixtures $jsonPath "$f.sidecar.json" --repo-type dataset
    if ($LASTEXITCODE -ne 0) { $failed += $f; continue }

    Write-Host "  $f.sidecar.safetensors"
    & huggingface-cli upload daydreamlive/demon-fixtures $sfPath "$f.sidecar.safetensors" --repo-type dataset
    if ($LASTEXITCODE -ne 0) { $failed += $f; continue }

    Write-Host "  ok $f" -ForegroundColor Green
}

# --- summary -----------------------------------------------------------
Write-Step "Summary"
if ($failed.Count -eq 0) {
    Write-Host "Done. $($Fixtures.Count) fixture(s) processed and uploaded." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Verify on HF: https://huggingface.co/datasets/daydreamlive/demon-fixtures"
    Write-Host "  2. Open a PR for the KNOWN_FIXTURES change in acestep/fixtures.py"
    exit 0
} else {
    Write-Host "Partial: $($failed.Count) failure(s):" -ForegroundColor Yellow
    foreach ($f in $failed) { Write-Host "  - $f" }
    Write-Host ""
    Write-Host "Rerun the script with the failed names once the issue is fixed."
    exit 1
}
