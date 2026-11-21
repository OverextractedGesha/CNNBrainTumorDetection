<#
.SYNOPSIS
    One-command setup, hardware-adaptive training, evaluation, and UI launcher for MRITumorDetection-CNN.

.PARAMETER NoUI
    Skip launching the Gradio web UI after evaluation completes.
#>
param(
    [switch]$NoUI
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = "1"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$CmdArgs = @(),
        [string]$FailureMessage = ""
    )
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        if ($FailureMessage) {
            throw "$FailureMessage (exit code $LASTEXITCODE)"
        }
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($CmdArgs -join ' ')"
    }
}

function Find-Python311 {
    $candidates = @(
        @{ Exe = "python"; Prefix = @() },
        @{ Exe = "py"; Prefix = @("-3.11") },
        @{ Exe = "py"; Prefix = @("-3") }
    )

    $prevPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        foreach ($cand in $candidates) {
            $cmd = Get-Command $cand.Exe -ErrorAction SilentlyContinue
            if (-not $cmd) {
                continue
            }
            $probeArgs = $cand.Prefix + @("-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}'); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)")
            $out = & $cand.Exe @probeArgs 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                return @{
                    Exe     = $cand.Exe
                    Prefix  = $cand.Prefix
                    Version = ($out | Select-Object -Last 1).ToString().Trim()
                }
            }
        }
    }
    finally {
        $ErrorActionPreference = $prevPref
    }

    throw "Python 3.11+ is required but was not found on PATH (tried 'python', 'py -3.11', 'py -3'). Install Python 3.11+ and retry."
}

function Test-TorchCuda {
    param([string]$PythonExe)
    $prevPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $res = & $PythonExe -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
        if ($LASTEXITCODE -eq 0 -and ($res | Select-Object -Last 1).ToString().Trim() -eq "1") {
            return $true
        }
    }
    finally {
        $ErrorActionPreference = $prevPref
    }
    return $false
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " MRITumorDetection-CNN: Automatic Setup & Pipeline Runner" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------
# Step 1: Python 3.11+ & virtual environment
# ------------------------------------------------------------------
Write-Host "`n[Step 1/3] Checking Python 3.11+ and virtual environment..." -ForegroundColor Yellow
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"
$VenvCli    = Join-Path $PSScriptRoot ".venv\Scripts\mri-tumor.exe"

if (-not (Test-Path $VenvPython)) {
    $pyInfo = Find-Python311
    Write-Host "Found Python $($pyInfo.Version) via '$($pyInfo.Exe) $($pyInfo.Prefix -join ' ')'. Creating .venv..."
    Invoke-Checked -Exe $pyInfo.Exe -CmdArgs ($pyInfo.Prefix + @("-m", "venv", ".venv")) -FailureMessage "Failed to create virtual environment .venv"
} else {
    $verOk = $false
    $prevPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $venvVer = & $VenvPython -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}'); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $verOk = $true
            Write-Host "Using existing .venv (Python $(($venvVer | Select-Object -Last 1).ToString().Trim()))."
        }
    }
    finally {
        $ErrorActionPreference = $prevPref
    }
    if (-not $verOk) {
        throw "Existing .venv uses Python < 3.11. Delete .venv and re-run run_all.ps1."
    }
}

Write-Host "Upgrading pip and installing package + dev dependencies..."
Invoke-Checked -Exe $VenvPython -CmdArgs @("-m", "pip", "install", "--upgrade", "pip") -FailureMessage "Failed to upgrade pip"
Invoke-Checked -Exe $VenvPython -CmdArgs @("-m", "pip", "install", "-e", ".[dev]") -FailureMessage "Failed to install mri-tumor package"

# ------------------------------------------------------------------
# Step 2: Auto-detect NVIDIA GPU & CUDA + Adaptive Training Config
# ------------------------------------------------------------------
Write-Host "`n[Step 2/3] Detecting hardware (GPU / VRAM / System RAM)..." -ForegroundColor Yellow

$GpuDetected = $false
$GpuName = "None"
$GpuVramMb = 0

$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $prevPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $smiLines = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $smiLines) {
            foreach ($line in $smiLines) {
                $parts = $line.ToString().Split(",")
                if ($parts.Count -ge 2) {
                    $parsedMem = 0
                    if ([int]::TryParse($parts[1].Trim(), [ref]$parsedMem)) {
                        if ($parsedMem -ge $GpuVramMb) {
                            $GpuDetected = $true
                            $GpuName = $parts[0].Trim()
                            $GpuVramMb = $parsedMem
                        }
                    }
                }
            }
        }
    }
    finally {
        $ErrorActionPreference = $prevPref
    }
}

if ($GpuDetected -and -not (Test-TorchCuda -PythonExe $VenvPython)) {
    Write-Host "NVIDIA GPU detected ($GpuName, $GpuVramMb MB VRAM) but PyTorch CUDA is unavailable. Installing CUDA PyTorch..."
    $cudaInstalled = $false
    foreach ($tag in @("cu130", "cu128", "cu126")) {
        $indexUrl = "https://download.pytorch.org/whl/$tag"
        Write-Host "Trying PyTorch CUDA wheel index: $indexUrl ..."
        $prevPref = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $VenvPython -m pip install --force-reinstall torch torchvision --index-url $indexUrl
            if ($LASTEXITCODE -eq 0) {
                $cudaInstalled = $true
                break
            }
        }
        finally {
            $ErrorActionPreference = $prevPref
        }
        Write-Warning "Install from $tag failed; trying fallback..."
    }
    if (-not $cudaInstalled) {
        Write-Warning "All CUDA PyTorch wheel installs failed; continuing to check device availability."
    }
}

$CudaAvailable = Test-TorchCuda -PythonExe $VenvPython
if ($CudaAvailable -and $GpuVramMb -eq 0) {
    $prevPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $torchProps = & $VenvPython -c "import torch; p = torch.cuda.get_device_properties(0); print(f'{p.name}|{p.total_memory // (1024*1024)}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $torchProps) {
            $tp = ($torchProps | Select-Object -Last 1).ToString().Split("|")
            if ($tp.Count -ge 2) {
                $GpuName = $tp[0].Trim()
                $GpuVramMb = [int]$tp[1].Trim()
            }
        }
    }
    finally {
        $ErrorActionPreference = $prevPref
    }
}

$SystemRamGb = [math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 1)

if ($CudaAvailable -and $GpuVramMb -ge 5500) {
    $ProfileMode = "GPU_HIGH"
    $BatchSize = 32
    $NumWorkers = if ($SystemRamGb -ge 12) { 2 } else { 0 }
    $EpochsS1 = 15
    $EpochsS2 = 20
    $Arch = "cnn"
    $TrainAllFolds = $true
}
elseif ($CudaAvailable) {
    $ProfileMode = "GPU_LOW"
    $BatchSize = 16
    $NumWorkers = 0
    $EpochsS1 = 15
    $EpochsS2 = 20
    $Arch = "cnn"
    $TrainAllFolds = $true
}
else {
    $ProfileMode = "CPU"
    $BatchSize = 16
    $NumWorkers = 0
    $EpochsS1 = 5
    $EpochsS2 = 5
    $Arch = "simple_cnn"
    $TrainAllFolds = $false
}

Write-Host "Hardware Summary: CUDA=$CudaAvailable | GPU=$GpuName ($GpuVramMb MB VRAM) | System RAM=$SystemRamGb GB"
Write-Host "Selected Profile: $ProfileMode | arch=$Arch | batch_size=$BatchSize | num_workers=$NumWorkers | epochs=(S1:$EpochsS1, S2:$EpochsS2) | all_folds=$TrainAllFolds" -ForegroundColor Green

# Sync configs/cascade.yaml stage2_checkpoints cleanly with the selected fold count
if ($TrainAllFolds) {
    $syncCascadePy = @'
import yaml
from pathlib import Path

p = Path("configs/cascade.yaml")
data = yaml.safe_load(p.read_text(encoding="utf-8"))
data["stage2_checkpoints"] = [
    "checkpoints/stage2_cnn_ablation/best_model.pt",
    "checkpoints/stage2_cnn_fold1/best_model.pt",
    "checkpoints/stage2_cnn_fold2/best_model.pt",
    "checkpoints/stage2_cnn_fold3/best_model.pt",
    "checkpoints/stage2_cnn_fold4/best_model.pt",
]
p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
'@
} else {
    $syncCascadePy = @'
import yaml
from pathlib import Path

p = Path("configs/cascade.yaml")
data = yaml.safe_load(p.read_text(encoding="utf-8"))
data["stage2_checkpoints"] = [
    "checkpoints/stage2_cnn_ablation/best_model.pt",
]
p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
'@
}
Invoke-Checked -Exe $VenvPython -CmdArgs @("-c", $syncCascadePy) -FailureMessage "Failed to update configs/cascade.yaml"

# ------------------------------------------------------------------
# Step 3: Verify, Prepare, Train, Evaluate, UI
# ------------------------------------------------------------------
Write-Host "`n[Step 3/3] Running tests, dataset prep, training, and evaluation..." -ForegroundColor Yellow

Write-Host "Running test suite..."
Invoke-Checked -Exe $VenvPytest -CmdArgs @("-p", "no:cacheprovider", "-q") -FailureMessage "Test suite failed"

$CascadeManifest = Join-Path $PSScriptRoot "data\manifests\cascade_test_manifest.csv"
if (-not (Test-Path $CascadeManifest)) {
    Write-Host "Preparing datasets and leak-free splits..."
    Invoke-Checked -Exe $VenvCli -CmdArgs @("prepare", "--config", "configs/cascade.yaml") -FailureMessage "Dataset preparation failed"
} else {
    Write-Host "Found existing $CascadeManifest; skipping dataset preparation."
}

Write-Host "Training Stage 1 (arch=$Arch, epochs=$EpochsS1)..."
Invoke-Checked -Exe $VenvCli -CmdArgs @(
    "train",
    "--config", "configs/stage1.yaml",
    "--arch", $Arch,
    "--epochs", "$EpochsS1",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--skip-if-exists"
) -FailureMessage "Stage 1 training failed"

Write-Host "Training Stage 2 Fold 0 -> checkpoints/stage2_cnn_ablation..."
Invoke-Checked -Exe $VenvCli -CmdArgs @(
    "train",
    "--config", "configs/stage2.yaml",
    "--arch", $Arch,
    "--fold", "0",
    "--epochs", "$EpochsS2",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--checkpoint-dir", "checkpoints/stage2_cnn_ablation",
    "--skip-if-exists"
) -FailureMessage "Stage 2 Fold 0 training failed"

if ($TrainAllFolds) {
    foreach ($foldIdx in 1..4) {
        $foldDir = "checkpoints/stage2_cnn_fold$foldIdx"
        Write-Host "Training Stage 2 Fold $foldIdx -> $foldDir..."
        Invoke-Checked -Exe $VenvCli -CmdArgs @(
            "train",
            "--config", "configs/stage2.yaml",
            "--arch", $Arch,
            "--fold", "$foldIdx",
            "--epochs", "$EpochsS2",
            "--batch-size", "$BatchSize",
            "--num-workers", "$NumWorkers",
            "--checkpoint-dir", $foldDir,
            "--skip-if-exists"
        ) -FailureMessage "Stage 2 Fold $foldIdx training failed"
    }
}

Write-Host "Evaluating cascade pipeline..."
Invoke-Checked -Exe $VenvCli -CmdArgs @("evaluate", "--config", "configs/cascade.yaml") -FailureMessage "Cascade evaluation failed"

if (-not $NoUI) {
    Write-Host "`nLaunching Gradio Web UI at http://127.0.0.1:7860 ..." -ForegroundColor Cyan
    Invoke-Checked -Exe $VenvCli -CmdArgs @("ui", "--config", "configs/cascade.yaml") -FailureMessage "Gradio UI exited with an error"
} else {
    Write-Host "`nPipeline complete (-NoUI specified; skipping Web UI launch)." -ForegroundColor Green
}
