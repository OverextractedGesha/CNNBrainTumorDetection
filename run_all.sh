#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONDONTWRITEBYTECODE=1

NO_UI=0
for arg in "$@"; do
    case "$arg" in
        --no-ui)
            NO_UI=1
            ;;
        -h|--help)
            echo "Usage: bash run_all.sh [--no-ui]"
            echo "  --no-ui   Skip launching the Gradio web UI after evaluation."
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$arg'. Use --no-ui or --help." >&2
            exit 1
            ;;
    esac
done

find_python311() {
    local cand
    for cand in python3 python3.12 python3.11 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}

torch_cuda_available() {
    "$VENV_PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1
}

get_system_ram_gb() {
    "$VENV_PYTHON" - <<'PY'
import os, platform, subprocess

ram_bytes = 0
system = platform.system()
if system == "Darwin":
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        ram_bytes = int(out)
    except Exception:
        ram_bytes = 0
elif os.path.exists("/proc/meminfo"):
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    ram_bytes = kb * 1024
                    break
    except Exception:
        ram_bytes = 0

if ram_bytes == 0 and hasattr(os, "sysconf"):
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            ram_bytes = pages * page_size
    except Exception:
        ram_bytes = 0

print(f"{ram_bytes / (1024 ** 3):.1f}")
PY
}

echo "============================================================"
echo " MRITumorDetection-CNN: Automatic Setup & Pipeline Runner"
echo "============================================================"

# ------------------------------------------------------------------
# Step 1: Python 3.11+ & virtual environment
# ------------------------------------------------------------------
echo ""
echo "[Step 1/3] Checking Python 3.11+ and virtual environment..."

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
VENV_PYTEST="$SCRIPT_DIR/.venv/bin/pytest"
VENV_CLI="$SCRIPT_DIR/.venv/bin/mri-tumor"

if [ ! -x "$VENV_PYTHON" ]; then
    SYS_PYTHON="$(find_python311 || true)"
    if [ -z "$SYS_PYTHON" ]; then
        echo "Error: Python 3.11+ is required but was not found on PATH." >&2
        exit 1
    fi
    PY_VER="$("$SYS_PYTHON" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")"
    echo "Found Python $PY_VER via '$SYS_PYTHON'. Creating .venv..."
    "$SYS_PYTHON" -m venv .venv
else
    if ! "$VENV_PYTHON" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
        echo "Error: Existing .venv uses Python < 3.11. Delete .venv and re-run run_all.sh." >&2
        exit 1
    fi
    PY_VER="$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")"
    echo "Using existing .venv (Python $PY_VER)."
fi

echo "Upgrading pip and installing package + dev dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e ".[dev]"

# ------------------------------------------------------------------
# Step 2: Auto-detect NVIDIA GPU & CUDA + Adaptive Training Config
# ------------------------------------------------------------------
echo ""
echo "[Step 2/3] Detecting hardware (GPU / VRAM / System RAM)..."

GPU_DETECTED=0
GPU_NAME="None"
GPU_VRAM_MB=0

if command -v nvidia-smi >/dev/null 2>&1; then
    SMI_OUTPUT="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -n "$SMI_OUTPUT" ]; then
        while IFS=',' read -r name mem; do
            name_trimmed="$(echo "${name:-}" | xargs)"
            mem_trimmed="$(echo "${mem:-}" | xargs)"
            if [[ "$mem_trimmed" =~ ^[0-9]+$ ]] && [ "$mem_trimmed" -ge "$GPU_VRAM_MB" ]; then
                GPU_DETECTED=1
                GPU_NAME="$name_trimmed"
                GPU_VRAM_MB="$mem_trimmed"
            fi
        done <<< "$SMI_OUTPUT"
    fi
fi

if [ "$GPU_DETECTED" -eq 1 ] && ! torch_cuda_available; then
    echo "NVIDIA GPU detected ($GPU_NAME, $GPU_VRAM_MB MB VRAM) but PyTorch CUDA is unavailable. Installing CUDA PyTorch..."
    CUDA_INSTALLED=0
    for tag in cu130 cu128 cu126; do
        INDEX_URL="https://download.pytorch.org/whl/$tag"
        echo "Trying PyTorch CUDA wheel index: $INDEX_URL ..."
        if "$VENV_PYTHON" -m pip install --force-reinstall torch torchvision --index-url "$INDEX_URL"; then
            CUDA_INSTALLED=1
            break
        else
            echo "Warning: Install from $tag failed; trying fallback..." >&2
        fi
    done
    if [ "$CUDA_INSTALLED" -eq 0 ]; then
        echo "Warning: All CUDA PyTorch wheel installs failed; continuing to check device availability." >&2
    fi
fi

CUDA_AVAILABLE=0
if torch_cuda_available; then
    CUDA_AVAILABLE=1
    if [ "$GPU_VRAM_MB" -eq 0 ]; then
        TORCH_PROPS="$("$VENV_PYTHON" -c "import torch; p = torch.cuda.get_device_properties(0); print(f'{p.name}|{p.total_memory // (1024*1024)}')" 2>/dev/null || true)"
        if [ -n "$TORCH_PROPS" ]; then
            GPU_NAME="${TORCH_PROPS%%|*}"
            GPU_VRAM_MB="${TORCH_PROPS##*|}"
        fi
    fi
fi

SYSTEM_RAM_GB="$(get_system_ram_gb)"

if [ "$CUDA_AVAILABLE" -eq 1 ] && [ "$GPU_VRAM_MB" -ge 5500 ]; then
    PROFILE_MODE="GPU_HIGH"
    BATCH_SIZE=32
    NUM_WORKERS="$("$VENV_PYTHON" -c "print(2 if float('${SYSTEM_RAM_GB}') >= 12.0 else 0)")"
    EPOCHS_S1=15
    EPOCHS_S2=20
    ARCH="cnn"
    TRAIN_ALL_FOLDS=1
elif [ "$CUDA_AVAILABLE" -eq 1 ]; then
    PROFILE_MODE="GPU_LOW"
    BATCH_SIZE=16
    NUM_WORKERS=0
    EPOCHS_S1=15
    EPOCHS_S2=20
    ARCH="cnn"
    TRAIN_ALL_FOLDS=1
else
    PROFILE_MODE="CPU"
    BATCH_SIZE=16
    NUM_WORKERS=0
    EPOCHS_S1=5
    EPOCHS_S2=5
    ARCH="simple_cnn"
    TRAIN_ALL_FOLDS=0
fi

echo "Hardware Summary: CUDA=$CUDA_AVAILABLE | GPU=$GPU_NAME ($GPU_VRAM_MB MB VRAM) | System RAM=$SYSTEM_RAM_GB GB"
echo "Selected Profile: $PROFILE_MODE | arch=$ARCH | batch_size=$BATCH_SIZE | num_workers=$NUM_WORKERS | epochs=(S1:$EPOCHS_S1, S2:$EPOCHS_S2) | all_folds=$TRAIN_ALL_FOLDS"

"$VENV_PYTHON" - "$TRAIN_ALL_FOLDS" <<'PY'
import sys
import yaml
from pathlib import Path

all_folds = sys.argv[1] == "1"
p = Path("configs/cascade.yaml")
data = yaml.safe_load(p.read_text(encoding="utf-8"))
if all_folds:
    data["stage2_checkpoints"] = [
        "checkpoints/stage2_cnn_ablation/best_model.pt",
        "checkpoints/stage2_cnn_fold1/best_model.pt",
        "checkpoints/stage2_cnn_fold2/best_model.pt",
        "checkpoints/stage2_cnn_fold3/best_model.pt",
        "checkpoints/stage2_cnn_fold4/best_model.pt",
    ]
else:
    data["stage2_checkpoints"] = [
        "checkpoints/stage2_cnn_ablation/best_model.pt",
    ]
p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

# ------------------------------------------------------------------
# Step 3: Verify, Prepare, Train, Evaluate, UI
# ------------------------------------------------------------------
echo ""
echo "[Step 3/3] Running tests, dataset prep, training, and evaluation..."

echo "Running test suite..."
"$VENV_PYTEST" -p no:cacheprovider -q

CASCADE_MANIFEST="$SCRIPT_DIR/data/manifests/cascade_test_manifest.csv"
if [ ! -f "$CASCADE_MANIFEST" ]; then
    echo "Preparing datasets and leak-free splits..."
    "$VENV_CLI" prepare --config configs/cascade.yaml
else
    echo "Found existing $CASCADE_MANIFEST; skipping dataset preparation."
fi

echo "Training Stage 1 (arch=$ARCH, epochs=$EPOCHS_S1)..."
"$VENV_CLI" train \
    --config configs/stage1.yaml \
    --arch "$ARCH" \
    --epochs "$EPOCHS_S1" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --skip-if-exists

echo "Training Stage 2 Fold 0 -> checkpoints/stage2_cnn_ablation..."
"$VENV_CLI" train \
    --config configs/stage2.yaml \
    --arch "$ARCH" \
    --fold 0 \
    --epochs "$EPOCHS_S2" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --checkpoint-dir checkpoints/stage2_cnn_ablation \
    --skip-if-exists

if [ "$TRAIN_ALL_FOLDS" -eq 1 ]; then
    for fold_idx in 1 2 3 4; do
        fold_dir="checkpoints/stage2_cnn_fold$fold_idx"
        echo "Training Stage 2 Fold $fold_idx -> $fold_dir..."
        "$VENV_CLI" train \
            --config configs/stage2.yaml \
            --arch "$ARCH" \
            --fold "$fold_idx" \
            --epochs "$EPOCHS_S2" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --checkpoint-dir "$fold_dir" \
            --skip-if-exists
    done
fi

echo "Evaluating cascade pipeline..."
"$VENV_CLI" evaluate --config configs/cascade.yaml

if [ "$NO_UI" -eq 0 ]; then
    echo ""
    echo "Launching Gradio Web UI at http://127.0.0.1:7860 ..."
    "$VENV_CLI" ui --config configs/cascade.yaml
else
    echo ""
    echo "Pipeline complete (--no-ui specified; skipping Web UI launch)."
fi
