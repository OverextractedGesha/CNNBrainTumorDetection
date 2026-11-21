# MRITumorDetection-CNN

A clinically grounded, two-stage cascade deep learning system for brain MRI screening and tumor subtyping with strict data hygiene, uncertainty calibration, and quantitative interpretability.

---

## 1. Architectural Overview

Brain MRI classification systems deployed in clinical triage must balance high screening sensitivity (avoiding false negatives) with fine-grained diagnostic specificity. `MRITumorDetection-CNN` implements a two-stage cascade architecture designed to eliminate patient-level data leakage, resolve cross-dataset contamination, and provide verifiable predictions.

```
                           [ Input MRI Slice ]
                                    │
                                    ▼
                    [ Preprocessing & Cranial Crop ]
                     (Otsu contour + [0.5, 99.5] %)
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  STAGE 1: BINARY SCREEN │
                       │    (Nickparvar Dataset) │
                       │   Trained for Screening │
                       └────────────┬────────────┘
                                    │
                         P(Tumor) >= \tau_screen (0.35)
                         [Target Sensitivity >= 98%]
                                    │
                     ┌──────────────┴──────────────┐
                     │ NO                          │ YES
                     ▼                             ▼
             [ Healthy / No Tumor ]   ┌─────────────────────────┐
                                      │  STAGE 2: SUBTYPING     │
                                      │    (Figshare Dataset)   │
                                      │ Meningioma / Glioma /   │
                                      │ Pituitary Subtyping     │
                                      └────────────┬────────────┘
                                                   │
                                     [ Temperature Scaling T ]
                                     Confidence = max softmax
                                                   │
                                     Confidence >= \tau_abstain (0.70)?
                                                   │
                                    ┌──────────────┴──────────────┐
                                    │ YES                         │ NO
                                    ▼                             ▼
                           [ Predicted Subtype ]         [ Abstain / Referral ]
                          + Grad-CAM Explanation        "Indeterminate / Low Conf"
```

### Key Components

1. **Two-Stage Clinical Cascade**:
   - **Stage 1 (Binary Screening)**: Distinguishes between healthy brain tissue and tumor presence using the Nickparvar dataset. Calibrated with a decision threshold $\tau \approx 0.35$ to guarantee $\ge 98\%$ sensitivity on prospective validation slices.
   - **Stage 2 (3-Class Subtyping)**: Subtypes detected tumors into Meningioma, Glioma, or Pituitary tumor using the benchmark Jun Cheng Figshare dataset. Out-of-distribution and ambiguous cases are gated by post-hoc temperature scaling and an abstention threshold ($\tau_{\text{abstain}} = 0.70$).

2. **Rigorous Data Hygiene**:
   - **Patient-Level Grouping**: 2D axial slices originating from the same 3D MRI volume share anatomical geometry and scanner noise. Standard random splitting causes acute data leakage and inflated accuracy (>99%). Stage 2 uses `StratifiedGroupKFold` strictly grouped by Patient ID (`PID`).
   - **pHash Cross-Dataset Deduplication**: Kaggle collections frequently aggregate re-encoded, lossy JPG copies of public clinical databases. We construct a 64-bit Discrete Cosine Transform (DCT) Perceptual Hash (`pHash`) with BK-Tree indexing to prune duplicated Figshare slices appearing in Nickparvar, ensuring clean evaluation boundaries.

3. **Model Architectures (Pure CNN)**:
   - **Primary Backbone (`TimmClassifier` — `efficientnet_b0`)**: Lightweight pretrained convolutional backbone (`4.01M` params) with `AdaptiveAvgPool2d`, dropout, and linear classification head used for Stage 1 screening and Stage 2 subtyping.
   - **Backbone & Baseline Comparisons**: Supports `TimmClassifier("resnet50")` (`23.51M` params) for capacity comparison and `SimpleCNN` (`0.39M` params, 4-block Conv-BN-ReLU-MaxPool network) as a fast from-scratch baseline.
   - **5-Fold Stage 2 Ensemble**: `CascadePredictor` supports averaging temperature-scaled softmax probabilities across 5 patient-grouped Stage 2 `TimmClassifier` folds (`stage2_checkpoints` in `configs/cascade.yaml`).

4. **Quantitative Interpretability**:
   - Grad-CAM heatmaps are evaluated against ground-truth tumor masks from Figshare via Pointing Game Hit Rate and Energy-Inside-Mask Ratio.

---

## 2. Repository Layout

```
.
├── configs/                   # Stage 1, Stage 2, and Cascade YAML configs
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── cascade.yaml
├── data/                      # Data manifests, processed splits, and caches (git-ignored)
│   ├── raw/
│   │   ├── figshare/
│   │   └── nickparvar/
│   └── manifests/
├── checkpoints/               # Saved model weights (git-ignored)
├── docs/                      # Clinical guides and milestone documentation
│   ├── milestone1_data_hygiene_and_leakage.md
│   ├── milestone2_models_and_training.md
│   ├── milestone3_evaluation_and_gradcam.md
│   └── milestone4_cascade_and_deployment.md
├── notebooks/                 # Exploratory and interactive tutorial notebooks
│   ├── 01_data_hygiene_and_deduplication.ipynb
│   ├── 02_models_and_training_walkthrough.ipynb
│   ├── 03_evaluation_and_gradcam_walkthrough.ipynb
│   └── 04_cascade_inference_walkthrough.ipynb
├── src/
│   └── mri_tumor/
│       ├── __init__.py
│       ├── app/               # CLI & Gradio interactive application
│       ├── data/              # Datasets, transforms, pHash deduplication, splits
│       ├── evaluation/        # Metrics, calibration, cascade evaluation
│       ├── explain/           # Grad-CAM and mask overlap metrics
│       ├── inference/         # Cascade pipeline and predictor
│       ├── models/            # SimpleCNN & TimmClassifier (efficientnet_b0 / resnet50)
│       └── training/          # Training loops, loss functions, learning rate schedules
├── tests/                     # Unit and integration test suites
├── pyproject.toml
├── requirements.txt
├── run_all.ps1                # Windows one-command setup, training, evaluation, and UI runner
└── run_all.sh                 # macOS/Linux one-command setup, training, evaluation, and UI runner
```

---

## 3. Dataset Setup & Download Tutorial

Raw datasets are not tracked in git. Before running `mri-tumor prepare` or `run_all`, download both public archives and place them in a `Dataset/` folder at the repository root (or pass their paths directly via `--figshare-zip` / `--nickparvar-zip`).

### Dataset 1 — Figshare Brain Tumor Dataset (Jun Cheng)
- **URL**: `https://figshare.com/articles/dataset/brain_tumor_dataset/1512427`
- **Steps**:
  1. Open the Figshare link in your browser and click **Download all** to download the dataset archive (contains **3,064** `.mat` slices across **233** patients packaged in 4 nested `brainTumorDataPublic_*.zip` shards).
  2. Move and rename the downloaded zip file to:
     `Dataset/Figshare Dataset.zip`
  *(Alternatively, keep it anywhere on disk and pass `--figshare-zip <path>` to `mri-tumor prepare`.)*

### Dataset 2 — Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle)
- **URL**: `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`
- **Option A — Browser Download**:
  1. Sign in to Kaggle, open the dataset page, and click **Download** to download the archive.
  2. Move and rename the downloaded zip file to:
     `Dataset/NickParvar Dataset.zip`
- **Option B — Kaggle CLI**:
  ```bash
  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
  ```
  Move/rename the resulting `brain-tumor-mri-dataset.zip` to `Dataset/NickParvar Dataset.zip` (or pass `--nickparvar-zip <path>` to `mri-tumor prepare`).

### Expected `Dataset/` Layout & Preparation

```
Dataset/
├── Figshare Dataset.zip       # Jun Cheng Brain Tumor Dataset (3,064 .mat slices with PIDs and masks)
└── NickParvar Dataset.zip     # Nickparvar 4-class / binary brain MRI dataset
```

Run preparation to extract both archives, run `pHash` cross-dataset deduplication, generate patient-grouped splits, and cache preprocessed slices:

```bash
mri-tumor prepare --config configs/cascade.yaml
```

*(If your archives live outside `Dataset/`, run: `mri-tumor prepare --config configs/cascade.yaml --figshare-zip path/to/figshare.zip --nickparvar-zip path/to/nickparvar.zip`.)*

Extracted files are stored under `data/raw/figshare/` and `data/raw/nickparvar/`, with clean deduplicated splits registered in `data/manifests/` (`stage1_manifest.csv`, `stage2_manifest.csv`, and `cascade_test_manifest.csv`).

---

## 4. Installation & Quick Start on a New PC

Requires **Python >= 3.11**. Clone the repository first:

```bash
git clone https://github.com/OverextractedGesha/CNNBrainTumorDetection.git
cd CNNBrainTumorDetection
```

### Option A: One-Command Auto Setup & Training (Recommended)

After placing the two dataset zips in `Dataset/`, run the automated runner from the repository root:

- **Windows (PowerShell)**:
  ```powershell
  .\run_all.ps1
  ```
  *(If PowerShell blocks local scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or run `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`. Pass `-NoUI` to skip launching Gradio at the end.)*

- **macOS / Linux**:
  ```bash
  bash run_all.sh
  ```
  *(Pass `--no-ui` to skip launching Gradio at the end.)*

What the auto-runner handles automatically:
1. Creates `.venv` (Python 3.11+) and installs `pip install -e ".[dev]"`.
2. Detects NVIDIA GPUs via `nvidia-smi`, installs CUDA PyTorch (`cu130`, falling back to `cu128` / `cu126`) if needed, and selects hardware-adaptive hyperparameters (`efficientnet_b0` + 5-fold Stage 2 on GPU; lightweight `simple_cnn` + Fold 0 on CPU) while syncing `configs/cascade.yaml`.
3. Runs `pytest -p no:cacheprovider -q`, `mri-tumor prepare`, Stage 1 & Stage 2 training (`--skip-if-exists`), `mri-tumor evaluate`, and launches the Gradio web UI at `http://127.0.0.1:7860`.

### Option B: Manual Setup

1. Create and activate a virtual environment, then install the package with dev dependencies:
   ```bash
   python -m venv .venv
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   # macOS / Linux:      source .venv/bin/activate

   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```
2. *(Optional — NVIDIA GPU only)* Install CUDA-enabled PyTorch (e.g., CUDA 13.0):
   ```bash
   pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
3. Verify the test suite:
   ```bash
   pytest -p no:cacheprovider -v
   ```

---

## 5. CLI Usage

The unified entry point `mri-tumor` provides end-to-end orchestration:

### 1. Data Preparation & Deduplication
```bash
mri-tumor prepare --config configs/cascade.yaml
```

### 2. Model Training
```bash
# Train Stage 1 Binary Screening -> checkpoints/stage1/
mri-tumor train --config configs/stage1.yaml

# Train Stage 2 Fold 0 -> checkpoints/stage2_cnn_ablation/
mri-tumor train --config configs/stage2.yaml --arch cnn --fold 0 --checkpoint-dir checkpoints/stage2_cnn_ablation
```

> **Note on 5-Fold Stage 2 Ensemble (`configs/cascade.yaml`)**:  
> `configs/cascade.yaml` defaults `stage2_checkpoints` to all 5 patient-grouped Stage 2 folds (`checkpoints/stage2_cnn_ablation/best_model.pt` for Fold 0, plus `checkpoints/stage2_cnn_fold1..4/best_model.pt` for Folds 1–4):
> ```bash
> mri-tumor train --config configs/stage2.yaml --arch cnn --fold 1 --checkpoint-dir checkpoints/stage2_cnn_fold1
> mri-tumor train --config configs/stage2.yaml --arch cnn --fold 2 --checkpoint-dir checkpoints/stage2_cnn_fold2
> mri-tumor train --config configs/stage2.yaml --arch cnn --fold 3 --checkpoint-dir checkpoints/stage2_cnn_fold3
> mri-tumor train --config configs/stage2.yaml --arch cnn --fold 4 --checkpoint-dir checkpoints/stage2_cnn_fold4
> ```
> If you only trained Fold 0 (`checkpoints/stage2_cnn_ablation`), remove or comment out folds 1–4 under `stage2_checkpoints:` in `configs/cascade.yaml` before running `evaluate`, `predict`, or `ui` (or use `run_all.ps1` / `run_all.sh`, which syncs `configs/cascade.yaml` automatically).

### 3. Evaluation & Cascade Benchmarking
```bash
mri-tumor evaluate --config configs/cascade.yaml
```

### 4. Single-Image Inference & Grad-CAM Output
```bash
mri-tumor predict --config configs/cascade.yaml --image path/to/axial_mri.jpg
```

### 5. Interactive Web Interface
```bash
mri-tumor ui --config configs/cascade.yaml --port 7860
```
