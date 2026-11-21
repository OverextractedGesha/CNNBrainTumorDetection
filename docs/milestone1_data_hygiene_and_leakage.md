# Milestone 1: Data Hygiene, Patient Leakage Prevention & Perceptual Deduplication

## 1. Introduction: The Clinical Pitfalls of Naive Deep Learning in Medical Imaging

In machine learning research on standard benchmark datasets (e.g., ImageNet, CIFAR-10), samples are generally presumed to be independent and identically distributed ($i.i.d.$). However, applying this naive assumption to 2D medical imaging datasets—specifically Magnetic Resonance Imaging (MRI)—is one of the most pervasive failure modes in clinical AI. 

When applied carelessly, deep learning architectures frequently achieve near-perfect validation accuracy ($>99\%$) on academic benchmark splits, yet fail catastrophically when evaluated on prospective patient cohorts or independent clinical sites. In this milestone, we dissect the three fundamental drivers of data corruption in brain MRI:
1. **Patient-Level Data Leakage** caused by slice-level random splitting.
2. **Cross-Dataset Contamination** caused by aggregation of re-encoded images across public repositories.
3. **Background Artifacts & Acquisition Shortcuts** caused by scanner black borders and non-standard intensity distributions.

---

## 2. Patient-Level Data Leakage in 2D Medical Imaging

### 2.1 The Volumetric Reality of 2D MRI Slices

Magnetic Resonance Imaging acquires volumetric signals ($k$-space frequency representations via Fourier encoding) of 3D biological tissues. Clinicians slice this 3D volume into contiguous 2D planar sections (axial, coronal, sagittal) with slice thicknesses typically ranging between $1\text{ mm}$ and $5\text{ mm}$, and inter-slice gaps often $<1\text{ mm}$.

Consequently, contiguous 2D slices extracted from a single patient's scan are **strongly correlated**:
- **Anatomical Geometry**: Ventricle size, cortical sulcal patterns, skull morphology, and vascular paths vary drastically between distinct human beings, but remain virtually identical between adjacent slices of the same patient.
- **Tumor Morphology & Necrotic Core**: An axial cross-section at slice $z=15$ exhibits almost the same boundary morphology and peritumoral edema as slice $z=16$.
- **Scanner Signature & Hardware Bias**: Scanner-specific magnetic field inhomogeneity ($B_0$ and $B_1$ field non-uniformities), RF coil sensitivity profiles, and gradient non-linearities leave subtle global spatial signatures across all slices from that scan session.

```
Patient A (3D MRI Volume)
├── Slice z=14  ───────────┐
├── Slice z=15 (Train)     │  Shared skull morphology, ventricles,
├── Slice z=16 (Test)  ────┼─ RF coil bias, and tumor texture!
└── Slice z=17             │
                           ▼
            Model memorizes Patient A's anatomy
                 "Shortcut Learning"
```

### 2.2 Mechanism of Shortcut Learning & Accuracy Inflation

When researchers flatten volumetric datasets into a collection of independent 2D image files and apply standard random train/test splits (e.g., `train_test_split(test_size=0.2)`):
- Slice $z=15$ from Patient A is placed into the **training set**.
- Slice $z=16$ from Patient A is placed into the **test set**.

A high-capacity convolutional neural network (e.g., EfficientNet, ResNet, DenseNet) easily minimizes training loss by exploiting high-frequency texture cues, unique ventricular contours, or specific noise patterns unique to Patient A. Rather than learning invariant pathological features of a *glioblastoma* or *meningioma*, the network acts as a nearest-neighbor memorizer of patient identity.

| Splitting Strategy | Test Accuracy | Generalization to New Patients | Clinical Reliability |
| :--- | :--- | :--- | :--- |
| **Random Slice Split** | **$99.2\% - 99.8\%$** | Fails ($<75\%$) | **Invalid** (Severe data leakage) |
| **Patient-Grouped Split** | **$86.0\% - 91.5\%$** | Real ($86.0\% - 91.5\%$) | **Valid** (Clinical-grade) |

### 2.3 Methodological Solution: `StratifiedGroupKFold`

To prevent patient identity leakage, the patient identifier (`PID`) must act as a hard constraint: **all slices from a single patient must reside exclusively in either the training set, the validation set, or the holdout test set**.

Furthermore, because class distributions across patients are often imbalanced (e.g., more glioma patients than pituitary tumor patients), we utilize `StratifiedGroupKFold`:
1. **Grouping constraint**: Every group $g \in \{\text{PID}_1, \text{PID}_2, \dots, \text{PID}_N\}$ is assigned to exactly one fold $k$.
2. **Stratification objective**: The marginal distribution of tumor subtypes across slices (or patient-level diagnoses) in each fold matches the dataset-wide class prior:
   $$D_{\text{KL}}\left(P_k(Y) \,||\, P_{\text{total}}(Y)\right) \approx 0 \quad \forall k \in \{1, \dots, K\}$$

---

## 3. Cross-Dataset Contamination & Perceptual Hashing (`pHash`)

### 3.1 Composite Benchmark Contamination

In the open-source machine learning community, secondary composite datasets on platforms like Kaggle are frequently assembled by merging pre-existing clinical archives. A prominent example is the Nickparvar Brain Tumor dataset:
- Slices labeled "glioma", "meningioma", and "pituitary" in Nickparvar were largely gathered from Jun Cheng's 2017 Figshare dataset (3,064 slices across 233 patients).
- Slices labeled "no_tumor" were gathered from alternative normal brain collections (e.g., Br35H or healthy volunteer scans).

If a practitioner trains a Stage 1 screening model on Nickparvar and evaluates Stage 2 subtyping on Figshare without deduplication:
- Slices in Figshare will have identical counterparts in Nickparvar.
- Evaluating cross-dataset performance or cascade calibration yields artificially optimistic performance because the models have already seen test slices under different file formats.

### 3.2 Failure of Cryptographic Hashing (MD5, SHA-256)

Cryptographic hash algorithms ($H: \{0, 1\}^* \to \{0, 1\}^b$) are deliberately engineered with high **avalanche effect**: altering a single bit in the input radically scrambles the output hash:
- When a MATLAB `.mat` matrix from Figshare is converted into a 24-bit JPEG or PNG for Nickparvar, discrete cosine quantization, 8-bit rounding, and compression artifacts change pixel intensity values by $\pm 1$ or $\pm 2$.
- Resizing from $512 \times 512$ to $224 \times 224$ completely alters pixel coordinate arrays.
- Consequently:
  $$\text{MD5}(I_{\text{figshare}}) \ne \text{MD5}(I_{\text{nickparvar}})$$
Cryptographic hashes are completely blind to perceptual visual equivalence.

### 3.3 The Mathematical Formulation of Perceptual Hashing (`pHash`)

Perceptual Hashing maps visually similar images to binary codes with small **Hamming distance**. We implement an 8-byte (64-bit) frequency-domain perceptual hash based on the 2D Discrete Cosine Transform (DCT):

```
[ Raw Image ] ──► [ Grayscale Resize 32x32 ] ──► [ 2D DCT Transform ] ──► [ Extract 8x8 Low-Freq ]
                                                                                   │
[ 64-bit Hash ] ◄── [ Binary Threshold ] ◄── [ Exclude DC (0,0) & Compute Median ] ◄┘
```

#### Step 1: Spatial Resampling & Grayscale Reduction
The image $I(x, y)$ is downsampled to a fixed low-resolution grid $N \times N = 32 \times 32$ using bilinear interpolation, eliminating high-frequency noise and high-resolution variations:
$$I_{32}(x, y) = \text{BilinearResize}(I, 32, 32)$$

#### Step 2: 2D Discrete Cosine Transform (DCT-II)
The 2D DCT expresses the image as a sum of sinusoidal basis functions varying at different spatial frequencies:
$$D(u, v) = \alpha(u) \alpha(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} I_{32}(x, y) \cos\left[ \frac{\pi (2x+1)u}{2N} \right] \cos\left[ \frac{\pi (2y+1)v}{2N} \right]$$
where:
$$\alpha(u) = \begin{cases} \sqrt{\frac{1}{N}}, & u = 0 \\ \sqrt{\frac{2}{N}}, & u > 0 \end{cases}$$

#### Step 3: Low-Frequency Extraction & DC Isolation
Human visual perception and broad anatomical structures reside in low spatial frequencies. We extract the top-left $M \times M = 8 \times 8$ submatrix:
$$\mathbf{D}_{\text{sub}} = \{ D(u, v) \mid 0 \le u < 8, \; 0 \le v < 8 \}$$
The component $D(0, 0)$ is the **DC coefficient**, representing the global average image brightness. To ensure invariance against overall brightness and contrast shifts, $D(0, 0)$ is excluded from median calculation.

#### Step 4: Median Binarization
We compute the scalar median of the remaining 63 AC frequency coefficients:
$$\mu = \text{median}\left(\{ D(u, v) \mid 0 \le u < 8, 0 \le v < 8, (u, v) \ne (0, 0) \}\right)$$
Each bit $b_{i} \in \{0, 1\}$ for index $i = 8u + v \in [0, 63]$ is determined by thresholding:
$$b_i = \begin{cases} 1, & \text{if } D(u, v) > \mu \\ 0, & \text{otherwise} \end{cases}$$

The 64 bits are packed into an unsigned 64-bit integer (`uint64`).

#### Step 5: Fast Near-Neighbor Search with BK-Tree
Given two perceptual hashes $h_1, h_2 \in \{0, 1\}^{64}$, their perceptual distance is the **Hamming distance**:
$$d_H(h_1, h_2) = \sum_{i=0}^{63} (h_1^{(i)} \oplus h_2^{(i)}) = \text{popcount}(h_1 \oplus h_2)$$

Two images are declared near-duplicates if $d_H(h_1, h_2) \le 6$ (out of 64 bits, representing $>90\%$ frequency congruence). To search for collisions across thousands of images in sub-linear time $O(\log N)$, we index the hashes in a **Burkhard-Keller Tree (BK-Tree)** satisfying the triangle inequality:
$$d_H(A, C) \le d_H(A, B) + d_H(B, C)$$

---

## 4. Cranial Contour Cropping & Intensity Normalization

### 4.1 Scanner Background Shortcuts & Field-of-View (FOV) Discrepancies

MRI scanners from different vendors (Siemens, GE, Philips) and hospital acquisition protocols format images with variable padding:
- Some datasets feature extensive black borders surrounding the skull.
- Others have tight crops, patient positioning markers, text annotations, or asymmetric head orientations.

If class A (e.g., healthy) originates from Hospital 1 with $20\%$ black border margins, and class B (e.g., tumor) originates from Hospital 2 with $5\%$ margins, a deep CNN will rapidly learn to measure the ratio of dark background pixels rather than identifying neuro-pathology.

### 4.2 Automated Cranial Isolation via Otsu Thresholding

To enforce geometric invariance and eliminate background bias, we crop the brain to its tightest anatomical bounding box:

```
[ Raw Slice ] ──► [ Gaussian Blur (5x5) ] ──► [ Otsu Threshold ] ──► [ Morph Close / Open ]
                                                                             │
[ Cropped Slice ] ◄── [ Add 5% Margin ] ◄── [ Bounding Box (x,y,w,h) ] ◄── [ Largest Contour ]
```

1. **Gaussian Smoothing**: Filter high-frequency noise spikes with a $5 \times 5$ Gaussian kernel ($\sigma = 1.0$).
2. **Otsu Binarization**: Automatically determine the optimal intensity threshold $T^*$ separating the cranium from background air by maximizing inter-class variance:
   $$\sigma_B^2(T) = \omega_0(T) \omega_1(T) \left[ \mu_0(T) - \mu_1(T) \right]^2$$
3. **Morphological Filtering**: Apply morphological closing ($\text{kernel} = 5 \times 5$) to bridge minor skull discontinuities, followed by opening to sever connection to stray artifacts.
4. **Largest External Contour Selection**: Extract all external contours $\mathcal{C} = \{C_1, C_2, \dots\}$. The cranial silhouette corresponds to the contour maximizing enclosed area:
   $$C^* = \arg\max_{C \in \mathcal{C}} \text{Area}(C)$$
5. **Bounding Box Extraction & Margin Padding**: Compute $(x_{\min}, y_{\min}, w, h)$. Expand each side by a configurable safety margin (default $5\%$) to guarantee peripheral cortical structures are preserved:
   $$x_0 = \max(0, x_{\min} - 0.05w), \quad y_0 = \max(0, y_{\min} - 0.05h)$$
   $$x_1 = \min(W, x_{\max} + 0.05w), \quad y_1 = \min(H, y_{\max} + 0.05h)$$

### 4.3 Robust Intensity Normalization: Percentile Clipping

Unlike Computed Tomography (CT), where pixel intensities correspond to calibrated physical units (Hounsfield Units, where Air $=-1000\text{ HU}$, Water $=0\text{ HU}$, Bone $=+1000\text{ HU}$), **raw MRI signals are not standardized**. Intensity scales depend on receiver coil gain, tissue relaxation times ($T_1, T_2, T_2^*$, FLAIR), magnetic field strength ($1.5\text{T}$ vs $3.0\text{T}$), and subject anatomy.

Furthermore, MRI images are subject to **Rician noise**—high-frequency magnetic resonance reconstruction spikes where a handful of pixels have extreme intensity values ($>4000$).

```
                Min-Max Scaling (Flawed)
   0 ──────────────────────────────●────────────── 4095 (Noise spike)
   All real brain tissue compressed into [0.0, 0.15]!

                Percentile Clipping (Robust)
   [ p0.5 ] ════════ Real Tissue Dynamic Range ════════ [ p99.5 ]
   Scaled cleanly to [0.0, 1.0] across all scans.
```

Standard min-max normalization:
$$I_{\text{bad}}(x, y) = \frac{I(x, y) - I_{\min}}{I_{\max} - I_{\min}}$$
compresses $99.9\%$ of meaningful brain tissue into a tiny fraction $[0.0, 0.15]$ due to a single noise spike.

We enforce **Robust Percentile Normalization**:
1. Calculate the $0.5^{\text{th}}$ and $99.5^{\text{th}}$ percentiles over non-zero tissue pixels:
   $$p_{0.5} = \text{Percentile}(I_{\text{brain}}, 0.5), \quad p_{99.5} = \text{Percentile}(I_{\text{brain}}, 99.5)$$
2. Clip outlier intensities:
   $$I_{\text{clipped}}(x, y) = \text{clip}(I(x, y), p_{0.5}, p_{99.5})$$
3. Rescale linear contrast into $[0, 1]$:
   $$I_{\text{norm}}(x, y) = \frac{I_{\text{clipped}}(x, y) - p_{0.5}}{p_{99.5} - p_{0.5} + \epsilon}$$

---

## 5. Summary Checklist for Clinical Rigor

Before launching Stage 1 or Stage 2 training in subsequent milestones, verify:
- [x] All 2D slices extracted from Jun Cheng Figshare retain their patient identifier (`PID`).
- [x] Cross-validation splits are created using `StratifiedGroupKFold(groups=PID)`.
- [x] 64-bit DCT perceptual hash (`pHash`) index is built across both datasets.
- [x] Cross-dataset duplicate pairs with Hamming distance $d_H \le 6$ are pruned from training manifests.
- [x] Cranial cropping with Otsu thresholding eliminates variable scanner background borders.
- [x] $[0.5, 99.5]$ percentile clipping ensures normalized contrast across vendors.
