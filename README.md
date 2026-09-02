# Pure Self-Supervised SSDU Conditional DDPM for FastMRI Knee Reconstruction

A self-supervised MRI reconstruction pipeline for **fastMRI Knee Multi-Coil** data using **SSDU (Self-Supervised Deep Learning for Undersampled MRI)** and a **conditional DDPM (Denoising Diffusion Probabilistic Model)**.

This implementation is designed so that **fully sampled ground-truth images are not used for the training loss**. Ground truth is reserved for final validation/test evaluation.

---

## Highlights

- Pure self-supervised **SSDU** training
- Conditional **DDPM** reconstruction
- Original **15-coil** fastMRI knee data
- **No coil compression**
- Cartesian **4× acceleration**
- ACS-based sensitivity estimation
- Hard **data consistency (DC)** in k-space
- Normalized **L1 + L2 SSDU loss**
- 1000-step diffusion training distribution
- 100-step respaced DDPM sampling at test time
- Seeded/reproducible validation and test sampling
- Optional learned k-space partitioning
- Optional learned sensitivity-map estimation (SME)
- Current default experiment uses a stable **fixed partition + fixed sensitivity** configuration

---

## Method Overview

The reconstruction pipeline can be summarized as:

```text
Fully sampled multi-coil k-space
            │
            ▼
      4× Cartesian mask
            │
            ▼
   Acquired k-space Ω
            │
            ▼
     SSDU partition
       Ω = Aleph + Upsilon
            │
      ┌─────┴─────┐
      ▼           ▼
   Aleph       Upsilon
      │           │
      │           └──────────────► Self-supervised k-space loss
      ▼
 SENSE adjoint / conditioning
      │
      ▼
 Hard data consistency
      │
      ▼
     x₀
      │
      ▼
  DDPM forward diffusion
      │
      ▼
      xₜ
      │
      ▼
 Conditional Physics U-Net
      │
      ▼
 Predicted x₀
      │
      ▼
 Hard data consistency
      │
      ▼
  SSDU loss on Upsilon
```

At test time, the model starts from Gaussian noise:

```text
x_T ~ N(0, I)
      │
      ▼
Respaced DDPM reverse sampling
      │
      ▼
Predicted x₀ + data consistency
      │
      ▼
Final reconstructed MRI
```

---

## Self-Supervised SSDU Principle

Let the acquired undersampled k-space be:

```text
Ω = Aleph ∪ Upsilon
Aleph ∩ Upsilon = ∅
```

The **Aleph** subset is used to construct the network input/conditioning representation.

The **Upsilon** subset is hidden from the reconstruction network and is used only as the self-supervised target.

Therefore:

```text
Input / conditioning  → Aleph
Loss target            → Upsilon
Fully sampled GT      → NOT used for training
```

This prevents the reconstruction network from directly seeing the held-out k-space samples used to evaluate the SSDU loss.

The implementation also explicitly avoids passing the Upsilon loss mask to the network, preventing a train/test mismatch and potential information leakage.

---

## Current Configuration

| Parameter | Value |
|---|---:|
| Dataset | fastMRI Knee Multi-Coil |
| Image size | 320 × 320 |
| Original coils | 15 |
| Coil compression | **Disabled** |
| Acceleration | 4× |
| Center fraction | 0.08 |
| ACS size | 24 |
| Contrast | `CORPD_FBK` |
| Number of contrasts | 1 |
| SSDU alpha | 0.85 |
| Learned partitioning | **False** |
| Learned sensitivity | **False** |
| Diffusion training steps | 1000 |
| Test sampling steps | 100 |
| Beta start | 1e-4 |
| Beta end | 0.02 |
| Base channels | 32 |
| Batch size | 2 |
| Epochs | 50 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| SSDU L2 weight | 0.5 |
| Optimizer | AdamW |
| LR scheduler | Cosine Annealing |
| Gradient clipping | 5.0 |
| GAN | Disabled |
| Internal DC | Enabled |
| Test start | `x_T ~ N(0,I)` |

The code verifies that all selected files contain the expected 15 coils and reports that coil compression is disabled.

---

## Model Architecture

### 1. Physics U-Net

The main reconstruction network is a diffusion-conditioned U-Net.

It contains:

- sinusoidal diffusion-time embedding
- time MLP
- residual convolution blocks
- GroupNorm
- SiLU activations
- encoder/decoder U-Net structure
- skip connections
- internal hard data consistency

For a single contrast, the network receives:

```text
5 input channels
= x_t + condition + measured-mask information
```

and predicts:

```text
2 output channels
= real + imaginary components of the reconstructed image
```

The self-supervised Upsilon mask is deliberately **not** supplied as a network input.

---

## 2. DDPM

The diffusion process uses:

```text
T = 1000
β_start = 1e-4
β_end   = 0.02
```

with a linear beta schedule.

During training:

```text
x₀ → q(xₜ | x₀) → xₜ
```

The network learns to predict the clean reconstruction `x₀` from the noisy state `xₜ`, conditioned on the measured MRI information.

---

## 3. SSDU Loss

The reconstruction is evaluated against the held-out Upsilon k-space samples.

The implemented SSDU objective is a normalized mixture of:

```text
L = (1 - λ) L1 + λ L2
```

with:

```text
λ = 0.5
```

The loss is computed only on the held-out Upsilon measurements.

---

## 4. Hard Data Consistency

Hard data consistency is applied to the predicted image using the measured k-space.

At test time, the implementation uses the **full acquired Ω** for final data consistency.

Conceptually:

```text
Predicted image
      ↓
SENSE forward operator
      ↓
Replace measured k-space samples
      ↓
Inverse SENSE
      ↓
Data-consistent reconstruction
```

---

## Learned Components

The code contains two optional trainable components.

### Learned k-space Partitioning

```python
USE_LEARNED_PARTITIONING = False
```

When disabled, the code uses the fixed SSDU split with:

```python
SSDU_ALPHA = 0.85
```

When enabled, a trainable per-line partitioning module can learn the probability of assigning acquired lines to Aleph. The implementation includes the hard threshold and straight-through gradient approximation corresponding to the intended learned-partition formulation.

### Learned Sensitivity Map Estimation

```python
USE_LEARNED_SENSITIVITY = False
```

When disabled, classical ACS + RSS sensitivity maps are used.

When enabled, a small trainable U-Net refines the per-coil ACS images before RSS normalization.

These options are intentionally disabled in the current stable baseline so that their contribution can be evaluated through controlled ablation experiments.

---

## Why Coil Compression Is Disabled

The raw fastMRI data used by this experiment contains 15 coils.

The current pipeline intentionally preserves all original channels:

```text
15 original coils
      ↓
NO PCA/SVD compression
      ↓
15-coil reconstruction
```

This avoids losing coil information through dimensionality reduction and makes the experiment directly operate on the original multi-coil measurements.

The trade-off is increased memory and computational cost.

---

## Training Strategy

Training is fully self-supervised.

For every training sample:

1. Load multi-coil k-space.
2. Crop to 320 × 320.
3. Apply the 4× Cartesian acquisition mask.
4. Obtain acquired k-space Ω.
5. Estimate sensitivity maps.
6. Split Ω into Aleph and Upsilon.
7. Build the SENSE-adjoint condition from Aleph.
8. Apply hard data consistency using Aleph.
9. Sample a random diffusion timestep `t`.
10. Add diffusion noise to obtain `x_t`.
11. Run the conditional Physics U-Net.
12. Apply internal data consistency.
13. Compare the prediction against the held-out Upsilon k-space.
14. Backpropagate the SSDU loss.

No fully sampled reconstruction target is required for the training objective.

---

## Validation

Validation is also self-supervised.

The validation checkpoint is selected using the held-out **Upsilon k-space SSDU loss**, rather than PSNR/SSIM against ground truth.

A fixed validation seed is used so that the stochastic components of validation are reproducible.

---

## Testing and Metrics

Ground truth is used **only for final evaluation**.

The implementation reports:

- **PSNR** — higher is better
- **SSIM** — higher is better
- **NMSE** — lower is better

The code also reports the improvement over the SENSE zero-filled baseline.

---

## Quantitative Results

For the reported `CORPD_FBK` test experiment:

| Method | PSNR (dB) | SSIM | NMSE |
|---|---:|---:|---:|
| SENSE Zero-Filled | **31.7438** | 0.8524 | 0.010830 |
| Pure SSDU Conditional DDPM | **35.5488** | **0.9010** | **0.004819** |
| Improvement | **+3.8049 dB** | **+0.0486** | **−0.006011** |

These values are the aggregate test results reported by the experiment.

### Representative qualitative result

The following visualization shows a representative test slice comparing the SENSE zero-filled reconstruction and the SSDU DDPM reconstruction.


<img width="2685" height="763" alt="example_001_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/f64b724c-825b-4b3f-b8a2-2061471ce358" />
<img width="2685" height="763" alt="example_010_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/af23c6b3-da60-4b29-8d22-7f18f0096769" />



The absolute-error map provides a visual indication of the remaining reconstruction error.

---

## Reproducibility

The implementation explicitly seeds:

```python
SPLIT_SEED = 42
VAL_SEED = 123456
TEST_SEED = 20260901
```

The test reverse diffusion process uses a dedicated random generator for both:

- initial Gaussian noise
- intermediate posterior-sampling noise

Therefore, repeated evaluation with the same checkpoint and configuration is designed to produce reproducible reconstructions and metrics.

---

## Data Split

The current experiment uses:

```text
Training files : 25
Validation files: 10
Test files      : 10
```

for the selected `CORPD_FBK` acquisition.

The reported run contains:

```text
Train slices : 771
Validation   : 310
Test         : 323
```

The exact number of usable slices depends on the edge-slice filtering and files available in the selected dataset mirror.

---

## Output Files

The script writes checkpoints to:

```text
/kaggle/working/ssdu_diffusion_checkpoints/
```

and reconstruction visualizations/results to:

```text
/kaggle/working/ssdu_diffusion_results/
```

Typical outputs include:

```text
best_ssdu_ddpm_audited_v4.pt
example_001_CORPD_FBK.png
example_002_CORPD_FBK.png
...
```

---

## Running the Code

### 1. Set the dataset path

Update:

```python
CFG.DATA_ROOT = "/kaggle/input/datasets/arafatshovon/fastmri-knee-multicoil"
```

to match the local dataset location.

### 2. Configure the experiment

The most important switches are:

```python
CFG.CONTRAST_ACQUISITIONS = ["CORPD_FBK"]

CFG.USE_LEARNED_PARTITIONING = False
CFG.USE_LEARNED_SENSITIVITY = False
```

### 3. Run

The script performs:

```text
Data discovery
    ↓
Coil verification
    ↓
Training
    ↓
Self-supervised validation
    ↓
Best-checkpoint loading
    ↓
Test reconstruction
    ↓
PSNR / SSIM / NMSE
    ↓
Qualitative visualization
```

---

## Recommended Ablation Study

For a research paper, the two optional learned components should be evaluated independently.

### A — Stable baseline

```text
Fixed Partition
+
Fixed Sensitivity
```

### B — Learned partition only

```text
Learned Partition
+
Fixed Sensitivity
```

### C — Learned sensitivity only

```text
Fixed Partition
+
Learned Sensitivity
```

### D — Full learned configuration

```text
Learned Partition
+
Learned Sensitivity
```

Compare all four configurations using the same:

- train/validation/test split
- random seeds
- acceleration
- epochs
- optimizer
- learning rate
- sampling steps
- preprocessing
- evaluation protocol

A component should be claimed as beneficial only if it consistently improves the evaluation metrics under a controlled ablation.

---

## Important Research Notes

### No GT training loss

The model is intentionally trained without a fully sampled image-domain target.

```text
Training:
SSDU held-out k-space loss only

Testing:
GT used for PSNR / SSIM / NMSE
```

This distinction is important when describing the method as self-supervised.

### Validation vs. test

Validation selects the checkpoint using self-supervised Upsilon k-space loss.

Test uses ground truth only to calculate final reconstruction metrics.

### Single-contrast default

The current configuration uses:

```python
["CORPD_FBK"]
```

so it is a true single-contrast experiment.

Multi-contrast mode is implemented, but should be enabled only when the selected acquisitions are appropriately paired for the intended experiment.

---

## Limitations

- Current reported experiment uses only the `CORPD_FBK` acquisition.
- The current training split is relatively small.
- 15-coil processing increases GPU memory and computational cost.
- DDPM sampling is more computationally expensive than a single-pass reconstruction network.
- Learned partitioning and learned sensitivity require additional controlled ablation before claiming an improvement.
- Results can depend on diffusion sampling steps, SSDU partition ratio, training budget, and random seeds.

---

## Project Structure

A recommended repository structure is:

```text
ssdu-ddpm-mri/
├── README.md
├── train_ssdu_ddpm.py
├── requirements.txt
├── checkpoints/
├── results/
│   └── qualitative_result.png
└── figures/
```

---

## Citation / Method References

The implementation is based on the SSDU self-supervised reconstruction principle and conditional diffusion reconstruction, with learned partitioning and sensitivity estimation implemented as optional components.

When publishing results, cite the original papers corresponding to:

- SSDU / self-supervised MRI reconstruction
- DDPM / diffusion probabilistic models
- learned sensitivity-map estimation / E2E-VarNet-style calibration
- learned k-space partitioning

---

## Summary

This project implements a **pure self-supervised 15-coil MRI reconstruction pipeline** combining:

```text
SSDU
  +
SENSE / sensitivity maps
  +
Hard Data Consistency
  +
Conditional Physics U-Net
  +
1000-step DDPM training
  +
100-step respaced DDPM inference
```

The reported `CORPD_FBK` experiment improves over the SENSE zero-filled baseline from:

```text
31.7438 dB → 35.5488 dB PSNR
0.8524   → 0.9010   SSIM
0.010830 → 0.004819 NMSE
```

while keeping the training objective fully self-supervised.
