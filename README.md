# Self-Supervised SSDU Conditional DDPM for Multi-Coil MRI Reconstruction

A pure self-supervised (SSDU-style) conditional diffusion model for
undersampled multi-coil MRI reconstruction, evaluated on the fastMRI knee
multi-coil dataset. No fully-sampled ground truth is used during training. The
model is trained with Self-Supervised learning via Data Undersampling
(SSDU, Yaman et al., MRM 2020), using a DDPM generator with a physics-informed
U-Net.

## Highlights

* Pure self-supervised SSDU training (held-out k-space loss only)
* Conditional DDPM with internal hard data consistency
* Original 15-coil acquisition, no coil compression
* Classical fixed sensitivity estimation (ACS + RSS)
* Fixed-probability SSDU k-space partitioning (`alpha = 0.7`, ACS always in the input set)
* Full 1000-step diffusion training
* Generalized respaced DDPM posterior sampler (250 test steps), starting from `x_T ~ N(0, I)`
* Data consistency with the full acquired k-space `Omega`
* RSS ground truth (official fastMRI convention, no sensitivity map)
* Volume-level PSNR / SSIM / NMSE (fastMRI, Zbontar et al. 2018, Eqs. 7-8)
* Optional learned k-space partitioning and learned sensitivity maps (disabled in the reported run)
* Modular Python package

---

## Method Overview

The acquired undersampled multi-coil k-space lines $\Omega$ are split by SSDU into two disjoint sets:

$$
\Omega = \aleph \cup \Upsilon
$$

where:

* $\aleph$ (Aleph) is fed to the network and used for data consistency (about 70% of the non-ACS acquired lines, plus all ACS lines).
* $\Upsilon$ (Upsilon) is held out and used only to compute the self-supervised loss.

The network never sees a fully sampled image as a training target.

Pipeline:

1. Multi-coil k-space loading, center crop to 320 x 320
2. Cartesian sampling mask generation (4x acceleration, 8% center fraction)
3. SSDU partitioning into $\aleph$ and $\Upsilon$ (ACS forced into $\aleph$)
4. Classical sensitivity map estimation from the ACS band
5. Conditional DDPM training: the network predicts a clean image, which is projected through hard data consistency on $\aleph$
6. Loss: normalized L1 + L2 mix between predicted and measured k-space on $\Upsilon$
7. Test: respaced DDPM sampling from pure noise, conditioned on the $\aleph$-derived image, with data consistency on the full $\Omega$
8. Final evaluation against the RSS ground truth (evaluation only)

---

# Repository Structure

```text
SSDU-DDPM-Self-Supervised-Diffusion-Based-MRI-Reconstruction/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
│
└── ssdu_ddpm/
    │
    ├── __init__.py
    ├── __main__.py
    │
    ├── config.py
    ├── data_discovery.py
    ├── dataset.py
    │
    ├── fft_utils.py
    ├── masks.py
    ├── partitioning.py
    ├── sensitivity.py
    ├── sense_ops.py
    │
    ├── network.py
    ├── diffusion_model.py
    │
    ├── train.py
    ├── test.py
    │
    ├── metrics.py
    └── reproducibility.py
```

---

# Module Description

| Module               | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `config.py`          | Central configuration (`CFG`), derived `ACS_SIZE`, config checks                              |
| `data_discovery.py`  | File discovery, acquisition labels, train/val/test split, coil-count verification             |
| `dataset.py`         | Multi-coil dataset, k-space scaling, RSS ground truth (test only), collate/batch helpers      |
| `fft_utils.py`       | Centered FFT/IFFT, RSS, center crop, complex <-> channel conversions                          |
| `masks.py`           | Cartesian undersampling mask, ACS line mask, fixed-probability SSDU split                     |
| `partitioning.py`    | Fixed split wrapper and optional learned partitioning (Kadota, Millard & Chiew, Eqs. 5-7)     |
| `sensitivity.py`     | Classical ACS sensitivity maps and optional learned sensitivity network (E2E-VarNet style)    |
| `sense_ops.py`       | SENSE forward/adjoint operators and hard data consistency                                     |
| `network.py`         | Time embedding, residual block, physics-informed U-Net                                        |
| `diffusion_model.py` | DDPM schedule, SSDU loss, training step, respaced sampler                                     |
| `train.py`           | Training loop and self-supervised validation                                                  |
| `test.py`            | Sampling, volume-level evaluation, visualizations                                             |
| `metrics.py`         | Volume-level PSNR, SSIM, NMSE                                                                 |
| `reproducibility.py` | Random seed utilities                                                                         |

---

# Experimental Configuration

```text
TRAINING      : PURE SELF-SUPERVISED SSDU (full 1000-step diffusion)
CONTRAST      : CORPD_FBK
ACCELERATION  : 4x, center fraction 0.08 (ACS size 26, derived)

PARTITIONING  : FIXED, alpha = 0.7
SENSITIVITY   : FIXED CLASSICAL (ACS + RSS)

DATA SPLIT    : 60 train / 8 val / 12 test volumes (split seed 42)
SLICES        : skip first 9 and last 2 slices of every volume
                1471 train / 176 val / 296 test slices

MODEL         : 14,512,716 parameters
                (includes 320 partitioning and 53,578 SME parameters,
                 which are unused while both learned options are off)
OPTIMIZER     : AdamW, lr 1e-4, weight decay 1e-4, cosine schedule
BATCH / EPOCH : batch size 2, 50 epochs (~5.7 min per epoch)

LOSS          : normalized L1 + L2 mix, L2 weight = 0.5
VALIDATION    : held-out Upsilon k-space loss, fixed-seed partition,
                fixed diffusion step t = T/2
CHECKPOINT    : best validation SSDU loss (epoch 50, 0.9082)

SAMPLER       : generalized respaced DDPM posterior
TEST SAMPLING : 250 steps out of 1000, x_T ~ N(0, I)
DATA CONSIST. : full acquired k-space Omega
MASK CHANNEL  : aleph at both train and test time
GROUND TRUTH  : plain RSS of fully sampled coil images (evaluation only)
METRICS       : volume-level (official fastMRI convention)
GAN           : disabled
COILS         : original 15, no compression
```

---

# Final Experimental Results

Evaluated on 12 held-out test volumes (296 slices), 4x Cartesian acceleration.
Each number is the mean of per-volume scores.

## CORPD_FBK Contrast

| Method                     |         PSNR ↑ |     SSIM ↑ |       NMSE ↓ |
| -------------------------- | -------------: | ---------: | -----------: |
| SENSE Zero-Filled Baseline |     31.8794 dB |     0.8261 |     0.011228 |
| Pure SSDU Conditional DDPM | **34.3753 dB** | **0.8626** | **0.006776** |

### Improvement Over SENSE Zero-Filled Baseline

| Metric |    Improvement |
| ------ | -------------: |
| PSNR   | **+2.4960 dB** |
| SSIM   |    **+0.0365** |
| NMSE   |  **−0.004452** |

## Per-Volume Results

| Volume           | Slices | ZF PSNR | DDPM PSNR | ZF SSIM | DDPM SSIM |
| ---------------- | -----: | ------: | --------: | ------: | --------: |
| file1002357.h5   |     26 |   30.18 |     32.74 |  0.7933 |    0.8424 |
| file1001605.h5   |     26 |   31.67 |     35.08 |  0.8225 |    0.8631 |
| file1001001.h5   |     23 |   31.09 |     32.61 |  0.8079 |    0.8361 |
| file1002471.h5   |     25 |   32.55 |     36.09 |  0.8573 |    0.8938 |
| file1001677.h5   |     22 |   31.54 |     33.01 |  0.7998 |    0.8377 |
| file1001861.h5   |     25 |   30.19 |     33.04 |  0.7944 |    0.8358 |
| file1002332.h5   |     34 |   32.05 |     34.54 |  0.8376 |    0.8748 |
| file1001120.h5   |     25 |   33.82 |     38.22 |  0.8854 |    0.9228 |
| file1000964.h5   |     21 |   31.70 |     33.57 |  0.8294 |    0.8674 |
| file1000084.h5   |     24 |   32.25 |     33.51 |  0.8070 |    0.8347 |
| file1001470.h5   |     26 |   30.99 |     33.24 |  0.7936 |    0.8309 |
| file1000601.h5   |     19 |   34.52 |     36.86 |  0.8845 |    0.9114 |

The DDPM improves over the zero-filled baseline on all 12 volumes in both PSNR and SSIM.

## Console Output

```text
--- Contrast: CORPD_FBK (12 volumes) ---
SENSE ZERO-FILLED BASELINE (classical fixed sensitivity)
  PSNR : 31.8794 dB
  SSIM : 0.8261
  NMSE : 0.011228
PURE SSDU CONDITIONAL DDPM
  PSNR : 34.3753 dB
  SSIM : 0.8626
  NMSE : 0.006776
IMPROVEMENT OVER SENSE-ZF
  PSNR gain: 2.4960 dB
  SSIM gain: 0.0365
  NMSE change: -0.004452
```

## Result Summary

* **PSNR increased by 2.4960 dB** (31.8794 to 34.3753 dB)
* **SSIM increased from 0.8261 to 0.8626**
* **NMSE decreased from 0.011228 to 0.006776**

<img width="2685" height="763" alt="example_001_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/c7d7f1d7-0024-46b9-9c17-b9335622ed12" />

<img width="2685" height="763" alt="example_010_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/962471d4-dd53-45d9-8948-116980361b19" />

*Each panel: RSS ground truth, SENSE zero-filled, SSDU DDPM reconstruction, absolute error.*

---

# Training Strategy

The training pipeline does not use fully sampled ground-truth images as supervision.

```text
Undersampled Multi-Coil k-Space (Omega)
                 │
                 ▼
          SSDU Partitioning
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
  Input mask ℵ       Loss mask Υ
  (incl. ACS)        (held out)
        │                  │
        ▼                  │
 Conditional DDPM          │
 + hard DC on ℵ            │
        │                  │
        ▼                  │
  Predicted image          │
  -> SENSE forward         │
  -> predicted k-space     │
        │                  │
        └────────┬─────────┘
                 ▼
 Normalized L1 + L2 loss on Υ lines
                 │
                 ▼
        Network Optimization
```

Fully sampled data is used **only** for final evaluation.

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Set data and output paths (defaults are Kaggle paths):

```bash
export SSDU_DATA_ROOT=/path/to/fastmri-knee-multicoil
export SSDU_CHECKPOINT_DIR=./ssdu_diffusion_checkpoints
export SSDU_OUTPUT_DIR=./ssdu_diffusion_results
```

Run training followed by testing:

```bash
python main.py
```

Alternatively:

```bash
python -m ssdu_ddpm
```

All hyperparameters are in `ssdu_ddpm/config.py`.

---

# Output

```text
ssdu_diffusion_checkpoints/
└── best_ssdu_ddpm_audited.pt

ssdu_diffusion_results/
├── example_001_CORPD_FBK.png
├── example_002_CORPD_FBK.png
├── ...
└── example_010_CORPD_FBK.png
```

Per-volume and overall metrics are printed to the console at the end of the run.

---

# Reproducibility

* Global seeding via `seed_everything` (Python, NumPy, PyTorch, CUDA), deterministic cuDNN
* Fixed file split (`SPLIT_SEED = 42`)
* Fixed-seed SSDU partition and fixed-seed noise for validation
* Fixed-seed partition at test time
* Sampling noise comes from the global PyTorch RNG, so results are repeatable for the same seed and hardware but not guaranteed bit-identical across devices

---

# Important Experimental Notes

* No adversarial training, GAN loss, or discriminator is used.
* No fully sampled ground truth is used as a training target.
* Ground truth is used only for final testing and metric calculation.
* The ground truth is the plain RSS of fully sampled coil images, independent of any sensitivity map. The SENSE zero-filled baseline uses the classical ACS sensitivity maps.
* Validation loss is a self-supervised proxy (SSDU loss at a fixed diffusion step), not a reconstruction-quality metric.
* Volume SSIM is the mean of per-slice scikit-image SSIM using the volume-wide data range, an approximation of the official implementation.
* Multi-contrast joint reconstruction is implemented but only one contrast is configured. fastMRI knee PD / PD-FS pairing would be by slice index, not by patient.
* The reported results correspond to this configuration and should be reproduced under the same dataset split, preprocessing, random seed, and evaluation protocol.

---

# Citation

If you use this repository in your research, please cite the repository and describe the SSDU and conditional diffusion methodology used in your experiments.

Related work:

* Yaman et al., *Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data*, MRM 2020.
* Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*, 2018.
* Ho et al., *Denoising Diffusion Probabilistic Models*, 2020.
* Kadota, Millard & Chiew, multi-contrast SSDU with learned partitioning.
* Sriram et al., *End-to-End Variational Networks for Accelerated MRI Reconstruction*, 2020.

---

# License

This project is released under the MIT License.

---

# Author

**Sagor**

Biomedical Engineering and Medical Image Reconstruction Research

---

## Repository

[SSDU-DDPM-Self-Supervised-Diffusion-Based-MRI-Reconstruction](https://github.com/sagor5271/SSDU-DDPM-Self-Supervised-Diffusion-Based-MRI-Reconstruction)
