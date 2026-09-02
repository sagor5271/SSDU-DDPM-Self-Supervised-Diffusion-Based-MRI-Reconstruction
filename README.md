# Self-Supervised SSDU Conditional DDPM for Multi-Coil MRI Reconstruction

A clean and modular implementation of **pure self-supervised MRI reconstruction** using **Self-Supervised Data Undersampling (SSDU)** and a **Conditional Denoising Diffusion Probabilistic Model (DDPM)**.

The framework is designed for **FastMRI multi-coil knee MRI reconstruction** and performs training without using fully sampled ground-truth images as reconstruction targets.

---

## Highlights

* Pure self-supervised SSDU training
* Conditional DDPM reconstruction
* Multi-coil MRI reconstruction
* Original 15-coil acquisition support
* No coil compression
* Classical fixed sensitivity estimation
* Fixed SSDU k-space partitioning
* Full 1000-step diffusion training
* Generalized respaced DDPM posterior sampling
* Reproducible seeded sampling
* Full acquired k-space data consistency
* PSNR, SSIM, and NMSE evaluation
* Modular and clean Python package structure

---

## Method Overview

The acquired undersampled multi-coil k-space is partitioned according to the SSDU strategy:

$$
\Omega = \Theta \cup \Lambda
$$

where:

* \(\Theta\) is used for reconstruction and data consistency.
* \(\Lambda\) is held out for self-supervised loss computation.

The model therefore learns MRI reconstruction without directly using fully sampled ground-truth images during training.

The reconstruction pipeline consists of:

1. Multi-coil k-space loading
2. Sampling mask generation
3. SSDU k-space partitioning
4. Classical sensitivity map estimation
5. Conditional diffusion training
6. Diffusion-based iterative reconstruction
7. Data consistency enforcement
8. Final evaluation against ground truth

---

# Repository Structure

```text
SSDU-DDPM-Self-Supervised-Diffusion-Based-MRI-Reconstruction/
│
├── main.py
├── requirements.txt
├── README.md
├── NOTES.md
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

| Module               | Description                                        |
| -------------------- | -------------------------------------------------- |
| `config.py`          | Central experiment and training configuration      |
| `data_discovery.py`  | MRI dataset discovery and file handling            |
| `dataset.py`         | Multi-coil MRI dataset loading and preprocessing   |
| `fft_utils.py`       | Centered FFT and inverse FFT utilities             |
| `masks.py`           | MRI undersampling mask generation                  |
| `partitioning.py`    | SSDU k-space partitioning                          |
| `sensitivity.py`     | Classical coil sensitivity map estimation          |
| `sense_ops.py`       | Multi-coil SENSE forward and adjoint operations    |
| `network.py`         | Conditional diffusion neural network architecture  |
| `diffusion_model.py` | DDPM schedule, training, and sampling              |
| `train.py`           | Self-supervised SSDU training pipeline             |
| `test.py`            | Reconstruction and final evaluation                |
| `metrics.py`         | PSNR, SSIM, and NMSE computation                   |
| `reproducibility.py` | Random seed and deterministic experiment utilities |

---

# Experimental Configuration

```text
TRAINING     : PURE SELF-SUPERVISED SSDU
CONTRASTS    : ['CORPD_FBK']

PARTITIONING : FIXED
ALPHA        : 0.85

SENSITIVITY  : FIXED CLASSICAL SENSITIVITY MAP

DIFFUSION TRAINING:
    Full 1000-step diffusion process

VALIDATION:
    Pure self-supervised SSDU
    Fixed-seed partition

TESTING:
    Fully sampled ground truth used for final evaluation only

LOSS:
    Normalized L1 + L2 mixed loss
    L2 weight = 0.5

GAN:
    Disabled

SAMPLER:
    Generalized respaced DDPM posterior
    Seeded and reproducible

INITIAL STATE:
    x_T ~ N(0, I)

TEST SAMPLING:
    100 steps out of 1000 training diffusion steps

DATA CONSISTENCY:
    Full acquired k-space Omega

COILS:
    Original 15 coils
    No coil compression
```

---

# Final Experimental Results

## CORPD_FBK Contrast

| Method                     |         PSNR ↑ |     SSIM ↑ |       NMSE ↓ |
| -------------------------- | -------------: | ---------: | -----------: |
| SENSE Zero-Filled Baseline |     31.7438 dB |     0.8524 |     0.010830 |
| Pure SSDU Conditional DDPM | **35.5488 dB** | **0.9010** | **0.004819** |

### Improvement Over SENSE Zero-Filled Baseline

| Metric |    Improvement |
| ------ | -------------: |
| PSNR   | **+3.8049 dB** |
| SSIM   |    **+0.0486** |
| NMSE   |  **−0.006011** |

---

## Overall Results

```text
================================================================================
FINAL RESULTS
================================================================================

--- Contrast: CORPD_FBK ---

SENSE ZERO-FILLED BASELINE
(classical fixed sensitivity)

PSNR : 31.7438 dB
SSIM : 0.8524
NMSE : 0.010830


PURE SSDU CONDITIONAL DDPM

PSNR : 35.5488 dB
SSIM : 0.9010
NMSE : 0.004819


IMPROVEMENT OVER SENSE-ZF

PSNR gain : +3.8049 dB
SSIM gain : +0.0486
NMSE change : -0.006011


--- OVERALL (averaged across all contrasts) ---

ZF PSNR/SSIM/NMSE:

31.7438 / 0.8524 / 0.010830


DDPM PSNR/SSIM/NMSE:

35.5488 / 0.9010 / 0.004819

================================================================================
```

---

# Result Summary

The proposed **Pure Self-Supervised SSDU Conditional DDPM** achieved substantial improvement over the classical SENSE zero-filled baseline.

* **PSNR increased by 3.8049 dB**
* **SSIM increased from 0.8524 to 0.9010**
* **NMSE decreased from 0.010830 to 0.004819**


<img width="2685" height="763" alt="example_001_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/c7d7f1d7-0024-46b9-9c17-b9335622ed12" />

<img width="2685" height="763" alt="example_010_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/962471d4-dd53-45d9-8948-116980361b19" />



These results indicate that the diffusion-based reconstruction effectively improves image quality while operating under a **pure self-supervised SSDU training framework**.

---

# Training Strategy

The training pipeline does not use fully sampled ground-truth images as direct supervision.

```text
Undersampled Multi-Coil k-Space
              │
              ▼
       Sampling Mask
              │
              ▼
       SSDU Partitioning
        ┌──────────────┐
        │              │
        ▼              ▼
 Reconstruction     Loss Mask
    Mask Θ           Mask Λ
        │              │
        ▼              │
 Conditional DDPM    │
 Reconstruction      │
        │              │
        └──────┬───────┘
               ▼
     Self-Supervised Loss
               │
               ▼
       Network Optimization
```

Fully sampled ground truth is reserved for **final evaluation only**.

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main training pipeline:

```bash
python main.py
```

Alternatively:

```bash
python -m ssdu_ddpm
```

---

# Output

Example output directories:

```text
ssdu_diffusion_checkpoints/
└── best_ssdu_ddpm_audited_v4.pt

ssdu_diffusion_results/
├── reconstructed_images/
├── comparisons/
└── metrics/
```

Example checkpoint:

```text
/kaggle/working/ssdu_diffusion_checkpoints/
best_ssdu_ddpm_audited_v4.pt
```

Example result directory:

```text
/kaggle/working/ssdu_diffusion_results
```

---

# Reproducibility

The framework supports reproducible experiments through:

* Fixed random seeds
* Fixed SSDU partitioning
* Deterministic validation
* Seeded DDPM sampling
* Controlled diffusion schedules

This helps ensure that reconstruction experiments can be repeated under consistent experimental conditions.

---

# Important Experimental Notes

* No adversarial training is used.
* No GAN loss is used.
* No discriminator is included.
* No fully sampled ground truth is used as the training target.
* Ground truth is used only during final testing and metric calculation.
* The reported results correspond to the configured experimental setup and should be reproduced under the same dataset split, preprocessing, random seed, and evaluation protocol.

---

# Citation

If you use this repository in your research, please cite the repository and clearly describe the SSDU and conditional diffusion methodology used in your experiments.

---

# License

This project is released under the MIT License.

---

# Author

**Sagor**

Biomedical Engineering and Medical Image Reconstruction Research

---

## Repository

[SSDU-DDPM-Self-Supervised-Diffusion-Based-MRI-Reconstruction](https://github.com/sagor5271/SSDU-DDPM-Self-Supervised-Diffusion-Based-MRI-Reconstruction?utm_source=chatgpt.com)
