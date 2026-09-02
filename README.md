# SSDU Conditional DDPM for Self-Supervised Multi-Coil MRI Reconstruction

A pure self-supervised (SSDU-style) conditional diffusion model for
undersampled multi-coil MRI reconstruction, evaluated on the FastMRI
knee multi-coil dataset. No fully-sampled ground truth is used during
training — the model is trained entirely with Self-Supervised
learning via Data Undersampling (SSDU, Yaman et al., MRM 2020),
extended with a DDPM generator.

## Features

- **Pure SSDU training** — the acquired k-space (Omega) is split into
  two disjoint sets, Aleph (network input) and Upsilon (self-supervised
  loss target); no fully-sampled ground truth is used for training.
- **Conditional DDPM generator** — a residual U-Net predicts `x0`
  directly at each diffusion step, conditioned on the Aleph zero-filled
  image, with a hard data-consistency layer folded into the forward
  pass.
- **Generalized respaced reverse sampler** — supports any number of
  test-time sampling steps (1 to `T_STEPS`) via a mathematically valid
  respaced DDPM posterior.
- *(Optional, off by default)* **Multi-contrast SSDU** — joint
  training across paired contrasts (e.g. PD / PD-FS).
- *(Optional, off by default)* **Learned k-space partitioning** — the
  Aleph/Upsilon split itself is a trainable distribution over k-space
  lines (Eqs. 5–7), with a straight-through gradient estimator.
- *(Optional, off by default)* **Learned coil sensitivity map
  estimation** — an E2E-VarNet-style refinement network replaces the
  classical ACS+RSS sensitivity formula for the reconstruction
  pipeline.

## Repository layout

```
ssdu-diffusion-mri/
├── main.py                  # entry point: discover data -> train -> test -> report
├── requirements.txt
├── NOTES.md                 # known issues / open design questions
├── ssdu_mri/
│   ├── config.py             # CFG: all hyperparameters and toggles
│   ├── reproducibility.py    # seed_everything()
│   ├── fft_utils.py          # centered FFT, RSS, cropping, complex<->channels
│   ├── masks.py               # undersampling mask, ACS mask, fixed SSDU split
│   ├── sensitivity.py         # fixed + learned coil sensitivity estimation
│   ├── sense_ops.py           # SENSE forward/adjoint, hard data consistency
│   ├── partitioning.py        # learned + fixed Aleph/Upsilon partitioning
│   ├── network.py             # PhysicsUNet generator
│   ├── diffusion_model.py     # SSDUDDPM: schedule, loss, train_step, sample
│   ├── data_discovery.py      # multi-contrast file discovery / splitting
│   ├── dataset.py             # PyTorch Dataset + collate
│   ├── metrics.py             # PSNR / SSIM / NMSE
│   └── train.py / test.py     # training loop / test-time evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

This code expects the [FastMRI knee multi-coil
dataset](https://fastmri.med.nyu.edu/) (registration required; the
dataset is **not** included in this repository). Point
`CFG.DATA_ROOT` in `ssdu_mri/config.py` at your local copy of the
`.h5` files before running.

## Running

```bash
python main.py
```

All hyperparameters, feature toggles (multi-contrast, learned
partitioning, learned sensitivity), and paths live in
`ssdu_mri/config.py` — edit the `CFG` class before running. Checkpoints
are written to `CFG.CHECKPOINT_DIR`; qualitative visualizations and
logs go to `CFG.OUTPUT_DIR`.

## Results

_Fill in after running on your data — e.g._

| | PSNR (dB) | SSIM | NMSE |
|---|---|---|---|
| SENSE zero-filled baseline |  31.74 | 0.85 | 0.01 |
| SSDU conditional DDPM (this repo) | 35.55 | 0.90 | 0.048 |


<img width="2685" height="763" alt="example_001_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/ee3a2615-b78c-43cc-8f4b-691cb56f5e5d" />
<img width="2685" height="763" alt="example_010_CORPD_FBK (1)" src="https://github.com/user-attachments/assets/f28c301b-b04f-4254-b21e-4173f98a7bc3" />



## Known issues / open design questions

See [`NOTES.md`](NOTES.md) — in particular, a train/test mismatch in
how the network's mask input and data-consistency step are conditioned
during `SSDUDDPM.sample()` versus `SSDUDDPM.train_step()` that has not
yet been resolved.

## References

- Yaman, B. et al. "Self-supervised learning of physics-guided
  reconstruction neural networks without fully sampled reference
  data." *Magnetic Resonance in Medicine*, 2020.
- Sriram, A. et al. "End-to-End Variational Networks for Accelerated
  MRI Reconstruction." *MICCAI*, 2020.
- Kadota et al. — multi-contrast self-supervised MRI reconstruction
  (see paper for the exact Eq. numbers referenced in
  `partitioning.py`).

## License

MIT — see [`LICENSE`](LICENSE).
