"""
Test-time evaluation: runs reverse-diffusion sampling on the held-out
test set, computes PSNR/SSIM/NMSE against ground truth, compares
against the classical SENSE zero-filled baseline, and saves a handful
of qualitative visualizations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CFG
from .dataset import move_batch
from .fft_utils import channels_to_complex_multi, complex_to_channels_multi
from .sense_ops import sense_adjoint_multi
from .metrics import psnr, ssim, nmse, normalize_for_metrics


# ============================================================================
# TEST
#
# Sensitivity maps for the reconstruction pipeline (condition +
# model.sample()'s internal/final DC) come from model.get_sens(komega)
# (learned or fixed, per CFG). The Dataset's fixed batch["sens"] is
# still used, unchanged, for the SENSE ZF baseline metric only -- a
# classical, model-independent reference point.
# ============================================================================

@torch.no_grad()
def test_model(model, loader, max_slices):

    model.eval()

    L = CFG.NUM_CONTRASTS
    labels = CFG.CONTRAST_ACQUISITIONS

    recon_psnr = {label: [] for label in labels}
    recon_ssim = {label: [] for label in labels}
    recon_nmse = {label: [] for label in labels}

    zf_psnr = {label: [] for label in labels}
    zf_ssim = {label: [] for label in labels}
    zf_nmse = {label: [] for label in labels}

    count = 0

    for batch in loader:

        if count >= max_slices:
            break

        batch = move_batch(batch)

        n = min(batch["komega"].shape[0], max_slices - count)

        komega = batch["komega"][:n]
        omega = batch["omega"][:n]
        fixed_sens = batch["sens"][:n]  # classical, ZF baseline ONLY
        target = batch["target"][:n]  # [n, L, H, W]

        # Reconstruction-pipeline sensitivity (learned or fixed, per CFG).
        recon_sens = model.get_sens(komega)

        # Test-time Aleph-derived condition (fixed seed => reproducible).
        aleph, _ = model.get_partition(omega, step_seed=999999)
        kaleph = komega * aleph.unsqueeze(2)

        condition_complex = sense_adjoint_multi(kaleph, recon_sens)
        condition = complex_to_channels_multi(condition_complex)

        # SENSE ZF baseline (full Omega, CLASSICAL fixed sensitivity --
        # a model-independent reference point).
        zf_complex = sense_adjoint_multi(komega, fixed_sens)  # [n,L,H,W]
        zf_mag = torch.abs(zf_complex).cpu().numpy()

        result = model.sample(condition, komega, recon_sens, omega)  # [n,2L,H,W]
        recon_complex = channels_to_complex_multi(result, L)
        recon_mag = torch.abs(recon_complex).cpu().numpy()

        target_np = target.cpu().numpy()

        for j in range(n):

            for c, label in enumerate(labels):

                t_img = target_np[j, c]
                z_img = zf_mag[j, c]
                r_img = recon_mag[j, c]

                zpsnr = psnr(t_img, z_img)
                zssim = ssim(t_img, z_img)
                znmse = nmse(t_img, z_img)

                rpsnr = psnr(t_img, r_img)
                rssim = ssim(t_img, r_img)
                rnmse = nmse(t_img, r_img)

                zf_psnr[label].append(zpsnr)
                zf_ssim[label].append(zssim)
                zf_nmse[label].append(znmse)

                recon_psnr[label].append(rpsnr)
                recon_ssim[label].append(rssim)
                recon_nmse[label].append(rnmse)

                if count < CFG.NUM_VIS_EXAMPLES:

                    tn, zn = normalize_for_metrics(t_img, z_img)
                    _, rn = normalize_for_metrics(t_img, r_img)

                    error = np.abs(rn - tn)

                    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

                    ax[0].imshow(tn, cmap="gray")
                    ax[0].set_title(f"SENSE GT ({label})")

                    ax[1].imshow(zn, cmap="gray")
                    ax[1].set_title(f"SENSE ZF\nPSNR {zpsnr:.2f} dB\nSSIM {zssim:.3f}")

                    ax[2].imshow(rn, cmap="gray")
                    ax[2].set_title(
                        f"SSDU DDPM ({CFG.TEST_SAMPLE_STEPS} steps)\n"
                        f"PSNR {rpsnr:.2f} dB\nSSIM {rssim:.3f}"
                    )

                    ax[3].imshow(error, cmap="hot")
                    ax[3].set_title("Absolute Error")

                    for a in ax:
                        a.axis("off")

                    plt.tight_layout()

                    save_path = os.path.join(
                        CFG.OUTPUT_DIR,
                        f"example_{count + 1:03d}_{label}.png"
                    )

                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close()

                    print("Saved visualization:", save_path)

                print(
                    f"Test slice {count + 1} [{label}]: "
                    f"ZF PSNR={zpsnr:.2f} | DDPM PSNR={rpsnr:.2f} | "
                    f"ZF SSIM={zssim:.4f} | DDPM SSIM={rssim:.4f}"
                )

            count += 1

            if count >= max_slices:
                break

    model.train()

    results = {}

    for label in labels:

        results[label] = {
            "zf_psnr": float(np.mean(zf_psnr[label])),
            "zf_ssim": float(np.mean(zf_ssim[label])),
            "zf_nmse": float(np.mean(zf_nmse[label])),
            "psnr": float(np.mean(recon_psnr[label])),
            "ssim": float(np.mean(recon_ssim[label])),
            "nmse": float(np.mean(recon_nmse[label]))
        }

    all_zf_psnr = [v for label in labels for v in zf_psnr[label]]
    all_zf_ssim = [v for label in labels for v in zf_ssim[label]]
    all_zf_nmse = [v for label in labels for v in zf_nmse[label]]
    all_psnr = [v for label in labels for v in recon_psnr[label]]
    all_ssim = [v for label in labels for v in recon_ssim[label]]
    all_nmse = [v for label in labels for v in recon_nmse[label]]

    results["overall"] = {
        "zf_psnr": float(np.mean(all_zf_psnr)),
        "zf_ssim": float(np.mean(all_zf_ssim)),
        "zf_nmse": float(np.mean(all_zf_nmse)),
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "nmse": float(np.mean(all_nmse))
    }

    return results
