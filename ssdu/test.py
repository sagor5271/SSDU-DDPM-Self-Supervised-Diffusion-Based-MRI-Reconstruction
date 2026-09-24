"""Test-time evaluation: DDPM sampling + volume-level metrics against RSS ground truth."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CFG
from .dataset import move_batch
from .fft_utils import channels_to_complex_multi, complex_to_channels_multi
from .metrics import clean_metric_arrays, volume_psnr_ssim_nmse
from .sense_ops import sense_adjoint_multi


def _mean(x):
    return float(np.mean(x)) if x else float("nan")


@torch.no_grad()
def test_model(model, loader, max_slices):

    model.eval()

    L = CFG.NUM_CONTRASTS
    labels = CFG.CONTRAST_ACQUISITIONS

    volumes = {label: {} for label in labels}

    count = 0
    vis_saved = 0

    for batch in loader:

        if count >= max_slices:
            break

        batch = move_batch(batch)

        n = min(batch["komega"].shape[0], max_slices - count)

        komega = batch["komega"][:n]
        omega = batch["omega"][:n]
        fixed_sens = batch["sens"][:n]
        target = batch["target"][:n]

        recon_sens = model.get_sens(komega)

        aleph, _ = model.get_partition(omega, step_seed=999999)
        kaleph = komega * aleph.unsqueeze(2)

        condition_complex = sense_adjoint_multi(kaleph, recon_sens)
        condition = complex_to_channels_multi(condition_complex)

        zf_complex = sense_adjoint_multi(komega, fixed_sens)
        zf_mag = torch.abs(zf_complex).cpu().numpy()

        # mask-channel = aleph (matches training); omega used only for DC
        result = model.sample(condition, komega, recon_sens, aleph, omega)
        recon_complex = channels_to_complex_multi(result, L)
        recon_mag = torch.abs(recon_complex).cpu().numpy()

        target_np = target.cpu().numpy()
        files_batch = batch["files"]

        for j in range(n):

            for c, label in enumerate(labels):

                t_img = target_np[j, c]
                z_img = zf_mag[j, c]
                r_img = recon_mag[j, c]

                fname = files_batch[j][c]

                if fname not in volumes[label]:
                    volumes[label][fname] = {"target": [], "zf": [], "recon": []}

                volumes[label][fname]["target"].append(t_img)
                volumes[label][fname]["zf"].append(z_img)
                volumes[label][fname]["recon"].append(r_img)

                if vis_saved < CFG.NUM_VIS_EXAMPLES:

                    t64, z64 = clean_metric_arrays(t_img, z_img)
                    _, r64 = clean_metric_arrays(t_img, r_img)

                    vis_scale = max(float(t64.max()), 1e-8)
                    tn, zn, rn = t64 / vis_scale, z64 / vis_scale, r64 / vis_scale
                    error = np.abs(rn - tn)

                    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

                    ax[0].imshow(tn, cmap="gray")
                    ax[0].set_title(f"RSS GT ({label})")

                    ax[1].imshow(zn, cmap="gray")
                    ax[1].set_title("SENSE ZF (slice preview)")

                    ax[2].imshow(rn, cmap="gray")
                    ax[2].set_title(f"SSDU DDPM ({CFG.TEST_SAMPLE_STEPS} steps)\n(slice preview)")

                    ax[3].imshow(error, cmap="hot")
                    ax[3].set_title("Absolute Error (slice preview)")

                    for a in ax:
                        a.axis("off")

                    plt.tight_layout()

                    save_path = os.path.join(
                        CFG.OUTPUT_DIR, f"example_{vis_saved + 1:03d}_{label}.png"
                    )
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close()

                    print("Saved visualization:", save_path)
                    vis_saved += 1

            count += 1
            if count >= max_slices:
                break

    model.train()

    results = {}

    all_r = {"psnr": [], "ssim": [], "nmse": []}
    all_z = {"psnr": [], "ssim": [], "nmse": []}

    for label in labels:

        lr = {"psnr": [], "ssim": [], "nmse": []}
        lz = {"psnr": [], "ssim": [], "nmse": []}

        for fname, buf in volumes[label].items():

            target_vol = np.stack(buf["target"], axis=0)
            zf_vol = np.stack(buf["zf"], axis=0)
            recon_vol = np.stack(buf["recon"], axis=0)

            zpsnr, zssim, znmse = volume_psnr_ssim_nmse(target_vol, zf_vol)
            rpsnr, rssim, rnmse = volume_psnr_ssim_nmse(target_vol, recon_vol)

            print(
                f"Volume [{label}] {fname} ({target_vol.shape[0]} slices): "
                f"ZF PSNR={zpsnr:.2f} | DDPM PSNR={rpsnr:.2f} | "
                f"ZF SSIM={zssim:.4f} | DDPM SSIM={rssim:.4f}"
            )

            for d, vals in ((lz, (zpsnr, zssim, znmse)), (lr, (rpsnr, rssim, rnmse))):
                d["psnr"].append(vals[0])
                d["ssim"].append(vals[1])
                d["nmse"].append(vals[2])

        results[label] = {
            "zf_psnr": _mean(lz["psnr"]),
            "zf_ssim": _mean(lz["ssim"]),
            "zf_nmse": _mean(lz["nmse"]),
            "psnr": _mean(lr["psnr"]),
            "ssim": _mean(lr["ssim"]),
            "nmse": _mean(lr["nmse"]),
            "num_volumes": len(volumes[label]),
        }

        for k in all_r:
            all_r[k].extend(lr[k])
            all_z[k].extend(lz[k])

    results["overall"] = {
        "zf_psnr": _mean(all_z["psnr"]),
        "zf_ssim": _mean(all_z["ssim"]),
        "zf_nmse": _mean(all_z["nmse"]),
        "psnr": _mean(all_r["psnr"]),
        "ssim": _mean(all_r["ssim"]),
        "nmse": _mean(all_r["nmse"]),
        "num_volumes": sum(len(volumes[label]) for label in labels),
    }

    return results, volumes
