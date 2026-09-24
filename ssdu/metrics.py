"""Volume-level PSNR / SSIM / NMSE (official fastMRI convention, Zbontar et al. 2018, Eqs. 7-8)."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def clean_metric_arrays(target, pred):
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    return target, pred


# ---- legacy per-slice metrics (kept for compatibility, not used for reported numbers) ----
def normalize_for_metrics(target, pred):
    target, pred = clean_metric_arrays(target, pred)
    scale = max(float(target.max()), 1e-8)
    return target / scale, pred / scale


def psnr(target, pred):
    target, pred = normalize_for_metrics(target, pred)
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def ssim(target, pred):
    target, pred = normalize_for_metrics(target, pred)
    return structural_similarity(target, pred, data_range=1.0)


def nmse(target, pred):
    target, pred = clean_metric_arrays(target, pred)
    return float(np.sum((target - pred) ** 2) / (np.sum(target ** 2) + 1e-8))


# ---- volume-level metrics ----
def volume_nmse(target_vol, pred_vol):
    target_vol, pred_vol = clean_metric_arrays(target_vol, pred_vol)
    numerator = np.sum((target_vol - pred_vol) ** 2)
    denominator = np.sum(target_vol ** 2) + 1e-8
    return float(numerator / denominator)


def volume_psnr(target_vol, pred_vol):
    target_vol, pred_vol = clean_metric_arrays(target_vol, pred_vol)

    max_val = max(float(target_vol.max()), 1e-8)
    mse = max(float(np.mean((target_vol - pred_vol) ** 2)), 1e-12)

    return float(10.0 * np.log10((max_val ** 2) / mse))


def volume_ssim(target_vol, pred_vol):
    """Mean of per-slice SSIM using the volume-wide data_range."""
    target_vol, pred_vol = clean_metric_arrays(target_vol, pred_vol)

    max_val = max(float(target_vol.max()), 1e-8)

    scores = [
        structural_similarity(target_vol[s], pred_vol[s], data_range=max_val)
        for s in range(target_vol.shape[0])
    ]

    return float(np.mean(scores)) if scores else float("nan")


def volume_psnr_ssim_nmse(target_vol, pred_vol):
    return (
        volume_psnr(target_vol, pred_vol),
        volume_ssim(target_vol, pred_vol),
        volume_nmse(target_vol, pred_vol),
    )
