"""
Evaluation metrics: PSNR, SSIM, NMSE, computed on normalized magnitude
images with NaN/Inf sanitization.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def clean_metric_arrays(target, pred):

    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    return target, pred


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

    numerator = np.sum((target - pred) ** 2)
    denominator = np.sum(target ** 2) + 1e-8

    return float(numerator / denominator)
