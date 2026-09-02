"""
Centered FFT, RSS, cropping, k-space scaling, and complex<->real-channel
conversion utilities shared across the whole pipeline.
"""

import torch
import torch.fft as tfft


# ============================================================================
# FFT
# ============================================================================

def fft2c(x):

    x = tfft.ifftshift(x, dim=(-2, -1))
    x = tfft.fft2(x, dim=(-2, -1), norm="ortho")
    x = tfft.fftshift(x, dim=(-2, -1))

    return x


def ifft2c(x):

    x = tfft.ifftshift(x, dim=(-2, -1))
    x = tfft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = tfft.fftshift(x, dim=(-2, -1))

    return x


# ============================================================================
# RSS
# ============================================================================

def rss(x, dim=0):

    return torch.sqrt(
        torch.sum(torch.abs(x) ** 2, dim=dim).clamp_min(1e-12)
    )


# ============================================================================
# CENTER CROP
# ============================================================================

def center_crop_complex(x, size):

    H, W = x.shape[-2:]

    if H < size or W < size:
        raise RuntimeError(f"Input {(H, W)} smaller than crop {size}")

    top = (H - size) // 2
    left = (W - size) // 2

    return x[..., top:top + size, left:left + size]


def crop_kspace(kspace, size):

    image = ifft2c(kspace)
    image = center_crop_complex(image, size)

    return fft2c(image)


# ============================================================================
# SCALE
# ============================================================================

def estimate_scale(kspace):

    image = ifft2c(kspace)
    magnitude = rss(image, dim=0)

    finite = magnitude[torch.isfinite(magnitude)]

    if finite.numel() == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    return torch.quantile(finite, 0.995).float().clamp_min(1e-8)


def normalize_kspace(kspace, scale):

    return kspace / scale.to(kspace.dtype)


# ============================================================================
# COMPLEX <-> CHANNELS  (single-contrast, [B,H,W] <-> [B,2,H,W])
#
# Also reused as-is (with a leading dim of any size N, not just batch) by
# the learned SME model to convert folded per-coil images to/from the
# 2-channel real/imag representation its U-Net operates on.
# ============================================================================

def complex_to_channels(x):

    return torch.stack([x.real, x.imag], dim=1)


def channels_to_complex(x):

    return torch.complex(x[:, 0], x[:, 1])


# ============================================================================
# COMPLEX <-> CHANNELS  (multi-contrast, [B,L,H,W] <-> [B,2L,H,W])
#
# Channel layout: [Re_1, Im_1, Re_2, Im_2, ..., Re_L, Im_L] -- i.e. the
# real/imag pair for each contrast stays adjacent, contrasts concatenated
# in order. With L=1 this is byte-for-byte equivalent to the single-
# contrast functions above.
# ============================================================================

def complex_to_channels_multi(x):
    """
    Input:  [B, L, H, W] complex
    Output: [B, 2L, H, W]
    """

    B, L, H, W = x.shape

    out = torch.stack([x.real, x.imag], dim=2)  # [B, L, 2, H, W]

    return out.reshape(B, L * 2, H, W)


def channels_to_complex_multi(x, num_contrasts):
    """
    Input:  [B, 2L, H, W]
    Output: [B, L, H, W] complex
    """

    B, C2, H, W = x.shape
    L = num_contrasts

    x = x.reshape(B, L, 2, H, W)

    return torch.complex(x[:, :, 0], x[:, :, 1])
