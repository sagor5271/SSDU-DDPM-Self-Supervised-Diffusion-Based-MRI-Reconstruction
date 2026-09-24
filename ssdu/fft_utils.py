"""Centered FFTs, RSS, cropping and complex<->channel conversions."""

import torch
import torch.fft as tfft


def fft2c(x):
    x = tfft.ifftshift(x, dim=(-2, -1))
    x = tfft.fft2(x, dim=(-2, -1), norm="ortho")
    return tfft.fftshift(x, dim=(-2, -1))


def ifft2c(x):
    x = tfft.ifftshift(x, dim=(-2, -1))
    x = tfft.ifft2(x, dim=(-2, -1), norm="ortho")
    return tfft.fftshift(x, dim=(-2, -1))


def rss(x, dim=0):
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=dim).clamp_min(1e-12))


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


# ---- single-contrast ----
def complex_to_channels(x):
    return torch.stack([x.real, x.imag], dim=1)


def channels_to_complex(x):
    return torch.complex(x[:, 0], x[:, 1])


# ---- multi-contrast ----
def complex_to_channels_multi(x):
    B, L, H, W = x.shape
    out = torch.stack([x.real, x.imag], dim=2)
    return out.reshape(B, L * 2, H, W)


def channels_to_complex_multi(x, num_contrasts):
    B, C2, H, W = x.shape
    L = num_contrasts
    x = x.reshape(B, L, 2, H, W)
    return torch.complex(x[:, :, 0], x[:, :, 1])
