"""SENSE forward / adjoint operators and hard data consistency (single + multi-contrast)."""

from .fft_utils import fft2c, ifft2c


def sense_forward(image, sens):
    return fft2c(sens * image.unsqueeze(1))


def sense_adjoint(kspace, sens):
    coil_images = ifft2c(kspace)
    numerator = (sens.conj() * coil_images).sum(dim=1)
    denominator = (sens.abs() ** 2).sum(dim=1).clamp_min(1e-8)
    return numerator / denominator


def hard_dc(image, measured_kspace, sens, mask):
    predicted_kspace = sense_forward(image, sens)
    M = mask.unsqueeze(1)
    corrected_kspace = M * measured_kspace + (1.0 - M) * predicted_kspace
    return sense_adjoint(corrected_kspace, sens)


def sense_forward_multi(image, sens):
    B, L, H, W = image.shape
    C = sens.shape[2]
    out = sense_forward(image.reshape(B * L, H, W), sens.reshape(B * L, C, H, W))
    return out.reshape(B, L, C, H, W)


def sense_adjoint_multi(kspace, sens):
    B, L, C, H, W = kspace.shape
    out = sense_adjoint(kspace.reshape(B * L, C, H, W), sens.reshape(B * L, C, H, W))
    return out.reshape(B, L, H, W)


def hard_dc_multi(image, measured_kspace, sens, mask):
    B, L, H, W = image.shape
    C = sens.shape[2]
    out = hard_dc(
        image.reshape(B * L, H, W),
        measured_kspace.reshape(B * L, C, H, W),
        sens.reshape(B * L, C, H, W),
        mask.reshape(B * L, H, W),
    )
    return out.reshape(B, L, H, W)
