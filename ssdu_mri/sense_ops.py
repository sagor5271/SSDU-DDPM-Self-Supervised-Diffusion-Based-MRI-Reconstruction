"""
SENSE forward/adjoint operators and hard (exact) data-consistency, for
both single-contrast and multi-contrast tensor shapes.
"""

from .fft_utils import fft2c, ifft2c


# ============================================================================
# SENSE OPERATORS  (single-contrast, [B,H,W] / [B,C,H,W])
# ============================================================================

def sense_forward(image, sens):
    """
    image: [B,H,W]
    sens : [B,C,H,W]
    return: [B,C,H,W]
    """

    return fft2c(sens * image.unsqueeze(1))


def sense_adjoint(kspace, sens):
    """
    kspace: [B,C,H,W]
    sens  : [B,C,H,W]
    """

    coil_images = ifft2c(kspace)

    numerator = (sens.conj() * coil_images).sum(dim=1)
    denominator = (sens.abs() ** 2).sum(dim=1).clamp_min(1e-8)

    return numerator / denominator


def hard_dc(image, measured_kspace, sens, mask):
    """
    Exact measured-data replacement in k-space.
    """

    predicted_kspace = sense_forward(image, sens)

    M = mask.unsqueeze(1)

    corrected_kspace = M * measured_kspace + (1.0 - M) * predicted_kspace

    return sense_adjoint(corrected_kspace, sens)


# ============================================================================
# SENSE OPERATORS  (multi-contrast, [B,L,H,W] / [B,L,C,H,W])
#
# Each of these simply flattens the (batch, contrast) dimensions into one
# combined batch dimension and calls the EXACT SAME single-contrast
# function above -- no new physics/math, just a reshape wrapper, so the
# underlying SENSE/DC math is guaranteed identical to the single-contrast
# case for every individual contrast.
# ============================================================================

def sense_forward_multi(image, sens):
    """
    image: [B,L,H,W] complex
    sens : [B,L,C,H,W] complex
    return: [B,L,C,H,W] complex
    """

    B, L, H, W = image.shape
    C = sens.shape[2]

    out = sense_forward(
        image.reshape(B * L, H, W),
        sens.reshape(B * L, C, H, W)
    )

    return out.reshape(B, L, C, H, W)


def sense_adjoint_multi(kspace, sens):
    """
    kspace: [B,L,C,H,W] complex
    sens  : [B,L,C,H,W] complex
    return: [B,L,H,W] complex
    """

    B, L, C, H, W = kspace.shape

    out = sense_adjoint(
        kspace.reshape(B * L, C, H, W),
        sens.reshape(B * L, C, H, W)
    )

    return out.reshape(B, L, H, W)


def hard_dc_multi(image, measured_kspace, sens, mask):
    """
    image          : [B,L,H,W] complex
    measured_kspace: [B,L,C,H,W] complex
    sens           : [B,L,C,H,W] complex
    mask           : [B,L,H,W]
    return: [B,L,H,W] complex
    """

    B, L, H, W = image.shape
    C = sens.shape[2]

    out = hard_dc(
        image.reshape(B * L, H, W),
        measured_kspace.reshape(B * L, C, H, W),
        sens.reshape(B * L, C, H, W),
        mask.reshape(B * L, H, W)
    )

    return out.reshape(B, L, H, W)
