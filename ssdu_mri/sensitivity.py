"""
Coil sensitivity map estimation: the classical fixed ACS+RSS formula,
and the optional learned (E2E-VarNet style) sensitivity refinement
network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fft_utils import ifft2c, rss, complex_to_channels, channels_to_complex


# ============================================================================
# SENSITIVITY MAP (FIXED / CLASSICAL)
#
# Used for: (a) the Dataset's per-slice "sens" (ZF baseline metric +
# test-time GT target only), and (b) the fallback path inside
# SSDUDDPM.get_sens() when CFG.USE_LEARNED_SENSITIVITY=False.
# ============================================================================

def estimate_sens(kspace, acs_size):

    C, H, W = kspace.shape

    center = H // 2
    half = acs_size // 2

    start = max(0, center - half)
    end = min(H, center + half)

    calib = torch.zeros_like(kspace)
    calib[:, start:end, :] = kspace[:, start:end, :]

    coil_images = ifft2c(calib)

    denom = rss(coil_images, dim=0).clamp_min(1e-8)
    sens = coil_images / denom.unsqueeze(0)

    sens = torch.complex(
        torch.nan_to_num(sens.real),
        torch.nan_to_num(sens.imag)
    )

    norm = torch.sqrt(
        torch.sum(torch.abs(sens) ** 2, dim=0, keepdim=True).clamp_min(1e-8)
    )

    sens = sens / norm

    return torch.complex(
        torch.nan_to_num(sens.real),
        torch.nan_to_num(sens.imag)
    )


def estimate_sens_multi_fixed(komega, acs_size):
    """
    komega: [B, L, C, H, W] complex
    Returns sens: [B, L, C, H, W] complex, using the ORIGINAL fixed
    (non-learned) estimate_sens() formula per (batch, contrast) slice.
    This is the CFG.USE_LEARNED_SENSITIVITY=False fallback path for the
    reconstruction pipeline (SSDUDDPM.get_sens()).
    """

    B, L, C, H, W = komega.shape

    out = []

    for b in range(B):
        for l in range(L):
            out.append(estimate_sens(komega[b, l], acs_size))

    return torch.stack(out, dim=0).reshape(B, L, C, H, W)


# ============================================================================
# LEARNED SENSITIVITY MAP ESTIMATION (E2E-VarNet style)
#
# extract_acs_images() reuses the EXACT same ACS-crop + IFFT logic as
# estimate_sens() above, just generalized to accept any number of
# leading batch dimensions (works for a single [C,H,W] slice OR a full
# [B,L,C,H,W] batch, since only the last two dims -- H,W -- are ever
# indexed).
# ============================================================================

def extract_acs_images(kspace, acs_size):
    """
    kspace: [..., C, H, W] complex, any number of leading dims.
    Returns raw (un-normalized) per-coil images from the ACS-only
    k-space, same [..., C, H, W] shape -- the same first step
    estimate_sens() performs, before its RSS-normalize.
    """

    H, W = kspace.shape[-2], kspace.shape[-1]

    center = H // 2
    half = acs_size // 2

    start = max(0, center - half)
    end = min(H, center + half)

    calib = torch.zeros_like(kspace)
    calib[..., start:end, :] = kspace[..., start:end, :]

    return ifft2c(calib)


class SensitivityRefineUNet(nn.Module):
    """
    Small shared U-Net that refines a SINGLE coil's raw ACS image
    (2-channel real/imag input/output). The SAME weights are applied to
    every coil, every contrast, and every batch element -- they are all
    folded into one leading batch dimension by the caller
    (LearnedSensitivityModel), exactly like E2E-VarNet's sensitivity
    model. No time/diffusion conditioning is needed here -- this is a
    one-shot calibration-refinement network, not part of the reverse
    diffusion chain.
    """

    def __init__(self, chans=8):

        super().__init__()

        def conv_block(in_ch, out_ch):

            groups = min(8, out_ch)

            while groups > 1 and out_ch % groups != 0:
                groups -= 1

            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
            )

        self.e1 = conv_block(2, chans)
        self.down1 = nn.Conv2d(chans, chans * 2, 4, 2, 1)

        self.e2 = conv_block(chans * 2, chans * 2)
        self.down2 = nn.Conv2d(chans * 2, chans * 4, 4, 2, 1)

        self.mid = conv_block(chans * 4, chans * 4)

        self.up2 = nn.ConvTranspose2d(chans * 4, chans * 2, 4, 2, 1)
        self.d2 = conv_block(chans * 4, chans * 2)

        self.up1 = nn.ConvTranspose2d(chans * 2, chans, 4, 2, 1)
        self.d1 = conv_block(chans * 2, chans)

        self.out = nn.Conv2d(chans, 2, 3, padding=1)

        # Zero-init the final layer so the module starts as a pure
        # identity/no-op residual (refined = raw + 0), i.e. training
        # begins EXACTLY at the classical fixed-sensitivity solution
        # and only departs from it as the loss actually rewards doing
        # so -- a stable starting point for this correction network.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """
        x: [N, 2, H, W]  (N = batch*contrast*coil, folded)
        Returns a residual correction, same shape.
        """

        e1 = self.e1(x)
        e2 = self.e2(self.down1(e1))
        m = self.mid(self.down2(e2))

        u2 = self.up2(m)

        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(
                u2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )

        d2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)

        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(
                u1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )

        d1 = self.d1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


class LearnedSensitivityModel(nn.Module):
    """
    Wraps SensitivityRefineUNet into the full sensitivity-estimation
    pipeline: ACS extraction -> per-coil residual refinement (shared
    weights across ALL coils/contrasts/batch) -> RSS-normalization to
    unit-norm sensitivity maps -- i.e. the same three conceptual steps
    as estimate_sens(), except step 2 (raw -> refined) now goes through
    a trainable network instead of being skipped entirely.
    """

    def __init__(self, acs_size, chans=8):

        super().__init__()

        self.acs_size = acs_size
        self.refine = SensitivityRefineUNet(chans=chans)

    def forward(self, komega):
        """
        komega: [B, L, C, H, W] complex, Omega-masked k-space (ACS band
        is always fully sampled within it, regardless of Aleph/Upsilon
        partitioning).

        Returns sens: [B, L, C, H, W] complex, unit-norm (over the coil
        dimension) sensitivity maps.
        """

        B, L, C, H, W = komega.shape

        coil_images = extract_acs_images(komega, self.acs_size)  # [B,L,C,H,W]

        flat = coil_images.reshape(B * L * C, H, W)

        x = complex_to_channels(flat)  # [B*L*C, 2, H, W]

        # Residual correction, zero-initialized (see SensitivityRefineUNet
        # docstring) so training starts exactly at the classical estimate.
        refined = x + self.refine(x)

        refined_complex = channels_to_complex(refined)  # [B*L*C, H, W]
        refined_complex = refined_complex.reshape(B, L, C, H, W)

        refined_complex = torch.complex(
            torch.nan_to_num(refined_complex.real),
            torch.nan_to_num(refined_complex.imag)
        )

        norm = torch.sqrt(
            torch.sum(
                torch.abs(refined_complex) ** 2, dim=2, keepdim=True
            ).clamp_min(1e-8)
        )

        sens = refined_complex / norm

        return torch.complex(
            torch.nan_to_num(sens.real),
            torch.nan_to_num(sens.imag)
        )
