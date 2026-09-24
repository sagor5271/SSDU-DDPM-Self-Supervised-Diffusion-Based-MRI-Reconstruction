"""Time embedding, residual block and the physics-informed U-Net (with internal hard DC)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .fft_utils import channels_to_complex_multi, complex_to_channels_multi
from .sense_ops import hard_dc_multi


class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )

        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)

        if self.dim % 2:
            emb = F.pad(emb, (0, 1))

        return emb


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        groups = min(8, out_ch)
        while groups > 1 and out_ch % groups != 0:
            groups -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)

        self.time = nn.Linear(time_dim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.skip(x)

        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))

        return h + residual


class PhysicsUNet(nn.Module):

    def __init__(self, base=32, time_dim=128, num_contrasts=1):
        super().__init__()

        self.num_contrasts = num_contrasts

        self.time_emb = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = 5 * num_contrasts   # xt (2L) + condition (2L) + mask (L)
        out_ch = 2 * num_contrasts

        self.e1 = ResBlock(in_ch, base, time_dim)

        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.e2 = ResBlock(base * 2, base * 2, time_dim)

        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.e3 = ResBlock(base * 4, base * 4, time_dim)

        self.down3 = nn.Conv2d(base * 4, base * 8, 4, 2, 1)
        self.mid = ResBlock(base * 8, base * 8, time_dim)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1)
        self.d3 = ResBlock(base * 8, base * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.d2 = ResBlock(base * 4, base * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.d1 = ResBlock(base * 2, base, time_dim)

        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, xt, condition, mask, t,
                measured_kspace=None, sens=None, dc_mask=None):

        tnorm = t.float() / max(CFG.T_STEPS - 1, 1)
        te = self.time_mlp(self.time_emb(tnorm))

        z = torch.cat([xt, condition, mask], dim=1)

        e1 = self.e1(z, te)
        e2 = self.e2(self.down1(e1), te)
        e3 = self.e3(self.down2(e2), te)
        m = self.mid(self.down3(e3), te)

        u3 = self.up3(m)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], dim=1), te)

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1), te)

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1), te)

        x0 = self.out(d1)

        if (
            CFG.USE_INTERNAL_DC
            and measured_kspace is not None
            and sens is not None
            and dc_mask is not None
        ):
            image_multi = channels_to_complex_multi(x0, self.num_contrasts)
            image_multi = hard_dc_multi(image_multi, measured_kspace, sens, dc_mask)
            x0 = complex_to_channels_multi(image_multi)

        return x0
