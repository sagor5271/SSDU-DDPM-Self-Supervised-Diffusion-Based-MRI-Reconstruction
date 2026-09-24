"""Learned k-space partitioning (Kadota, Millard & Chiew, Eqs. 5-7) and fixed-probability split."""

import math

import torch
import torch.nn as nn

from .config import CFG
from .masks import split_ssdu


class PartitionThreshold(torch.autograd.Function):
    """Straight-through hard threshold."""

    @staticmethod
    def forward(ctx, z, slope_s):
        ctx.save_for_backward(z)
        ctx.slope_s = slope_s
        return (z > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        s = ctx.slope_s
        grad_approx = torch.sigmoid(s * z) * torch.sigmoid(s * (z - 1.0))
        return grad_output * grad_approx, None


class LearnedPartitioning(nn.Module):

    def __init__(self, num_contrasts, H, init_alpha, slope_t, slope_s):
        super().__init__()
        init_logit = math.log(init_alpha / (1.0 - init_alpha)) / slope_t
        self.logits = nn.Parameter(torch.full((num_contrasts, H), float(init_logit)))
        self.slope_t, self.slope_s = slope_t, slope_s

    def current_probabilities(self):
        return torch.sigmoid(self.slope_t * self.logits).detach()

    def forward(self, omega, acs_mask, seed=None, deterministic=False):
        B, L, H, W = omega.shape
        probs = torch.sigmoid(self.slope_t * self.logits)
        acquired = omega[..., 0] > 0.5
        acs = acs_mask[..., 0] > 0.5
        eligible = acquired & (~acs)
        p = probs[None].expand(B, L, H)

        if deterministic:
            lines = acs.clone()
            scores = p.masked_fill(~eligible, -float("inf"))
            for bi in range(B):
                for li in range(L):
                    n = int(eligible[bi, li].sum())
                    k = int(round(CFG.SSDU_ALPHA * n))
                    if k:
                        idx = torch.topk(scores[bi, li], k=k).indices
                        lines[bi, li, idx] = True
            aleph_lines = lines.float()
        else:
            g = torch.Generator(device=omega.device)
            if seed is not None:
                g.manual_seed(int(seed))
            hard = (torch.rand((B, L, H), device=omega.device, generator=g) < p).float()
            sampled = hard + p - p.detach()
            aleph_lines = acs.float() + (1.0 - acs.float()) * sampled * eligible.float()

        aleph_lines = aleph_lines * acquired.float()
        upsilon_lines = acquired.float() * (1.0 - aleph_lines)

        return (
            aleph_lines[..., None].expand(B, L, H, W),
            upsilon_lines[..., None].expand(B, L, H, W),
            probs,
        )

    def fraction_penalty(self, omega, acs_mask):
        acquired = omega[..., 0] > 0.5
        acs = acs_mask[..., 0] > 0.5
        eligible = (acquired & ~acs).float()
        p = torch.sigmoid(self.slope_t * self.logits)[None].expand_as(eligible)
        frac = (p * eligible).sum(-1) / eligible.sum(-1).clamp_min(1.0)
        return ((frac - CFG.SSDU_ALPHA) ** 2).mean()


def fixed_probability_split_multi(omega, acs_mask, alpha, seed):
    B, L, H, W = omega.shape
    device = omega.device

    aleph_list, upsilon_list = [], []

    for b in range(B):
        for l in range(L):
            split_seed = seed + b * 7919 + l * 104729
            a, u = split_ssdu(
                omega[b, l].detach().cpu(), alpha, CFG.ACS_SIZE, split_seed
            )
            aleph_list.append(a.to(device))
            upsilon_list.append(u.to(device))

    aleph = torch.stack(aleph_list, dim=0).reshape(B, L, H, W)
    upsilon = torch.stack(upsilon_list, dim=0).reshape(B, L, H, W)

    return aleph, upsilon
