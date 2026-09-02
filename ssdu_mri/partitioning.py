"""
Learned k-space partitioning (Section II.B, Eqs. 5-7) and the fixed-
probability SSDU split fallback.
"""

import math

import torch
import torch.nn as nn

from .config import CFG
from .masks import split_ssdu


class PartitionThreshold(torch.autograd.Function):
    """
    Forward:  M = (Z > 0)                                        [Eq. 6]
    Backward: dM/dZ ~= sigmoid(s*Z) * sigmoid(s*(Z - 1))          [Eq. 7]

    This is the paper's straight-through gradient approximation for the
    hard threshold used to sample the partitioning mask, implemented as
    a custom autograd.Function so the forward pass is an EXACT 0/1 mask
    (no soft blending), while gradients still flow to the underlying
    learnable partitioning logits through the smooth approximation.

    Called by LearnedPartitioning.forward() below.
    """

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
    """
    Seeded learned SSDU partitioner with deterministic evaluation.

    Sampling (forward(), deterministic=False): draws a reparameterized
    Bernoulli(p) sample via z = p - u, u ~ Uniform(0,1)
    (so P(z > 0) = P(u < p) = p, exactly the intended per-line inclusion
    probability), then calls PartitionThreshold.apply(z, slope_s) for
    the hard 0/1 forward pass with the paper's Eq. 7 smooth gradient
    approximation on the backward pass -- this is what Section II.B
    actually specifies.

    Evaluation (deterministic=True): a seeded, reproducible top-k
    selection over the learned per-line probabilities, rather than a
    stochastic draw.
    """

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
            # Reparameterized hard Bernoulli(p) sample: z = p - u,
            # u ~ Uniform(0,1) => P(z > 0) = P(u < p) = p exactly.
            # PartitionThreshold.apply gives the exact 0/1 forward mask
            # AND the intended smooth straight-through backward (Eq. 7).
            u = torch.rand((B, L, H), device=omega.device, generator=g)
            z = p - u
            sampled = PartitionThreshold.apply(z, self.slope_s)
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
    """
    Fallback path used when CFG.USE_LEARNED_PARTITIONING = False.
    Reuses the ORIGINAL, non-differentiable split_ssdu() once per
    (batch, contrast) slice.

    omega, acs_mask: [B, L, H, W]  (acs_mask unused here -- split_ssdu
    recomputes the ACS band internally from CFG.ACS_SIZE, kept as an
    argument only so this function has the same signature as the
    learned-partitioning path).
    """

    B, L, H, W = omega.shape
    device = omega.device

    aleph_list = []
    upsilon_list = []

    for b in range(B):
        for l in range(L):

            split_seed = seed + b * 7919 + l * 104729

            a, u = split_ssdu(
                omega[b, l].detach().cpu(),
                alpha,
                CFG.ACS_SIZE,
                split_seed
            )

            aleph_list.append(a.to(device))
            upsilon_list.append(u.to(device))

    aleph = torch.stack(aleph_list, dim=0).reshape(B, L, H, W)
    upsilon = torch.stack(upsilon_list, dim=0).reshape(B, L, H, W)

    return aleph, upsilon
