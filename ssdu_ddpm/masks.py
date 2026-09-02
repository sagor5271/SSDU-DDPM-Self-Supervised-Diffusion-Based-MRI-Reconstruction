"""
Cartesian undersampling mask generation, the fixed ACS line mask, and
the fixed-probability SSDU Aleph/Upsilon split.
"""

import numpy as np
import torch


# ============================================================================
# CARTESIAN MASK
# ============================================================================

def create_mask(H, W, acceleration, center_fraction, seed):

    rng = np.random.RandomState(seed)

    mask_1d = np.zeros(H, dtype=np.float32)

    center_lines = max(1, int(round(H * center_fraction)))
    center = H // 2

    start = max(0, center - center_lines // 2)
    end = min(H, start + center_lines)

    mask_1d[start:end] = 1.0

    target_lines = max(center_lines, int(round(H / acceleration)))
    remaining = max(0, target_lines - int(mask_1d.sum()))

    candidates = np.where(mask_1d == 0)[0]

    if remaining > 0:

        chosen = rng.choice(
            candidates,
            size=min(remaining, len(candidates)),
            replace=False
        )

        mask_1d[chosen] = 1.0

    return torch.from_numpy(np.tile(mask_1d[:, None], (1, W))).float()


def build_acs_line_mask(H, W, acs_size):
    """
    Deterministic [H,W] 0/1 mask marking the ACS (auto-calibration)
    region -- the same fixed center band used everywhere else in this
    package, but expressed as a mask so it can be combined with the
    learned partitioning module's tensors.
    """

    center = H // 2
    half = acs_size // 2

    start = max(0, center - half)
    end = min(H, center + half)

    m = torch.zeros(H, dtype=torch.float32)
    m[start:end] = 1.0

    return m[:, None].expand(H, W).contiguous()


# ============================================================================
# FIXED-PROBABILITY SSDU SPLIT (fallback when USE_LEARNED_PARTITIONING
#     is False)
# ============================================================================

def split_ssdu(omega, alpha, acs_size, seed):
    """
    Omega = Aleph + Upsilon. ACS is forced into Aleph. alpha: fraction of
    non-ACS acquired lines assigned to Aleph.
    """

    H, W = omega.shape

    rng = np.random.RandomState(seed)

    lines = omega[:, 0].cpu().numpy().astype(np.float32)

    center = H // 2
    half = acs_size // 2

    acs_start = max(0, center - half)
    acs_end = min(H, center + half)

    acquired = np.where(lines > 0)[0]

    non_acs = [h for h in acquired if not (acs_start <= h < acs_end)]

    keep_non_acs = int(round(alpha * len(non_acs)))
    keep_non_acs = min(keep_non_acs, len(non_acs))

    if keep_non_acs > 0:
        kept = rng.choice(non_acs, size=keep_non_acs, replace=False)
    else:
        kept = np.array([], dtype=int)

    aleph_1d = np.zeros_like(lines)
    aleph_1d[acs_start:acs_end] = lines[acs_start:acs_end]
    aleph_1d[kept] = 1.0

    upsilon_1d = np.clip(lines - aleph_1d, 0.0, 1.0)

    aleph = torch.from_numpy(np.tile(aleph_1d[:, None], (1, W))).float()
    upsilon = torch.from_numpy(np.tile(upsilon_1d[:, None], (1, W))).float()

    if not torch.equal(omega.cpu(), aleph + upsilon):
        raise RuntimeError("SSDU split failed: Omega != Aleph + Upsilon")

    if torch.any(upsilon[acs_start:acs_end] > 0):
        raise RuntimeError("ACS leaked into Upsilon.")

    return aleph, upsilon
