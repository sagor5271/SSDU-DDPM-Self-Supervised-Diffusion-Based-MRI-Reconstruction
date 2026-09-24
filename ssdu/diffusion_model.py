"""SSDU-DDPM: training step (Aleph->Upsilon k-space loss) and respaced DDPM sampler."""

import numpy as np
import torch
import torch.nn as nn

from .config import CFG
from .fft_utils import channels_to_complex_multi, complex_to_channels_multi
from .masks import build_acs_line_mask
from .network import PhysicsUNet
from .partitioning import LearnedPartitioning, fixed_probability_split_multi
from .sense_ops import sense_adjoint_multi, sense_forward_multi, hard_dc_multi
from .sensitivity import LearnedSensitivityModel, estimate_sens_multi_fixed


def linear_schedule(T, beta_start, beta_end):
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = torch.cat([torch.ones(1, dtype=torch.float32), abar[:-1]], dim=0)
    return betas, alphas, abar, abar_prev


class SSDUDDPM(nn.Module):

    def __init__(self, num_contrasts):
        super().__init__()

        self.num_contrasts = num_contrasts

        betas, alphas, abar, abar_prev = linear_schedule(
            CFG.T_STEPS, CFG.BETA_START, CFG.BETA_END
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("abar_prev", abar_prev)

        posterior_var = betas * (1.0 - abar_prev) / (1.0 - abar).clamp_min(1e-8)
        posterior_mean_coef1 = betas * torch.sqrt(abar_prev) / (1.0 - abar).clamp_min(1e-8)
        posterior_mean_coef2 = (
            (1.0 - abar_prev) * torch.sqrt(alphas) / (1.0 - abar).clamp_min(1e-8)
        )

        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        self.generator = PhysicsUNet(
            CFG.BASE_CH, CFG.TIME_DIM, num_contrasts=num_contrasts
        )

        self.partitioner = LearnedPartitioning(
            num_contrasts=num_contrasts,
            H=CFG.IMAGE_SIZE,
            init_alpha=CFG.SSDU_ALPHA,
            slope_t=CFG.PARTITION_SLOPE_T,
            slope_s=CFG.PARTITION_SLOPE_S,
        )

        self.sens_model = LearnedSensitivityModel(
            acs_size=CFG.ACS_SIZE, chans=CFG.SENS_UNET_CHANS
        )

        self.register_buffer(
            "acs_line_mask",
            build_acs_line_mask(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, CFG.ACS_SIZE),
        )

    # ------------------------------------------------------------------
    def get_sens(self, komega):
        if CFG.USE_LEARNED_SENSITIVITY:
            return self.sens_model(komega)
        return estimate_sens_multi_fixed(komega, CFG.ACS_SIZE)

    def get_partition(self, omega, step_seed):
        B, L, H, W = omega.shape
        acs_mask = self.acs_line_mask[None, None, :, :].expand(B, L, H, W)

        if CFG.USE_LEARNED_PARTITIONING:
            aleph, upsilon, _ = self.partitioner(
                omega, acs_mask, seed=step_seed, deterministic=(not self.training)
            )
        else:
            aleph, upsilon = fixed_probability_split_multi(
                omega, acs_mask, CFG.SSDU_ALPHA, step_seed
            )

        return aleph, upsilon

    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        ab = self.abar[t][:, None, None, None]
        return torch.sqrt(ab) * x0 + torch.sqrt((1.0 - ab).clamp_min(1e-8)) * noise

    def respaced_posterior(self, x0, xt, t, s):
        ab_t = self.abar[t][:, None, None, None]
        ab_s = self.abar[s][:, None, None, None]

        alpha_ratio = (ab_t / ab_s.clamp_min(1e-8)).clamp_min(1e-8)
        denom = (1.0 - ab_t).clamp_min(1e-8)

        c1 = torch.sqrt(ab_s) * (1.0 - alpha_ratio) / denom
        c2 = torch.sqrt(alpha_ratio) * (1.0 - ab_s) / denom

        mean = c1 * x0 + c2 * xt
        var = ((1.0 - ab_s) / denom * (1.0 - alpha_ratio)).clamp_min(0.0)
        var = torch.where(s[:, None, None, None] == 0, torch.zeros_like(var), var)

        return mean, var

    # ------------------------------------------------------------------
    def ssdu_loss(self, x0_pred, kupsilon, upsilon, sens):
        image = channels_to_complex_multi(x0_pred, self.num_contrasts)
        predicted_kspace = sense_forward_multi(image, sens)

        mask = upsilon.unsqueeze(2)

        error = (predicted_kspace - kupsilon) * mask
        target = kupsilon * mask

        l1_num = torch.abs(error).sum(dim=(2, 3, 4))
        l1_den = torch.abs(target).sum(dim=(2, 3, 4)).clamp_min(1e-8)
        l1_per = l1_num / l1_den

        l2_num = torch.sqrt((error.abs() ** 2).sum(dim=(2, 3, 4)).clamp_min(1e-12))
        l2_den = torch.sqrt((target.abs() ** 2).sum(dim=(2, 3, 4)).clamp_min(1e-12))
        l2_per = l2_num / l2_den

        w = CFG.SSDU_L2_WEIGHT
        per_sample_per_contrast = (1.0 - w) * l1_per + w * l2_per
        per_sample = per_sample_per_contrast.mean(dim=1)

        return per_sample.mean(), per_sample.detach()

    def train_step(self, komega, omega, step_seed):
        B = komega.shape[0]
        device = komega.device

        sens = self.get_sens(komega)
        aleph, upsilon = self.get_partition(omega, step_seed)

        kaleph = komega * aleph.unsqueeze(2)
        kupsilon = komega * upsilon.unsqueeze(2)

        condition_complex = sense_adjoint_multi(kaleph, sens)
        condition = complex_to_channels_multi(condition_complex)

        t = torch.randint(0, CFG.T_STEPS, (B,), device=device, dtype=torch.long)

        x0_complex = hard_dc_multi(condition_complex, kaleph, sens, aleph)
        x0 = complex_to_channels_multi(x0_complex)

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        x0_pred = self.generator(
            xt, condition, aleph, t,
            measured_kspace=kaleph, sens=sens, dc_mask=aleph,
        )

        ssdu_loss, per_sample = self.ssdu_loss(x0_pred, kupsilon, upsilon, sens)

        partition_penalty = torch.zeros((), device=device)
        if CFG.USE_LEARNED_PARTITIONING:
            acs_mask = self.acs_line_mask[None, None].expand_as(omega)
            partition_penalty = self.partitioner.fraction_penalty(omega, acs_mask)

        loss = ssdu_loss + CFG.PARTITION_FRACTION_WEIGHT * partition_penalty

        return {
            "loss": loss,
            "ssdu_loss": ssdu_loss.detach(),
            "partition_penalty": partition_penalty.detach(),
            "x0_pred": x0_pred,
            "loss_per_sample": per_sample,
            "aleph_fraction": aleph.mean().detach().item(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, condition, measured_kspace, sens, aleph, omega):
        """
        condition       : [B, 2L, H, W]   Aleph-derived zero-filled input
        measured_kspace : [B, L, C, H, W] FULL Omega k-space
        sens            : [B, L, C, H, W]
        aleph           : [B, L, H, W]    mask-channel input (matches training)
        omega           : [B, L, H, W]    FULL acquired mask, used ONLY for data consistency
        """
        B = condition.shape[0]
        device = condition.device
        L = self.num_contrasts

        x = torch.randn(B, 2 * L, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, device=device)

        steps = int(CFG.TEST_SAMPLE_STEPS)
        raw_timesteps = np.linspace(CFG.T_STEPS - 1, 0, num=steps, dtype=np.int64)
        timesteps = np.unique(raw_timesteps)[::-1].copy()

        for i, t_value in enumerate(timesteps):

            t = torch.full((B,), int(t_value), dtype=torch.long, device=device)

            x0_pred = self.generator(
                x, condition, aleph, t,
                measured_kspace=measured_kspace, sens=sens, dc_mask=omega,
            )

            x0_complex = channels_to_complex_multi(x0_pred, L)
            x0_complex = hard_dc_multi(x0_complex, measured_kspace, sens, omega)
            x0_pred = complex_to_channels_multi(x0_complex)

            if i == len(timesteps) - 1:
                return x0_pred

            next_t_value = int(timesteps[i + 1])
            s = torch.full((B,), next_t_value, dtype=torch.long, device=device)

            mean, var = self.respaced_posterior(x0_pred, x, t, s)

            if next_t_value > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var.clamp_min(1e-20)) * noise
            else:
                x = mean

        return x
