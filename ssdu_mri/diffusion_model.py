"""
DDPM noise schedule and the SSDUDDPM model: forward diffusion, the
generalized respaced reverse posterior, the SSDU self-supervised loss,
the training step, and reproducible reverse-diffusion sampling.
"""

import numpy as np
import torch
import torch.nn as nn

from .config import CFG
from .fft_utils import channels_to_complex_multi, complex_to_channels_multi
from .sense_ops import sense_adjoint_multi, sense_forward_multi, hard_dc_multi
from .sensitivity import estimate_sens_multi_fixed, LearnedSensitivityModel
from .partitioning import LearnedPartitioning, fixed_probability_split_multi
from .masks import build_acs_line_mask
from .network import PhysicsUNet


# ============================================================================
# DDPM SCHEDULE
# ============================================================================

def linear_schedule(T, beta_start, beta_end):

    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    abar_prev = torch.cat(
        [torch.ones(1, dtype=torch.float32), abar[:-1]], dim=0
    )

    return betas, alphas, abar, abar_prev


# ============================================================================
# SSDU DDPM  (multi-contrast + learned partitioning + learned SME)
# ============================================================================

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

        posterior_var = (
            betas * (1.0 - abar_prev) / (1.0 - abar).clamp_min(1e-8)
        )

        posterior_mean_coef1 = (
            betas * torch.sqrt(abar_prev) / (1.0 - abar).clamp_min(1e-8)
        )

        posterior_mean_coef2 = (
            (1.0 - abar_prev) * torch.sqrt(alphas) / (1.0 - abar).clamp_min(1e-8)
        )

        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        self.generator = PhysicsUNet(
            CFG.BASE_CH, CFG.TIME_DIM, num_contrasts=num_contrasts
        )

        # ---------------------------------------------------------------
        # LEARNED PARTITIONING MODULE. A normal nn.Module, so its
        # parameters (self.partitioner.logits) are automatically picked
        # up by `model.parameters()` and trained end-to-end by the SAME
        # optimizer as the reconstruction network -- no separate
        # training loop needed.
        # ---------------------------------------------------------------

        self.partitioner = LearnedPartitioning(
            num_contrasts=num_contrasts,
            H=CFG.IMAGE_SIZE,
            init_alpha=CFG.SSDU_ALPHA,
            slope_t=CFG.PARTITION_SLOPE_T,
            slope_s=CFG.PARTITION_SLOPE_S
        )

        # ---------------------------------------------------------------
        # LEARNED SENSITIVITY MAP ESTIMATION MODULE. Also a normal
        # nn.Module -- self.sens_model.refine.* parameters are likewise
        # automatically included in model.parameters() and trained by
        # the SAME optimizer. Always constructed (even if
        # USE_LEARNED_SENSITIVITY=False) so the toggle can be flipped
        # without changing the checkpoint's parameter structure; simply
        # unused (and untrained, since it never appears in the forward
        # graph) when disabled.
        # ---------------------------------------------------------------

        self.sens_model = LearnedSensitivityModel(
            acs_size=CFG.ACS_SIZE,
            chans=CFG.SENS_UNET_CHANS
        )

        self.register_buffer(
            "acs_line_mask",
            build_acs_line_mask(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, CFG.ACS_SIZE)
        )

    # ========================================================================
    # SENSITIVITY HELPER -- dispatches to learned or fixed estimation.
    #
    # komega: [B, L, C, H, W] complex, Omega-masked k-space. Used for
    # EVERY sensitivity map that participates in the actual
    # reconstruction physics (condition, hard-DC, SSDU loss, sampling).
    # The Dataset's separately-precomputed "sens" (always the fixed
    # formula) remains reserved for the ZF baseline metric and the
    # test-time GT target only.
    # ========================================================================

    def get_sens(self, komega):
        """
        komega: [B, L, C, H, W] complex
        Returns sens: [B, L, C, H, W] complex
        """

        if CFG.USE_LEARNED_SENSITIVITY:

            return self.sens_model(komega)

        else:

            return estimate_sens_multi_fixed(komega, CFG.ACS_SIZE)

    # ========================================================================
    # PARTITION HELPER -- dispatches to learned or fixed-probability split
    # ========================================================================

    def get_partition(self, omega, step_seed):
        """
        omega: [B, L, H, W]
        Returns aleph, upsilon: [B, L, H, W]
        """

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

    # ========================================================================
    # FORWARD DIFFUSION
    # ========================================================================

    def q_sample(self, x0, t, noise):

        ab = self.abar[t][:, None, None, None]

        return (
            torch.sqrt(ab) * x0
            + torch.sqrt((1.0 - ab).clamp_min(1e-8)) * noise
        )

    # ========================================================================
    # GENERALIZED RESPACED POSTERIOR (operates elementwise over the
    # channel dimension via broadcasting, so it is completely agnostic
    # to whether that dimension holds 2 or 2L channels)
    # ========================================================================

    def respaced_posterior(self, x0, xt, t, s):

        ab_t = self.abar[t][:, None, None, None]
        ab_s = self.abar[s][:, None, None, None]

        alpha_ratio = (ab_t / ab_s.clamp_min(1e-8)).clamp_min(1e-8)
        denom = (1.0 - ab_t).clamp_min(1e-8)

        c1 = torch.sqrt(ab_s) * (1.0 - alpha_ratio) / denom
        c2 = torch.sqrt(alpha_ratio) * (1.0 - ab_s) / denom

        mean = c1 * x0 + c2 * xt

        var = ((1.0 - ab_s) / denom * (1.0 - alpha_ratio)).clamp_min(0.0)

        var = torch.where(
            s[:, None, None, None] == 0, torch.zeros_like(var), var
        )

        return mean, var

    # ========================================================================
    # SSDU LOSS -- multi-contrast, normalized L1+L2 mix, averaged over
    # contrasts (1/L sum, Eq. 3/8a of the multi-contrast paper).
    # ========================================================================

    def ssdu_loss(self, x0_pred, kupsilon, upsilon, sens):
        """
        x0_pred : [B, 2L, H, W]
        kupsilon: [B, L, C, H, W]
        upsilon : [B, L, H, W]
        sens    : [B, L, C, H, W]
        """

        image = channels_to_complex_multi(x0_pred, self.num_contrasts)
        predicted_kspace = sense_forward_multi(image, sens)  # [B,L,C,H,W]

        mask = upsilon.unsqueeze(2)  # [B,L,1,H,W]

        error = (predicted_kspace - kupsilon) * mask
        target = kupsilon * mask

        l1_num = torch.abs(error).sum(dim=(2, 3, 4))
        l1_den = torch.abs(target).sum(dim=(2, 3, 4)).clamp_min(1e-8)
        l1_per = l1_num / l1_den  # [B, L]

        l2_num = torch.sqrt((error.abs() ** 2).sum(dim=(2, 3, 4)).clamp_min(1e-12))
        l2_den = torch.sqrt((target.abs() ** 2).sum(dim=(2, 3, 4)).clamp_min(1e-12))
        l2_per = l2_num / l2_den  # [B, L]

        w = CFG.SSDU_L2_WEIGHT

        per_sample_per_contrast = (1.0 - w) * l1_per + w * l2_per  # [B, L]

        # Average over contrasts (1/L sum, matching the paper).
        per_sample = per_sample_per_contrast.mean(dim=1)  # [B]

        loss = per_sample.mean()

        return loss, per_sample.detach()

    # ========================================================================
    # TRAIN STEP
    #
    # Sensitivity maps for the reconstruction pipeline come from
    # self.get_sens(komega) (learned or fixed, per CFG).
    # ========================================================================

    def train_step(self, komega, omega, step_seed):
        """
        komega: [B, L, C, H, W]  Omega-masked, normalized k-space
        omega : [B, L, H, W]     acquired-line mask
        step_seed: int, used only by the fixed-probability partition
                   fallback split
        """

        B = komega.shape[0]
        device = komega.device

        sens = self.get_sens(komega)  # [B,L,C,H,W], learned or fixed

        aleph, upsilon = self.get_partition(omega, step_seed)

        kaleph = komega * aleph.unsqueeze(2)
        kupsilon = komega * upsilon.unsqueeze(2)

        condition_complex = sense_adjoint_multi(kaleph, sens)  # [B,L,H,W]
        condition = complex_to_channels_multi(condition_complex)  # [B,2L,H,W]

        t = torch.randint(0, CFG.T_STEPS, (B,), device=device, dtype=torch.long)

        # Training x0 is constructed ONLY from Aleph.
        x0_complex = hard_dc_multi(condition_complex, kaleph, sens, aleph)
        x0 = complex_to_channels_multi(x0_complex)

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        x0_pred = self.generator(
            xt,
            condition,
            aleph,
            t,
            measured_kspace=kaleph,
            sens=sens,
            dc_mask=aleph
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
            "aleph_fraction": aleph.mean().detach().item()
        }

    # ========================================================================
    # RESPACED DDPM SAMPLING (test-time) -- reproducible.
    #
    # NOTE ON MASK/DC CONVENTION: at test time, this method is called
    # with mask=omega (the FULL acquired set) for both the network's
    # mask input channel and internal/final DC -- i.e. it conditions and
    # enforces DC on everything measured, not just an Aleph subset. This
    # differs from train_step(), which uses the Aleph subset for both.
    # If your intended design is "condition on the same Aleph-derived
    # representation as training, DC on full Omega only at the final
    # step" (as some SSDU papers describe), you will need to pass an
    # Aleph-derived condition/mask through the loop and only substitute
    # full-Omega DC on the last iteration -- verify which behavior you
    # actually want before reporting results.
    #
    # Every random draw in this reverse chain (the initial x_T and every
    # intermediate posterior noise) goes through an explicit
    # torch.Generator seeded with CFG.TEST_SEED, so repeated test runs
    # on the same checkpoint produce identical reconstructions/metrics.
    # ========================================================================

    @torch.no_grad()
    def sample(self, condition, measured_kspace, sens, omega):
        """
        condition       : [B, 2L, H, W]   Aleph-derived zero-filled input
        measured_kspace : [B, L, C, H, W] FULL Omega k-space
        sens            : [B, L, C, H, W]
        omega           : [B, L, H, W]
        """

        B = condition.shape[0]
        device = condition.device
        L = self.num_contrasts

        sample_gen = torch.Generator(device=device)
        sample_gen.manual_seed(CFG.TEST_SEED)

        x = torch.randn(
            (B, 2 * L, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            device=device, generator=sample_gen
        )

        steps = int(CFG.TEST_SAMPLE_STEPS)

        raw_timesteps = np.linspace(CFG.T_STEPS - 1, 0, num=steps, dtype=np.int64)
        timesteps = np.unique(raw_timesteps)[::-1].copy()

        for i, t_value in enumerate(timesteps):

            t = torch.full((B,), int(t_value), dtype=torch.long, device=device)

            x0_pred = self.generator(
                x,
                condition,
                omega,
                t,
                measured_kspace=measured_kspace,
                sens=sens,
                dc_mask=omega
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

                noise = torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=sample_gen
                )
                x = mean + torch.sqrt(var.clamp_min(1e-20)) * noise

            else:

                x = mean

        return x
