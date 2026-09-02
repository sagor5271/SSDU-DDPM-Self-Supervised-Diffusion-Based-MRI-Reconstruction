"""
Self-supervised validation and the main training loop.
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import CFG
from .dataset import FastMRISSDUMultiContrast, collate_fn, move_batch
from .diffusion_model import SSDUDDPM
from .fft_utils import complex_to_channels_multi
from .sense_ops import sense_adjoint_multi, hard_dc_multi


# ============================================================================
# SELF-SUPERVISED VALIDATION
#
# Sensitivity maps come from model.get_sens(komega) (learned or fixed,
# per CFG). Uses the SAME partitioning mechanism as training (learned or
# fixed). Reproducibility comes entirely from the isolated,
# CFG.VAL_SEED-seeded generator used for the noise draw below -- no
# global RNG state is touched.
# ============================================================================

@torch.no_grad()
def validate(model, loader, max_slices):

    model.eval()

    total = 0.0
    count = 0

    val_gen = torch.Generator(device=CFG.DEVICE)
    val_gen.manual_seed(CFG.VAL_SEED)

    for batch in loader:

        if count >= max_slices:
            break

        batch = move_batch(batch)

        n = min(batch["komega"].shape[0], max_slices - count)

        komega = batch["komega"][:n]
        omega = batch["omega"][:n]

        B = n

        sens = model.get_sens(komega)

        aleph, upsilon = model.get_partition(omega, step_seed=123456)

        kaleph = komega * aleph.unsqueeze(2)
        kupsilon = komega * upsilon.unsqueeze(2)

        condition_complex = sense_adjoint_multi(kaleph, sens)
        condition = complex_to_channels_multi(condition_complex)

        t_value = CFG.T_STEPS // 2
        t = torch.full((B,), t_value, dtype=torch.long, device=CFG.DEVICE)

        x0_complex = hard_dc_multi(condition_complex, kaleph, sens, aleph)
        x0 = complex_to_channels_multi(x0_complex)

        noise = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=val_gen
        )
        xt = model.q_sample(x0, t, noise)

        # PhysicsUNet signature is: (xt, condition, mask, t, ...).
        # Upsilon is a self-supervised TARGET only and must never be
        # passed as an extra positional network argument.
        x0_pred = model.generator(
            xt, condition, aleph, t,
            measured_kspace=kaleph, sens=sens, dc_mask=aleph
        )

        loss, _ = model.ssdu_loss(x0_pred, kupsilon, upsilon, sens)

        total += loss.item() * n
        count += n

    model.train()

    return total / max(count, 1)


# ============================================================================
# TRAIN
# ============================================================================

def train(train_files, val_files):

    train_ds = FastMRISSDUMultiContrast(train_files, seed=100, training=True)
    val_ds = FastMRISSDUMultiContrast(val_files, seed=10000, training=True)

    print("Train slices (paired across contrasts):", len(train_ds))
    print("Val slices (paired across contrasts)  :", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=CFG.TRAIN_DROP_LAST,
        collate_fn=collate_fn,
        persistent_workers=CFG.PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=CFG.VAL_DROP_LAST,
        collate_fn=collate_fn,
        persistent_workers=CFG.PERSISTENT_WORKERS
    )

    model = SSDUDDPM(num_contrasts=CFG.NUM_CONTRASTS).to(CFG.DEVICE)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    partition_params = sum(p.numel() for p in model.partitioner.parameters())
    sens_params = sum(p.numel() for p in model.sens_model.parameters())

    print(f"Trainable parameters (total)      : {params:,}")
    print(f"  of which partitioning parameters: {partition_params:,}")
    print(f"  of which SME (sensitivity) params: {sens_params:,}")

    optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    best_val = float("inf")

    best_path = os.path.join(
        CFG.CHECKPOINT_DIR,
        "best_ssdu_ddpm_audited_v4.pt"
    )

    for epoch in range(CFG.EPOCHS):

        train_ds.set_epoch(epoch)
        val_ds.set_epoch(0)

        model.train()

        running = 0.0
        running_aleph_frac = 0.0
        t0 = time.time()

        for step_idx, batch in enumerate(train_loader):

            batch = move_batch(batch)

            step_seed = (
                (epoch * 1000003) + (step_idx * 9973) + 777
            )

            out = model.train_step(
                batch["komega"], batch["omega"], step_seed
            )

            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)

            optimizer.step()

            running += loss.item()
            running_aleph_frac += out["aleph_fraction"]

            if (step_idx + 1) % 25 == 0:

                d = step_idx + 1

                print(
                    f"Epoch {epoch + 1}/{CFG.EPOCHS} | "
                    f"Step {d}/{len(train_loader)} | "
                    f"SSDU-loss {running / d:.8f} | "
                    f"mean Aleph frac {running_aleph_frac / d:.4f}"
                )

        scheduler.step()

        epoch_loss = running / max(len(train_loader), 1)
        elapsed = (time.time() - t0) / 60.0

        print()
        print(f"Epoch {epoch + 1} finished in {elapsed:.2f} min")
        print(f"Train SSDU loss: {epoch_loss:.8f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        if CFG.USE_LEARNED_PARTITIONING:

            probs = model.partitioner.current_probabilities()  # [L,H]

            for c, label in enumerate(CFG.CONTRAST_ACQUISITIONS):

                r_tilde = probs[c].mean().item()

                print(
                    f"  Learned partition R~ for {label}: {r_tilde:.4f} "
                    f"(avg per-line Aleph probability, non-ACS lines)"
                )

        val_loss = validate(model, val_loader, min(len(val_ds), CFG.VAL_SLICES))

        print(f"Val SSDU loss: {val_loss:.8f}")

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val": best_val
                },
                best_path
            )

            print("BEST SELF-SUPERVISED CHECKPOINT SAVED.")

        print()

    return model, best_path
