"""
Entry point: discover data, verify coils, train, load the best
checkpoint, run test-time evaluation, and print final results.

Usage:
    python main.py

Edit ssdu_mri/config.py (the CFG class) to change hyperparameters,
dataset paths, and feature toggles before running.
"""

import os

import torch
from torch.utils.data import DataLoader

from ssdu_mri.config import CFG
from ssdu_mri.reproducibility import seed_everything
from ssdu_mri.data_discovery import get_files_multi_contrast, verify_original_coils_multi
from ssdu_mri.dataset import FastMRISSDUMultiContrast, collate_fn
from ssdu_mri.train import train
from ssdu_mri.test import test_model


def print_startup_banner():

    print("=" * 80)
    print("PURE SELF-SUPERVISED SSDU CONDITIONAL DDPM MRI")
    print("RESPACED SAMPLER VERSION -- ORIGINAL 15-COIL (NO COMPRESSION)")
    print("=" * 80)

    print("Device                 :", CFG.DEVICE)
    print("Code version           :", CFG.CODE_VERSION)
    print("Image size             :", CFG.IMAGE_SIZE)
    print("Original coils         :", CFG.ORIGINAL_COILS)
    print("Contrasts              :", CFG.CONTRAST_ACQUISITIONS)
    print("Num contrasts (L)      :", CFG.NUM_CONTRASTS)
    print("Acceleration           :", CFG.ACCELERATION)
    print("SSDU alpha (init)      :", CFG.SSDU_ALPHA)
    print("Learned partitioning   :", CFG.USE_LEARNED_PARTITIONING)
    if CFG.USE_LEARNED_PARTITIONING:
        print("  slope_t              :", CFG.PARTITION_SLOPE_T)
        print("  slope_s              :", CFG.PARTITION_SLOPE_S)
    print("Learned sensitivity    :", CFG.USE_LEARNED_SENSITIVITY)
    if CFG.USE_LEARNED_SENSITIVITY:
        print("  SME U-Net channels   :", CFG.SENS_UNET_CHANS)
    print("ACS size               :", CFG.ACS_SIZE)
    print("Train T_STEPS          :", CFG.T_STEPS)
    print("Test steps             :", CFG.TEST_SAMPLE_STEPS)
    print("Batch size             :", CFG.BATCH_SIZE)
    print("Epochs                 :", CFG.EPOCHS)

    print()

    if CFG.NUM_CONTRASTS > 1:
        print("MULTI-CONTRAST CAVEAT  : fastMRI knee PD/PD-FS pairing is BY")
        print("                         SLICE INDEX ONLY, not by patient --")
        print("                         see README.md for details.")

    print(
        f"LOSS                   : NORMALIZED L1+L2 MIX "
        f"(L2 weight={CFG.SSDU_L2_WEIGHT}), averaged over {CFG.NUM_CONTRASTS} contrast(s)"
    )

    active_features = []
    if CFG.NUM_CONTRASTS > 1:
        active_features.append("multi-contrast")
    if CFG.USE_LEARNED_PARTITIONING:
        active_features.append("learned k-space partitioning")
    if CFG.USE_LEARNED_SENSITIVITY:
        active_features.append("learned sensitivity map estimation (SME)")

    print(
        "ACTIVE OPTIONAL FEATURES: "
        + (", ".join(active_features) if active_features else "NONE (single-contrast, fixed split, fixed sensitivity)")
    )

    print(
        "SENSITIVITY SOURCE     : "
        + (
            "LEARNED (E2E-VarNet-style SME network, ZF baseline/GT still fixed)"
            if CFG.USE_LEARNED_SENSITIVITY
            else "FIXED (classical ACS+RSS formula, everywhere)"
        )
    )
    print(
        "PARTITIONING SOURCE    : "
        + (
            "LEARNED (Eqs. 5-7, deterministic top-k at eval)"
            if CFG.USE_LEARNED_PARTITIONING
            else f"FIXED (random split, alpha={CFG.SSDU_ALPHA})"
        )
    )
    print("GAN                    : DISABLED")
    print("REVERSE SAMPLER        : GENERALIZED RESPACED DDPM POSTERIOR (seeded, reproducible)")
    print("DC                     : FULL ACQUIRED OMEGA")
    print("=" * 80)


def main():

    seed_everything(CFG.SPLIT_SEED)

    print_startup_banner()

    # ========================================================================
    # DATA
    # ========================================================================

    print("\nDISCOVERING DATA")

    train_files, val_files, test_files = get_files_multi_contrast()

    for label in CFG.CONTRAST_ACQUISITIONS:
        print(
            f"{label}: train={len(train_files[label])} "
            f"val={len(val_files[label])} test={len(test_files[label])}"
        )

    all_files_dict = {
        label: (
            train_files[label] + val_files[label] + test_files[label]
        )
        for label in CFG.CONTRAST_ACQUISITIONS
    }

    verify_original_coils_multi(all_files_dict)

    # ========================================================================
    # TRAIN
    # ========================================================================

    model, best_path = train(train_files, val_files)

    # ========================================================================
    # LOAD BEST CHECKPOINT
    # ========================================================================

    if os.path.exists(best_path):

        ckpt = torch.load(best_path, map_location=CFG.DEVICE)
        model.load_state_dict(ckpt["model"])

        print("=" * 80)
        print("BEST CHECKPOINT LOADED")
        print("Epoch:", ckpt["epoch"])
        print("Best validation SSDU:", ckpt["best_val"])
        print("=" * 80)

    # ========================================================================
    # TEST
    # ========================================================================

    test_ds = FastMRISSDUMultiContrast(test_files, seed=50000, training=False)
    test_ds.set_epoch(0)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=CFG.TEST_DROP_LAST,
        collate_fn=collate_fn,
        persistent_workers=CFG.PERSISTENT_WORKERS
    )

    print("Test slices available (paired):", len(test_ds))
    print("Test slices evaluated         :", min(len(test_ds), CFG.TEST_SLICES))

    results = test_model(model, test_loader, CFG.TEST_SLICES)

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for label in CFG.CONTRAST_ACQUISITIONS:

        r = results[label]

        print(f"\n--- Contrast: {label} ---")
        print("SENSE ZERO-FILLED BASELINE (classical fixed sensitivity)")
        print(f"  PSNR : {r['zf_psnr']:.4f} dB")
        print(f"  SSIM : {r['zf_ssim']:.4f}")
        print(f"  NMSE : {r['zf_nmse']:.6f}")

        print("PURE SSDU CONDITIONAL DDPM")
        print(f"  PSNR : {r['psnr']:.4f} dB")
        print(f"  SSIM : {r['ssim']:.4f}")
        print(f"  NMSE : {r['nmse']:.6f}")

        print("IMPROVEMENT OVER SENSE-ZF")
        print(f"  PSNR gain: {r['psnr'] - r['zf_psnr']:.4f} dB")
        print(f"  SSIM gain: {r['ssim'] - r['zf_ssim']:.4f}")
        print(f"  NMSE change: {r['nmse'] - r['zf_nmse']:.6f}")

    overall = results["overall"]

    print("\n--- OVERALL (averaged across all contrasts) ---")
    print(f"  ZF   PSNR/SSIM/NMSE : {overall['zf_psnr']:.4f} / {overall['zf_ssim']:.4f} / {overall['zf_nmse']:.6f}")
    print(f"  DDPM PSNR/SSIM/NMSE : {overall['psnr']:.4f} / {overall['ssim']:.4f} / {overall['nmse']:.6f}")

    if CFG.USE_LEARNED_PARTITIONING:

        probs = model.partitioner.current_probabilities()

        print("\nFINAL LEARNED PARTITIONING (avg non-ACS-line Aleph probability):")

        for c, label in enumerate(CFG.CONTRAST_ACQUISITIONS):
            print(f"  {label}: {probs[c].mean().item():.4f}")

    print()
    print("Checkpoint:", best_path)
    print("Results   :", CFG.OUTPUT_DIR)

    print()
    print("TRAINING     : PURE SELF-SUPERVISED SSDU (full 1000-step dist.)")
    print("CONTRASTS    :", CFG.CONTRAST_ACQUISITIONS)
    print("PARTITIONING :", "LEARNED (Eqs. 5-7)" if CFG.USE_LEARNED_PARTITIONING else f"FIXED (alpha={CFG.SSDU_ALPHA})")
    print("SENSITIVITY  :", "LEARNED SME (E2E-VarNet style)" if CFG.USE_LEARNED_SENSITIVITY else "FIXED (classical)")
    print("VALIDATION   : PURE SELF-SUPERVISED SSDU (fixed-seed partition)")
    print("TEST GT      : FINAL EVALUATION ONLY")
    print(f"LOSS         : NORMALIZED L1+L2 MIX (L2 weight={CFG.SSDU_L2_WEIGHT})")
    print("GAN          : DISABLED")
    print("SAMPLER      : GENERALIZED RESPACED DDPM POSTERIOR (seeded/reproducible)")
    print("START        : x_T ~ N(0,I)")
    print(f"TEST STEPS   : {CFG.TEST_SAMPLE_STEPS} (of {CFG.T_STEPS})")
    print("DC           : FULL ACQUIRED OMEGA")
    print("COILS        : ORIGINAL", CFG.ORIGINAL_COILS, "(NO COMPRESSION)")
    print("=" * 80)


if __name__ == "__main__":
    main()
