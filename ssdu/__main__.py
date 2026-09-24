"""Entry point:  python -m ssdu_ddpm_mri"""

import os

import torch
from torch.utils.data import DataLoader

from .config import CFG
from .data_discovery import get_files_multi_contrast, verify_original_coils_multi
from .dataset import FastMRISSDUMultiContrast, collate_fn
from .reproducibility import seed_everything
from .test import test_model
from .train import train


def print_banner():
    print("=" * 80)
    print("PURE SELF-SUPERVISED SSDU CONDITIONAL DDPM MRI")
    print("RESPACED SAMPLER VERSION -- ORIGINAL 15-COIL (NO COMPRESSION)")
    print("=" * 80)
    print("Device                 :", CFG.DEVICE)
    print("Image size             :", CFG.IMAGE_SIZE)
    print("Original coils         :", CFG.ORIGINAL_COILS)
    print("Contrasts              :", CFG.CONTRAST_ACQUISITIONS)
    print("Acceleration           :", CFG.ACCELERATION)
    print("Center fraction        :", CFG.CENTER_FRACTION)
    print("Derived ACS size       :", CFG.ACS_SIZE)
    print("SSDU alpha (init)      :", CFG.SSDU_ALPHA)
    print("Learned partitioning   :", CFG.USE_LEARNED_PARTITIONING)
    print("Learned sensitivity    :", CFG.USE_LEARNED_SENSITIVITY)
    print("Skip start/end slices  :", CFG.SKIP_START_SLICES, "/", CFG.SKIP_END_SLICES)
    print("Train T_STEPS          :", CFG.T_STEPS)
    print("Test steps             :", CFG.TEST_SAMPLE_STEPS)
    print("Batch size / epochs    :", CFG.BATCH_SIZE, "/", CFG.EPOCHS)
    print(f"LOSS                   : NORMALIZED L1+L2 MIX (L2 weight={CFG.SSDU_L2_WEIGHT})")
    print("GROUND TRUTH           : PLAIN RSS (no sensitivity map)")
    print("MASK CHANNEL           : train=aleph, test=aleph")
    print("METRICS                : VOLUME-LEVEL (fastMRI Eqs. 7-8)")
    print("=" * 80)


def main():
    seed_everything(CFG.SPLIT_SEED)
    print_banner()

    print("\nDISCOVERING DATA")
    train_files, val_files, test_files = get_files_multi_contrast()

    for label in CFG.CONTRAST_ACQUISITIONS:
        print(
            f"{label}: train={len(train_files[label])} "
            f"val={len(val_files[label])} test={len(test_files[label])}"
        )

    all_files_dict = {
        label: train_files[label] + val_files[label] + test_files[label]
        for label in CFG.CONTRAST_ACQUISITIONS
    }
    verify_original_coils_multi(all_files_dict)

    model, best_path = train(train_files, val_files)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=CFG.DEVICE)
        model.load_state_dict(ckpt["model"])
        print("=" * 80)
        print("BEST CHECKPOINT LOADED")
        print("Epoch:", ckpt["epoch"])
        print("Best validation SSDU:", ckpt["best_val"])
        print("=" * 80)

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
        persistent_workers=CFG.PERSISTENT_WORKERS,
    )

    print("Test slices available (paired):", len(test_ds))
    print("Test slices evaluated         :", min(len(test_ds), CFG.TEST_SLICES))

    results, _ = test_model(model, test_loader, CFG.TEST_SLICES)

    print()
    print("=" * 80)
    print("FINAL RESULTS (VOLUME-LEVEL, official fastMRI convention, RSS GT)")
    print("=" * 80)

    for label in CFG.CONTRAST_ACQUISITIONS:
        r = results[label]
        print(f"\n--- Contrast: {label} ({r['num_volumes']} volumes) ---")
        print("SENSE ZERO-FILLED BASELINE")
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

    o = results["overall"]
    print(f"\n--- OVERALL ({o['num_volumes']} volumes) ---")
    print(f"  ZF   PSNR/SSIM/NMSE : {o['zf_psnr']:.4f} / {o['zf_ssim']:.4f} / {o['zf_nmse']:.6f}")
    print(f"  DDPM PSNR/SSIM/NMSE : {o['psnr']:.4f} / {o['ssim']:.4f} / {o['nmse']:.6f}")

    print()
    print("Checkpoint:", best_path)
    print("Results   :", CFG.OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
