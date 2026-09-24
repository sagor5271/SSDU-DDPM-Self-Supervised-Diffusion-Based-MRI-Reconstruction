"""Global configuration. Edit values in `CFG` (or override DATA_ROOT / output dirs via env vars)."""

import os
import torch


class CFG:

    DATA_ROOT = os.environ.get(
        "SSDU_DATA_ROOT",
        "/kaggle/input/datasets/arafatshovon/fastmri-knee-multicoil",
    )

    IMAGE_SIZE = 320
    ORIGINAL_COILS = 15

    ACCELERATION = 4
    CENTER_FRACTION = 0.08

    # Derived from CENTER_FRACTION right after this class (never hardcode).
    ACS_SIZE = None

    CONTRAST_ACQUISITIONS = ["CORPD_FBK"]
    NUM_CONTRASTS = len(CONTRAST_ACQUISITIONS)

    SSDU_ALPHA = 0.7

    USE_LEARNED_PARTITIONING = False
    PARTITION_SLOPE_T = 1.0
    PARTITION_SLOPE_S = 5.0
    PARTITION_FRACTION_WEIGHT = 0.1

    USE_LEARNED_SENSITIVITY = False
    SENS_UNET_CHANS = 8

    SSDU_L2_WEIGHT = 0.5

    T_STEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02

    TEST_SAMPLE_STEPS = 250

    BASE_CH = 64
    TIME_DIM = 128

    BATCH_SIZE = 2
    EPOCHS = 50

    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    TRAIN_FILES = 60
    VAL_FILES = 8
    TEST_FILES = 12
    SPLIT_SEED = 42

    SKIP_START_SLICES = 9
    SKIP_END_SLICES = 2

    VAL_SLICES = 10**9
    TEST_SLICES = 10**9

    NUM_VIS_EXAMPLES = 10

    NUM_WORKERS = 2
    PIN_MEMORY = torch.cuda.is_available()

    TRAIN_DROP_LAST = True
    VAL_DROP_LAST = False
    TEST_DROP_LAST = False
    PERSISTENT_WORKERS = False

    USE_INTERNAL_DC = True

    CHECKPOINT_DIR = os.environ.get(
        "SSDU_CHECKPOINT_DIR", "/kaggle/working/ssdu_diffusion_checkpoints"
    )
    OUTPUT_DIR = os.environ.get(
        "SSDU_OUTPUT_DIR", "/kaggle/working/ssdu_diffusion_results"
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Same formula create_mask() uses for its center band -> can never drift.
CFG.ACS_SIZE = max(1, int(round(CFG.IMAGE_SIZE * CFG.CENTER_FRACTION)))


def check_config():
    if not (1 <= CFG.TEST_SAMPLE_STEPS <= CFG.T_STEPS):
        raise ValueError(
            f"TEST_SAMPLE_STEPS ({CFG.TEST_SAMPLE_STEPS}) must be between "
            f"1 and T_STEPS ({CFG.T_STEPS}) inclusive."
        )
    if not (0.0 <= CFG.SSDU_L2_WEIGHT <= 1.0):
        raise ValueError(f"SSDU_L2_WEIGHT ({CFG.SSDU_L2_WEIGHT}) must be in [0, 1].")
    if not (0.0 < CFG.SSDU_ALPHA < 1.0):
        raise ValueError(f"SSDU_ALPHA ({CFG.SSDU_ALPHA}) must be strictly between 0 and 1.")
    if CFG.NUM_CONTRASTS < 1:
        raise ValueError("CONTRAST_ACQUISITIONS must contain at least one entry.")
    if CFG.SENS_UNET_CHANS < 1:
        raise ValueError("SENS_UNET_CHANS must be >= 1.")
    if CFG.SKIP_START_SLICES < 0:
        raise ValueError("SKIP_START_SLICES must be >= 0.")
    if CFG.SKIP_END_SLICES < 0:
        raise ValueError("SKIP_END_SLICES must be >= 0.")
    if CFG.ACS_SIZE < 1:
        raise ValueError("Derived ACS_SIZE must be >= 1.")


check_config()

os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
