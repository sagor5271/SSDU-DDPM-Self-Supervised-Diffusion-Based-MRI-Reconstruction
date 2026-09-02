"""
Global configuration for the SSDU conditional DDPM pipeline.

Import CFG from this module everywhere else. validate_config() is called
automatically on import, and checkpoint/output directories are created
automatically on import as well.
"""

import os
import torch


class CFG:

    CODE_VERSION = "SSDU-DDPM-AUDITED-v4"

    DATA_ROOT = (
        "/kaggle/input/datasets/"
        "arafatshovon/fastmri-knee-multicoil"
    )

    IMAGE_SIZE = 320

    # ------------------------------------------------------------------------
    # COILS
    # ------------------------------------------------------------------------

    ORIGINAL_COILS = 15

    # ------------------------------------------------------------------------
    # Cartesian sampling
    # ------------------------------------------------------------------------

    ACCELERATION = 4
    CENTER_FRACTION = 0.08
    ACS_SIZE = 24

    # ------------------------------------------------------------------------
    # MULTI-CONTRAST
    #
    # Each string must match the "acquisition" HDF5 attribute stored on
    # every fastMRI knee .h5 file (standard fastMRI knee values are
    # "CORPD_FBK" and "CORPDFS_FBK"). Order matters: index l in this
    # list is contrast channel l everywhere else in the script (network
    # I/O, loss averaging, etc). Set to a single-element list to fully
    # recover the original single-contrast behavior.
    #
    # If discover_acquisition_labels() (called at startup) reports
    # different label strings for your particular Kaggle mirror, update
    # this list to match before training.
    # ------------------------------------------------------------------------

    CONTRAST_ACQUISITIONS = ["CORPD_FBK"]  # clean single-contrast baseline; enable multi-contrast only with true matched pairs
    NUM_CONTRASTS = len(CONTRAST_ACQUISITIONS)

    # ------------------------------------------------------------------------
    # SSDU
    #
    # SSDU_ALPHA is the target fraction of non-ACS acquired lines placed
    # in Aleph. When USE_LEARNED_PARTITIONING=True, this value is used
    # ONLY to initialize the learned distribution (so training starts
    # from a sensible 85/15-ish split); the actual split is then learned.
    # When USE_LEARNED_PARTITIONING=False, this value is used directly
    # by the original fixed-probability random split_ssdu().
    #
    # ACS remains in Aleph in both cases.
    # ------------------------------------------------------------------------

    SSDU_ALPHA = 0.85

    # ------------------------------------------------------------------------
    # LEARNED PARTITIONING (Section II.B, Eqs. 5-7)
    # ------------------------------------------------------------------------

    USE_LEARNED_PARTITIONING = False  # stable baseline; optional learned mode below is deterministic at eval

    # Lambda_l = sigmoid(PARTITION_SLOPE_T * W_l)  (forward probability map)
    PARTITION_SLOPE_T = 1.0

    # Backward gradient approximation slope (Eq. 7):
    # d(M)/dZ ~= sigmoid(PARTITION_SLOPE_S * Z) * sigmoid(PARTITION_SLOPE_S * (Z - 1))
    PARTITION_SLOPE_S = 5.0
    PARTITION_FRACTION_WEIGHT = 0.1

    # ------------------------------------------------------------------------
    # LEARNED SENSITIVITY MAP ESTIMATION
    #
    # When True, coil sensitivity maps used throughout the RECONSTRUCTION
    # pipeline (condition, internal/final hard-DC, SSDU loss) come from a
    # small trainable U-Net (SensitivityRefineUNet) applied to the raw
    # per-coil ACS images, trained jointly with everything else via the
    # same optimizer (E2E-VarNet style). The ZF baseline metric and the
    # test-time GT target still always use the classical FIXED
    # estimate_sens() formula (see sensitivity.py for why).
    #
    # When False, sensitivity maps for the reconstruction pipeline also
    # fall back to the classical fixed estimate_sens() formula (looped
    # per batch/contrast slice), exactly reproducing the previous
    # version's behavior.
    # ------------------------------------------------------------------------

    USE_LEARNED_SENSITIVITY = False  # keep reconstruction/GT representation consistent by default

    # Base channel width of the shared per-coil sensitivity-refinement
    # U-Net. Kept small -- this is a calibration-refinement network, not
    # the main reconstruction network, and it runs once per training
    # step on every coil (15 coils x up to 2 contrasts = up to 30 small
    # forward passes folded into one batched call).
    SENS_UNET_CHANS = 8

    # ------------------------------------------------------------------------
    # SSDU LOSS (normalized L1 + L2 mix, original SSDU paper:
    # Yaman et al., MRM 2020), computed per contrast then averaged over
    # contrasts (1/L sum, matching Eq. 3/8a of the multi-contrast paper).
    #
    # loss = (1 - SSDU_L2_WEIGHT) * normalized_L1 + SSDU_L2_WEIGHT * normalized_L2
    # ------------------------------------------------------------------------

    SSDU_L2_WEIGHT = 0.5

    # ------------------------------------------------------------------------
    # DDPM
    # ------------------------------------------------------------------------

    T_STEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02

    TEST_SAMPLE_STEPS = 100

    # ------------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------------

    BASE_CH = 32
    TIME_DIM = 128

    # ------------------------------------------------------------------------
    # Training
    #
    # NOTE: each training step now runs the generator on NUM_CONTRASTS
    # channel-concatenated contrasts at once (5*L input channels instead
    # of 5), so memory/compute per step scales up with NUM_CONTRASTS. The
    # learned SME network adds a modest extra cost (small U-Net, run
    # once per coil per contrast per step). If you hit CUDA OOM, lower
    # BATCH_SIZE first, then BASE_CH, then SENS_UNET_CHANS.
    # ------------------------------------------------------------------------

    BATCH_SIZE = 2
    EPOCHS = 10

    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    # ------------------------------------------------------------------------
    # File split
    #
    # These counts are PER CONTRAST -- e.g. TRAIN_FILES=50 means 50
    # CORPD_FBK files AND 50 CORPDFS_FBK files (100 files total) are
    # used for training.
    # ------------------------------------------------------------------------

    TRAIN_FILES = 25
    VAL_FILES = 10
    TEST_FILES = 10
    SPLIT_SEED = 42

    # ------------------------------------------------------------------------
    # Slices
    # ------------------------------------------------------------------------

    SKIP_EDGE_SLICES = 2
    VAL_SLICES = 10**9
    TEST_SLICES = 10**9

    # Fixed seeds for reproducible validation and test-time sampling.
    VAL_SEED = 123456
    TEST_SEED = 20260901

    # ------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------

    NUM_VIS_EXAMPLES = 10

    # ------------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------------

    NUM_WORKERS = 2
    PIN_MEMORY = torch.cuda.is_available()

    TRAIN_DROP_LAST = True
    VAL_DROP_LAST = False
    TEST_DROP_LAST = False
    PERSISTENT_WORKERS = False

    # ------------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------------

    USE_INTERNAL_DC = True

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    CHECKPOINT_DIR = (
        "/kaggle/working/"
        "ssdu_diffusion_checkpoints"
    )

    OUTPUT_DIR = (
        "/kaggle/working/"
        "ssdu_diffusion_results"
    )

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def validate_config():
    """
    Sanity-checks CFG values and creates checkpoint/output directories.
    Called automatically on import of this module; safe to call again
    (e.g. after mutating CFG values at runtime) if needed.
    """

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    if not (1 <= CFG.TEST_SAMPLE_STEPS <= CFG.T_STEPS):
        raise ValueError(
            f"TEST_SAMPLE_STEPS ({CFG.TEST_SAMPLE_STEPS}) must be between "
            f"1 and T_STEPS ({CFG.T_STEPS}) inclusive."
        )

    if not (0.0 <= CFG.SSDU_L2_WEIGHT <= 1.0):
        raise ValueError(
            f"SSDU_L2_WEIGHT ({CFG.SSDU_L2_WEIGHT}) must be between "
            f"0.0 and 1.0 inclusive."
        )

    if not (0.0 < CFG.SSDU_ALPHA < 1.0):
        raise ValueError(
            f"SSDU_ALPHA ({CFG.SSDU_ALPHA}) must be strictly between 0 and 1."
        )

    if CFG.NUM_CONTRASTS < 1:
        raise ValueError("CONTRAST_ACQUISITIONS must contain at least one entry.")

    if CFG.SENS_UNET_CHANS < 1:
        raise ValueError("SENS_UNET_CHANS must be >= 1.")


validate_config()
