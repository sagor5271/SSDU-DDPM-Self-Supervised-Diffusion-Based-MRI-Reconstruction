"""
Multi-contrast .h5 file discovery, contrast-label inspection, train/
val/test splitting per contrast, and coil-count verification.
"""

import glob
import os

import h5py
import numpy as np

from .config import CFG


def get_contrast_label(path):

    with h5py.File(path, "r") as hf:

        acq = hf.attrs.get("acquisition", None)

        if acq is None:
            return "UNKNOWN"

        if isinstance(acq, bytes):
            acq = acq.decode("utf-8")

        return str(acq)


def discover_acquisition_labels(files, sample_n=25):
    """
    Diagnostic helper: prints the distribution of "acquisition" HDF5
    attribute values found across a sample of files, so CFG.CONTRAST_
    ACQUISITIONS can be verified/corrected against the actual dataset
    mirror before training.
    """

    sample = files[:sample_n]

    counts = {}

    for path in sample:

        label = get_contrast_label(path)
        counts[label] = counts.get(label, 0) + 1

    print("=" * 80)
    print(f"DISCOVERED ACQUISITION LABELS (sample of {len(sample)} files)")
    print("=" * 80)

    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label!r}: {n}")

    print(
        "Configured CFG.CONTRAST_ACQUISITIONS:",
        CFG.CONTRAST_ACQUISITIONS
    )

    print("=" * 80)


def get_files_multi_contrast():
    """
    Groups all .h5 files under CFG.DATA_ROOT by their "acquisition"
    attribute, then independently shuffles and splits each contrast
    group into train/val/test using CFG.TRAIN_FILES / VAL_FILES /
    TEST_FILES (PER CONTRAST, not total).

    Returns three dicts (train, val, test), each mapping
    contrast_label -> list of file paths.
    """

    all_files = sorted(
        glob.glob(os.path.join(CFG.DATA_ROOT, "**", "*.h5"), recursive=True)
    )

    if len(all_files) == 0:
        raise RuntimeError(f"No .h5 files found under {CFG.DATA_ROOT}")

    discover_acquisition_labels(all_files)

    groups = {label: [] for label in CFG.CONTRAST_ACQUISITIONS}

    for path in all_files:

        label = get_contrast_label(path)

        if label in groups:
            groups[label].append(path)

    rng = np.random.RandomState(CFG.SPLIT_SEED)

    train_files = {}
    val_files = {}
    test_files = {}

    required = CFG.TRAIN_FILES + CFG.VAL_FILES + CFG.TEST_FILES

    for label in CFG.CONTRAST_ACQUISITIONS:

        files = list(groups[label])
        rng.shuffle(files)

        if len(files) < required:
            raise RuntimeError(
                f"Contrast '{label}': found {len(files)} files, "
                f"but {required} required (TRAIN_FILES + VAL_FILES + "
                f"TEST_FILES). Lower these CFG values, or verify the "
                f"acquisition label matches your dataset (see the "
                f"DISCOVERED ACQUISITION LABELS printout above)."
            )

        a = CFG.TRAIN_FILES
        b = a + CFG.VAL_FILES
        c = b + CFG.TEST_FILES

        train_files[label] = files[:a]
        val_files[label] = files[a:b]
        test_files[label] = files[b:c]

    return train_files, val_files, test_files


def verify_original_coils_multi(files_dict):

    print("=" * 80)
    print("VERIFYING ORIGINAL COIL COUNT (NO COMPRESSION)")
    print("=" * 80)

    total = 0

    for label, files in files_dict.items():

        for path in files:

            with h5py.File(path, "r") as hf:

                if "kspace" not in hf:
                    raise RuntimeError(f"'kspace' not found in {path}")

                shape = hf["kspace"].shape

            if len(shape) != 4:
                raise RuntimeError(f"Unexpected kspace shape in {path}: {shape}")

            coils = int(shape[1])

            if coils != CFG.ORIGINAL_COILS:
                raise RuntimeError(
                    f"{os.path.basename(path)} ({label}) has {coils} coils, "
                    f"expected {CFG.ORIGINAL_COILS}."
                )

            total += 1

        print(f"  {label}: {len(files)} files OK")

    print(f"Verified {total} files total across {len(files_dict)} contrasts.")
    print("Coils per file  :", CFG.ORIGINAL_COILS)
    print("Coil compression: DISABLED")
    print("=" * 80)
