"""
Multi-contrast FastMRI SSDU Dataset, its collate function, and the
helper that moves a collated batch onto CFG.DEVICE.

For each contrast in CFG.CONTRAST_ACQUISITIONS, an INDEPENDENT (fid,
slice) index is built. The combined dataset length is the MINIMUM
across all contrasts' slice counts; item i of contrast 0 is paired with
item i of contrast 1, etc. (see the multi-contrast caveat in README.md
about index-only pairing on fastMRI knee).

Aleph/Upsilon are NOT computed here (that must happen INSIDE the model
for learned partitioning, where gradients can flow to the partitioning
parameters). The Dataset's "sens" field IS still computed here, using
the classical FIXED estimate_sens() formula -- but it is used ONLY for
the ZF baseline metric and the test-time GT target; the actual
reconstruction pipeline gets its sensitivity maps from
SSDUDDPM.get_sens(komega) instead (learned or fixed, per CFG).
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CFG
from .fft_utils import crop_kspace, estimate_scale, normalize_kspace
from .masks import create_mask
from .sensitivity import estimate_sens
from .sense_ops import sense_adjoint


class FastMRISSDUMultiContrast(Dataset):

    def __init__(self, files_per_contrast, seed, training=True):
        """
        files_per_contrast: dict[label] -> list[path], one entry per
        CFG.CONTRAST_ACQUISITIONS label.
        """

        self.contrast_labels = list(CFG.CONTRAST_ACQUISITIONS)
        self.files_per_contrast = [
            files_per_contrast[label] for label in self.contrast_labels
        ]

        self.seed = seed
        self.training = training
        self.epoch = 0

        self.index_per_contrast = []

        for files in self.files_per_contrast:

            idx = []

            for fid, path in enumerate(files):

                with h5py.File(path, "r") as hf:
                    shape = hf["kspace"].shape
                    n = shape[0]
                    n_coils = shape[1]

                if n_coils != CFG.ORIGINAL_COILS:
                    raise RuntimeError(
                        f"{path}: found {n_coils} coils; "
                        f"expected {CFG.ORIGINAL_COILS}."
                    )

                start = CFG.SKIP_EDGE_SLICES
                end = max(start, n - CFG.SKIP_EDGE_SLICES)

                for sid in range(start, end):
                    idx.append((fid, sid))

            self.index_per_contrast.append(idx)

        self.length = min(len(idx) for idx in self.index_per_contrast)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def _load_one_contrast(self, c, idx):

        files = self.files_per_contrast[c]
        index_list = self.index_per_contrast[c]

        fid, sid = index_list[idx]
        path = files[fid]

        with h5py.File(path, "r") as hf:
            raw = np.asarray(hf["kspace"][sid])

        kspace = torch.from_numpy(raw).to(torch.complex64)

        if kspace.shape[0] != CFG.ORIGINAL_COILS:
            raise RuntimeError(
                f"Loaded {kspace.shape[0]} coils; expected {CFG.ORIGINAL_COILS}."
            )

        kspace = crop_kspace(kspace, CFG.IMAGE_SIZE)

        C, H, W = kspace.shape

        if C != CFG.ORIGINAL_COILS:
            raise RuntimeError(f"After crop: {C} coils; expected {CFG.ORIGINAL_COILS}.")

        mask_seed = (
            self.seed + idx * 1009 + self.epoch * 1000003 + c * 500009
        )

        omega = create_mask(H, W, CFG.ACCELERATION, CFG.CENTER_FRACTION, mask_seed)

        komega_raw = kspace * omega.unsqueeze(0)
        scale = estimate_scale(komega_raw)
        komega = normalize_kspace(komega_raw, scale)

        # Fixed/classical sensitivity -- reserved for the ZF baseline
        # metric and the test-time GT target ONLY. The actual
        # reconstruction pipeline gets its sensitivity maps from
        # SSDUDDPM.get_sens(komega) at train/val/test time instead.
        sens = estimate_sens(komega, CFG.ACS_SIZE)

        if self.training:

            target = torch.empty(0, dtype=torch.float32)

        else:

            full_kspace = normalize_kspace(kspace, scale)

            target_complex = sense_adjoint(
                full_kspace.unsqueeze(0), sens.unsqueeze(0)
            )

            target = torch.abs(target_complex)[0].float()

        return {
            "komega": komega,
            "omega": omega,
            "sens": sens,
            "target": target,
            "file": os.path.basename(path),
            "slice": sid
        }

    def __getitem__(self, idx):

        per_contrast = [
            self._load_one_contrast(c, idx)
            for c in range(len(self.contrast_labels))
        ]

        komega = torch.stack([d["komega"] for d in per_contrast], dim=0)  # [L,C,H,W]
        omega = torch.stack([d["omega"] for d in per_contrast], dim=0)    # [L,H,W]
        sens = torch.stack([d["sens"] for d in per_contrast], dim=0)      # [L,C,H,W]
        target = torch.stack([d["target"] for d in per_contrast], dim=0)  # [L,H,W] or [L,0]

        return {
            "komega": komega,
            "omega": omega,
            "sens": sens,
            "target": target,
            "files": [d["file"] for d in per_contrast],
            "slices": [d["slice"] for d in per_contrast]
        }


def collate_fn(batch):

    return {

        "komega": torch.stack([x["komega"] for x in batch]),
        "omega": torch.stack([x["omega"] for x in batch]),
        "sens": torch.stack([x["sens"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),

        "files": [x["files"] for x in batch],
        "slices": [x["slices"] for x in batch]
    }


def move_batch(batch):

    tensor_keys = ["komega", "omega", "sens"]

    out = {}

    for key in tensor_keys:
        out[key] = batch[key].to(CFG.DEVICE, non_blocking=True)

    out["target"] = batch["target"]
    out["files"] = batch["files"]
    out["slices"] = batch["slices"]

    return out
