"""Dataset class, DataLoader factory, augmentation pipelines for lens distortion correction."""

import csv
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import kornia.filters
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import Config, StageConfig, cfg


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _build_paired_augmentation(resolution: int) -> A.Compose:
    """Augmentations applied identically to both distorted and corrected images."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=resolution, width=resolution, p=0.3),
        ],
        additional_targets={"target": "image"},
    )


def _build_distorted_only_augmentation() -> A.Compose:
    """Augmentations applied only to the distorted image (after paired transforms)."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(10, 50), p=0.3),
            A.ImageCompression(quality_range=(70, 100), p=0.3),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LensDataset(Dataset):
    """Paired distorted / corrected image dataset for lens-correction training."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        resolution: int = 512,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.resolution = resolution
        self.augment = augment

        if augment:
            self.paired_aug = _build_paired_augmentation(resolution)
            self.distorted_aug = _build_distorted_only_augmentation()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        distorted_path, corrected_path = self.pairs[idx]

        # Load images (BGR -> RGB)
        distorted = cv2.imread(distorted_path, cv2.IMREAD_COLOR)
        corrected = cv2.imread(corrected_path, cv2.IMREAD_COLOR)

        if distorted is None:
            raise FileNotFoundError(f"Cannot read distorted image: {distorted_path}")
        if corrected is None:
            raise FileNotFoundError(f"Cannot read corrected image: {corrected_path}")

        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        # Resize to target resolution
        distorted = cv2.resize(distorted, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        corrected = cv2.resize(corrected, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

        # Augmentations
        if self.augment:
            # Paired augmentations (identical transform for both images)
            paired = self.paired_aug(image=distorted, target=corrected)
            distorted = paired["image"]
            corrected = paired["target"]

            # Distorted-only augmentations
            distorted = self.distorted_aug(image=distorted)["image"]

        # Convert to float32 tensors in [0, 1], shape (3, H, W)
        distorted_t = torch.from_numpy(distorted.astype(np.float32) / 255.0).permute(2, 0, 1)
        corrected_t = torch.from_numpy(corrected.astype(np.float32) / 255.0).permute(2, 0, 1)

        return distorted_t, corrected_t


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in _IMG_EXTS


def discover_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """Scan *data_dir* for paired distorted / corrected images.

    Supported layouts
    -----------------
    1. ``data_dir/distorted/`` + ``data_dir/corrected/`` with matching filenames.
    2. ``data_dir/train/distorted/`` + ``data_dir/train/corrected/``.
    3. Naming convention inside a flat directory: ``*_distorted.*`` / ``*_corrected.*``.
    """
    data_path = Path(data_dir)
    pairs: List[Tuple[str, str]] = []

    # Layout 1: top-level distorted/ and corrected/ directories
    dist_dir = data_path / "distorted"
    corr_dir = data_path / "corrected"
    if dist_dir.is_dir() and corr_dir.is_dir():
        pairs.extend(_match_dirs(dist_dir, corr_dir))

    # Layout 2: train/ subdirectory with distorted/ and corrected/
    train_dist = data_path / "train" / "distorted"
    train_corr = data_path / "train" / "corrected"
    if train_dist.is_dir() and train_corr.is_dir():
        pairs.extend(_match_dirs(train_dist, train_corr))

    # Layout 3: naming convention (*_distorted / *_corrected)
    if not pairs and data_path.is_dir():
        pairs.extend(_match_naming_convention(data_path))

    # Deduplicate while preserving order, then sort by distorted filename
    seen = set()
    unique: List[Tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    unique.sort(key=lambda t: Path(t[0]).name)
    return unique


def _match_dirs(dist_dir: Path, corr_dir: Path) -> List[Tuple[str, str]]:
    """Return pairs where filenames match between two directories."""
    corrected_names = {f.name for f in corr_dir.iterdir() if f.is_file() and _is_image(str(f))}
    pairs: List[Tuple[str, str]] = []
    for f in sorted(dist_dir.iterdir()):
        if f.is_file() and _is_image(str(f)) and f.name in corrected_names:
            pairs.append((str(f), str(corr_dir / f.name)))
    return pairs


def _match_naming_convention(directory: Path) -> List[Tuple[str, str]]:
    """Match paired images by naming convention.

    Supported patterns (distorted -> corrected):
    - ``*_distorted.*`` / ``*_corrected.*``
    - ``*_generated.*`` / ``*_original.*``  (AutoHDR Kaggle competition format)
    """
    files = {f.name: f for f in directory.iterdir() if f.is_file() and _is_image(str(f))}
    pairs: List[Tuple[str, str]] = []

    for name, fpath in sorted(files.items()):
        # Pattern 1: _distorted / _corrected
        if "_distorted" in name:
            partner = name.replace("_distorted", "_corrected")
            if partner in files:
                pairs.append((str(fpath), str(files[partner])))
        # Pattern 2: _original (distorted) / _generated (corrected)
        # In AutoHDR competition: original=distorted input, generated=corrected GT
        elif "_original" in name:
            partner = name.replace("_original", "_generated")
            if partner in files:
                pairs.append((str(fpath), str(files[partner])))

    return pairs


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str,
    stage_config: Optional[StageConfig] = None,
    cfg: Config = cfg,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Parameters
    ----------
    data_dir : str
        Root directory containing the image pairs.
    stage_config : StageConfig, optional
        Stage-specific configuration (resolution, batch_size, etc.).
        Defaults to ``cfg.s01``.
    cfg : Config
        Global configuration.

    Returns
    -------
    (train_loader, val_loader)
    """
    if stage_config is None:
        stage_config = cfg.s01

    pairs = discover_pairs(data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image pairs found in {data_dir}")

    # Reproducible 80/20 split
    rng = random.Random(cfg.SEED)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    split_idx = int(len(pairs) * (1.0 - cfg.val_split))
    train_indices = sorted(indices[:split_idx])
    val_indices = sorted(indices[split_idx:])

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]

    # Persist splits to CSV for reproducibility
    _save_pairs_csv(cfg.train_csv, train_pairs)
    _save_pairs_csv(cfg.val_csv, val_pairs)

    # Build datasets
    train_ds = LensDataset(train_pairs, resolution=stage_config.resolution, augment=True)
    val_ds = LensDataset(val_pairs, resolution=stage_config.resolution, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=stage_config.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=stage_config.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def _save_pairs_csv(path: str, pairs: List[Tuple[str, str]]) -> None:
    """Write a list of (distorted, corrected) pairs to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["distorted", "corrected"])
        for d, c in pairs:
            writer.writerow([d, c])


# ---------------------------------------------------------------------------
# Validation metric functions
# ---------------------------------------------------------------------------

def compute_edge_similarity(pred: Tensor, target: Tensor) -> float:
    """Compute edge-based similarity using Sobel filtering.

    Parameters
    ----------
    pred, target : Tensor
        Shape ``(B, 3, H, W)`` or ``(B, 1, H, W)`` float tensors.

    Returns
    -------
    float
        ``1.0 - L1(sobel(pred), sobel(target))``.
    """
    # Convert to grayscale if 3-channel
    if pred.shape[1] == 3:
        pred = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
    if target.shape[1] == 3:
        target = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

    sobel_pred = kornia.filters.sobel(pred)
    sobel_target = kornia.filters.sobel(target)

    return 1.0 - F.l1_loss(sobel_pred, sobel_target).item()


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM between two uint8 images.

    Parameters
    ----------
    pred, target : np.ndarray
        Shape ``(H, W, C)`` uint8 arrays.

    Returns
    -------
    float
        SSIM value.
    """
    return float(structural_similarity(pred, target, channel_axis=2))


def compute_pixel_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error normalised to [0, 1].

    Parameters
    ----------
    pred, target : np.ndarray
        Shape ``(H, W, C)`` uint8 arrays.

    Returns
    -------
    float
        ``mean(|pred - target|) / 255``.
    """
    return float(np.mean(np.abs(pred.astype(float) - target.astype(float))) / 255.0)


def compute_composite(
    edge: float,
    ssim: float,
    pixel_mae: float,
) -> float:
    """Weighted composite quality score using real metrics only.

    Simplified from the competition formula (which includes line/gradient proxies
    we can't compute locally). Uses only the three metrics we can actually measure,
    reweighted to sum to 1.0:
        0.55 * edge + 0.30 * ssim + 0.15 * (1 - pixel_mae)
    """
    return (
        0.55 * edge
        + 0.30 * ssim
        + 0.15 * (1.0 - pixel_mae)
    )
