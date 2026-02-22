"""Configuration module for AutoHDR Lens Correction project."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class StageConfig:
    """Per-stage training configuration."""
    resolution: int
    batch_size: int
    lr: float
    epochs: int
    stage: str


@dataclass
class LossWeights:
    """Loss component weights."""
    edge: float = 0.40
    grad_orientation: float = 0.20
    ssim: float = 0.20
    l1_pixel: float = 0.10
    perceptual: float = 0.10


@dataclass
class Config:
    """Global project configuration."""

    # Reproducibility
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage configs
    s01: StageConfig = field(default_factory=lambda: StageConfig(
        resolution=512, batch_size=2, lr=1e-4, epochs=50, stage="s01"
    ))
    s02: StageConfig = field(default_factory=lambda: StageConfig(
        resolution=768, batch_size=1, lr=5e-5, epochs=15, stage="s02"
    ))

    # Loss weights
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # Displacement field
    displacement_clamp: float = 0.1

    # Checkpointing
    checkpoint_freq: int = 500  # steps

    # Optimizer
    optimizer: str = "AdamW"
    weight_decay: float = 1e-4

    # Gradient clipping
    grad_clip_max_norm: float = 1.0

    # Gradient accumulation (effective batch = batch_size * grad_accum_steps)
    grad_accum_steps: int = 4

    # Encoder freeze
    encoder_freeze_epochs: int = 1

    # Early stopping
    early_stopping_patience: int = 5

    # Data paths (Kaggle: /kaggle/input/automatic-lens-correction)
    data_dir: str = "/kaggle/input/automatic-lens-correction/lens-correction-train-cleaned"
    test_dir: str = "/kaggle/input/automatic-lens-correction/test-originals"
    train_csv: str = "/kaggle/working/train.csv"
    val_csv: str = "/kaggle/working/val.csv"

    # Checkpoint directory
    checkpoint_dir: str = "/kaggle/working/checkpoints"

    # Output and logging
    output_dir: str = "/kaggle/working/outputs"
    log_dir: str = "/kaggle/working/logs"

    # Mixed precision
    amp_enabled: bool = True

    # DataLoader
    num_workers: int = 4

    # Validation split
    val_split: float = 0.2

    def get_stage_config(self, stage: str) -> StageConfig:
        """Return the config for a given stage name."""
        if stage == "s01":
            return self.s01
        elif stage == "s02":
            return self.s02
        else:
            raise ValueError(f"Unknown stage: {stage}")


# Default global config instance
cfg = Config()
