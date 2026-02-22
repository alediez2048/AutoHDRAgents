"""Training loop with checkpoint save/resume, metric logging, and early stopping."""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config, cfg
from model import LensCorrector
from losses import CompositeLoss
from data import (
    create_dataloaders,
    compute_edge_similarity,
    compute_ssim,
    compute_pixel_mae,
    compute_composite,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages training, validation, checkpointing, and logging."""

    def __init__(
        self,
        stage: str = "s01",
        resume_from: Optional[str] = None,
        cfg: Config = cfg,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.stage_cfg = cfg.get_stage_config(stage)
        self.device = torch.device(cfg.DEVICE)

        # Model
        self.model = LensCorrector().to(self.device)

        # Loss
        self.loss_fn = CompositeLoss(weights=cfg.loss_weights, device=cfg.DEVICE)

        # Data
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=cfg.data_dir,
            stage_config=self.stage_cfg,
            cfg=cfg,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.stage_cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler: 1-epoch linear warmup then cosine decay
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.stage_cfg.epochs
        )
        self.warmup_epochs = 1

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp_enabled)

        # Tracking state
        self.current_epoch: int = 0
        self.best_composite: float = -1.0
        self.global_step: int = 0
        self.run_id: str = uuid.uuid4().hex[:8]
        self.epochs_without_improvement: int = 0

        # Resume if requested
        if resume_from is not None:
            self.load_checkpoint(resume_from)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> float:
        """Run one training epoch. Returns average loss.

        Supports gradient accumulation (cfg.grad_accum_steps) to simulate
        larger effective batch sizes on memory-constrained GPUs.
        """
        self.model.train()

        # Encoder freeze logic (epoch-level, not per-step)
        if self.current_epoch < self.cfg.encoder_freeze_epochs:
            self.model.freeze_encoder()
        else:
            self.model.unfreeze_encoder()

        running_loss = 0.0
        num_batches = 0
        accum_steps = getattr(self.cfg, 'grad_accum_steps', 1)

        self.optimizer.zero_grad()

        for batch_idx, (distorted, target) in enumerate(self.train_loader):
            distorted = distorted.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.amp_enabled):
                corrected, displacement = self.model(distorted)
                total_loss, loss_dict = self.loss_fn(corrected, target, displacement=displacement)
                # Scale loss by accumulation steps for correct gradient magnitude
                scaled_loss = total_loss / accum_steps

            self.scaler.scale(scaled_loss).backward()

            # Step optimizer every accum_steps batches (or on last batch)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += total_loss.item()
            num_batches += 1

            # Logging every 50 steps
            if self.global_step % 50 == 0:
                eff_bs = distorted.shape[0] * accum_steps
                parts = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
                print(
                    f"  [step {self.global_step}] loss: {total_loss.item():.4f} | {parts}"
                    + (f" | eff_bs={eff_bs}" if accum_steps > 1 else "")
                )

            # Periodic checkpoint
            if (
                self.cfg.checkpoint_freq > 0
                and self.global_step > 0
                and self.global_step % self.cfg.checkpoint_freq == 0
            ):
                self.save_checkpoint(self.current_epoch, {"step_loss": total_loss.item()})

            self.global_step += 1

        # LR scheduling: linear warmup for first epoch(s), then cosine
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.stage_cfg.lr * warmup_factor
        else:
            self.scheduler.step()

        avg_loss = running_loss / max(num_batches, 1)
        return avg_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on the validation set. Returns metrics dict."""
        self.model.eval()

        total_loss = 0.0
        total_edge = 0.0
        total_ssim = 0.0
        total_mae = 0.0
        num_batches = 0

        for distorted, target in self.val_loader:
            distorted = distorted.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.amp_enabled):
                corrected, displacement = self.model(distorted)
                loss, _ = self.loss_fn(corrected, target)

            total_loss += loss.item()

            # Edge similarity (tensor-based)
            total_edge += compute_edge_similarity(corrected, target)

            # SSIM and MAE require numpy uint8
            pred_np = (
                corrected.clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            tgt_np = (
                target.clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)

            batch_ssim = 0.0
            batch_mae = 0.0
            for i in range(pred_np.shape[0]):
                # (C, H, W) -> (H, W, C)
                p = pred_np[i].transpose(1, 2, 0)
                t = tgt_np[i].transpose(1, 2, 0)
                batch_ssim += compute_ssim(p, t)
                batch_mae += compute_pixel_mae(p, t)
            total_ssim += batch_ssim / pred_np.shape[0]
            total_mae += batch_mae / pred_np.shape[0]

            num_batches += 1

        n = max(num_batches, 1)
        avg_loss = total_loss / n
        avg_edge = total_edge / n
        avg_ssim = total_ssim / n
        avg_mae = total_mae / n
        composite = compute_composite(avg_edge, avg_ssim, avg_mae)

        return {
            "val_loss": avg_loss,
            "edge_sim": avg_edge,
            "ssim": avg_ssim,
            "pixel_mae": avg_mae,
            "composite": composite,
        }

    # ------------------------------------------------------------------
    # Visual comparison
    # ------------------------------------------------------------------

    @torch.no_grad()
    def save_visual_comparison(self, epoch: int, num_samples: int = 3) -> None:
        """Save a figure with best and worst predictions from a small val sample."""
        self.model.eval()
        results: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

        # Only check first 10 batches to avoid OOM from storing all val images
        for batch_idx, (distorted, target) in enumerate(self.val_loader):
            if batch_idx >= 10:
                break
            distorted = distorted.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.amp_enabled):
                corrected, _ = self.model(distorted)

            for i in range(distorted.shape[0]):
                p = corrected[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                t = target[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                d = distorted[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                mae = float(np.mean(np.abs(p - t)))
                results.append((mae, d, p, t))

            del distorted, target, corrected
            torch.cuda.empty_cache()

        # Sort by MAE: best = lowest, worst = highest
        results.sort(key=lambda x: x[0])
        best = results[:num_samples]
        worst = results[-num_samples:]
        selected = best + worst
        labels = [f"best-{i+1}" for i in range(len(best))] + [
            f"worst-{i+1}" for i in range(len(worst))
        ]

        fig, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)))
        if len(selected) == 1:
            axes = axes[np.newaxis, :]
        for row, ((mae, d, p, t), label) in enumerate(zip(selected, labels)):
            axes[row, 0].imshow(d)
            axes[row, 0].set_title(f"{label} | distorted")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(p)
            axes[row, 1].set_title(f"predicted (MAE={mae:.4f})")
            axes[row, 1].axis("off")
            axes[row, 2].imshow(t)
            axes[row, 2].set_title("target")
            axes[row, 2].axis("off")

        fig.suptitle(f"Epoch {epoch + 1} — Visual Comparison", fontsize=14)
        fig.tight_layout()

        out_dir = Path(self.cfg.output_dir) / "visuals"
        os.makedirs(out_dir, exist_ok=True)
        fig_path = out_dir / f"{self.run_id}_epoch{epoch}.png"
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        print(f"  Visual comparison saved: {fig_path}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save a training checkpoint."""
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        composite = metrics.get("composite", 0.0)
        filename = f"{self.run_id}_{self.stage}_{epoch}_{composite:.4f}.pth"
        path = ckpt_dir / filename

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": metrics,
            "cfg": asdict(self.cfg),
            "run_id": self.run_id,
        }
        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")

        # Save as best if composite improved
        if composite > self.best_composite:
            best_path = ckpt_dir / "best.pth"
            torch.save(state, best_path)
            print(f"  New best checkpoint: {best_path} (composite={composite:.4f})")

    def load_checkpoint(self, path: str) -> None:
        """Load state from a checkpoint file."""
        print(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1  # resume from next epoch
        self.global_step = ckpt["global_step"]
        self.run_id = ckpt.get("run_id", self.run_id)

        # Restore best composite from saved metrics
        saved_metrics = ckpt.get("metrics", {})
        saved_composite = saved_metrics.get("composite", -1.0)
        if saved_composite > self.best_composite:
            self.best_composite = saved_composite

        print(
            f"  Resumed at epoch {self.current_epoch}, "
            f"global_step {self.global_step}, "
            f"best_composite {self.best_composite:.4f}"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full training loop with validation, logging, and early stopping."""
        log_dir = Path(self.cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = log_dir / f"{self.run_id}.json"

        print(
            f"Training stage={self.stage} | epochs={self.stage_cfg.epochs} | "
            f"lr={self.stage_cfg.lr} | batch={self.stage_cfg.batch_size} | "
            f"device={self.device} | run_id={self.run_id}"
        )

        epoch_logs: list[dict] = []

        for epoch in range(self.current_epoch, self.stage_cfg.epochs):
            self.current_epoch = epoch
            print(f"\n--- Epoch {epoch + 1}/{self.stage_cfg.epochs} ---")

            avg_loss = self.train_one_epoch()
            metrics = self.validate()

            composite = metrics["composite"]
            print(
                f"  train_loss={avg_loss:.4f} | "
                f"val_edge={metrics['edge_sim']:.4f} | "
                f"val_ssim={metrics['ssim']:.4f} | "
                f"val_mae={metrics['pixel_mae']:.4f} | "
                f"composite={composite:.4f}"
            )

            # Visual comparison every 5 epochs and on the final epoch
            if (epoch + 1) % 5 == 0 or epoch == self.stage_cfg.epochs - 1:
                self.save_visual_comparison(epoch)

            # Save checkpoint if composite improved
            if composite > self.best_composite:
                self.save_checkpoint(epoch, metrics)
                self.best_composite = composite
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Append to per-epoch log
            epoch_log = {
                "epoch": epoch,
                "train_loss": avg_loss,
                **metrics,
            }
            epoch_logs.append(epoch_log)
            with open(log_path, "w") as f:
                json.dump(epoch_logs, f, indent=2)

            # Early stopping
            if self.epochs_without_improvement >= self.cfg.early_stopping_patience:
                print(
                    f"Early stopping: no improvement for "
                    f"{self.cfg.early_stopping_patience} epochs."
                )
                break

        print(f"\nTraining complete. Best composite: {self.best_composite:.4f}")
        print(f"Logs saved to: {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AutoHDR Lens Correction Training")
    parser.add_argument(
        "--stage",
        type=str,
        default="s01",
        choices=["s01", "s02"],
        help="Training stage (default: s01)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--config",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override config values, e.g. --config checkpoint_dir=/tmp/ckpts amp_enabled=False",
    )
    args = parser.parse_args()

    # Apply config overrides
    config = cfg
    for override in args.config:
        if "=" not in override:
            parser.error(f"Config override must be KEY=VALUE, got: {override}")
        key, value = override.split("=", 1)
        if not hasattr(config, key):
            parser.error(f"Unknown config key: {key}")
        field_type = type(getattr(config, key))
        if field_type is bool:
            parsed = value.lower() in ("true", "1", "yes")
        elif field_type is int:
            parsed = int(value)
        elif field_type is float:
            parsed = float(value)
        else:
            parsed = value
        object.__setattr__(config, key, parsed)

    set_seed(config.SEED)
    trainer = Trainer(stage=args.stage, resume_from=args.resume, cfg=config)
    trainer.train()


if __name__ == "__main__":
    main()
