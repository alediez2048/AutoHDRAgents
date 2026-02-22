"""Visualization helpers, checkpoint utilities, seed setting, tensor conversion."""

import random
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import Tensor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Tensor <-> image conversion
# ---------------------------------------------------------------------------

def tensor_to_image(tensor: Tensor) -> np.ndarray:
    """Convert a (C,H,W) or (B,C,H,W) float [0,1] tensor to (H,W,C) uint8 [0,255] numpy array.

    If batched, uses the first image in the batch.
    """
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    # (C, H, W) -> (H, W, C)
    img = t.permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def image_to_tensor(image: np.ndarray) -> Tensor:
    """Convert a (H,W,C) uint8 numpy array to (C,H,W) float [0,1] tensor."""
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _to_displayable(img: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor (C,H,W) or numpy (H,W,C) to displayable (H,W,C) uint8."""
    if isinstance(img, Tensor):
        return tensor_to_image(img)
    # Already numpy -- ensure uint8
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def visualize_pair(
    distorted: Union[Tensor, np.ndarray],
    target: Union[Tensor, np.ndarray],
    pred: Optional[Union[Tensor, np.ndarray]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Show 2 or 3 panels side by side: Distorted, Ground Truth, [Predicted]."""
    n_panels = 3 if pred is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    axes[0].imshow(_to_displayable(distorted))
    axes[0].set_title("Distorted")
    axes[0].axis("off")

    axes[1].imshow(_to_displayable(target))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    if pred is not None:
        axes[2].imshow(_to_displayable(pred))
        axes[2].set_title("Predicted")
        axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_displacement(
    displacement: Tensor,
    save_path: Optional[str] = None,
) -> None:
    """Visualize a (2,H,W) or (B,2,H,W) displacement field as dx/dy heatmaps."""
    d = displacement.detach().cpu().float()
    if d.ndim == 4:
        d = d[0]
    dx = d[0].numpy()
    dy = d[1].numpy()

    mag = np.sqrt(dx ** 2 + dy ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(dx, cmap="coolwarm")
    axes[0].set_title(f"dx  (min={dx.min():.4f}, max={dx.max():.4f})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(dy, cmap="coolwarm")
    axes[1].set_title(f"dy  (min={dy.min():.4f}, max={dy.max():.4f})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Displacement magnitude: mean={mag.mean():.5f}, max={mag.max():.5f}",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_metrics(metrics: dict) -> str:
    """Pretty-print a metrics dict as an aligned string for logging."""
    if not metrics:
        return ""
    max_key_len = max(len(str(k)) for k in metrics)
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {str(k):<{max_key_len}}  {v:.6f}")
        else:
            lines.append(f"  {str(k):<{max_key_len}}  {v}")
    return "\n".join(lines)
