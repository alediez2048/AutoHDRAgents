"""Edge, gradient orientation, SSIM, L1 pixel, and perceptual loss components with composite wrapper."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters
import kornia.losses
from torchvision.models import vgg16, VGG16_Weights

from config import cfg, LossWeights


class EdgeLoss(nn.Module):
    """L1 loss between Sobel edge maps of prediction and target."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edges_pred = kornia.filters.sobel(pred)
        edges_target = kornia.filters.sobel(target)
        return F.l1_loss(edges_pred, edges_target)


class GradientOrientationLoss(nn.Module):
    """Penalises angular difference in gradient orientation, masked to strong-gradient regions."""

    def __init__(self, magnitude_threshold: float = 0.01) -> None:
        super().__init__()
        self.magnitude_threshold = magnitude_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # spatial_gradient returns (B, C, 2, H, W) where dim 2 is [dx, dy]
        grad_pred = kornia.filters.spatial_gradient(pred)    # (B, C, 2, H, W)
        grad_target = kornia.filters.spatial_gradient(target)

        dx_pred = grad_pred[:, :, 0]   # (B, C, H, W)
        dy_pred = grad_pred[:, :, 1]
        dx_target = grad_target[:, :, 0]
        dy_target = grad_target[:, :, 1]

        angle_pred = torch.atan2(dy_pred, dx_pred)
        angle_target = torch.atan2(dy_target, dx_target)

        # Mask: keep only pixels where the target gradient magnitude exceeds threshold
        mag_target = torch.sqrt(dx_target ** 2 + dy_target ** 2)
        mask = mag_target > self.magnitude_threshold

        diff = 1.0 - torch.cos(angle_pred - angle_target)

        if mask.any():
            return diff[mask].mean()
        return diff.mean()


class SSIMLoss(nn.Module):
    """1 - SSIM structural similarity loss."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.ssim = kornia.losses.SSIMLoss(window_size=window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ssim(pred, target)


class L1PixelLoss(nn.Module):
    """Simple L1 pixel loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class PerceptualLoss(nn.Module):
    """L2 feature-matching loss using frozen VGG16 layers {3, 8, 15}."""

    def __init__(self) -> None:
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # We only need features up to index 15 (inclusive)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        # Freeze completely
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        self.extract_layers = {3, 8, 15}

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "PerceptualLoss":
        """Keep VGG in eval mode regardless of outer training state."""
        super().train(mode)
        self.features.eval()
        return self

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.extract_layers:
                feats.append(x)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)

        feats_pred = self._extract(pred_n)
        feats_target = self._extract(target_n)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for fp, ft in zip(feats_pred, feats_target):
            loss = loss + F.mse_loss(fp, ft)
        return loss / len(feats_pred)


class DisplacementSmoothnessLoss(nn.Module):
    """Penalises spatial discontinuities in the displacement field."""

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
        dx = displacement[:, :, :, 1:] - displacement[:, :, :, :-1]
        return (dx.pow(2).mean() + dy.pow(2).mean()) * 0.5


class CompositeLoss(nn.Module):
    """Weighted sum of all loss components."""

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.weights = weights if weights is not None else cfg.loss_weights

        self.edge_loss = EdgeLoss()
        self.grad_loss = GradientOrientationLoss()
        self.ssim_loss = SSIMLoss()
        self.l1_loss = L1PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        self.disp_smooth_loss = DisplacementSmoothnessLoss()

        self.to(device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        displacement: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_edge = self.edge_loss(pred, target)
        l_grad = self.grad_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_l1 = self.l1_loss(pred, target)
        l_perc = self.perceptual_loss(pred, target)

        total = (
            self.weights.edge * l_edge
            + self.weights.grad_orientation * l_grad
            + self.weights.ssim * l_ssim
            + self.weights.l1_pixel * l_l1
            + self.weights.perceptual * l_perc
        )

        components = {
            "edge": l_edge.item(),
            "grad": l_grad.item(),
            "ssim": l_ssim.item(),
            "l1": l_l1.item(),
            "perceptual": l_perc.item(),
        }

        if displacement is not None:
            l_smooth = self.disp_smooth_loss(displacement)
            total = total + self.weights.displacement_smoothness * l_smooth
            components["disp_smooth"] = l_smooth.item()

        return total, components
