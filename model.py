"""ResNet50-UNet encoder-decoder with displacement field prediction and grid_sample warp layer."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from config import cfg


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """ResNet50 backbone that extracts multi-scale feature maps.

    Feature outputs (for input of size H x W):
        f_init : (B,   64, H/4,  W/4)   — after conv1 + bn1 + relu + maxpool
        f1     : (B,  256, H/4,  W/4)   — after layer1
        f2     : (B,  512, H/8,  W/8)   — after layer2
        f3     : (B, 1024, H/16, W/16)  — after layer3
        f4     : (B, 2048, H/32, W/32)  — after layer4
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Initial stem (conv1 → bn1 → relu → maxpool)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Residual stages
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    # -- forward ----------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (f_init, f1, f2, f3, f4)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f_init = self.maxpool(x)          # (B,  64, H/4,  W/4)

        f1 = self.layer1(f_init)          # (B, 256, H/4,  W/4)
        f2 = self.layer2(f1)              # (B, 512, H/8,  W/8)
        f3 = self.layer3(f2)              # (B,1024, H/16, W/16)
        f4 = self.layer4(f3)              # (B,2048, H/32, W/32)

        return f_init, f1, f2, f3, f4

    # -- freeze / unfreeze ------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters and keep BN in eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters (BN stays in eval via train())."""
        for param in self.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True) -> "Encoder":
        """Override to always keep BatchNorm layers in eval mode."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
        return self


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Up-sample → concat skip → two (Conv → BN → ReLU) blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle spatial size mismatch between upsampled and skip
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """UNet-style decoder producing a 2-channel displacement field.

    Path (channels):
        f4(2048) + f3(1024) skip → 512
        512      + f2(512)  skip → 256
        256      + f1(256)  skip → 128
        128      + f_init(64) skip → 64
        upsample ×4 → Conv → 2
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = DecoderBlock(in_ch=2048, skip_ch=1024, out_ch=512)
        self.block2 = DecoderBlock(in_ch=512,  skip_ch=512,  out_ch=256)
        self.block3 = DecoderBlock(in_ch=256,  skip_ch=256,  out_ch=128)
        self.block4 = DecoderBlock(in_ch=128,  skip_ch=64,   out_ch=64)

        # Final upsample from H/4 → H, then 1×1 conv to 2 channels
        self.final_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

        # Initialise final conv with small random weights so gradients can flow
        # (exact zeros trap the model at identity displacement)
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        f_init: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> torch.Tensor:
        """Return displacement field of shape (B, 2, H, W)."""
        x = self.block1(f4, f3)           # (B, 512, H/16, W/16)
        x = self.block2(x, f2)            # (B, 256, H/8,  W/8)
        x = self.block3(x, f1)            # (B, 128, H/4,  W/4)
        x = self.block4(x, f_init)        # (B,  64, H/4,  W/4)
        x = self.final_up(x)             # (B,  64, H,    W)
        return self.final_conv(x)         # (B,   2, H,    W)


# ---------------------------------------------------------------------------
# Warp layer
# ---------------------------------------------------------------------------

class WarpLayer(nn.Module):
    """Differentiable spatial transformer that applies a displacement field."""

    def forward(self, input_image: torch.Tensor, displacement_field: torch.Tensor) -> torch.Tensor:
        """Warp *input_image* according to *displacement_field*.

        Args:
            input_image:       (B, C, H, W)
            displacement_field: (B, 2, H, W)  — values in normalised [-1, 1] space

        Returns:
            Warped image (B, C, H, W).
        """
        B, _, H, W = displacement_field.shape

        # Build base grid of normalised [-1, 1] coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=displacement_field.device, dtype=displacement_field.dtype),
            torch.linspace(-1.0, 1.0, W, device=displacement_field.device, dtype=displacement_field.dtype),
            indexing="ij",
        )
        # (1, H, W, 2) — last dim is (x, y) as expected by grid_sample
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # displacement_field: (B, 2, H, W) → (B, H, W, 2)
        disp = displacement_field.permute(0, 2, 3, 1)

        sample_grid = base_grid + disp
        return F.grid_sample(
            input_image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LensCorrector(nn.Module):
    """ResNet50-UNet that predicts a displacement field and warps the input."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.warp_layer = WarpLayer()

    def forward(
        self, distorted_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (corrected_image, displacement_field)."""
        f_init, f1, f2, f3, f4 = self.encoder(distorted_image)
        displacement = self.decoder(f_init, f1, f2, f3, f4)
        displacement = torch.clamp(displacement, -cfg.displacement_clamp, cfg.displacement_clamp)
        corrected = self.warp_layer(distorted_image, displacement)
        return corrected, displacement

    # -- convenience wrappers ---------------------------------------------

    def freeze_encoder(self) -> None:
        self.encoder.freeze_encoder()

    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_encoder()
