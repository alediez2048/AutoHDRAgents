"""7-check sanity suite and pre-submission validation checklist."""

import os
import sys
import tempfile
import traceback
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config, cfg
from model import LensCorrector
from losses import CompositeLoss
from utils import set_seed, tensor_to_image, image_to_tensor, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataloader(
    batch_size: int = 2,
    n_samples: int = 4,
    resolution: int = 512,
) -> DataLoader:
    """Create a synthetic dataloader with random image pairs for validation checks."""
    distorted = torch.rand(n_samples, 3, resolution, resolution)
    # Make target slightly different from distorted
    target = torch.clamp(distorted + 0.1 * torch.randn_like(distorted), 0.0, 1.0)
    dataset = TensorDataset(distorted, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Check 1: Data pipeline
# ---------------------------------------------------------------------------

def check_data_pipeline(dataloader: DataLoader) -> bool:
    """Load 1 batch and verify shapes, value ranges, and that distorted != target."""
    name = "check_data_pipeline"
    try:
        batch = next(iter(dataloader))
        distorted, target = batch[0], batch[1]

        B = distorted.shape[0]
        assert distorted.ndim == 4 and distorted.shape[1] == 3, (
            f"Distorted shape {distorted.shape} is not (B, 3, H, W)"
        )
        assert target.ndim == 4 and target.shape[1] == 3, (
            f"Target shape {target.shape} is not (B, 3, H, W)"
        )
        assert distorted.shape == target.shape, (
            f"Shape mismatch: distorted {distorted.shape} vs target {target.shape}"
        )

        # Value range
        assert distorted.min() >= 0.0 and distorted.max() <= 1.0, (
            f"Distorted range [{distorted.min():.4f}, {distorted.max():.4f}] outside [0,1]"
        )
        assert target.min() >= 0.0 and target.max() <= 1.0, (
            f"Target range [{target.min():.4f}, {target.max():.4f}] outside [0,1]"
        )

        # distorted should differ from target
        mse = torch.nn.functional.mse_loss(distorted, target).item()
        assert mse > 0.001, f"Distorted and target are too similar (MSE={mse:.6f})"

        print(f"[PASS] {name}")
        print(f"       Batch shape: distorted={distorted.shape}, target={target.shape}")
        print(f"       Distorted range: [{distorted.min():.4f}, {distorted.max():.4f}]")
        print(f"       Target    range: [{target.min():.4f}, {target.max():.4f}]")
        print(f"       MSE(distorted, target): {mse:.6f}")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 2: Identity warp at init
# ---------------------------------------------------------------------------

def check_identity_warp(model: torch.nn.Module, device: str) -> bool:
    """Verify that a freshly initialised model produces near-identity warp."""
    name = "check_identity_warp"
    try:
        model.eval()
        x = torch.rand(1, 3, 512, 512, device=device)
        with torch.no_grad():
            corrected, displacement = model(x)

        disp_max = displacement.abs().max().item()
        mse = torch.nn.functional.mse_loss(corrected, x).item()

        assert disp_max < 0.01, (
            f"Displacement too large at init: max abs = {disp_max:.6f}"
        )
        assert mse < 1e-4, (
            f"Output differs from input at init: MSE = {mse:.6f}"
        )

        print(f"[PASS] {name}")
        print(f"       Displacement max abs: {disp_max:.6f}")
        print(f"       MSE(output, input):   {mse:.6f}")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 3: Output shapes
# ---------------------------------------------------------------------------

def check_output_shapes(model: torch.nn.Module, device: str) -> bool:
    """Verify output shapes at 512 and 768 resolution."""
    name = "check_output_shapes"
    try:
        model.eval()
        for res in [512, 768]:
            x = torch.rand(1, 3, res, res, device=device)
            with torch.no_grad():
                corrected, displacement = model(x)

            assert corrected.shape == (1, 3, res, res), (
                f"Corrected shape {corrected.shape} != expected (1, 3, {res}, {res})"
            )
            assert displacement.shape == (1, 2, res, res), (
                f"Displacement shape {displacement.shape} != expected (1, 2, {res}, {res})"
            )
            print(f"       Resolution {res}: corrected={corrected.shape}, displacement={displacement.shape}")

        print(f"[PASS] {name}")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 4: Displacement clamping
# ---------------------------------------------------------------------------

def check_displacement_clamp(model: torch.nn.Module, device: str) -> bool:
    """Verify displacement values are within cfg.displacement_clamp."""
    name = "check_displacement_clamp"
    try:
        model.eval()
        x = torch.rand(1, 3, 512, 512, device=device)
        with torch.no_grad():
            _, displacement = model(x)

        disp_max = displacement.abs().max().item()
        limit = cfg.displacement_clamp + 1e-6

        assert disp_max <= limit, (
            f"Displacement {disp_max:.6f} exceeds clamp {cfg.displacement_clamp} + eps"
        )

        print(f"[PASS] {name}")
        print(f"       Displacement max abs: {disp_max:.6f} (limit: {cfg.displacement_clamp})")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 5: Loss gradients
# ---------------------------------------------------------------------------

def check_loss_gradients(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str,
) -> bool:
    """Verify loss is positive, finite, and all trainable params get gradients."""
    name = "check_loss_gradients"
    try:
        model.train()
        model.zero_grad()

        pred = torch.rand(1, 3, 256, 256, device=device, requires_grad=True)
        target = torch.rand(1, 3, 256, 256, device=device)

        total_loss, loss_dict = loss_fn(pred, target)

        assert total_loss.item() > 0, f"Loss is not positive: {total_loss.item()}"
        assert not torch.isnan(total_loss), "Loss is NaN"
        assert not torch.isinf(total_loss), "Loss is Inf"

        total_loss.backward()

        # Check that all trainable params in model have gradients
        # (We use pred here; model params won't have grads from this pass since
        #  we used random tensors, not model output. So let's do a proper forward.)
        model.zero_grad()
        x = torch.rand(1, 3, 256, 256, device=device)
        corrected, disp = model(x)
        loss, _ = loss_fn(corrected, target)
        loss.backward()

        missing_grad = []
        nan_grad = []
        for pname, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    missing_grad.append(pname)
                elif torch.isnan(p.grad).any():
                    nan_grad.append(pname)

        assert len(missing_grad) == 0, f"Params with missing gradients: {missing_grad[:5]}"
        assert len(nan_grad) == 0, f"Params with NaN gradients: {nan_grad[:5]}"

        print(f"[PASS] {name}")
        print(f"       Loss: {loss.item():.6f}")
        print(f"       Components: {loss_dict}")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 6: Overfit single batch
# ---------------------------------------------------------------------------

def check_overfit_single(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    steps: int = 50,
) -> bool:
    """Train on a single batch for N steps and verify loss decreases by >= 50%."""
    name = "check_overfit_single"
    try:
        model.train()
        batch = next(iter(dataloader))
        distorted, target = batch[0].to(device), batch[1].to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None

        for step in range(steps + 1):
            optimizer.zero_grad()
            corrected, _ = model(distorted)
            loss, _ = loss_fn(corrected, target)
            if step == 0:
                initial_loss = loss.item()
            if step == steps:
                final_loss = loss.item()
                break

            loss.backward()
            optimizer.step()

            if step in (0, steps // 2):
                print(f"       Step {step:3d}: loss = {loss.item():.6f}")

        print(f"       Step {steps:3d}: loss = {final_loss:.6f}")

        assert final_loss < 0.5 * initial_loss, (
            f"Loss did not decrease enough: {initial_loss:.6f} -> {final_loss:.6f} "
            f"(ratio {final_loss / initial_loss:.3f}, need < 0.5)"
        )

        print(f"[PASS] {name}")
        print(f"       Loss: {initial_loss:.6f} -> {final_loss:.6f} "
              f"(ratio {final_loss / initial_loss:.3f})")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 7: Inference round-trip
# ---------------------------------------------------------------------------

def check_inference_roundtrip(
    model: torch.nn.Module,
    device: str,
    tmp_dir: Optional[str] = None,
) -> bool:
    """Create a dummy image, run inference, verify output shape and value range."""
    name = "check_inference_roundtrip"
    try:
        import cv2

        model.eval()
        cleanup = tmp_dir is None
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="validate_")

        # Create and save a dummy test image
        dummy_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        input_path = os.path.join(tmp_dir, "test_input.png")
        cv2.imwrite(input_path, dummy_img)

        # Load and run inference
        loaded = cv2.imread(input_path, cv2.IMREAD_COLOR)
        loaded_rgb = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        tensor_in = image_to_tensor(loaded_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            corrected, _ = model(tensor_in)

        output_img = tensor_to_image(corrected)

        # Verify output
        assert output_img.shape == (512, 512, 3), (
            f"Output shape {output_img.shape} != expected (512, 512, 3)"
        )
        assert output_img.dtype == np.uint8, f"Output dtype {output_img.dtype} != uint8"
        assert output_img.min() >= 0 and output_img.max() <= 255, (
            f"Output range [{output_img.min()}, {output_img.max()}] outside [0, 255]"
        )

        # Save output
        output_path = os.path.join(tmp_dir, "test_output.png")
        cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        assert os.path.exists(output_path), "Output file was not created"

        # Clean up
        if cleanup:
            os.remove(input_path)
            os.remove(output_path)
            os.rmdir(tmp_dir)

        print(f"[PASS] {name}")
        print(f"       Input shape:  {loaded_rgb.shape}")
        print(f"       Output shape: {output_img.shape}, dtype: {output_img.dtype}")
        return True

    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_checks(cfg_override: Optional[Config] = None) -> bool:
    """Run all 7 sanity checks and print a summary.

    Returns True only if ALL checks pass. Target runtime: < 90 seconds.
    """
    config = cfg_override if cfg_override is not None else cfg

    print("=" * 60)
    print("  AutoHDR Lens Correction - Sanity Suite (7 checks)")
    print("=" * 60)

    set_seed(config.SEED)
    device = config.DEVICE

    # --- Setup ---------------------------------------------------------------
    print(f"\nDevice: {device}")

    print("\n[Setup] Creating model ...")
    model = LensCorrector().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"        Parameters: {total_params:,} total, {trainable_params:,} trainable")

    print("[Setup] Creating loss function ...")
    loss_fn = CompositeLoss(weights=config.loss_weights, device=device)

    print("[Setup] Creating synthetic dataloader ...")
    dataloader = _make_synthetic_dataloader(batch_size=2, n_samples=4, resolution=512)

    # --- Run checks ----------------------------------------------------------
    results = []

    print("\n--- Check 1/7: Data Pipeline ---")
    results.append(check_data_pipeline(dataloader))

    print("\n--- Check 2/7: Identity Warp ---")
    results.append(check_identity_warp(model, device))

    print("\n--- Check 3/7: Output Shapes (512 only on T4) ---")
    # Skip 768 test to save VRAM on T4
    model.eval()
    x = torch.rand(1, 3, 512, 512, device=device)
    with torch.no_grad():
        c, d = model(x)
    ok3 = c.shape == (1, 3, 512, 512) and d.shape == (1, 2, 512, 512)
    print(f"       512: corrected={c.shape}, displacement={d.shape}")
    print(f"[{'PASS' if ok3 else 'FAIL'}] check_output_shapes")
    results.append(ok3)
    del x, c, d
    torch.cuda.empty_cache()

    print("\n--- Check 4/7: Displacement Clamp ---")
    results.append(check_displacement_clamp(model, device))

    torch.cuda.empty_cache()
    print("\n--- Check 5/7: Loss Gradients ---")
    results.append(check_loss_gradients(model, loss_fn, device))

    print("\n--- Check 6/7: Overfit Single Batch ---")
    # Reuse existing model/loss to avoid OOM on T4 (don't create a second copy)
    torch.cuda.empty_cache()
    results.append(check_overfit_single(model, loss_fn, dataloader, device))

    print("\n--- Check 7/7: Inference Round-trip ---")
    results.append(check_inference_roundtrip(model, device))

    # --- Summary -------------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Summary: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("  All checks PASSED.")
    else:
        failed_indices = [i + 1 for i, r in enumerate(results) if not r]
        print(f"  FAILED checks: {failed_indices}")

    return passed == total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run validation suite and exit with appropriate code."""
    passed = run_all_checks()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
