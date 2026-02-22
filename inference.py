"""Standalone inference pipeline: load checkpoint, process test images, identity fallback, ZIP output."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from config import Config
from model import LensCorrector


# ---------------------------------------------------------------------------
# QA checks
# ---------------------------------------------------------------------------

def no_nan_inf(image: np.ndarray) -> bool:
    """Return True if image contains no NaN or Inf values."""
    return not (np.isnan(image).any() or np.isinf(image).any())


def not_blank(image: np.ndarray) -> bool:
    """Return True if image is not all-black (mean > 10)."""
    return image.mean() > 10


def not_saturated(image: np.ndarray) -> bool:
    """Return True if image is not all-white (mean < 245)."""
    return image.mean() < 245


def edge_density_ok(output_image: np.ndarray, input_image: np.ndarray) -> bool:
    """Return True if output edge density >= 50% of input edge density."""
    out_gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY) if output_image.ndim == 3 else output_image
    in_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) if input_image.ndim == 3 else input_image

    out_sobel_x = cv2.Sobel(out_gray, cv2.CV_64F, 1, 0, ksize=3)
    out_sobel_y = cv2.Sobel(out_gray, cv2.CV_64F, 0, 1, ksize=3)
    out_edges = np.abs(out_sobel_x).sum() + np.abs(out_sobel_y).sum()

    in_sobel_x = cv2.Sobel(in_gray, cv2.CV_64F, 1, 0, ksize=3)
    in_sobel_y = cv2.Sobel(in_gray, cv2.CV_64F, 0, 1, ksize=3)
    in_edges = np.abs(in_sobel_x).sum() + np.abs(in_sobel_y).sum()

    return out_edges > 0.5 * in_edges


def dimensions_match(output: np.ndarray, original_h: int, original_w: int) -> bool:
    """Return True if output dimensions match original."""
    return output.shape[0] == original_h and output.shape[1] == original_w


def run_qa_checks(
    output_image: np.ndarray,
    input_image: np.ndarray,
    original_h: int,
    original_w: int,
) -> Tuple[bool, List[str]]:
    """Run all QA checks. Return (passed, list_of_failure_names)."""
    failures: List[str] = []

    if not no_nan_inf(output_image):
        failures.append("no_nan_inf")
    if not not_blank(output_image):
        failures.append("not_blank")
    if not not_saturated(output_image):
        failures.append("not_saturated")
    if not edge_density_ok(output_image, input_image):
        failures.append("edge_density_ok")
    if not dimensions_match(output_image, original_h, original_w):
        failures.append("dimensions_match")

    passed = len(failures) == 0
    return passed, failures


# ---------------------------------------------------------------------------
# Identity fallback
# ---------------------------------------------------------------------------

def identity_fallback(input_path: str, output_path: str) -> None:
    """Copy input image unchanged to output path as a safe fallback."""
    shutil.copy2(input_path, output_path)
    print(f"FALLBACK: {input_path} — copying input as output (score ~0.3 > 0.0)")


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Load a trained LensCorrector checkpoint and run inference on test images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg or Config()

        # Resolve device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Determine model resolution from checkpoint config or default
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            if isinstance(ckpt_cfg, dict):
                self.model_resolution = ckpt_cfg.get("resolution", 512)
            elif hasattr(ckpt_cfg, "s01"):
                # It's a Config object — use the highest stage resolution available
                self.model_resolution = getattr(ckpt_cfg, "s02", ckpt_cfg.s01).resolution
            else:
                self.model_resolution = 512
        else:
            self.model_resolution = 512

        # Create model and load weights
        self.model = LensCorrector()

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            # Try loading the dict directly as a state_dict
            self.model.load_state_dict(checkpoint)
        else:
            # Raw state_dict (OrderedDict)
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Model resolution: {self.model_resolution}")

    def process_single_image(self, input_path: str, output_path: str) -> bool:
        """Process a single image through the model.

        Returns True on success, False if fallback was triggered.
        """
        # Read image (BGR -> RGB)
        bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"ERROR: Could not read {input_path}")
            identity_fallback(input_path, output_path)
            return False

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb.shape[:2]

        # Resize to model resolution (square)
        resized = cv2.resize(rgb, (self.model_resolution, self.model_resolution), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor [0, 1], add batch dim, move to device
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0  # (C, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

        # Inference
        with torch.no_grad():
            corrected, displacement = self.model(tensor)

        # Remove batch dim, convert to numpy (C, H, W) -> (H, W, C)
        corrected_np = corrected.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

        # Resize back to original dimensions
        corrected_np = cv2.resize(corrected_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Clamp and convert to uint8
        corrected_np = np.clip(corrected_np, 0.0, 1.0)
        output_uint8 = (corrected_np * 255.0).astype(np.uint8)

        # QA checks
        input_uint8 = rgb  # original input in RGB uint8
        passed, failures = run_qa_checks(output_uint8, input_uint8, original_h, original_w)

        if not passed:
            print(f"QA FAILED for {input_path}: {failures}")
            identity_fallback(input_path, output_path)
            return False

        # Save as PNG (convert RGB -> BGR for cv2)
        output_bgr = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
        return True

    def process_all(self, test_dir: str, output_dir: str) -> Dict:
        """Process all images in test_dir, saving results to output_dir.

        Returns a stats dictionary.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find all images
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        image_paths: List[str] = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))

        # Remove duplicates and sort
        image_paths = sorted(set(image_paths))

        total = len(image_paths)
        succeeded = 0
        fallback_count = 0
        failed_checks: List[str] = []

        print(f"Processing {total} images from {test_dir}")

        for i, img_path in enumerate(image_paths):
            filename = Path(img_path).stem + ".png"
            output_path = os.path.join(output_dir, filename)

            success = self.process_single_image(img_path, output_path)

            if success:
                succeeded += 1
            else:
                fallback_count += 1
                failed_checks.append(img_path)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{total} images processed")

        print(f"Processed {total} images: {succeeded} succeeded, {fallback_count} fallbacks")

        return {
            "total": total,
            "succeeded": succeeded,
            "fallback_count": fallback_count,
            "failed_checks": failed_checks,
        }


# ---------------------------------------------------------------------------
# Submission utilities
# ---------------------------------------------------------------------------

def create_submission_zip(output_dir: str, zip_path: str | None = None) -> str:
    """Zip all files in output_dir into a submission archive.

    Returns the path to the created ZIP file.
    """
    if zip_path is None:
        zip_path = output_dir.rstrip("/") + ".zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath):
                zf.write(filepath, arcname=filename)

    # Verify integrity
    with zipfile.ZipFile(zip_path, "r") as zf:
        n_files = len(zf.namelist())

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"ZIP created: {zip_path} ({n_files} files, {size_mb:.1f} MB)")

    return zip_path


def validate_submission(output_dir: str, test_dir: str) -> bool:
    """Validate that output_dir contains a valid submission.

    Checks:
      - Number of output files matches number of test files
      - All output files are valid PNG images

    Returns True if all checks pass.
    """
    # Count test images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    test_files: List[str] = []
    for ext in extensions:
        test_files.extend(glob.glob(os.path.join(test_dir, ext)))
        test_files.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    test_files = sorted(set(test_files))
    n_test = len(test_files)

    # Count output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "*")))
    output_files = [f for f in output_files if os.path.isfile(f)]
    n_output = len(output_files)

    overall = True

    # Check 1: file count
    if n_output == n_test:
        print(f"PASS: File count matches ({n_output} == {n_test})")
    else:
        print(f"FAIL: File count mismatch (output={n_output}, test={n_test})")
        overall = False

    # Check 2: all outputs are valid PNGs
    invalid_pngs = 0
    for fpath in output_files:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"FAIL: Cannot read {fpath}")
            invalid_pngs += 1

    if invalid_pngs == 0:
        print(f"PASS: All {n_output} output files are valid images")
    else:
        print(f"FAIL: {invalid_pngs} output files are invalid")
        overall = False

    return overall


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AutoHDR Lens Correction Inference Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory of test images")
    parser.add_argument("--output_dir", type=str, default="outputs/test", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Create submission ZIP after processing")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect if not set)")
    args = parser.parse_args()

    # Create pipeline
    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Process all images
    stats = pipeline.process_all(args.test_dir, args.output_dir)

    # Validate submission
    print("\n--- Submission Validation ---")
    valid = validate_submission(args.output_dir, args.test_dir)

    if not valid:
        print("\nWARNING: Submission validation failed. Review output before submitting.")

    # Optionally create ZIP
    if args.zip:
        print("\n--- Creating Submission ZIP ---")
        create_submission_zip(args.output_dir)


if __name__ == "__main__":
    main()
