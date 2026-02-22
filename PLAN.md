# AutoHDR Lens Correction - Project Plan

**Competition:** Kaggle — Automatic Lens Correction
**Deadline:** Sunday, Feb 22, 2026, 2:00 PM
**Environment:** Kaggle Notebook, Tesla T4 GPU (15GB VRAM)
**Repository:** github.com/alediez2048/AutoHDRAgents

---

## Phase 1: Foundation (Hours 0-6) — MUST HAVE

### TICKET-001: Environment & Project Setup
**Status:** COMPLETED
- [x] Create folder structure (8 Python modules + configs)
- [x] Set up GitHub repo (private, alediez2048/AutoHDRAgents)
- [x] Install dependencies (kornia, albumentations, opencv, scikit-image, timm)
- [x] Freeze random seeds (42)
- [x] Pivot from Colab to Kaggle (data already mounted)
- [x] Configure Kaggle paths in config.py

### TICKET-002: Data Exploration & Visual Inspection
**Status:** COMPLETED
- [x] Inspect training pairs (23,118 pairs discovered)
- [x] Profile resolution distribution (2048x~1367, 3:2 aspect ratio)
- [x] Assess distortion severity (subtle barrel/pincushion)
- [x] Count test images (1,000 in test-originals/)
- [x] Fix critical pair ordering bug (original=distorted, generated=corrected)

### TICKET-003: Data Pipeline & Validation Split
**Status:** COMPLETED
- [x] LensDataset class with paired loading and augmentations
- [x] DataLoader factory (batch_size=4 for T4, num_workers=4)
- [x] 80/20 train/val split (seed=42)
- [x] Geometry-safe augmentations (flip, crop, brightness, noise, compression)
- [x] Fix albumentations API (std_range, quality_range)
- [x] Validation metrics (edge similarity, SSIM, pixel MAE, composite)

### TICKET-004: Composite Loss Function
**Status:** COMPLETED
- [x] EdgeLoss (Sobel L1) — weight 0.40
- [x] GradientOrientationLoss (atan2 angular diff) — weight 0.20
- [x] SSIMLoss (1 - SSIM) — weight 0.20
- [x] L1PixelLoss — weight 0.10
- [x] PerceptualLoss (VGG16 features) — weight 0.10
- [x] CompositeLoss wrapper with configurable weights

### TICKET-005: Model Architecture (ResNet50-UNet)
**Status:** COMPLETED
- [x] Encoder: ResNet50 (ImageNet pre-trained), multi-scale features
- [x] Decoder: UNet-style with skip connections, 2-channel output (dx, dy)
- [x] WarpLayer: base_grid + displacement -> grid_sample
- [x] Displacement clamp: +/-0.1
- [x] Identity warp verified at initialization

### TICKET-006: Sanity Check Suite
**Status:** COMPLETED (abbreviated for T4)
- [x] Model shape consistency at 512x512
- [x] Loss computation + gradient flow
- [x] Overfit test (inconclusive at 128x128 — artifact of tiny resolution)
- [x] OOM fixes for T4 (single model instance, VRAM cleanup)
- [ ] ~~Full 7-check suite~~ (skipped — OOM on T4 with dual models)

---

## Phase 2: Baseline Training & First Submission (Hours 6-12) — MUST HAVE

### TICKET-007: Training Loop & Checkpoint Infrastructure
**Status:** COMPLETED
- [x] AdamW optimizer (lr=1e-4, wd=1e-4)
- [x] CosineAnnealingLR scheduler
- [x] AMP (mixed precision) for memory efficiency
- [x] Gradient clipping (max_norm=1.0)
- [x] Encoder freeze for first 5 epochs
- [x] Checkpoint save every 500 steps + best model tracking
- [x] Resume-from-checkpoint support
- [x] Per-epoch metric logging to JSON
- [x] Visual comparison every 5 epochs (best/worst predictions)

### TICKET-008: S01 Baseline Training (512x512)
**Status:** IN PROGRESS
- [x] Training launched (batch=4, 512x512, 50 epochs max)
- [ ] Monitor val metrics per epoch
- [ ] Early stopping if no improvement for 5 epochs
- [ ] Best checkpoint saved to /kaggle/working/checkpoints/best.pth
- **Estimated completion:** ~1.5-2 hours on T4
- **First step loss:** 0.3021

### TICKET-009: Inference Pipeline & Identity Fallback
**Status:** COMPLETED (code ready, awaiting checkpoint)
- [x] Standalone inference: load checkpoint -> process test images -> save outputs
- [x] Resize to model input -> forward pass -> resize back to original dims
- [x] Output as .jpg with quality=95 (matching test filenames)
- [x] QA checks: no NaN/Inf, not blank, not saturated, edge density, dimensions
- [x] Identity fallback: copy input unchanged if QA fails
- [x] ZIP creation + integrity verification
- [ ] Run on actual test set (waiting for trained model)

### TICKET-010: First Kaggle Submission (HOUR 10 GATE)
**Status:** PENDING
- [ ] Run inference on 1,000 test images with best S01 checkpoint
- [ ] Run pre-submission QA checks
- [ ] Spot-check 10 outputs visually
- [ ] Create submission.zip
- [ ] Upload to bounty.autohdr.com -> download CSV
- [ ] Submit CSV to Kaggle
- [ ] Record public score
- [ ] Git tag: sub-001-score-X.XX

**Prepared cells ready to execute:**
```
Cell 5: Inference + ZIP creation
Cell 6: S02 fine-tune
```

---

## Phase 3: Progressive Resolution & Experiments (Hours 12-22) — SHOULD HAVE

### TICKET-011: S02 Fine-Tune (768x768)
**Status:** PENDING
- [ ] Load best S01 checkpoint
- [ ] Fine-tune at 768x768 (10-15 epochs, lr=5e-5, batch=2 on T4)
- [ ] Early stopping patience=5
- [ ] Compare val metrics vs S01

### TICKET-012: S02 Submission & Proxy Calibration
**Status:** PENDING
- [ ] Run inference with S02 checkpoint
- [ ] Submit to Kaggle
- [ ] Compare S01 vs S02 public scores
- [ ] Calibrate proxy drift (local vs public score delta)
- [ ] Identify current best model

### TICKET-013: X01 — Edge Loss Weight Tuning (0.40 -> 0.50)
**Status:** PENDING
- [ ] Clone best config, increase edge weight to 0.50
- [ ] Train 10-15 epochs from best checkpoint
- [ ] Evaluate against promotion criteria

### TICKET-014: X02 — Gradient Loss Weight Tuning (0.20 -> 0.25)
**Status:** PENDING
- [ ] Clone best config, increase gradient weight to 0.25
- [ ] Train and evaluate

### TICKET-015: X03 — Post-Warp Unsharp Mask
**Status:** PENDING
- [ ] Add unsharp mask post-processing (sigma=1, strength=0.3)
- [ ] Evaluate edge proxy improvement
- [ ] Reject if artifacts appear

### TICKET-016: X04 — Brown-Conrady Parametric Baseline
**Status:** PENDING (P2 — only if time allows)
- [ ] Implement parametric model (k1, k2, k3 coefficients)
- [ ] Train on same data
- [ ] Evaluate as ensemble candidate

### TICKET-017: Architecture Lock (HOUR 22 GATE)
**Status:** PENDING
- [ ] Review all experiment results
- [ ] Identify best model with score evidence
- [ ] Lock architecture — no new models after this

---

## Phase 4: Hardening & Ensemble (Hours 22-30) — SHOULD HAVE

### TICKET-018: Stochastic Weight Averaging (SWA)
**Status:** PENDING
- [ ] Apply SWA to best locked model
- [ ] Average last 3-5 checkpoints

### TICKET-019: Displacement Field Ensemble
**Status:** PENDING (requires TICKET-016)
- [ ] Average displacement fields from S-track + parametric
- [ ] Evaluate composite score

### TICKET-020: Test-Time Augmentation (TTA)
**Status:** PENDING
- [ ] D4 symmetry TTA (original, H-flip, V-flip, 180-rotation)
- [ ] Median average of displacement fields

### TICKET-021: Full Test Inference + QA Pass
**Status:** PENDING
- [ ] Run inference with best model (+ SWA/TTA/ensemble if ready)
- [ ] Full QA pass on all 1,000 outputs
- [ ] Submit and confirm score

---

## Phase 5: Lock & Ship (Hours 30-36) — MUST HAVE

### TICKET-022: Backup Model Submission
**Status:** PENDING
- [ ] Submit second-best model as insurance

### TICKET-023: Reproducibility Verification
**Status:** PENDING
- [ ] Re-run inference — outputs must be identical
- [ ] Verify all seeds fixed
- [ ] Verify requirements.txt complete

### TICKET-024: Final Video Recording
**Status:** PENDING
- [ ] Record < 1 min video (problem, architecture, key insight, results)

### TICKET-025: Final Submission & Code Cleanup
**Status:** PENDING
- [ ] Clean code, ensure runnable
- [ ] Final git push
- [ ] Submit video + code via Google Form

---

## Progress Summary

| Phase | Tickets | Completed | In Progress | Pending |
|-------|---------|-----------|-------------|---------|
| 1: Foundation | 6 | 6 | 0 | 0 |
| 2: Baseline | 4 | 2 | 1 | 1 |
| 3: Experiments | 5 | 0 | 0 | 5 |
| 4: Hardening | 4 | 0 | 0 | 4 |
| 5: Ship | 4 | 0 | 0 | 4 |
| **Total** | **23** | **8** | **1** | **14** |

## Critical Path (Next Actions)

```
[NOW]  TICKET-008: S01 training running (~1.5-2 hrs)
  |
  v
[NEXT] TICKET-010: First submission (inference + upload to bounty.autohdr.com)
  |
  v
[THEN] TICKET-011: S02 fine-tune at 768x768
  |
  v
[THEN] TICKET-012: S02 submission + compare scores
  |
  v
[IF TIME] TICKET-013-016: X-track experiments
```

## Non-Negotiable Gates

| Gate | Ticket | Deadline | Status |
|------|--------|----------|--------|
| First Kaggle submission | TICKET-010 | Hour 10 | PENDING |
| Architecture lock | TICKET-017 | Hour 22 | PENDING |
| Experiment lock | — | Hour 30 | PENDING |
| Final submission | TICKET-025 | Hour 36 | PENDING |

## Architecture Summary

```
Input Image (distorted)
    |
    v
[ResNet50 Encoder] (ImageNet pretrained, frozen first 5 epochs)
    |
    v  (multi-scale features f1-f4)
[UNet Decoder] (skip connections, ConvTranspose2d upsampling)
    |
    v  (2-channel displacement field, clamped +/-0.1)
[WarpLayer] (base_grid + displacement -> F.grid_sample)
    |
    v
Output Image (corrected)
```

## Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| config.py | 100 | Dataclass config (stages, loss weights, paths) |
| model.py | 228 | ResNet50-UNet encoder-decoder + WarpLayer |
| data.py | 354 | Dataset, augmentations, dataloaders, metrics |
| losses.py | 171 | 5-component composite loss |
| train.py | 465 | Training loop, checkpointing, early stopping |
| inference.py | 366 | Inference pipeline, QA checks, identity fallback |
| validate.py | 445 | 7-check sanity suite |
| utils.py | 166 | Seed, tensor/image conversion, visualization |
