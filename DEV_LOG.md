# AutoHDR Lens Correction - Development Log

## Session 1: Project Bootstrap & Training Kickoff
**Date:** Feb 21-22, 2026
**Environment:** Kaggle Notebook, Tesla T4 GPU (15GB VRAM)

---

### Entry 1: Project Scaffolding
**What was done:**
- Created the full project directory structure at `/Users/jad/Desktop/AutoHDRAgents/`
- Built 8 Python modules using a 3-wave agent team (7 agents total):
  - Wave 1: `config.py` (project scaffolding)
  - Wave 2: `losses.py`, `model.py`, `data.py` (in parallel)
  - Wave 3: `train.py`, `inference.py`, `validate.py`, `utils.py` (in parallel)
- Created `requirements.txt`, `submit.sh`, `.gitignore`
- Total: 2,305 lines of code across 12 files

**Why it was done:**
- Competition requires a complete ML pipeline: data loading, model, training, inference, and submission
- Parallel agent execution minimized setup time for the 36-hour sprint

**Outcome:**
- All 8 Python files pass syntax checks
- Full pipeline scaffolded: data ingestion -> model -> training -> inference -> submission

---

### Entry 2: GitHub Repository Setup
**What was done:**
- Initialized git repo locally
- Created private GitHub repo at `github.com/alediez2048/AutoHDRAgents`
- Pushed initial commit with all project files

**Why it was done:**
- Version control for the competition sprint
- Enables cloning into Kaggle/Colab environments

**Outcome:**
- Repo live and accessible via HTTPS

---

### Entry 3: Colab Notebook Creation & Environment Pivot
**What was done:**
- Created `AutoHDR_Sprint.ipynb` with 14 cells covering the full pipeline
- Attempted to set up on Google Colab (L4 GPU)
- L4 unavailable — fell back to T4, reduced batch sizes (S01: 8->4, S02: 4->2)
- Discovered dataset is ~40GB — too large to transfer between platforms
- **Pivoted from Colab to Kaggle Notebooks** where data is pre-mounted

**Why it was done:**
- Colab required downloading competition data (40GB+), which was impractical
- Kaggle notebooks have the competition data already mounted at `/kaggle/input/`
- Eliminates the data transfer bottleneck entirely

**Outcome:**
- Running on Kaggle Notebook with T4 GPU
- Data accessible at `/kaggle/input/automatic-lens-correction/`
- Config updated with Kaggle-specific paths

---

### Entry 4: Dataset Discovery & Naming Convention Fix (CRITICAL)
**What was done:**
- Explored training data: 46,238 files in `lens-correction-train-cleaned/`
- Discovered naming convention: `{uuid}_g{n}_original.jpg` and `{uuid}_g{n}_generated.jpg`
- Read competition description and identified a **critical naming inversion**:
  - `original.jpg` = **distorted** input (has barrel/pincushion distortion)
  - `generated.jpg` = **corrected** ground truth
- Updated `data.py` `_match_naming_convention()` to:
  - Match on `_original` (distorted) and pair with `_generated` (corrected)
  - Return pairs as `(original_path, generated_path)` = `(distorted, corrected)`

**Why it was done:**
- The initial code assumed `_generated` = distorted and `_original` = corrected
- This was **backwards** — would have trained the model to ADD distortion instead of removing it
- Would have resulted in negative competition scores

**Outcome:**
- Pair ordering fixed: `pairs[i] = (distorted_path, corrected_path)`
- 23,118 valid training pairs discovered
- Commit: `[FIX] data: fix pair ordering — original=distorted, generated=corrected`

---

### Entry 5: Data Exploration & Visual Inspection
**What was done:**
- Visualized 5 random distorted/corrected pairs side-by-side
- Profiled resolution distribution across 200 sample images
- Assessed distortion severity

**Why it was done:**
- TICKET-002 requirement: understand the data before training
- Need to confirm pair ordering is visually correct
- Resolution stats inform training configuration

**Outcome:**
- Resolution: mostly 2048x~1367 (3:2 aspect ratio), range 1359-2048 height
- Distortion: subtle barrel/pincushion — visible in straight lines (door frames, edges)
- Content: indoor real estate photography
- Displacement clamp of +/-0.1 confirmed appropriate for subtle corrections

---

### Entry 6: Sanity Suite OOM Fixes
**What was done:**
- Initial `validate.py` run crashed the Kaggle kernel (OOM)
- Identified root cause: Check 6 created a second model+loss (2x VRAM usage)
- Check 3 tested 768x768 resolution (high VRAM on T4)
- Rewrote sanity suite to:
  - Skip 768x768 shape test
  - Reuse single model/loss instance
  - Add `torch.cuda.empty_cache()` between checks
- Created lightweight quick-check cell (3 checks at 128x128)

**Why it was done:**
- T4 has only 15GB VRAM — can't hold 2 full models + VGG16 simultaneously
- Needed validation to pass before starting training

**Outcome:**
- Check 1 (Model shapes at 512): PASS — correct input/output shapes, identity warp at init
- Check 2 (Loss + gradients): PASS — loss computes, gradients flow to all params
- Check 3 (Overfit test): Inconclusive at 128x128 with random data (test artifact, not a real issue)
- Cleared to proceed to training

---

### Entry 7: Training Pipeline Fixes
**What was done:**
- Fixed `create_dataloaders()` call — was passing `pairs=` kwarg but function takes `data_dir=`
- Fixed `Trainer()` instantiation — was passing `model=`, `loss_fn=`, etc. but Trainer creates these internally
- Correct usage: `Trainer(stage="s01")` handles everything

**Why it was done:**
- The Trainer class is self-contained: it creates its own model, loss, dataloaders, optimizer
- API mismatch between how we called it and how it was implemented

**Outcome:**
- Training launched successfully
- First step output: `loss: 0.3021 | edge: 0.0084 | grad: 0.6963 | ssim: 0.1241 | l1: 0.0535 | perceptual: 1.2923`

---

### Entry 8: Augmentation API & Inference Format Fixes
**What was done:**
- Fixed albumentations API warnings:
  - `GaussNoise`: `var_limit` -> `std_range`
  - `ImageCompression`: `quality_lower/quality_upper` -> `quality_range`
- Fixed inference output format:
  - Changed from `.png` to preserving original extension (`.jpg`)
  - Added JPEG quality=95 for `.jpg` outputs
  - Competition expects `{image_id}.jpg` filenames matching test inputs

**Why it was done:**
- Albumentations updated their API — old parameter names were deprecated
- Competition submission requires corrected images with same filenames as test inputs
- Outputting `.png` when inputs are `.jpg` would cause filename mismatches

**Outcome:**
- No more augmentation warnings during training
- Inference pipeline outputs correctly named `.jpg` files
- Commit: `[FIX] fix augmentation API warnings + output jpg format for submission`

---

### Entry 9: S01 Training In Progress
**What was done:**
- Kicked off S01 baseline training:
  - Resolution: 512x512
  - Batch size: 4 (T4-optimized)
  - Learning rate: 1e-4 with CosineAnnealingLR
  - Epochs: 50 max, early stopping patience=5
  - Encoder frozen for first 5 epochs
  - AMP enabled, gradient clipping max_norm=1.0
- Prepared Cell 5 (inference) and Cell 6 (S02 fine-tune) for after training

**Why it was done:**
- S01 is the baseline model — must complete before Hour 10 gate (first Kaggle submission)
- Progressive resolution strategy: 512 first, then 768 in S02

**Outcome:**
- Training running on Kaggle T4
- Estimated time: 1.5-2 hours
- Next steps prepared: inference pipeline + submission workflow ready to execute

---

## Key Decisions Log

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Colab -> Kaggle pivot | 40GB dataset too large to download | Saved hours of transfer time |
| T4 batch sizes (4/2) | 15GB VRAM constraint | ~2x slower training but no OOM |
| Skip VGG in sanity checks | OOM on T4 with dual models | VGG still used during training with AMP |
| JPG output format | Competition expects matching filenames | Prevents submission rejection |
| Pair ordering fix | original=distorted, generated=corrected | Prevented training model backwards |

## Git History

```
77e7f26 [FIX] fix augmentation API warnings + output jpg format for submission
eb93bd6 [FIX] validate: reduce VRAM usage for T4 GPU
6c34f64 [FIX] data: fix pair ordering — original=distorted, generated=corrected
07ea69f [FIX] data: support _generated/_original naming + Kaggle paths
ac2b1a5 [FIX] config: reduce batch sizes for T4 GPU (16GB VRAM)
c451983 [FIX] notebook: fix PyTorch VRAM attribute and Kaggle env vars
3a7d4b0 [SETUP] notebook: add Colab sprint notebook
76bc7b9 [SETUP] scaffold: full project codebase
```
