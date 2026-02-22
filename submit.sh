#!/bin/bash
# AutoHDR Submission Script
# Usage: bash submit.sh <checkpoint_path> <test_dir>
echo "Running inference..."
python inference.py --checkpoint "$1" --test_dir "$2" --output_dir outputs/test --zip
echo "Inference complete. ZIP created at outputs/test.zip"
echo "NEXT STEPS:"
echo "1. Upload outputs/test.zip to https://bounty.autohdr.com"
echo "2. Download submission.csv from scoring service"
echo "3. Submit CSV to Kaggle leaderboard"
