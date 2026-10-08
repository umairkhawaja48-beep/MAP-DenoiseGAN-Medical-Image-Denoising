#!/bin/bash
# Train the per-dataset noise-level estimators used for estimated routing.
# Set MAP_DATA_ROOT / MAP_RESULTS_ROOT first (see README).
set -e
cd "$(dirname "$0")/../src"
python train_noise_estimator.py --epochs 30
