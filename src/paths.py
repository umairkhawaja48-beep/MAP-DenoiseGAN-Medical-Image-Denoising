"""
Central path configuration for MAP-DenoiseGAN.

All scripts read their data / results / external-code locations from here.
Override any of them with environment variables, e.g.

    export MAP_DATA_ROOT=/path/to/data
    export MAP_RESULTS_ROOT=/path/to/outputs
    export MAP_EXTERNAL_DIR=/path/to/baseline/model/files
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Datasets (prepared .npy files, raw downloads)
DATA_ROOT = os.environ.get("MAP_DATA_ROOT", str(REPO_ROOT / "data"))

# Checkpoints, CSV results, generated tables and figures
RESULTS_ROOT = os.environ.get("MAP_RESULTS_ROOT", str(REPO_ROOT / "outputs"))

# Baseline / shared model definitions (DnCNN, Restormer-lite, SwinIR-lite,
# segmentation UNet, MAP agent definitions). See src/external/README.md.
EXTERNAL_DIR = os.environ.get("MAP_EXTERNAL_DIR", str(REPO_ROOT / "src" / "external"))
