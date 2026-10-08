"""
Builds the FIXED-sigma (not range) test sets at sigma in {0.05, 0.10, 0.20}
used for the final comparison table, mirroring exactly the evaluation
protocol of the original MAP-DenoiseGAN feasibility experiment's
eval_map_denoise.py (wigan_project/scripts/map_denoise/, read-only
reference - not modified): each fixed-sigma test set is evaluated against
its "matching" specialized agent (0.05->agent1_low, 0.10->agent2_medium,
0.20->agent3_high) via KNOWN routing, plus the joint model and classical
baselines, all on the SAME images.

Uses the dataset's own held-out test split (already-saved
test_joint_clean.npy from prepare_cross_dataset_data.py - the full test
set's clean images, unmodified copy regardless of which agent's file it's
read from) as the source of clean images.
"""
import argparse
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np

MAP_PROJECT_SCRIPTS = Path(EXTERNAL_DIR)
sys.path.insert(0, str(MAP_PROJECT_SCRIPTS))
from utils import add_noise  # noqa: E402

DATA_BASE = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
FIXED_SIGMAS = [0.05, 0.10, 0.20]
SEED = 4242


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    data_dir = DATA_BASE / args.dataset
    clean = np.load(data_dir / "test_joint_clean.npy").astype(np.float32)

    for sigma in FIXED_SIGMAS:
        noisy = np.stack([
            add_noise(clean[i:i + 1], gaussian_sigma=sigma, seed=int(SEED + sigma * 10000) * 100000 + i)[0]
            for i in range(len(clean))
        ])
        tag = f"{sigma:.2f}".replace("0.", "0")
        np.save(data_dir / f"test_fixedsigma{tag}_clean.npy", clean)
        np.save(data_dir / f"test_fixedsigma{tag}_noisy.npy", noisy)
        print(f"[prepare_fixed_sigma_eval] {args.dataset}: sigma={sigma} n={len(clean)} "
              f"saved test_fixedsigma{tag}_{{clean,noisy}}.npy", flush=True)


if __name__ == "__main__":
    main()
