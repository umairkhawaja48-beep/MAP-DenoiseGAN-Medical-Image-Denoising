"""
Cross-dataset generalization: prepares per-agent noise-range datasets for
ANY base dataset (nodule3d_128, brain_tumor_128, cbsd68), reusing the
EXACT same noise ranges and protocol as the original BraTS-based
MAP-DenoiseGAN feasibility experiment:
  agent1_low:    sigma ~ Uniform[0.01, 0.05)
  agent2_medium: sigma ~ Uniform[0.05, 0.15)
  agent3_high:   sigma ~ Uniform[0.15, 0.25)
  joint:         sigma ~ Uniform[0.01, 0.25)

IMPORTANT DEVIATION FROM THE BRIEF, LOGGED HERE NOT HIDDEN: the brief
asks for "the same 8,000-slice subsample" as BraTS for every dataset.
nodule3d_128 only has 1,158 training images total; brain_tumor_128 only
has 177. Neither can supply an 8,000-image subsample. This script uses
EACH dataset's FULL available training set instead of subsampling to
8,000 - the honest choice given the data that actually exists, not a
silent reinterpretation. This means BraTS (8,000 train images) and these
two datasets are NOT trained on equal-sized data - a real, reportable
difference the cross-dataset comparison must account for, not obscure.
"""
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np

MAP_PROJECT_SCRIPTS = Path(EXTERNAL_DIR)
sys.path.insert(0, str(MAP_PROJECT_SCRIPTS))
from utils import add_noise  # noqa: E402

WIGAN_DATA_DIR = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets"))
OUT_BASE = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))

AGENT_RANGES = {
    "agent1_low": (0.01, 0.05),
    "agent2_medium": (0.05, 0.15),
    "agent3_high": (0.15, 0.25),
}
JOINT_RANGE = (0.01, 0.25)
SEED = 42

DATASET_FILES = {
    "nodule3d_128": {
        "train": "nodule3d_128_train.npy", "val": "nodule3d_128_val.npy", "test": "nodule3d_128_test.npy",
    },
    "brain_tumor_128": {
        "train": "brain_tumor_128_train.npy", "val": "brain_tumor_128_val.npy", "test": "brain_tumor_128_test.npy",
    },
    "cbsd68_128": {
        "train": "cbsd68_128_train.npy", "val": "cbsd68_128_val.npy", "test": "cbsd68_128_test.npy",
    },
}


def make_range_noisy(clean, lo, hi, seed):
    rng = np.random.default_rng(seed)
    sigmas = rng.uniform(lo, hi, size=len(clean))
    noisy = np.stack([add_noise(clean[i:i + 1], gaussian_sigma=float(sigmas[i]), seed=seed * 100000 + i)[0]
                       for i in range(len(clean))])
    return noisy, sigmas


def build_for_split(dataset_name, split_name, source_path, seed_offset, out_dir):
    clean = np.load(source_path).astype(np.float32)
    manifest = {"source": str(source_path), "n_images": len(clean), "agents": {}}

    for name, (lo, hi) in AGENT_RANGES.items():
        noisy, sigmas = make_range_noisy(clean, lo, hi, seed=SEED + seed_offset)
        np.save(out_dir / f"{split_name}_{name}_clean.npy", clean)
        np.save(out_dir / f"{split_name}_{name}_noisy.npy", noisy)
        manifest["agents"][name] = {"range": [lo, hi], "mean_sigma": float(sigmas.mean())}
        print(f"[prepare_cross_dataset] {dataset_name}/{split_name}/{name}: range=[{lo},{hi}) "
              f"n={len(clean)} mean_sigma={sigmas.mean():.4f}", flush=True)

    noisy_joint, sigmas_joint = make_range_noisy(clean, *JOINT_RANGE, seed=SEED + seed_offset + 500)
    np.save(out_dir / f"{split_name}_joint_clean.npy", clean)
    np.save(out_dir / f"{split_name}_joint_noisy.npy", noisy_joint)
    manifest["agents"]["joint"] = {"range": list(JOINT_RANGE), "mean_sigma": float(sigmas_joint.mean())}
    print(f"[prepare_cross_dataset] {dataset_name}/{split_name}/joint: range={JOINT_RANGE} "
          f"n={len(clean)} mean_sigma={sigmas_joint.mean():.4f}", flush=True)

    return manifest


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_FILES.keys()))
    args = ap.parse_args()

    files = DATASET_FILES[args.dataset]
    out_dir = OUT_BASE / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = {"dataset": args.dataset}
    for i, split in enumerate(["train", "val", "test"]):
        source_path = WIGAN_DATA_DIR / files[split]
        full_manifest[split] = build_for_split(args.dataset, split, source_path, seed_offset=i, out_dir=out_dir)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"[prepare_cross_dataset] DONE for {args.dataset}, manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
