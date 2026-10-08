"""
Preprocesses CBSD68 (grayscale variant: "BSD68") + its conventional
companion training set (BSD400) into the same 128x128 float32 [0,1]
.npy format used by nodule3d_128/brain_tumor_128, so
prepare_cross_dataset_data.py can run unmodified on this dataset too.

SOURCE (logged here, not hidden): SaoYan/DnCNN-PyTorch on GitHub
(https://github.com/SaoYan/DnCNN-PyTorch), a widely-used, standard
reference reimplementation of Zhang et al.'s DnCNN - cloned read-only to
$MAP_DATA_ROOT/cbsd68_raw/. data/train/ = 400 grayscale 180x180
patches (the standard "BSD400" training set used throughout the
denoising literature alongside BSD68); data/Set68/ = 68 grayscale
variable-size images (the standard "BSD68"/"CBSD68-grayscale" TEST-ONLY
benchmark set).

DEVIATION FROM THE BRIEF, LOGGED NOT HIDDEN: CBSD68 itself is
conventionally a 68-image TEST-ONLY set and was never intended to supply
its own training data. Per standard practice in the denoising literature
(DnCNN, FFDNet, and most follow-on papers), we pair it with BSD400 as the
training/validation source and hold BSD68 out ENTIRELY as the final test
set (never seen during training) - this is the standard, defensible
convention, not an improvised substitute. BSD400 (400 images) is itself
well short of the brief's "8,000-slice subsample," logged as the same
class of unavoidable data-availability deviation as nodule3d_128/
brain_tumor_128.

Split: BSD400's 400 images -> 360 train / 40 val (90/10, seeded shuffle).
BSD68's 68 images -> all 68 held out as test (matching the standard
convention of never training or validating on BSD68).
"""
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
from pathlib import Path

import numpy as np
from PIL import Image

RAW_DIR = Path(os.path.expanduser(DATA_ROOT + "/cbsd68_raw/DnCNN-PyTorch/data"))
OUT_DIR = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets"))
IMG_SIZE = 128
SEED = 42


def load_and_resize(paths):
    arr = []
    for p in paths:
        im = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(arr)


def main():
    train_paths = sorted((RAW_DIR / "train").glob("*.png"))
    test_paths = sorted((RAW_DIR / "Set68").glob("*.png"))
    assert len(train_paths) == 400, f"expected 400 BSD400 images, found {len(train_paths)}"
    assert len(test_paths) == 68, f"expected 68 BSD68 images, found {len(test_paths)}"

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_paths))
    n_val = 40
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_imgs = load_and_resize([train_paths[i] for i in train_idx])
    val_imgs = load_and_resize([train_paths[i] for i in val_idx])
    test_imgs = load_and_resize(test_paths)

    np.save(OUT_DIR / "cbsd68_128_train.npy", train_imgs)
    np.save(OUT_DIR / "cbsd68_128_val.npy", val_imgs)
    np.save(OUT_DIR / "cbsd68_128_test.npy", test_imgs)

    print(f"[prepare_cbsd68] train={train_imgs.shape} val={val_imgs.shape} test={test_imgs.shape}")
    print(f"[prepare_cbsd68] pixel range check: train min/max = {train_imgs.min():.4f}/{train_imgs.max():.4f}")
    print(f"[prepare_cbsd68] saved to {OUT_DIR}/cbsd68_128_{{train,val,test}}.npy")


if __name__ == "__main__":
    main()
