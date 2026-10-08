"""
Downstream BraTS segmentation for the RANGE-trained Restormer-lite.

Necessary for honesty: the paper's clinical claim ("classical baselines
collapse at sigma=0.20 while MAP-DenoiseGAN holds") was measured against
baselines trained at a single fixed sigma=0.1, i.e. it inherits exactly the
out-of-distribution confound that Experiment 1 was run to remove for PSNR. If
a range-trained baseline also holds up downstream, the clinical claim must be
restated. Inference only.

Appends to tables/table5_downstream_dice.csv's companion file.
"""
import csv
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

MAP_SCRIPTS = Path(EXTERNAL_DIR)
WIGAN = Path(EXTERNAL_DIR)
for p in (str(MAP_SCRIPTS), str(WIGAN), str(WIGAN / "phase2")):
    sys.path.insert(0, p)

from unet_model import UNet2D, dice_coefficient  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402

DATA = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
P2 = Path(os.path.expanduser(RESULTS_ROOT + "/wigan_phase2"))
CKPT = Path(os.path.expanduser(
    RESULTS_ROOT + "/map_project/brats_extra/restormer_range/checkpoints/restormer_range_best.pt"))
OUT = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
SIGMAS = [0.05, 0.10, 0.20]
BATCH = 100


def hausdorff(pred, gt):
    diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    pe, ge = pred.sum() == 0, gt.sum() == 0
    if pe and ge:
        return 0.0
    if pe or ge:
        return diag
    return float(max(distance_transform_edt(1 - gt)[pred.astype(bool)].max(),
                     distance_transform_edt(1 - pred)[gt.astype(bool)].max()))


def apply(model, imgs, device):
    outs = []
    t = torch.from_numpy(imgs).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), BATCH):
            outs.append(model(t[i:i + BATCH].to(device)).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = np.load(DATA / "phase2_test_5000_mask.npy").astype(np.float32)

    seg = UNet2D().to(device)
    seg.load_state_dict(torch.load(P2 / "segmentation_unet" / "checkpoints" / "unet_best.pt",
                                   map_location=device, weights_only=True))
    seg.eval()

    den = RestormerLite().to(device)
    den.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    den.eval()
    print(f"[downstream_range] loaded {CKPT}", flush=True)

    rows = []
    for sigma in SIGMAS:
        tag = f"{sigma:.2f}".replace("0.", "0")
        degraded = np.load(DATA / f"degraded_sigma{tag}.npy").astype(np.float32)
        probs = apply(seg, apply(den, degraded, device), device)
        pred = (probs > 0.5).astype(np.float32)
        d = dice_coefficient(pred, masks)
        h = np.array([hausdorff(pred[i], masks[i]) for i in range(len(masks))])
        rows.append({"model": "restormer_range", "sigma": sigma,
                     "mean_dice": round(float(d.mean()), 4), "std_dice": round(float(d.std()), 4),
                     "mean_hausdorff": round(float(h.mean()), 3),
                     "std_hausdorff": round(float(h.std()), 3), "n_images": len(masks)})
        print(f"[downstream_range] sigma={sigma}: Dice={d.mean():.4f}+-{d.std():.4f} "
              f"HD={h.mean():.3f}", flush=True)

    out = OUT / "table5_downstream_range.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[downstream_range] wrote {out}")


if __name__ == "__main__":
    main()
