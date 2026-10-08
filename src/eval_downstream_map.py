"""
BSPC Task C1.3 / Table 5: downstream clinical-task (BraTS tumour segmentation)
evaluation of the MAP-DenoiseGAN agents.

This was never run for MAP-DenoiseGAN before -- Phase 2's harness only
evaluated DnCNN / Restormer-lite / SwinIR-lite / WI-QGAN. INFERENCE ONLY: it
loads existing trained checkpoints and the existing trained segmentation
U-Net; nothing is trained here.

Protocol is identical to Phase 2's eval_downstream.py (same fixed degraded
test sets, same segmentation U-Net checkpoint, same Dice/Hausdorff
definitions) so the numbers drop straight into the same table as the
already-published classical-baseline rows. Phase 2 files are imported
read-only and not modified; results are written to map_project's own
directory, never to Phase 2's downstream_results.csv.

Usage:  python eval_downstream_map.py
"""
import csv
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

WIGAN_SCRIPTS = Path(EXTERNAL_DIR)
PHASE2_DIR = WIGAN_SCRIPTS / "phase2"
for p in (str(WIGAN_SCRIPTS), str(PHASE2_DIR)):
    sys.path.insert(0, p)

from unet_model import UNet2D, dice_coefficient  # noqa: E402

DATA_DIR = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
PHASE2_RESULTS = Path(os.path.expanduser(RESULTS_ROOT + "/wigan_phase2"))
UNET_CKPT = PHASE2_RESULTS / "segmentation_unet" / "checkpoints" / "unet_best.pt"
MAP_RESULTS = Path(os.path.expanduser(RESULTS_ROOT + "/map_denoise_feasibility"))
OUT_DIR = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
OUT_CSV = OUT_DIR / "table5_downstream_dice.csv"

SIGMA_LEVELS = [0.05, 0.10, 0.20]
EVAL_BATCH = 100
SIGMA_TO_AGENT = {0.05: "agent1_low", 0.10: "agent2_medium", 0.20: "agent3_high"}


def hausdorff_distance(pred_mask, gt_mask):
    diag = np.sqrt(pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2)
    pred_empty = pred_mask.sum() == 0
    gt_empty = gt_mask.sum() == 0
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return diag
    dt_gt = distance_transform_edt(1 - gt_mask)
    dt_pred = distance_transform_edt(1 - pred_mask)
    return float(max(dt_gt[pred_mask.astype(bool)].max(), dt_pred[gt_mask.astype(bool)].max()))


def load_unet(path, device):
    m = UNet2D().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


def apply_model(model, images_np, device):
    outs = []
    t = torch.from_numpy(images_np).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), EVAL_BATCH):
            outs.append(model(t[i:i + EVAL_BATCH].to(device)).cpu().numpy()[:, 0])
    return np.concatenate(outs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = np.load(DATA_DIR / "phase2_test_5000.npy").astype(np.float32)
    masks = np.load(DATA_DIR / "phase2_test_5000_mask.npy").astype(np.float32)
    seg_unet = load_unet(UNET_CKPT, device)
    print(f"[eval_downstream_map] test set {clean.shape}, segmentation U-Net from {UNET_CKPT}", flush=True)

    agents = {}
    for name in ["agent1_low", "agent2_medium", "agent3_high", "single_agent_joint"]:
        ckpt = MAP_RESULTS / name / "checkpoints" / "generator_best.pt"
        agents[name] = load_unet(ckpt, device)
        print(f"[eval_downstream_map] loaded MAP model {name}", flush=True)

    rows = []
    for sigma in SIGMA_LEVELS:
        tag = f"{sigma:.2f}".replace("0.", "0")
        degraded = np.load(DATA_DIR / f"degraded_sigma{tag}.npy").astype(np.float32)
        matching = SIGMA_TO_AGENT[sigma]

        variants = {
            "no_denoising": degraded,
            "oracle_clean": clean,
            f"map_denoisegan_routed({matching})": apply_model(agents[matching], degraded, device),
            "map_single_agent_joint": apply_model(agents["single_agent_joint"], degraded, device),
        }

        for label, seg_input in variants.items():
            probs = apply_model(seg_unet, seg_input, device)
            pred = (probs > 0.5).astype(np.float32)
            dices = dice_coefficient(pred, masks)
            hds = np.array([hausdorff_distance(pred[i], masks[i]) for i in range(len(masks))])
            rows.append({
                "model": label, "sigma": sigma,
                "mean_dice": round(float(dices.mean()), 4), "std_dice": round(float(dices.std()), 4),
                "mean_hausdorff": round(float(hds.mean()), 3), "std_hausdorff": round(float(hds.std()), 3),
                "n_images": len(masks),
            })
            print(f"[eval_downstream_map] sigma={sigma} {label}: Dice={dices.mean():.4f}"
                  f"+-{dices.std():.4f} HD={hds.mean():.3f}", flush=True)

    fields = ["model", "sigma", "mean_dice", "std_dice", "mean_hausdorff", "std_hausdorff", "n_images"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[eval_downstream_map] wrote {len(rows)} rows to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
