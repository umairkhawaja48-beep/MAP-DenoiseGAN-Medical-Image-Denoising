"""
Final comparison eval for one cross-dataset dataset, mirroring exactly the
protocol of the original MAP-DenoiseGAN feasibility experiment's
eval_map_denoise.py (wigan_project/scripts/map_denoise/, read-only
reference only).

At each fixed sigma in {0.05, 0.10, 0.20}, compares (on the SAME images):
  - none (no denoising)
  - dncnn, restormer, swinir (this dataset's own from-scratch baselines,
    trained at fixed sigma=0.1 on this dataset's own training set)
  - map_denoise_<agent> (the specialized agent matching that sigma via
    KNOWN routing - agent1_low for 0.05, agent2_medium for 0.10,
    agent3_high for 0.20 - isolating specialization/inheritance quality
    from router accuracy, exactly as the original protocol does)
  - single_agent_joint (trained on the full noise range, same budget)

Output: $MAP_RESULTS_ROOT/map_project/cross_dataset/<dataset>/comparison_results.csv
"""
import argparse
import csv
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch

MAP_PROJECT_SCRIPTS = Path(EXTERNAL_DIR)
sys.path.insert(0, str(MAP_PROJECT_SCRIPTS))
from models import UNet2D  # noqa: E402
from utils import batch_psnr_ssim  # noqa: E402

WIGAN_PHASE2 = Path(EXTERNAL_DIR)
sys.path.insert(0, str(WIGAN_PHASE2))
from dncnn_model import DnCNN  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402
from swinir_lite_model import SwinIRLite  # noqa: E402

DATA_BASE = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
RESULTS_BASE = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))
EVAL_BATCH = 200

SIGMA_TO_AGENT = {0.05: "agent1_low", 0.10: "agent2_medium", 0.20: "agent3_high"}


def denoise_full_image(model, noisy_np, device):
    outs = []
    t = torch.from_numpy(noisy_np).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), EVAL_BATCH):
            chunk = t[i:i + EVAL_BATCH].to(device)
            out = model(chunk)
            outs.append(out.cpu().numpy()[:, 0])
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = DATA_BASE / args.dataset
    out_dir = RESULTS_BASE / args.dataset
    ckpt_root = out_dir

    dncnn = DnCNN().to(device)
    dncnn.load_state_dict(torch.load(ckpt_root / "dncnn" / "checkpoints" / "dncnn_best.pt",
                                      map_location=device, weights_only=True))
    dncnn.eval()

    restormer = RestormerLite().to(device)
    restormer.load_state_dict(torch.load(ckpt_root / "restormer" / "checkpoints" / "restormer_best.pt",
                                          map_location=device, weights_only=True))
    restormer.eval()

    swinir = SwinIRLite().to(device)
    swinir.load_state_dict(torch.load(ckpt_root / "swinir" / "checkpoints" / "swinir_best.pt",
                                       map_location=device, weights_only=True))
    swinir.eval()

    agents = {}
    for name in ["agent1_low", "agent2_medium", "agent3_high"]:
        m = UNet2D().to(device)
        ckpt = ckpt_root / name / "checkpoints" / "generator_best.pt"
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        agents[name] = m

    single_agent = UNet2D().to(device)
    single_agent.load_state_dict(torch.load(ckpt_root / "joint" / "checkpoints" / "generator_best.pt",
                                             map_location=device, weights_only=True))
    single_agent.eval()

    rows = []
    for sigma, agent_name in SIGMA_TO_AGENT.items():
        tag = f"{sigma:.2f}".replace("0.", "0")
        clean = np.load(data_dir / f"test_fixedsigma{tag}_clean.npy").astype(np.float32)
        degraded = np.load(data_dir / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32)
        psnr_n, ssim_n = batch_psnr_ssim(clean, degraded)

        methods = {
            "none": degraded,
            "dncnn": denoise_full_image(dncnn, degraded, device),
            "restormer": denoise_full_image(restormer, degraded, device),
            "swinir": denoise_full_image(swinir, degraded, device),
            f"map_denoise_{agent_name}": denoise_full_image(agents[agent_name], degraded, device),
            "single_agent_joint": denoise_full_image(single_agent, degraded, device),
        }

        for method_name, denoised in methods.items():
            psnr_d, ssim_d = batch_psnr_ssim(clean, denoised)
            rows.append({
                "dataset": args.dataset, "sigma": sigma, "method": method_name,
                "psnr_noisy": round(psnr_n, 4), "psnr_denoised": round(psnr_d, 4),
                "delta_psnr": round(psnr_d - psnr_n, 4),
                "ssim_noisy": round(ssim_n, 5), "ssim_denoised": round(ssim_d, 5),
                "delta_ssim": round(ssim_d - ssim_n, 5),
            })
            print(f"[eval_cross_dataset] {args.dataset} sigma={sigma} method={method_name}: "
                  f"PSNR {psnr_n:.2f}->{psnr_d:.2f} (dPSNR={psnr_d-psnr_n:+.2f}) "
                  f"SSIM {ssim_n:.3f}->{ssim_d:.3f}", flush=True)

    out_csv = out_dir / "comparison_results.csv"
    fields = ["dataset", "sigma", "method", "psnr_noisy", "psnr_denoised", "delta_psnr",
              "ssim_noisy", "ssim_denoised", "delta_ssim"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[eval_cross_dataset] wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
