"""
Tasks 2-4: estimator accuracy on the fixed-sigma test sets, and PSNR/SSIM
under four routing strategies.

Strategies, all reusing the EXISTING trained agents (nothing is retrained):
  A  expert1_only  - always agent1_low
  B  random        - uniformly random agent per image
  C  oracle        - route on the true sigma (reproduces the existing results)
  D  estimated     - route on sigma_hat from the Task-1 estimator

ROUTING RULE, applied identically to the true sigma (C) and the estimate (D):
    sigma <= 0.05          -> agent1_low
    0.05 < sigma <= 0.15   -> agent2_medium
    sigma >  0.15          -> agent3_high
The upper-closed boundaries are chosen so the rule reproduces the existing
oracle assignment exactly at the three test levels (0.05 -> agent1,
0.10 -> agent2, 0.20 -> agent3). Note this makes sigma=0.05 a boundary case:
any over-estimate there routes to agent2, which is visible in the routing
accuracy rather than smoothed away.

Agents exist at 4 seeds per dataset, so every strategy is evaluated per seed
and reported as mean +/- std across seeds. The estimator is a single model per
dataset (as specified), shared across the agent seeds.
"""
import argparse
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch

MAP_SCRIPTS = Path(EXTERNAL_DIR)
EST_DIR = MAP_SCRIPTS / "estrouting"
for p in (str(MAP_SCRIPTS), str(EST_DIR)):
    sys.path.insert(0, p)
from models import UNet2D  # noqa: E402
from utils import batch_psnr_ssim  # noqa: E402
from train_noise_estimator import NoiseEstimator, SIGMA_SCALE  # noqa: E402

RES = Path(os.path.expanduser(RESULTS_ROOT + ""))
CROSS_RES = RES / "map_project" / "cross_dataset"
BRATS_FEAS = RES / "map_denoise_feasibility"
CROSS_DATA = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
BRATS_P2 = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
OUT = RES / "map_project" / "estimated_routing"

DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]
PRETTY = {"BraTS2021": "BraTS 2021", "nodule3d_128": "nodule3d", "cbsd68_128": "CBSD68"}
SIGMAS = [0.05, 0.10, 0.20]
AGENTS = ["agent1_low", "agent2_medium", "agent3_high"]
SEEDS = [42, 43, 44, 45]
BATCH = 100


def route(sig):
    """Shared rule for true sigma and sigma_hat -> agent index."""
    return np.where(sig <= 0.05, 0, np.where(sig <= 0.15, 1, 2)).astype(int)


def agent_dir(ds, seed, name):
    if seed == 42:
        if ds == "BraTS2021":
            return BRATS_FEAS / name
        return CROSS_RES / ds / name
    root = CROSS_RES / "brats_extra" / "seeds" if ds == "BraTS2021" else CROSS_RES / ds / "seeds"
    return root / f"seed{seed}" / name


def load_test(ds, sigma):
    tag = f"{sigma:.2f}".replace("0.", "0")
    if ds == "BraTS2021":
        return (np.load(BRATS_P2 / "phase2_test_5000.npy").astype(np.float32),
                np.load(BRATS_P2 / f"degraded_sigma{tag}.npy").astype(np.float32))
    return (np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_clean.npy").astype(np.float32),
            np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32))


def denoise(model, noisy, device):
    outs = []
    t = torch.from_numpy(noisy).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), BATCH):
            outs.append(model(t[i:i + BATCH].to(device)).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def assemble(per_agent, assign):
    """Pick each image's output from the agent its routing assigned."""
    out = np.empty_like(per_agent[0])
    for j in range(len(per_agent)):
        m = assign == j
        if m.any():
            out[m] = per_agent[j][m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    mae_rows, cmp_rows = [], []

    for ds in DATASETS:
        est = NoiseEstimator().to(device)
        est.load_state_dict(torch.load(OUT / "checkpoints" / f"estimator_{ds}.pt",
                                       map_location=device, weights_only=True))
        est.eval()
        print(f"=== {ds} ===", flush=True)

        for sigma in SIGMAS:
            clean, noisy = load_test(ds, sigma)

            # --- Task 2: estimator accuracy on this fixed-sigma test set ---
            preds = []
            with torch.no_grad():
                t = torch.from_numpy(noisy).unsqueeze(1)
                for i in range(0, len(t), 256):
                    preds.append((est(t[i:i + 256].to(device)) * SIGMA_SCALE).cpu().numpy())
            shat = np.concatenate(preds)
            mae = float(np.mean(np.abs(shat - sigma)))
            bias = float(np.mean(shat - sigma))
            true_assign = route(np.full(len(shat), sigma, dtype=np.float32))
            est_assign = route(shat)
            acc = float((est_assign == true_assign).mean())
            mae_rows.append({"dataset": PRETTY[ds], "sigma": sigma, "n_images": len(shat),
                             "mae": round(mae, 6), "bias": round(bias, 6),
                             "sigma_hat_mean": round(float(shat.mean()), 6),
                             "sigma_hat_std": round(float(shat.std()), 6),
                             "routing_accuracy": round(acc, 4)})
            print(f"  sigma={sigma}: MAE={mae:.5f} bias={bias:+.5f} "
                  f"sigma_hat={shat.mean():.4f}+/-{shat.std():.4f} "
                  f"routing_acc={100 * acc:.1f}%", flush=True)

            # --- Task 3: four routing strategies, per agent seed ---
            per_seed = {k: [] for k in ["expert1", "random", "oracle", "estimated"]}
            per_seed_ssim = {k: [] for k in per_seed}
            rng = np.random.default_rng(0)
            rand_assign = rng.integers(0, len(AGENTS), size=len(clean))

            for seed in SEEDS:
                models = []
                for a in AGENTS:
                    m = UNet2D().to(device)
                    m.load_state_dict(torch.load(
                        agent_dir(ds, seed, a) / "checkpoints" / "generator_best.pt",
                        map_location=device, weights_only=True))
                    m.eval()
                    models.append(m)
                outs = [denoise(m, noisy, device) for m in models]

                strat = {
                    "expert1": np.zeros(len(clean), dtype=int),
                    "random": rand_assign,
                    "oracle": true_assign,
                    "estimated": est_assign,
                }
                for k, assign in strat.items():
                    p, s = batch_psnr_ssim(clean, assemble(outs, assign))
                    per_seed[k].append(p)
                    per_seed_ssim[k].append(s)
                for m in models:
                    del m
                del outs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            row = {"dataset": PRETTY[ds], "sigma": sigma, "n_seeds": len(SEEDS),
                   "routing_accuracy": round(acc, 4), "mae": round(mae, 6)}
            for k in per_seed:
                row[f"{k}_psnr_mean"] = round(float(np.mean(per_seed[k])), 4)
                row[f"{k}_psnr_std"] = round(float(np.std(per_seed[k], ddof=1)), 4)
                row[f"{k}_ssim_mean"] = round(float(np.mean(per_seed_ssim[k])), 5)
                row[f"{k}_ssim_std"] = round(float(np.std(per_seed_ssim[k], ddof=1)), 5)
            cmp_rows.append(row)
            print(f"           expert1={row['expert1_psnr_mean']:.2f} "
                  f"random={row['random_psnr_mean']:.2f} "
                  f"oracle={row['oracle_psnr_mean']:.2f} "
                  f"estimated={row['estimated_psnr_mean']:.2f}  (dB, mean over "
                  f"{len(SEEDS)} seeds)", flush=True)

        del est
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(OUT / "estimator_mae.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mae_rows[0].keys()))
        w.writeheader()
        for r in mae_rows:
            w.writerow(r)
    with open(OUT / "routing_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        for r in cmp_rows:
            w.writerow(r)
    json.dump({"mae": mae_rows, "comparison": cmp_rows},
              open(OUT / "estimated_routing.json", "w"), indent=2)
    print(f"\n[tasks2-4] wrote {OUT / 'estimator_mae.csv'}", flush=True)
    print(f"[tasks2-4] wrote {OUT / 'routing_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
