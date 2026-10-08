"""
Task 1: train a noise-level estimator, one per dataset.

Predicts sigma_hat from a noisy image so that routing can use an ESTIMATED
noise level instead of the true one. No existing agent is retrained; this is
the only new model.

DATA NOTE: per-image sigma was not stored when the agents' training data was
generated (the manifests record only each agent's range and mean sigma), so the
(noisy image, true sigma) pairs are regenerated here from the stored CLEAN
training images using the same Gaussian+Poisson degradation the agents saw.
Sigma is drawn per image from U[0.01, 0.25), the joint operating range, because
the router must cover the whole range rather than any single agent's regime.
Generation is seeded, so the set is reproducible.

Architecture (as specified): 4 stride-2 conv blocks 1->32->64->128->128 with
ReLU, global average pool, FC 128->64->1, sigmoid, scaled to [0, 0.25].
Loss is L1 in normalised units (sigma / 0.25).
"""
import argparse
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MAP_SCRIPTS = Path(EXTERNAL_DIR)
sys.path.insert(0, str(MAP_SCRIPTS))
from utils import add_noise, seed_everything  # noqa: E402

CROSS = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
BRATS = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/map_denoise"))
OUT = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/estimated_routing"))

SIGMA_MIN, SIGMA_MAX = 0.01, 0.25
SIGMA_SCALE = 0.25          # normalisation constant: sigma / SIGMA_SCALE in [0,1]
DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]


def data_dir(ds):
    return BRATS if ds == "BraTS2021" else CROSS / ds


class NoiseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                  nn.Linear(64, 1))

    def forward(self, x):
        h = self.features(x)
        h = h.mean(dim=(2, 3))                      # global average pool
        return torch.sigmoid(self.head(h)).squeeze(1)   # normalised sigma in [0,1]


def make_pairs(clean, seed, tag):
    """Regenerate (noisy, true sigma) pairs with sigma ~ U[0.01, 0.25)."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(SIGMA_MIN, SIGMA_MAX, size=len(clean)).astype(np.float32)
    noisy = np.stack([
        add_noise(clean[i:i + 1], gaussian_sigma=float(sig[i]), seed=seed * 100003 + i)[0]
        for i in range(len(clean))])
    print(f"    {tag}: n={len(clean)} sigma in [{sig.min():.4f},{sig.max():.4f}] "
          f"mean={sig.mean():.4f}", flush=True)
    return noisy, sig


def evaluate(model, noisy, sig, device, batch=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(noisy), batch):
            xb = torch.from_numpy(noisy[i:i + batch]).unsqueeze(1).to(device)
            preds.append((model(xb) * SIGMA_SCALE).cpu().numpy())
    model.train()
    p = np.concatenate(preds)
    return float(np.mean(np.abs(p - sig))), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    device = torch.device(args.device)
    rows = []

    for ds in args.datasets.split(","):
        print(f"=== {ds} ===", flush=True)
        d = data_dir(ds)
        tr_clean = np.load(d / "train_joint_clean.npy").astype(np.float32)
        va_clean = np.load(d / "val_joint_clean.npy").astype(np.float32)
        print("  regenerating (noisy, sigma) pairs", flush=True)
        tr_noisy, tr_sig = make_pairs(tr_clean, args.seed, "train")
        va_noisy, va_sig = make_pairs(va_clean, args.seed + 1, "val")

        seed_everything(args.seed)
        model = NoiseEstimator().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        l1 = nn.L1Loss()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  estimator parameters: {n_par:,}", flush=True)

        idx_all = np.arange(len(tr_noisy))
        rng = np.random.default_rng(args.seed)
        best_val, best_ep, hist = float("inf"), -1, []
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            rng.shuffle(idx_all)
            losses = []
            for i in range(0, len(idx_all) - args.batch_size + 1, args.batch_size):
                b = idx_all[i:i + args.batch_size]
                xb = torch.from_numpy(tr_noisy[b]).unsqueeze(1).to(device)
                yb = torch.from_numpy(tr_sig[b] / SIGMA_SCALE).to(device)
                loss = l1(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            val_mae, _ = evaluate(model, va_noisy, va_sig, device)
            hist.append(val_mae)
            if val_mae < best_val:
                best_val, best_ep = val_mae, ep
                torch.save(model.state_dict(), OUT / "checkpoints" / f"estimator_{ds}.pt")
            if ep % 5 == 0 or ep in (1, args.epochs):
                print(f"  [ep {ep:2d}/{args.epochs}] train_L1(norm)={np.mean(losses):.5f} "
                      f"val_MAE(sigma)={val_mae:.5f}", flush=True)
        wall = time.time() - t0

        # convergence check: val MAE must have improved over the run and be
        # meaningfully better than predicting the mean of the sigma range
        naive_mae = float(np.mean(np.abs(va_sig - va_sig.mean())))
        converged = bool(best_val < naive_mae * 0.5 and best_ep > 1)
        print(f"  best val MAE = {best_val:.5f} at epoch {best_ep} "
              f"(constant-mean baseline {naive_mae:.5f}) -> "
              f"{'CONVERGED' if converged else 'NOT CONVERGED'}  [{wall:.0f}s]\n", flush=True)

        rows.append({"dataset": ds, "n_train": len(tr_clean), "n_val": len(va_clean),
                     "parameters": n_par, "epochs": args.epochs,
                     "best_val_mae": round(best_val, 6), "best_epoch": best_ep,
                     "constant_mean_baseline_mae": round(naive_mae, 6),
                     "converged": converged, "wall_s": round(wall, 1),
                     "val_mae_history": ";".join(f"{h:.6f}" for h in hist)})

    with open(OUT / "estimator_training.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(OUT / "estimator_training.json", "w"), indent=2)
    print(f"[task1] wrote {OUT / 'estimator_training.csv'}", flush=True)
    bad = [r["dataset"] for r in rows if not r["converged"]]
    print(f"[task1] all estimators converged: {not bad}"
          + (f"  (failed: {bad})" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
