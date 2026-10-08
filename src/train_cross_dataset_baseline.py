"""
Trains one classical baseline (DnCNN / Restormer-lite / SwinIR-lite) on a
given cross-dataset dataset, at FIXED sigma=0.1 on the full training set -
exactly mirroring the original MAP-DenoiseGAN feasibility experiment's
train_classical_baseline.py protocol (wigan_project/scripts/phase2/,
READ-ONLY import of the model classes only; that file itself is not
modified, and no quantum-project file is written to).

Usage:
    python train_cross_dataset_baseline.py --dataset nodule3d_128 --model dncnn
    python train_cross_dataset_baseline.py --dataset nodule3d_128 --model restormer
    python train_cross_dataset_baseline.py --dataset nodule3d_128 --model swinir
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
from torch.utils.data import DataLoader, Dataset

MAP_PROJECT_SCRIPTS = Path(EXTERNAL_DIR)
sys.path.insert(0, str(MAP_PROJECT_SCRIPTS))
from utils import add_noise, batch_psnr_ssim, seed_everything  # noqa: E402

WIGAN_PHASE2 = Path(EXTERNAL_DIR)
sys.path.insert(0, str(WIGAN_PHASE2))
from dncnn_model import DnCNN  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402
from swinir_lite_model import SwinIRLite  # noqa: E402

DATA_BASE = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
RESULTS_BASE = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))
TRAIN_SIGMA = 0.1

MODEL_BUILDERS = {
    "dncnn": lambda: DnCNN(),
    "restormer": lambda: RestormerLite(),
    "swinir": lambda: SwinIRLite(),
}


class DenoiseDataset(Dataset):
    def __init__(self, images, seed):
        clean = images.astype(np.float32)
        noisy = np.stack([add_noise(clean[i:i + 1], gaussian_sigma=TRAIN_SIGMA, seed=seed * 100000 + i)[0]
                           for i in range(len(clean))])
        self.clean = torch.from_numpy(clean).unsqueeze(1)
        self.noisy = torch.from_numpy(noisy).unsqueeze(1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def evaluate(model, loader, device):
    model.eval()
    all_clean, all_noisy, all_denoised = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)
            all_clean.append(clean.cpu().numpy()[:, 0])
            all_noisy.append(noisy.cpu().numpy()[:, 0])
            all_denoised.append(denoised.cpu().numpy()[:, 0])
    model.train()
    clean = np.concatenate(all_clean)
    noisy = np.concatenate(all_noisy)
    denoised = np.concatenate(all_denoised)
    psnr_noisy, ssim_noisy = batch_psnr_ssim(clean, noisy)
    psnr_den, ssim_den = batch_psnr_ssim(clean, denoised)
    return {
        "psnr_noisy": psnr_noisy, "ssim_noisy": ssim_noisy,
        "psnr_denoised": psnr_den, "ssim_denoised": ssim_den,
        "delta_psnr": psnr_den - psnr_noisy, "delta_ssim": ssim_den - ssim_noisy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    data_dir = DATA_BASE / args.dataset
    out_dir = RESULTS_BASE / args.dataset / args.model
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = np.load(data_dir / "train_joint_clean.npy").astype(np.float32)
    val_imgs = np.load(data_dir / "val_joint_clean.npy").astype(np.float32)
    test_imgs = np.load(data_dir / "test_joint_clean.npy").astype(np.float32)

    train_ds = DenoiseDataset(train_imgs, seed=args.seed)
    val_ds = DenoiseDataset(val_imgs, seed=args.seed + 1)
    test_ds = DenoiseDataset(test_imgs, seed=args.seed + 2)

    batch_size = min(args.batch_size, max(1, len(train_ds) // 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=(len(train_ds) > batch_size))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[train_cross_dataset_baseline] dataset={args.dataset} model={args.model} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} batch_size={batch_size} "
          f"device={device} train_sigma={TRAIN_SIGMA}", flush=True)

    model = MODEL_BUILDERS[args.model]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "epoch_time_s", "val_psnr_denoised", "val_delta_psnr"])

    best_val_psnr = -1.0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = []
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)
            loss = l1(denoised, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        epoch_time = time.time() - t0

        row = [epoch, np.mean(train_losses), epoch_time, "", ""]
        if epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1:
            metrics = evaluate(model, val_loader, device)
            row[3] = metrics["psnr_denoised"]
            row[4] = metrics["delta_psnr"]
            print(f"[epoch {epoch}/{args.epochs}] train_loss={np.mean(train_losses):.4f} "
                  f"val_PSNR={metrics['psnr_denoised']:.2f} (dPSNR={metrics['delta_psnr']:+.2f}) "
                  f"[{epoch_time:.1f}s]", flush=True)
            if metrics["psnr_denoised"] > best_val_psnr:
                best_val_psnr = metrics["psnr_denoised"]
                torch.save(model.state_dict(), ckpt_dir / f"{args.model}_best.pt")
        else:
            print(f"[epoch {epoch}/{args.epochs}] train_loss={np.mean(train_losses):.4f} [{epoch_time:.1f}s]",
                  flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        torch.save(model.state_dict(), ckpt_dir / f"{args.model}_last.pt")

    total_wall_time = time.time() - t_start
    model.load_state_dict(torch.load(ckpt_dir / f"{args.model}_best.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({**test_metrics, "dataset": args.dataset, "model": args.model, "epochs": args.epochs,
                   "batch_size": batch_size, "train_samples": len(train_ds),
                   "total_wall_time_s": total_wall_time}, f, indent=2)
    print(f"[train_cross_dataset_baseline] {args.dataset}/{args.model} FINAL TEST: PSNR "
          f"{test_metrics['psnr_noisy']:.2f} -> {test_metrics['psnr_denoised']:.2f} "
          f"(dPSNR={test_metrics['delta_psnr']:+.2f}) total_wall_time={total_wall_time:.1f}s", flush=True)
    print(f"[train_cross_dataset_baseline] results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
