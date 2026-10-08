"""
Cross-dataset generalization: trains one MAP-DenoiseGAN agent (or the
joint baseline) on a given base dataset. Reuses (import, unmodified)
map_project's UNet2D + PatchGANDiscriminator + noise/metric utilities -
the EXACT same architecture and GAN recipe as the original BraTS-based
feasibility experiment and Phase A1/A1.5/A1.6, so any cross-dataset
differences reflect the DATA, not an architecture change.

Usage:
    python train_cross_dataset_agent.py --dataset nodule3d_128 --agent agent1_low --out-dir nodule3d_128/agent1_low
    python train_cross_dataset_agent.py --dataset nodule3d_128 --agent agent2_medium --out-dir nodule3d_128/agent2_medium \\
        --init-from .../nodule3d_128/agent1_low/checkpoints/generator_last.pt
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
from models import UNet2D, PatchGANDiscriminator  # noqa: E402
from utils import batch_psnr_ssim, seed_everything  # noqa: E402

DATA_BASE = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
RESULTS_BASE = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))


class PairDataset(Dataset):
    def __init__(self, clean_path, noisy_path):
        self.clean = torch.from_numpy(np.load(clean_path).astype(np.float32)).unsqueeze(1)
        self.noisy = torch.from_numpy(np.load(noisy_path).astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def evaluate(generator, loader, device):
    generator.eval()
    all_clean, all_noisy, all_denoised = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = generator(noisy)
            all_clean.append(clean.cpu().numpy()[:, 0])
            all_noisy.append(noisy.cpu().numpy()[:, 0])
            all_denoised.append(denoised.cpu().numpy()[:, 0])
    generator.train()
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
    ap.add_argument("--data-dir", default=None,
                    help="override the data directory; needed for BraTS, whose per-agent "
                         "range datasets live in wigan_datasets/map_denoise/ rather than "
                         "under cross_dataset/. File naming is identical either way.")
    ap.add_argument("--agent", required=True, choices=["agent1_low", "agent2_medium", "agent3_high", "joint"])
    ap.add_argument("--out-dir", required=True, help="relative to $MAP_RESULTS_ROOT/map_project/cross_dataset/")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-rec", type=float, default=70.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    data_dir = Path(os.path.expanduser(args.data_dir)) if args.data_dir else DATA_BASE / args.dataset
    out_dir = RESULTS_BASE / args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairDataset(data_dir / f"train_{args.agent}_clean.npy", data_dir / f"train_{args.agent}_noisy.npy")
    val_ds = PairDataset(data_dir / f"val_{args.agent}_clean.npy", data_dir / f"val_{args.agent}_noisy.npy")
    test_ds = PairDataset(data_dir / f"test_{args.agent}_clean.npy", data_dir / f"test_{args.agent}_noisy.npy")

    batch_size = min(args.batch_size, max(1, len(train_ds) // 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=(len(train_ds) > batch_size))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[train_cross_dataset_agent] dataset={args.dataset} agent={args.agent} out_dir={out_dir} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} batch_size={batch_size} "
          f"device={device} init_from={args.init_from}", flush=True)

    generator = UNet2D().to(device)
    if args.init_from:
        state = torch.load(os.path.expanduser(args.init_from), map_location=device, weights_only=True)
        generator.load_state_dict(state, strict=True)
        with torch.no_grad():
            init_metrics = evaluate(generator, val_loader, device)
        print(f"[train_cross_dataset_agent] INHERITED from {args.init_from} | immediate val dPSNR = "
              f"{init_metrics['delta_psnr']:+.4f}", flush=True)
        with open(out_dir / "inheritance_log.json", "w") as f:
            json.dump({"init_from": args.init_from, **init_metrics}, f, indent=2)

    discriminator = PatchGANDiscriminator(in_channels=2).to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "d_loss", "g_loss", "adv_loss", "rec_loss", "epoch_time_s",
                                 "val_psnr_denoised", "val_delta_psnr"])

    best_val_psnr = -1.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        d_losses, g_losses, adv_losses, rec_losses = [], [], [], []
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            with torch.no_grad():
                denoised = generator(noisy)
            d_real = discriminator(noisy, clean)
            d_fake = discriminator(noisy, denoised)
            real_labels = torch.full_like(d_real, 0.9)
            fake_labels = torch.zeros_like(d_fake)
            d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            denoised = generator(noisy)
            d_fake_for_g = discriminator(noisy, denoised)
            adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            rec_loss = l1(denoised, clean)
            g_loss = adv_loss + args.lambda_rec * rec_loss
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            adv_losses.append(adv_loss.item())
            rec_losses.append(rec_loss.item())

        epoch_time = time.time() - t0
        row = [epoch, np.mean(d_losses), np.mean(g_losses), np.mean(adv_losses), np.mean(rec_losses),
               epoch_time, "", ""]

        if epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1:
            metrics = evaluate(generator, val_loader, device)
            row[6] = metrics["psnr_denoised"]
            row[7] = metrics["delta_psnr"]
            print(f"[epoch {epoch}/{args.epochs}] d_loss={row[1]:.4f} g_loss={row[2]:.4f} "
                  f"rec={row[4]:.4f} val_PSNR={metrics['psnr_denoised']:.2f} "
                  f"(dPSNR={metrics['delta_psnr']:+.2f}) [{epoch_time:.1f}s]", flush=True)
            if metrics["psnr_denoised"] > best_val_psnr:
                best_val_psnr = metrics["psnr_denoised"]
                best_epoch = epoch
                torch.save(generator.state_dict(), ckpt_dir / "generator_best.pt")
        else:
            print(f"[epoch {epoch}/{args.epochs}] d_loss={row[1]:.4f} g_loss={row[2]:.4f} "
                  f"rec={row[4]:.4f} [{epoch_time:.1f}s]", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        torch.save(generator.state_dict(), ckpt_dir / "generator_last.pt")

    total_wall_time = time.time() - t_start

    generator.load_state_dict(torch.load(ckpt_dir / "generator_best.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(generator, test_loader, device)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({**test_metrics, "dataset": args.dataset, "agent": args.agent, "epochs": args.epochs,
                   "batch_size": batch_size, "init_from": args.init_from,
                   "best_epoch": best_epoch, "total_wall_time_s": total_wall_time,
                   "train_samples": len(train_ds)}, f, indent=2)

    print(f"[train_cross_dataset_agent] {args.dataset}/{args.agent} FINAL TEST: PSNR {test_metrics['psnr_noisy']:.2f} -> "
          f"{test_metrics['psnr_denoised']:.2f} (dPSNR={test_metrics['delta_psnr']:+.2f}) "
          f"best_epoch={best_epoch}/{args.epochs} total_wall_time={total_wall_time:.1f}s", flush=True)
    print(f"[train_cross_dataset_agent] results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
