"""
Experiment 1 (reviewer objection 5): RANGE-trained classical baseline.

The classical baselines in the main study follow their original protocols and
are trained at a single fixed sigma=0.1. At sigma=0.05 and sigma=0.20 they are
therefore operating outside their training distribution, which confounds
"regime-aware vs fixed-sigma training" with "in-distribution vs
out-of-distribution". This script removes that confound: it trains the same
Restormer-lite over the FULL noise range sigma ~ U[0.01, 0.25) -- the identical
noise distribution, and the identical stored noisy training data, used by the
single-agent joint baseline -- so the remaining difference between it and
MAP-DenoiseGAN is regime partitioning alone.

Uses each dataset's stored `*_joint_{clean,noisy}.npy` files, so the range
baseline sees bit-identical training data to the joint MAP model.

Nothing about the existing baselines is modified; this adds a new model.

Usage:
    python train_range_baseline.py --dataset BraTS2021
    python train_range_baseline.py --dataset nodule3d_128
    python train_range_baseline.py --dataset cbsd68_128
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

MAP_SCRIPTS = Path(EXTERNAL_DIR)
WIGAN_SCRIPTS = Path(EXTERNAL_DIR)
PHASE2_DIR = WIGAN_SCRIPTS / "phase2"
for p in (str(MAP_SCRIPTS), str(WIGAN_SCRIPTS), str(PHASE2_DIR)):
    sys.path.insert(0, p)

from utils import batch_psnr_ssim, seed_everything  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402
from swinir_lite_model import SwinIRLite  # noqa: E402

CROSS_DATA = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
BRATS_DATA = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/map_denoise"))
BRATS_P2 = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
RESULTS = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))
BRATS_RESULTS = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/brats_extra"))

MODELS = {"restormer": RestormerLite, "swinir": SwinIRLite}
SIGMAS = [0.05, 0.10, 0.20]


def data_dir(ds):
    return BRATS_DATA if ds == "BraTS2021" else CROSS_DATA / ds


def out_base(ds):
    return BRATS_RESULTS if ds == "BraTS2021" else RESULTS / ds


class PairDataset(Dataset):
    def __init__(self, clean_path, noisy_path):
        self.clean = torch.from_numpy(np.load(clean_path).astype(np.float32)).unsqueeze(1)
        self.noisy = torch.from_numpy(np.load(noisy_path).astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return self.noisy[i], self.clean[i]


def evaluate(model, loader, device):
    model.eval()
    cs, ns, ds = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            cs.append(clean.cpu().numpy()[:, 0])
            ns.append(noisy.cpu().numpy()[:, 0])
            ds.append(out.cpu().numpy()[:, 0])
    model.train()
    c, n, d = np.concatenate(cs), np.concatenate(ns), np.concatenate(ds)
    pn, sn = batch_psnr_ssim(c, n)
    pd_, sd = batch_psnr_ssim(c, d)
    return {"psnr_noisy": pn, "ssim_noisy": sn, "psnr_denoised": pd_, "ssim_denoised": sd,
            "delta_psnr": pd_ - pn, "delta_ssim": sd - sn}


def fixed_sigma_eval(model, ds, device):
    """Evaluate on the same fixed-sigma test sets used by the main comparison."""
    out = {}
    for sigma in SIGMAS:
        tag = f"{sigma:.2f}".replace("0.", "0")
        if ds == "BraTS2021":
            clean = np.load(BRATS_P2 / "phase2_test_5000.npy").astype(np.float32)
            noisy = np.load(BRATS_P2 / f"degraded_sigma{tag}.npy").astype(np.float32)
        else:
            clean = np.load(data_dir(ds) / f"test_fixedsigma{tag}_clean.npy").astype(np.float32)
            noisy = np.load(data_dir(ds) / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32)
        den = []
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(noisy).unsqueeze(1)
            for i in range(0, len(t), 200):
                den.append(model(t[i:i + 200].to(device)).cpu().numpy()[:, 0])
        model.train()
        den = np.concatenate(den)
        pn, sn = batch_psnr_ssim(clean, noisy)
        pd_, sd = batch_psnr_ssim(clean, den)
        out[sigma] = {"psnr_noisy": pn, "psnr_denoised": pd_, "delta_psnr": pd_ - pn,
                      "ssim_noisy": sn, "ssim_denoised": sd, "delta_ssim": sd - sn}
        print(f"[range_baseline] {ds} sigma={sigma}: PSNR {pn:.2f}->{pd_:.2f} "
              f"(dPSNR={pd_ - pn:+.2f}) SSIM {sd:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["BraTS2021", "nodule3d_128", "cbsd68_128"])
    ap.add_argument("--model", default="restormer", choices=list(MODELS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    dd = data_dir(args.dataset)
    out_dir = out_base(args.dataset) / f"{args.model}_range"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairDataset(dd / "train_joint_clean.npy", dd / "train_joint_noisy.npy")
    val_ds = PairDataset(dd / "val_joint_clean.npy", dd / "val_joint_noisy.npy")
    bs = min(args.batch_size, max(1, len(train_ds) // 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2,
                              drop_last=(len(train_ds) > bs))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[range_baseline] dataset={args.dataset} model={args.model} "
          f"train={len(train_ds)} val={len(val_ds)} batch={bs} device={device} "
          f"noise=range[0.01,0.25) (same stored data as the joint MAP baseline)", flush=True)

    model = MODELS[args.model]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()

    log = out_dir / "train_log.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "epoch_time_s", "val_psnr_denoised"])

    best = -1.0
    t0all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = l1(model(noisy), clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        et = time.time() - t0
        row = [epoch, float(np.mean(losses)), et, ""]
        if epoch % args.eval_every == 0 or epoch in (1, args.epochs):
            m = evaluate(model, val_loader, device)
            row[3] = m["psnr_denoised"]
            print(f"[epoch {epoch}/{args.epochs}] loss={row[1]:.4f} "
                  f"val_PSNR={m['psnr_denoised']:.2f} [{et:.1f}s]", flush=True)
            if m["psnr_denoised"] > best:
                best = m["psnr_denoised"]
                torch.save(model.state_dict(), ckpt_dir / f"{args.model}_range_best.pt")
        else:
            print(f"[epoch {epoch}/{args.epochs}] loss={row[1]:.4f} [{et:.1f}s]", flush=True)
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow(row)
    wall = time.time() - t0all

    model.load_state_dict(torch.load(ckpt_dir / f"{args.model}_range_best.pt",
                                     map_location=device, weights_only=True))
    fixed = fixed_sigma_eval(model, args.dataset, device)
    json.dump({"dataset": args.dataset, "model": f"{args.model}_range",
               "train_noise": "range[0.01,0.25)", "epochs": args.epochs,
               "batch_size": bs, "train_images": len(train_ds),
               "total_wall_time_s": wall,
               "fixed_sigma": {str(k): v for k, v in fixed.items()}},
              open(out_dir / "test_metrics.json", "w"), indent=2)
    print(f"[range_baseline] DONE {args.dataset}/{args.model}_range wall={wall:.1f}s -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
