"""
Task B1.4: controlled Adam-vs-Muon comparison on brain_tumor_128 (177 training
images), the dataset excluded from the main study because adversarial training
was unstable at that scale.

This is a strict A/B: identical data, architecture, schedule, batch size, seed
and loss. The ONLY difference is the optimiser used for the generator and
discriminator. Adam results already exist from the main study and are not
re-run; this script produces the Muon arm and, with --optimizer adam, can
reproduce the Adam arm under the identical harness as a sanity check.

Pre-registered success criterion (from the task brief): Muon stabilises
training if BOTH the matched agent and the joint model achieve positive
delta-PSNR, i.e. they improve on their noisy inputs rather than degrading them.
Under Adam both are negative at sigma=0.05 (-2.82 and -4.26 dB).
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
PAPER_DIR = MAP_SCRIPTS / "paper"
for p in (str(MAP_SCRIPTS), str(PAPER_DIR)):
    sys.path.insert(0, p)
from models import UNet2D, PatchGANDiscriminator  # noqa: E402
from utils import batch_psnr_ssim, seed_everything  # noqa: E402
from muon import Muon  # noqa: E402

DATA = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset/brain_tumor_128"))
OUT_BASE = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/muon_study"))
SIGMAS = [0.05, 0.10, 0.20]


class PairDataset(Dataset):
    def __init__(self, c, n):
        self.clean = torch.from_numpy(np.load(c).astype(np.float32)).unsqueeze(1)
        self.noisy = torch.from_numpy(np.load(n).astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return self.noisy[i], self.clean[i]


def evaluate(gen, loader, device):
    gen.eval()
    cs, ns, ds = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            ds.append(gen(noisy).cpu().numpy()[:, 0])
            cs.append(clean.cpu().numpy()[:, 0])
            ns.append(noisy.cpu().numpy()[:, 0])
    gen.train()
    c, n, d = np.concatenate(cs), np.concatenate(ns), np.concatenate(ds)
    pn, sn = batch_psnr_ssim(c, n)
    pd_, sd = batch_psnr_ssim(c, d)
    return {"psnr_noisy": pn, "psnr_denoised": pd_, "delta_psnr": pd_ - pn,
            "ssim_denoised": sd, "delta_ssim": sd - sn}


def fixed_sigma_eval(gen, device):
    out = {}
    gen.eval()
    for s in SIGMAS:
        tag = f"{s:.2f}".replace("0.", "0")
        clean = np.load(DATA / f"test_fixedsigma{tag}_clean.npy").astype(np.float32)
        noisy = np.load(DATA / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32)
        den = []
        with torch.no_grad():
            t = torch.from_numpy(noisy).unsqueeze(1)
            for i in range(0, len(t), 64):
                den.append(gen(t[i:i + 64].to(device)).cpu().numpy()[:, 0])
        den = np.concatenate(den)
        pn, _ = batch_psnr_ssim(clean, noisy)
        pd_, sd = batch_psnr_ssim(clean, den)
        out[str(s)] = {"psnr_noisy": pn, "psnr_denoised": pd_, "delta_psnr": pd_ - pn,
                       "ssim_denoised": sd}
        print(f"    sigma={s}: PSNR {pn:.2f}->{pd_:.2f} (dPSNR={pd_ - pn:+.2f})", flush=True)
    gen.train()
    return out


def build_opt(kind, params, lr, muon_lr=2e-2):
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    return Muon(params, lr=muon_lr, momentum=0.95, nesterov=True, adam_lr=lr,
                betas=(0.9, 0.999))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", required=True, choices=["adam", "muon"])
    ap.add_argument("--agent", required=True,
                    choices=["agent1_low", "agent2_medium", "agent3_high", "joint"])
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-rec", type=float, default=70.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--muon-lr", type=float, default=2e-2,
                    help="Muon learning rate for >=2-D parameters; the Adam "
                         "fallback for biases/1-D params uses --lr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    tag = args.optimizer if args.optimizer == "adam" else ("muon_lr%g" % args.muon_lr)
    out_dir = OUT_BASE / tag / args.agent
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    tr = PairDataset(DATA / f"train_{args.agent}_clean.npy", DATA / f"train_{args.agent}_noisy.npy")
    va = PairDataset(DATA / f"val_{args.agent}_clean.npy", DATA / f"val_{args.agent}_noisy.npy")
    bs = min(args.batch_size, max(1, len(tr) // 4))
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2, drop_last=(len(tr) > bs))
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False)
    print(f"[muon_study] opt={args.optimizer} agent={args.agent} train={len(tr)} "
          f"val={len(va)} bs={bs} seed={args.seed}", flush=True)

    gen = UNet2D().to(device)
    if args.init_from:
        gen.load_state_dict(torch.load(os.path.expanduser(args.init_from),
                                       map_location=device, weights_only=True))
        print(f"[muon_study] inherited from {args.init_from}", flush=True)
    disc = PatchGANDiscriminator(in_channels=2).to(device)

    g_opt = build_opt(args.optimizer, gen.parameters(), args.lr, args.muon_lr)
    d_opt = build_opt(args.optimizer, disc.parameters(), args.lr, args.muon_lr)
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()

    log = out_dir / "train_log.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "d_loss", "g_loss", "rec_loss",
                                "val_psnr_denoised", "val_delta_psnr"])
    best, best_ep, t0all = -1e9, -1, time.time()

    for ep in range(1, args.epochs + 1):
        dl_, gl_, rl_ = [], [], []
        for noisy, clean in tl:
            noisy, clean = noisy.to(device), clean.to(device)
            with torch.no_grad():
                fake = gen(noisy)
            dr, df = disc(noisy, clean), disc(noisy, fake)
            d_loss = bce(dr, torch.full_like(dr, 0.9)) + bce(df, torch.zeros_like(df))
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            fake = gen(noisy)
            dfg = disc(noisy, fake)
            adv = bce(dfg, torch.ones_like(dfg))
            rec = l1(fake, clean)
            g_loss = adv + args.lambda_rec * rec
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
            dl_.append(d_loss.item()); gl_.append(g_loss.item()); rl_.append(rec.item())

        m = evaluate(gen, vl, device)
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep, np.mean(dl_), np.mean(gl_), np.mean(rl_),
                                    m["psnr_denoised"], m["delta_psnr"]])
        if ep % 5 == 0 or ep in (1, args.epochs):
            print(f"  [ep {ep}/{args.epochs}] d={np.mean(dl_):.4f} g={np.mean(gl_):.4f} "
                  f"rec={np.mean(rl_):.4f} val_dPSNR={m['delta_psnr']:+.2f}", flush=True)
        if m["psnr_denoised"] > best:
            best, best_ep = m["psnr_denoised"], ep
            torch.save(gen.state_dict(), out_dir / "checkpoints" / "generator_best.pt")
        torch.save(gen.state_dict(), out_dir / "checkpoints" / "generator_last.pt")

    wall = time.time() - t0all
    gen.load_state_dict(torch.load(out_dir / "checkpoints" / "generator_best.pt",
                                   map_location=device, weights_only=True))
    print(f"  fixed-sigma evaluation ({args.optimizer}/{args.agent}):", flush=True)
    fixed = fixed_sigma_eval(gen, device)
    json.dump({"optimizer": args.optimizer, "muon_lr": args.muon_lr, "agent": args.agent, "seed": args.seed,
               "epochs": args.epochs, "batch_size": bs, "best_epoch": best_ep,
               "total_wall_time_s": wall, "train_images": len(tr),
               "init_from": args.init_from, "fixed_sigma": fixed},
              open(out_dir / "test_metrics.json", "w"), indent=2)
    print(f"[muon_study] DONE {args.optimizer}/{args.agent} best_ep={best_ep} "
          f"wall={wall:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
