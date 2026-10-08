"""
MAP-DenoiseGAN Path A (map_project): shared utilities. Fresh self-contained
copy (per "Path A is a fresh classical-only project" - do not import across
into the quantum wigan_project code) of the noise model, PSNR/SSIM metric,
and seeding helper already established and validated in the earlier
feasibility work (qgan_utils_wigan.py), kept identical for continuity of
methodology across the whole research program.
"""
import random

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_noise(images, gaussian_sigma=0.1, poisson_scale=1.0, seed=None):
    """Add Gaussian(0, sigma) + Poisson shot noise to a batch of [0,1] images."""
    r = np.random.default_rng(seed)
    images = images.astype(np.float32)
    peak = 255.0 * poisson_scale
    poisson_noisy = r.poisson(images * peak) / peak
    gaussian_noise = r.normal(0.0, gaussian_sigma, size=images.shape)
    noisy = poisson_noisy + gaussian_noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def batch_psnr_ssim(clean, other):
    """clean, other: numpy arrays (N, H, W) in [0,1]. Returns mean (psnr, ssim)."""
    psnrs, ssims = [], []
    for c, o in zip(clean, other):
        psnrs.append(sk_psnr(c, o, data_range=1.0))
        ssims.append(sk_ssim(c, o, data_range=1.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))
