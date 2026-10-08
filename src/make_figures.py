"""
Task 4: builds the 3 required cross-dataset figures.
  1. PSNR vs noise level overlay (all 4 datasets, joint vs matching agent)
  2. Training curves inherited vs from-scratch (agent3_high chain, all 4 datasets)
  3. Crossover point per dataset (delta = agent_psnr - joint_psnr vs sigma)
"""
import csv
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_BASE = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))
BRATS_CSV = Path(os.path.expanduser(RESULTS_ROOT + "/map_denoise_feasibility/comparison_results.csv"))
OUT_DIR = RESULTS_BASE
SIGMAS = [0.05, 0.10, 0.20]
SIGMA_TO_AGENT = {0.05: "agent1_low", 0.10: "agent2_medium", 0.20: "agent3_high"}

DATASETS = ["BraTS_original", "nodule3d_128", "brain_tumor_128", "cbsd68_128"]
COLORS = {"BraTS_original": "tab:blue", "nodule3d_128": "tab:orange",
          "brain_tumor_128": "tab:green", "cbsd68_128": "tab:red"}


def load_comparison(dataset):
    if dataset == "BraTS_original":
        path = BRATS_CSV
    else:
        path = RESULTS_BASE / dataset / "comparison_results.csv"
    rows = list(csv.DictReader(open(path)))
    return rows


def get_metric(rows, sigma, method_prefix):
    for r in rows:
        if abs(float(r["sigma"]) - sigma) < 1e-6 and r["method"].startswith(method_prefix):
            return float(r["psnr_denoised"])
    return None


def fig1_psnr_vs_noise():
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in DATASETS:
        rows = load_comparison(ds)
        joint_vals = [get_metric(rows, s, "single_agent_joint") for s in SIGMAS]
        agent_vals = [get_metric(rows, s, "map_denoise_") for s in SIGMAS]
        ax.plot(SIGMAS, joint_vals, "--", color=COLORS[ds], marker="o", label=f"{ds} joint")
        ax.plot(SIGMAS, agent_vals, "-", color=COLORS[ds], marker="s", label=f"{ds} matching agent")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Denoised PSNR (dB)")
    ax.set_title("PSNR vs noise level: joint vs matching specialized agent, all datasets")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_psnr_vs_noise.png", dpi=150)
    plt.close(fig)
    print("[make_figures] saved fig1_psnr_vs_noise.png")


def load_train_log(dataset, name):
    if dataset == "BraTS_original":
        return None
    path = RESULTS_BASE / dataset / name / "train_log.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    epochs, psnrs = [], []
    for r in rows:
        if r["val_psnr_denoised"] not in ("", None):
            epochs.append(int(r["epoch"]))
            psnrs.append(float(r["val_psnr_denoised"]))
    return epochs, psnrs


def fig2_training_curves():
    new_datasets = [d for d in DATASETS if d != "BraTS_original"]
    fig, axes = plt.subplots(1, len(new_datasets), figsize=(5 * len(new_datasets), 4.5), sharey=False)
    for ax, ds in zip(axes, new_datasets):
        inh = load_train_log(ds, "agent3_high")
        scr = load_train_log(ds, "agent3_high_scratch")
        if inh:
            ax.plot(inh[0], inh[1], "-o", label="inherited (from agent2_medium)", color="tab:blue")
        if scr:
            ax.plot(scr[0], scr[1], "-s", label="from scratch", color="tab:gray")
        ax.set_title(f"{ds}: agent3_high")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val PSNR (dB)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Training curves: inherited vs from-scratch (agent3_high)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_training_curves.png", dpi=150)
    plt.close(fig)
    print("[make_figures] saved fig2_training_curves.png")


def fig3_crossover():
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in DATASETS:
        rows = load_comparison(ds)
        deltas = []
        for s in SIGMAS:
            joint = get_metric(rows, s, "single_agent_joint")
            agent = get_metric(rows, s, "map_denoise_")
            deltas.append(agent - joint)
        ax.plot(SIGMAS, deltas, "-o", color=COLORS[ds], label=ds)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Agent PSNR - Joint PSNR (dB)")
    ax.set_title("Crossover: specialist advantage over joint vs noise level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_crossover_point.png", dpi=150)
    plt.close(fig)
    print("[make_figures] saved fig3_crossover_point.png")


if __name__ == "__main__":
    fig1_psnr_vs_noise()
    fig2_training_curves()
    fig3_crossover()
