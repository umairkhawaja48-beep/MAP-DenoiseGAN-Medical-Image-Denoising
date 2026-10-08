"""
BSPC Task C1.2: generates Figures 1-6 as vector PDF + 300 dpi PNG.
Reads only existing result files and existing checkpoints (Fig 6 runs
inference for the qualitative panels). Nothing is trained.

Output: $MAP_RESULTS_ROOT/map_project/paper/figures/
"""
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MAP_SCRIPTS = Path(EXTERNAL_DIR)
WIGAN_SCRIPTS = Path(EXTERNAL_DIR)
PHASE2_DIR = WIGAN_SCRIPTS / "phase2"
for p in (str(MAP_SCRIPTS), str(WIGAN_SCRIPTS), str(PHASE2_DIR)):
    sys.path.insert(0, p)

from models import UNet2D  # noqa: E402
from dncnn_model import DnCNN  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402

FIG = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/figures"))
TAB = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
BRATS_DATA = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
BRATS_RES = Path(os.path.expanduser(RESULTS_ROOT + "/map_denoise_feasibility"))
PHASE2_RES = Path(os.path.expanduser(RESULTS_ROOT + "/wigan_phase2"))
CROSS_RES = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))

SIGMAS = [0.05, 0.10, 0.20]
DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]
PRETTY = {"BraTS2021": "BraTS 2021 (brain MRI)", "nodule3d_128": "nodule3d (lung CT)",
          "cbsd68_128": "BSD68 (natural images)"}
SHORT = {"BraTS2021": "BraTS 2021", "nodule3d_128": "nodule3d", "cbsd68_128": "CBSD68"}
DS_COLOR = {"BraTS2021": "#1f77b4", "nodule3d_128": "#d62728", "cbsd68_128": "#2ca02c"}

# Published display label for the natural-image dataset. The directory and
# CSV key ("cbsd68_128" / "CBSD68") are internal historical paths and are
# deliberately left untouched; only the label the reader sees is mapped.
DISPLAY = {"CBSD68": "BSD68"}


def disp(name):
    return DISPLAY.get(name, name)


plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def base_dir(ds):
    return BRATS_RES if ds == "BraTS2021" else CROSS_RES / ds


def joint_name(ds):
    return "single_agent_joint" if ds == "BraTS2021" else "joint"


def read_curve(path):
    rows = list(csv.DictReader(open(path)))
    out = [(int(r["epoch"]), float(r["val_psnr_denoised"]))
           for r in rows if r.get("val_psnr_denoised", "") not in ("", None)]
    return out


def load_t3():
    table = {}
    for r in csv.DictReader(open(TAB / "table3_main_psnr.csv")):
        table.setdefault(r["dataset"], {}).setdefault(float(r["sigma"]), {})[r["method"]] = {
            "psnr": float(r["psnr_db"]), "ssim": float(r["ssim"])}
    return table


# ------------------------------------------------------------------- Fig 1
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#333333", fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                     linewidth=1.0, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", color="#333333", lw=1.1, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                      mutation_scale=9, linewidth=lw, color=color, linestyle=ls))

    stages = [
        (0.35, "Agent 1\n$\\sigma\\in[0.01,0.05)$", "#dbe9f6"),
        (3.65, "Agent 2\n$\\sigma\\in[0.05,0.15)$", "#c3dcf0"),
        (6.95, "Agent 3\n$\\sigma\\in[0.15,0.25)$", "#a9cde9"),
    ]
    for x, label, fc in stages:
        box(x, 2.55, 2.7, 1.5, label, fc, bold=True, fs=8.5)
        box(x + 0.15, 1.35, 2.4, 0.85, "U-Net generator\n+ PatchGAN critic", "#ffffff", fs=7.2)
        arrow(x + 1.35, 2.55, x + 1.35, 2.22)

    for x0, _, _ in stages[:-1]:
        arrow(x0 + 2.7, 3.3, x0 + 3.65, 3.3, lw=1.5, color="#b8341f")
    ax.text(3.35, 3.62, "weight\ninheritance", ha="center", va="bottom", fontsize=7.4,
            color="#b8341f", fontweight="bold", linespacing=1.2)
    ax.text(6.62, 3.62, "weight\ninheritance", ha="center", va="bottom", fontsize=7.4,
            color="#b8341f", fontweight="bold", linespacing=1.2)

    box(0.35, 4.35, 9.3, 0.72,
        "Training curriculum: increasing noise severity  $\\longrightarrow$", "#f2f2f2", fs=8.5)
    box(0.35, 0.18, 9.3, 0.82,
        "Inference: each image is routed to the agent matching its noise level "
        "(one generator per image)", "#eef7ee", fs=8)
    for x, _, _ in stages:
        arrow(x + 1.35, 1.35, x + 1.35, 1.02, ls=(0, (2, 1.6)), color="#2f6b2f")

    save(fig, "fig1_architecture")


# ------------------------------------------------------------------- Fig 2
def fig2_training_curves():
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for ds in DATASETS:
        b = base_dir(ds)
        inh, scr = b / "agent2_medium" / "train_log.csv", b / "agent2_medium_scratch" / "train_log.csv"
        if not (inh.exists() and scr.exists()):
            continue
        ci, cs = read_curve(inh), read_curve(scr)
        c = DS_COLOR[ds]
        ax.plot([e for e, _ in ci], [p for _, p in ci], "-o", color=c, ms=3.4, lw=1.5,
                label=f"{disp(SHORT[ds])} — inherited")
        ax.plot([e for e, _ in cs], [p for _, p in cs], "--s", color=c, ms=3.4, lw=1.3,
                alpha=0.75, label=f"{disp(SHORT[ds])} — from scratch")
        ax.annotate("", xy=(1, ci[0][1]), xytext=(1, cs[0][1]),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=0.9, alpha=0.65))
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("Weight inheritance vs. from-scratch training (agent 2)")
    ax.legend(loc="lower right", ncol=1, framealpha=0.92)
    save(fig, "fig2_training_curves")


# ------------------------------------------------------------------- Fig 3
def fig3_psnr_vs_noise(t3):
    # dashed = trained at a single fixed sigma (evaluated out of distribution
    # away from 0.1); solid = trained over the full noise range
    methods = [("No denoising", "#999999", ":", "x"),
               ("DnCNN ($\\sigma{=}0.1$)", "#8c564b", "--", "v"),
               ("Restormer-lite ($\\sigma{=}0.1$)", "#9467bd", "--", "^"),
               ("SwinIR-lite ($\\sigma{=}0.1$)", "#ff7f0e", "--", "D"),
               ("Restormer-lite (range-trained)", "#2ca02c", "-", "P"),
               ("Single-agent joint", "#1f77b4", "-", "s"),
               ("MAP-DenoiseGAN (matched agent)", "#d62728", "-", "o")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        key = SHORT[ds] if SHORT[ds] != "BraTS 2021" else "BraTS 2021"
        for label, color, ls, mk in methods:
            ys = [t3[key][s][label]["psnr"] for s in SIGMAS if label in t3[key][s]]
            if len(ys) != len(SIGMAS):
                continue
            ax.plot(SIGMAS, ys, ls, color=color, marker=mk, ms=3.6,
                    lw=2.0 if "MAP" in label else 1.2, label=label)
        ax.set_title(PRETTY[ds], fontsize=8.5)
        ax.set_xlabel("Noise level $\\sigma$")
        ax.set_xticks(SIGMAS)
    axes[0].set_ylabel("PSNR (dB)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.20), fontsize=7.4)
    fig.tight_layout()
    save(fig, "fig3_psnr_vs_noise")


# ------------------------------------------------------------------- Fig 4
def fig4_high_noise_win():
    rows = list(csv.DictReader(open(TAB / "table6_replication.csv")))
    labels = [r["dataset"] for r in rows]
    d_joint = [float(r["delta_vs_joint_s020"]) for r in rows]
    d_fixed = [float(r["delta_vs_fixed_sigma_classical_s020"]) for r in rows]
    d_range = [float(r["delta_vs_range_classical_s020"]) if r["delta_vs_range_classical_s020"]
               else np.nan for r in rows]

    # seed error bars on the vs-joint comparison, where they exist
    err = [np.nan] * len(labels)
    sp = TAB / "table7_seed_stats.json"
    if sp.exists():
        stats = json.load(open(sp))
        for i, lab in enumerate(labels):
            e = stats.get(lab, {}).get("0.2")
            if e:
                d_joint[i] = e["gap_mean"]
                err[i] = e["gap_std"]

    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    b1 = ax.bar(x - w, d_joint, w, yerr=err, capsize=3, label="vs. single-agent joint",
                color="#1f77b4")
    b2 = ax.bar(x, d_fixed, w, label="vs. fixed-$\\sigma$ classical", color="#bbbbbb")
    b3 = ax.bar(x + w, d_range, w, label="vs. range-trained classical", color="#d62728")
    for bars, errs in ((b1, err), (b2, [0] * len(labels)), (b3, [0] * len(labels))):
        for bb, e in zip(bars, errs):
            h = bb.get_height()
            if np.isnan(h):
                continue
            # lift the label clear of the error bar so the two never overlap
            top = h + (e if e and not np.isnan(e) else 0.0)
            ax.annotate(f"{h:+.2f}", (bb.get_x() + bb.get_width() / 2, top),
                        textcoords="offset points", xytext=(0, 3.0), ha="center", fontsize=7.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([disp(l) for l in labels])
    ax.set_ylabel("$\\Delta$PSNR at $\\sigma=0.20$ (dB)")
    ax.set_title("High-noise advantage, against fair and unfair baselines")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, max(v for v in d_fixed if not np.isnan(v)) * 1.25)
    save(fig, "fig4_high_noise_win")


# ------------------------------------------------------------------- Fig 5
def fig5_downstream_dice():
    rows = list(csv.DictReader(open(TAB / "table5_downstream_summary.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[float(r["sigma"])] = (float(r["mean_dice"]),
                                                              float(r["std_dice"]))
    order = ["No denoising", "DnCNN ($\\sigma{=}0.1$)", "Restormer-lite ($\\sigma{=}0.1$)",
             "SwinIR-lite ($\\sigma{=}0.1$)", "Restormer-lite (range-trained)",
             "Single-agent joint", "MAP-DenoiseGAN (matched agent)"]
    colors = ["#999999", "#8c564b", "#9467bd", "#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
    order, colors = zip(*[(o, c) for o, c in zip(order, colors) if o in by])
    x = np.arange(len(SIGMAS)); w = 0.118
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (label, c) in enumerate(zip(order, colors)):
        vals = [by[label][s][0] for s in SIGMAS]
        ax.bar(x + (i - len(order) / 2 + 0.5) * w, vals, w, label=label, color=c)
    oracle = by["Oracle (clean image)"][0.05][0]
    ax.axhline(oracle, color="black", ls="--", lw=1.1)
    ax.text(len(SIGMAS) - 1 + 0.40, oracle + 0.008, f"oracle (clean) {oracle:.3f}",
            ha="right", fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels([f"$\\sigma={s:g}$" for s in SIGMAS])
    ax.set_ylabel("Tumour segmentation Dice")
    ax.set_title("Downstream BraTS segmentation after denoising")
    ax.set_ylim(0.40, 0.90)
    ax.legend(loc="lower left", ncol=2, fontsize=7)
    save(fig, "fig5_downstream_dice")


# ------------------------------------------------------------------- Fig 6
def fig6_qualitative(device):
    clean = np.load(BRATS_DATA / "phase2_test_5000.npy").astype(np.float32)
    masks = np.load(BRATS_DATA / "phase2_test_5000_mask.npy").astype(np.float32)
    degraded = np.load(BRATS_DATA / "degraded_sigma020.npy").astype(np.float32)
    idx = int(np.argmax(masks.reshape(len(masks), -1).sum(1)))  # most tumour tissue

    def load(ctor, path):
        m = ctor().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    models = {
        "DnCNN": load(DnCNN, PHASE2_RES / "baselines" / "dncnn" / "checkpoints" / "dncnn_best.pt"),
        "Restormer-lite": load(RestormerLite,
                               PHASE2_RES / "baselines" / "restormer" / "checkpoints" / "restormer_best.pt"),
        "Single-agent joint": load(UNet2D,
                                   BRATS_RES / "single_agent_joint" / "checkpoints" / "generator_best.pt"),
        "MAP-DenoiseGAN\n(agent 3)": load(UNet2D,
                                          BRATS_RES / "agent3_high" / "checkpoints" / "generator_best.pt"),
    }
    x = torch.from_numpy(degraded[idx:idx + 1]).unsqueeze(1).to(device)
    panels = [("Clean", clean[idx]), ("Noisy ($\\sigma=0.20$)", degraded[idx])]
    with torch.no_grad():
        for name, m in models.items():
            panels.append((name, m(x).cpu().numpy()[0, 0]))

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return 10 * np.log10(1.0 / max(mse, 1e-12))

    fig, axes = plt.subplots(2, len(panels), figsize=(1.42 * len(panels), 3.3))
    ref = clean[idx]
    for j, (name, img) in enumerate(panels):
        axes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        title = name if j == 0 else f"{name}\n{psnr(ref, img):.2f} dB"
        axes[0, j].set_title(title, fontsize=7.2, linespacing=1.3)
        zc, zs = 64, 40
        axes[1, j].imshow(img[zc - zs // 2:zc + zs // 2, zc - zs // 2:zc + zs // 2],
                          cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for a in (axes[0, j], axes[1, j]):
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    axes[1, 0].set_ylabel("zoom", fontsize=7.5)
    fig.suptitle("Qualitative comparison at high noise (BraTS test slice)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_qualitative")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[make_bspc_figures] device={device}")
    t3 = load_t3()
    print("Fig 1"); fig1_architecture()
    print("Fig 2"); fig2_training_curves()
    print("Fig 3"); fig3_psnr_vs_noise(t3)
    print("Fig 4"); fig4_high_noise_win()
    print("Fig 5"); fig5_downstream_dice()
    print("Fig 6"); fig6_qualitative(device)
    print(f"[make_bspc_figures] all figures written to {FIG}")


if __name__ == "__main__":
    main()
