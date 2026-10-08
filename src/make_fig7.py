"""
Phase C1.2: Figure 7 -- Theorem 2.4 decomposition across noise levels.

Three panels:
  (a) the three terms (specialization gain, routing regret, diversity) vs sigma
  (b) the resulting condition SG + D - RR, with the sign change marked
  (c) the mechanism: fraction of images for which the sigma-matched agent is
      also the per-image optimal agent -- i.e. why routing regret vanishes
"""
import csv
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAB = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
FIG = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/figures"))
SIGMAS = [0.05, 0.10, 0.20]
DS = ["BraTS 2021", "nodule3d", "CBSD68"]
COLOR = {"BraTS 2021": "#1f77b4", "nodule3d": "#d62728", "CBSD68": "#2ca02c"}

# Published display label for the natural-image dataset. The directory and
# CSV key ("cbsd68_128" / "CBSD68") are internal historical paths and are
# deliberately left untouched; only the label the reader sees is mapped.
DISPLAY = {"CBSD68": "BSD68"}


def disp(name):
    return DISPLAY.get(name, name)


plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    rows = list(csv.DictReader(open(TAB / "table8_theorem_terms.csv")))
    d = {}
    for r in rows:
        d.setdefault(r["dataset"], {})[float(r["sigma"])] = r

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

    # (a) the three terms, in units of 1e-4 MSE for readability
    ax = axes[0]
    for ds in DS:
        sg = [float(d[ds][s]["specialization_gain_mse"]) * 1e4 for s in SIGMAS]
        rr = [float(d[ds][s]["routing_regret_hard_mse"]) * 1e4 for s in SIGMAS]
        ax.plot(SIGMAS, sg, "-o", color=COLOR[ds], ms=3.6, lw=1.6, label=f"{disp(ds)}: gain")
        ax.plot(SIGMAS, rr, "--s", color=COLOR[ds], ms=3.2, lw=1.1, alpha=0.8,
                label=f"{disp(ds)}: regret")
    ax.set_xlabel("Noise level $\\sigma$")
    ax.set_ylabel("MSE $\\times 10^{-4}$")
    ax.set_title("(a) Specialisation gain vs.\\ routing regret", fontsize=8.6)
    ax.set_xticks(SIGMAS)
    ax.legend(fontsize=5.8, ncol=1, loc="upper left")

    # (b) the condition itself
    ax = axes[1]
    for ds in DS:
        cond = [float(d[ds][s]["condition_hard_mse"]) * 1e4 for s in SIGMAS]
        ax.plot(SIGMAS, cond, "-o", color=COLOR[ds], ms=4, lw=1.8, label=disp(ds))
    ax.axhline(0, color="black", lw=1.0, ls=":")
    ax.annotate("condition $>0$:\nrouted beats joint", xy=(0.105, 0.6),
                xycoords=("data", "axes fraction"), fontsize=6.6, ha="left", color="#333333")
    ax.set_xlabel("Noise level $\\sigma$")
    ax.set_ylabel("$SG + D - RR$  (MSE $\\times 10^{-4}$)")
    ax.set_title("(b) Theorem 2.4 condition", fontsize=8.6)
    ax.set_xticks(SIGMAS)
    ax.legend(loc="upper left", fontsize=6.8)

    # (c) why regret vanishes
    ax = axes[2]
    for ds in DS:
        frac = [100 * float(d[ds][s]["oracle_picks_matching_agent_frac"]) for s in SIGMAS]
        ax.plot(SIGMAS, frac, "-o", color=COLOR[ds], ms=4, lw=1.8, label=disp(ds))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Noise level $\\sigma$")
    ax.set_ylabel("\\% images where matched\nagent is per-image optimal")
    ax.set_title("(c) Why routing regret vanishes", fontsize=8.6)
    ax.set_xticks(SIGMAS)
    ax.legend(loc="lower right", fontsize=6.8)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig7_theorem_validation.pdf")
    fig.savefig(FIG / "fig7_theorem_validation.png", dpi=300)
    plt.close(fig)
    print("wrote fig7_theorem_validation.pdf / .png")

    # console summary used when writing the manuscript
    print("\nsign of condition (hard routing):")
    for ds in DS:
        signs = ["+" if float(d[ds][s]["condition_hard_mse"]) > 0 else "-" for s in SIGMAS]
        print(f"  {ds:12s}: " + "  ".join(f"sigma={s}: {sg}" for s, sg in zip(SIGMAS, signs)))
    print("\nuniform-ensemble condition vs actual ensemble outcome:")
    for ds in DS:
        for s in SIGMAS:
            r = d[ds][s]
            pred = float(r["condition_uniform_mse"]) > 0
            actual = float(r["psnr_uniform_ens_db"]) > float(r["psnr_joint_db"])
            print(f"  {ds:12s} sigma={s}: predicted ens>joint={pred}, actual={actual}, "
                  f"{'MATCH' if pred == actual else 'MISMATCH'}")


if __name__ == "__main__":
    main()
