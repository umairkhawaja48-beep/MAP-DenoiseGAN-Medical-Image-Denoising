"""
Task 5: Figure 8 -- PSNR against sigma under each routing strategy, one panel
per dataset, with oracle and estimated plotted so their overlap is visible.

Reads only routing_comparison.csv. Draws nothing that is not in that file.
"""
import csv
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/estimated_routing"))
SIGMAS = [0.05, 0.10, 0.20]
DS = ["BraTS 2021", "nodule3d", "CBSD68"]

# Published display label for the natural-image dataset. The directory and
# CSV key ("cbsd68_128" / "CBSD68") are internal historical paths and are
# deliberately left untouched; only the label the reader sees is mapped.
DISPLAY = {"CBSD68": "BSD68"}


def disp(name):
    return DISPLAY.get(name, name)


STRATS = [
    ("expert1",   "Expert 1 only",    "#999999", ":",  "x", 1.2),
    ("random",    "Random routing",   "#8c564b", "--", "v", 1.2),
    ("oracle",    "Oracle ($\\sigma$)", "#1f77b4", "-",  "o", 2.4),
    ("estimated", "Estimated ($\\hat{\\sigma}$)", "#d62728", "--", "s", 1.6),
]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.4, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    rows = list(csv.DictReader(open(OUT / "routing_comparison.csv")))
    d = {}
    for r in rows:
        d.setdefault(r["dataset"], {})[float(r["sigma"])] = r

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for ax, ds in zip(axes, DS):
        for key, label, color, ls, mk, lw in STRATS:
            mean = [float(d[ds][s][f"{key}_psnr_mean"]) for s in SIGMAS]
            std = [float(d[ds][s][f"{key}_psnr_std"]) for s in SIGMAS]
            ax.errorbar(SIGMAS, mean, yerr=std, fmt=ls, color=color, marker=mk,
                        ms=4, lw=lw, capsize=2.5, label=label)
        ax.set_title(disp(ds), fontsize=9)
        ax.set_xlabel("Noise level $\\sigma$")
        ax.set_xticks(SIGMAS)
    axes[0].set_ylabel("PSNR (dB)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15),
               fontsize=7.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_estimated_routing.pdf")
    fig.savefig(OUT / "fig8_estimated_routing.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT / 'fig8_estimated_routing.pdf'}")
    print(f"wrote {OUT / 'fig8_estimated_routing.png'}")

    print("\noracle vs estimated, PSNR gap (dB):")
    for ds in DS:
        for s in SIGMAS:
            o = float(d[ds][s]["oracle_psnr_mean"])
            e = float(d[ds][s]["estimated_psnr_mean"])
            print(f"  {ds:<12} sigma={s:<5} oracle={o:7.3f}  estimated={e:7.3f}  "
                  f"gap={e - o:+.3f}")


if __name__ == "__main__":
    main()
