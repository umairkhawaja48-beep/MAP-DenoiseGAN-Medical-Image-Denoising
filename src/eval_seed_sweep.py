"""
Experiment 2 analysis: evaluates every seed's agent/joint checkpoints on the
SAME fixed-sigma test sets used by the main comparison, then aggregates the
headline (agent - joint) gap across seeds with paired significance tests.

Inference only on existing checkpoints.

Output:
  tables/table7_seed_variance.{csv,tex}
  tables/table7_seed_stats.json
"""
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch

MAP_SCRIPTS = Path(EXTERNAL_DIR)
sys.path.insert(0, str(MAP_SCRIPTS))
from models import UNet2D  # noqa: E402
from utils import batch_psnr_ssim  # noqa: E402

RES = Path(os.path.expanduser(RESULTS_ROOT + ""))
CROSS = RES / "map_project" / "cross_dataset"          # results/checkpoints live here
CROSS_DATA = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))  # test .npy live here
BRATS_FEAS = RES / "map_denoise_feasibility"
BRATS_P2 = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
OUT = RES / "map_project" / "paper" / "tables"

DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]
PRETTY = {"BraTS2021": "BraTS 2021", "nodule3d_128": "nodule3d", "cbsd68_128": "CBSD68"}
SIGMAS = [0.05, 0.10, 0.20]
SIGMA_TO_AGENT = {0.05: "agent1_low", 0.10: "agent2_medium", 0.20: "agent3_high"}
SEEDS = [42, 43, 44, 45]


def seed_dir(ds, seed, model):
    """Locate one (dataset, seed, model) run directory, or None."""
    if seed == 42:
        if ds == "BraTS2021":
            name = "single_agent_joint" if model == "joint" else model
            return BRATS_FEAS / name
        return CROSS / ds / model
    root = CROSS / "brats_extra" / "seeds" if ds == "BraTS2021" else CROSS / ds / "seeds"
    return root / f"seed{seed}" / model


def load_test(ds, sigma):
    tag = f"{sigma:.2f}".replace("0.", "0")
    if ds == "BraTS2021":
        clean = np.load(BRATS_P2 / "phase2_test_5000.npy").astype(np.float32)
        noisy = np.load(BRATS_P2 / f"degraded_sigma{tag}.npy").astype(np.float32)
    else:
        clean = np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_clean.npy").astype(np.float32)
        noisy = np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32)
    return clean, noisy


def denoise(model, noisy, device, batch=200):
    outs = []
    t = torch.from_numpy(noisy).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), batch):
            outs.append(model(t[i:i + batch].to(device)).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in DATASETS:
        cache = {s: load_test(ds, s) for s in SIGMAS}
        for seed in SEEDS:
            models = {}
            missing = []
            for m in ["agent1_low", "agent2_medium", "agent3_high", "joint"]:
                d = seed_dir(ds, seed, m)
                ck = d / "checkpoints" / "generator_best.pt"
                if not ck.exists():
                    missing.append(str(ck))
                    continue
                net = UNet2D().to(device)
                net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
                net.eval()
                models[m] = net
            if missing:
                print(f"  [skip] {ds} seed {seed}: missing {len(missing)} checkpoint(s), "
                      f"first={missing[0]}")
                continue

            for sigma in SIGMAS:
                clean, noisy = cache[sigma]
                agent_name = SIGMA_TO_AGENT[sigma]
                pn, _ = batch_psnr_ssim(clean, noisy)
                pa, sa = batch_psnr_ssim(clean, denoise(models[agent_name], noisy, device))
                pj, sj = batch_psnr_ssim(clean, denoise(models["joint"], noisy, device))
                rows.append({"dataset": PRETTY[ds], "seed": seed, "sigma": sigma,
                             "agent": agent_name,
                             "agent_psnr": round(pa, 4), "joint_psnr": round(pj, 4),
                             "gap_agent_minus_joint": round(pa - pj, 4),
                             "agent_ssim": round(sa, 5), "joint_ssim": round(sj, 5),
                             "psnr_noisy": round(pn, 4)})
                print(f"  {ds} seed{seed} sigma={sigma}: agent={pa:.3f} joint={pj:.3f} "
                      f"gap={pa - pj:+.3f}", flush=True)
            for net in models.values():
                del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        print("no completed seeds found")
        return
    with open(OUT / "table7_seed_variance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT / 'table7_seed_variance.csv'} ({len(rows)} rows)")

    stats = {}
    for ds in DATASETS:
        p = PRETTY[ds]
        stats[p] = {}
        for sigma in SIGMAS:
            sel = [r for r in rows if r["dataset"] == p and r["sigma"] == sigma]
            if not sel:
                continue
            a = np.array([r["agent_psnr"] for r in sel])
            j = np.array([r["joint_psnr"] for r in sel])
            g = a - j
            e = {"n_seeds": len(sel), "seeds": [r["seed"] for r in sel],
                 "agent_mean": float(a.mean()), "agent_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                 "joint_mean": float(j.mean()), "joint_std": float(j.std(ddof=1)) if len(j) > 1 else 0.0,
                 "gap_mean": float(g.mean()), "gap_std": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
                 "gap_min": float(g.min()), "gap_max": float(g.max()),
                 "n_positive": int((g > 0).sum())}
            if len(g) > 2:
                try:
                    from scipy import stats as st
                    e["paired_t_p"] = float(st.ttest_rel(a, j).pvalue)
                    e["wilcoxon_p"] = float(st.wilcoxon(a, j).pvalue)
                except Exception as exc:
                    print(f"  [warn] tests failed: {exc}")
            stats[p][str(sigma)] = e
    json.dump(stats, open(OUT / "table7_seed_stats.json", "w"), indent=2)
    print(f"wrote {OUT / 'table7_seed_stats.json'}")

    body = []
    for ds in DATASETS:
        p = PRETTY[ds]
        if not stats.get(p):
            continue
        for sigma in SIGMAS:
            e = stats[p].get(str(sigma))
            if not e:
                continue
            cells = [p if sigma == SIGMAS[0] else "", f"{sigma:g}",
                     rf"{e['agent_mean']:.2f} $\pm$ {e['agent_std']:.2f}",
                     rf"{e['joint_mean']:.2f} $\pm$ {e['joint_std']:.2f}",
                     rf"\textbf{{{e['gap_mean']:+.2f}}} $\pm$ {e['gap_std']:.2f}",
                     f"{e['n_positive']}/{e['n_seeds']}"]
            if "paired_t_p" in e:
                cells.append(f"{e['paired_t_p']:.3f}")
            else:
                cells.append("--")
            body.append(cells)
        body.append("MIDRULE")
    if body and body[-1] == "MIDRULE":
        body.pop()

    n = stats[PRETTY[DATASETS[0]]][str(SIGMAS[0])]["n_seeds"]
    lines = [r"\begin{table}[!t]", r"\centering",
             rf"\caption{{Seed variance of the headline comparison. Each entry is the mean $\pm$ "
             rf"standard deviation over {n} independently seeded repetitions of the full protocol "
             r"(agent~1 $\rightarrow$ agent~2 $\rightarrow$ agent~3 chain plus the compute-matched "
             r"joint baseline). \emph{Wins} counts seeds in which the matched agent beat the joint "
             r"baseline. The high-noise advantage is consistent across seeds, whereas the "
             r"low-noise differences are within seed noise.}}",
             r"\label{tab:seedvar}",
             r"\begin{tabular}{llrrrcc}", r"\toprule",
             r"Dataset & $\sigma$ & Agent PSNR (dB) & Joint PSNR (dB) & Gap (dB) & Wins & "
             r"paired $p$ \\", r"\midrule"]
    for r in body:
        lines.append(r"\midrule" if r == "MIDRULE" else " & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table7_seed_variance.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'table7_seed_variance.tex'}")

    print("\n=== headline summary (sigma=0.20) ===")
    for ds in DATASETS:
        e = stats.get(PRETTY[ds], {}).get("0.2")
        if e:
            print(f"  {PRETTY[ds]}: gap {e['gap_mean']:+.3f} +- {e['gap_std']:.3f} dB "
                  f"over n={e['n_seeds']} seeds, {e['n_positive']}/{e['n_seeds']} positive"
                  + (f", paired p={e['paired_t_p']:.4f}" if "paired_t_p" in e else ""))


if __name__ == "__main__":
    main()
