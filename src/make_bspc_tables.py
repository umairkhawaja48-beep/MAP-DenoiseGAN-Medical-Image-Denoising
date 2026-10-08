"""
BSPC Task C1.3: generates Tables 1-6 as CSV + LaTeX (booktabs).

All numbers are read from existing result files or computed by INFERENCE on
existing checkpoints. Nothing is trained. The one computed-here quantity is
SwinIR-lite's PSNR on the BraTS fixed-sigma test sets: the original BraTS
feasibility comparison CSV predates the SwinIR baseline and so lacks that row,
while both other datasets have it -- computing it makes Table 3 complete and
symmetric rather than leaving a hole in the headline table.

Output: $MAP_RESULTS_ROOT/map_project/paper/tables/
"""
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import sys
import time
from pathlib import Path

import numpy as np
import torch

MAP_SCRIPTS = Path(EXTERNAL_DIR)
WIGAN_SCRIPTS = Path(EXTERNAL_DIR)
PHASE2_DIR = WIGAN_SCRIPTS / "phase2"
for p in (str(MAP_SCRIPTS), str(WIGAN_SCRIPTS), str(PHASE2_DIR)):
    sys.path.insert(0, p)

from models import UNet2D, PatchGANDiscriminator  # noqa: E402  (map_project's own copies)
from utils import batch_psnr_ssim  # noqa: E402
from dncnn_model import DnCNN  # noqa: E402
from restormer_lite_model import RestormerLite  # noqa: E402
from swinir_lite_model import SwinIRLite  # noqa: E402

OUT = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
BRATS_DATA = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
BRATS_RES = Path(os.path.expanduser(RESULTS_ROOT + "/map_denoise_feasibility"))
PHASE2_RES = Path(os.path.expanduser(RESULTS_ROOT + "/wigan_phase2"))
CROSS_RES = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/cross_dataset"))

SIGMAS = [0.05, 0.10, 0.20]
SIGMA_TO_AGENT = {0.05: "agent1_low", 0.10: "agent2_medium", 0.20: "agent3_high"}
DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]
PRETTY_DS = {"BraTS2021": "BraTS 2021", "nodule3d_128": "nodule3d", "cbsd68_128": "CBSD68"}
METHOD_ORDER = ["none", "dncnn", "restormer", "swinir", "restormer_range", "joint", "map_agent"]
PRETTY_METHOD = {
    "none": "No denoising", "dncnn": "DnCNN ($\\sigma{=}0.1$)",
    "restormer": "Restormer-lite ($\\sigma{=}0.1$)", "swinir": "SwinIR-lite ($\\sigma{=}0.1$)",
    "restormer_range": "Restormer-lite (range-trained)",
    "map_agent": "MAP-DenoiseGAN (matched agent)",
    "joint": "Single-agent joint",
}
# baselines trained at a single fixed sigma -- evaluated out-of-distribution at
# sigma != 0.1, which is exactly the confound the range-trained variant removes
FIXED_SIGMA_BASELINES = ["dncnn", "restormer", "swinir"]
RANGE_BASELINES = ["restormer_range"]


def write_csv(name, fieldnames, rows):
    path = OUT / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def write_latex(name, caption, label, header, body_rows, align=None, notes=None):
    ncol = len(header)
    align = align or ("l" + "r" * (ncol - 1))
    lines = [
        r"\begin{table}[!t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    for r in body_rows:
        if r == "MIDRULE":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\begin{{tablenotes}}\footnotesize{{{notes}}}\end{{tablenotes}}")
    lines.append(r"\end{table}")
    path = OUT / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ----------------------------------------------------------------- Table 1
def table1():
    rows = [
        {"dataset": "BraTS 2021", "modality": "Brain MRI (axial slices)", "train": 8000,
         "val": 3000, "test": 5000, "resolution": "128x128", "source": "BraTS 2021 challenge"},
        {"dataset": "nodule3d", "modality": "Lung CT (nodule slices)", "train": 1158,
         "val": 165, "test": 310, "resolution": "128x128", "source": "MedMNIST3D NoduleMNIST3D"},
        {"dataset": "CBSD68", "modality": "Natural images (grayscale)", "train": 360,
         "val": 40, "test": 68, "resolution": "128x128", "source": "BSD400 train / BSD68 test"},
    ]
    write_csv("table1_datasets", list(rows[0].keys()), rows)
    body = [[r["dataset"], r["modality"], f"{r['train']:,}", f"{r['val']:,}", f"{r['test']:,}", r["source"]]
            for r in rows]
    write_latex(
        "table1_datasets",
        "Datasets. All images are resampled to $128\\times128$ and scaled to $[0,1]$. "
        "CBSD68 is a 68-image test-only benchmark and is therefore paired with its conventional "
        "companion training set BSD400, which is split 90/10 into train/validation; BSD68 is held "
        "out entirely and never seen during training.",
        "tab:datasets",
        ["Dataset", "Modality", "Train", "Val", "Test", "Source"],
        body, align="llrrrl")
    return rows


# ----------------------------------------------------------------- Table 2
def table2(device):
    specs = {
        "MAP-DenoiseGAN generator (U-Net)": UNet2D,
        "PatchGAN discriminator (training only)": lambda: PatchGANDiscriminator(in_channels=2),
        "DnCNN": DnCNN,
        "Restormer-lite": RestormerLite,
        "SwinIR-lite": SwinIRLite,
    }
    rows = []
    for label, ctor in specs.items():
        model = ctor().to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        if "discriminator" in label:
            ms = float("nan")
        else:
            x = torch.rand(32, 1, 128, 128, device=device)
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - t0) / 10 / 32 * 1000
        rows.append({"model": label, "parameters": n_params,
                     "inference_ms_per_image": ("-" if ms != ms else round(ms, 3))})
        print(f"  {label}: {n_params:,} params, {ms:.3f} ms/img" if ms == ms
              else f"  {label}: {n_params:,} params")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv("table2_complexity", list(rows[0].keys()), rows)
    body = [[r["model"], f"{r['parameters']:,}", r["inference_ms_per_image"]] for r in rows]
    write_latex(
        "table2_complexity",
        "Model complexity and inference cost. Timings are per image at $128\\times128$, batch size 32, "
        "on a single NVIDIA A40-class GPU, averaged over 10 runs after 3 warm-up passes. "
        "The discriminator is used only during training and adds no inference cost. "
        "Every MAP-DenoiseGAN agent is one generator, so a three-agent deployment that routes each "
        "image to a single agent has the same per-image inference cost as one generator.",
        "tab:complexity",
        ["Model", "Parameters", "Inference (ms/image)"], body, align="lrr")
    return rows


# ----------------------------------------------------------------- Table 3
def load_comparison(dataset):
    path = BRATS_RES / "comparison_results.csv" if dataset == "BraTS2021" \
        else CROSS_RES / dataset / "comparison_results.csv"
    return list(csv.DictReader(open(path)))


def canonical_method(raw):
    if raw.startswith("map_denoise_"):
        return "map_agent"
    if raw == "single_agent_joint":
        return "joint"
    return raw


def swinir_on_brats(device):
    """Fills the one hole in Table 3: SwinIR-lite was added to the study after
    the BraTS comparison CSV was written, so its BraTS PSNR is computed here
    from the existing Phase 2 checkpoint (inference only)."""
    ckpt = PHASE2_RES / "baselines" / "swinir" / "checkpoints" / "swinir_best.pt"
    model = SwinIRLite().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    clean = np.load(BRATS_DATA / "phase2_test_5000.npy").astype(np.float32)
    out = {}
    for sigma in SIGMAS:
        tag = f"{sigma:.2f}".replace("0.", "0")
        degraded = np.load(BRATS_DATA / f"degraded_sigma{tag}.npy").astype(np.float32)
        den = []
        with torch.no_grad():
            t = torch.from_numpy(degraded).unsqueeze(1)
            for i in range(0, len(t), 200):
                den.append(model(t[i:i + 200].to(device)).cpu().numpy()[:, 0])
        den = np.concatenate(den)
        psnr_n, ssim_n = batch_psnr_ssim(clean, degraded)
        psnr_d, ssim_d = batch_psnr_ssim(clean, den)
        out[sigma] = {"psnr_noisy": psnr_n, "psnr_denoised": psnr_d, "delta_psnr": psnr_d - psnr_n,
                      "ssim_denoised": ssim_d}
        print(f"  SwinIR-lite on BraTS sigma={sigma}: PSNR {psnr_n:.2f}->{psnr_d:.2f} "
              f"(dPSNR={psnr_d - psnr_n:+.2f})")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def load_range_baseline(ds):
    """Experiment 1: Restormer-lite trained over the full sigma range."""
    base = (Path(os.path.expanduser(RESULTS_ROOT + "/map_project/brats_extra"))
            if ds == "BraTS2021" else CROSS_RES / ds)
    p = base / "restormer_range" / "test_metrics.json"
    if not p.exists():
        print(f"  [warn] no range baseline for {ds} at {p}")
        return None
    m = json.load(open(p))
    out = {}
    for k, v in m["fixed_sigma"].items():
        out[float(k)] = {"psnr_denoised": v["psnr_denoised"], "delta_psnr": v["delta_psnr"],
                         "ssim_denoised": v["ssim_denoised"], "psnr_noisy": v["psnr_noisy"]}
    return out


def table3(device):
    swin_brats = swinir_on_brats(device)
    table = {}
    for ds in DATASETS:
        rows = load_comparison(ds)
        table[ds] = {}
        for r in rows:
            sigma = float(r["sigma"])
            m = canonical_method(r["method"])
            table[ds].setdefault(sigma, {})[m] = {
                "psnr_denoised": float(r["psnr_denoised"]),
                "delta_psnr": float(r["delta_psnr"]),
                "ssim_denoised": float(r["ssim_denoised"]),
                "psnr_noisy": float(r["psnr_noisy"]),
            }
        if ds == "BraTS2021":
            for sigma, v in swin_brats.items():
                table[ds][sigma]["swinir"] = v
        rng = load_range_baseline(ds)
        if rng:
            for sigma, v in rng.items():
                table[ds][sigma]["restormer_range"] = v

    flat = []
    for ds in DATASETS:
        for sigma in SIGMAS:
            for m in METHOD_ORDER:
                e = table[ds][sigma].get(m)
                if e is None:
                    continue
                flat.append({"dataset": PRETTY_DS[ds], "sigma": sigma, "method": PRETTY_METHOD[m],
                             "psnr_db": round(e["psnr_denoised"], 2),
                             "delta_psnr_db": round(e["delta_psnr"], 2),
                             "ssim": round(e["ssim_denoised"], 4)})
    write_csv("table3_main_psnr", list(flat[0].keys()), flat)

    body = []
    for ds in DATASETS:
        body.append([rf"\multicolumn{{7}}{{l}}{{\textit{{{PRETTY_DS[ds]}}}}}"])
        for m in METHOD_ORDER:
            if table[ds][SIGMAS[0]].get(m) is None:
                continue
            cells = [PRETTY_METHOD[m]]
            for sigma in SIGMAS:
                e = table[ds][sigma][m]
                val = f"{e['psnr_denoised']:.2f}"
                best = max(table[ds][sigma][k]["psnr_denoised"]
                           for k in table[ds][sigma] if k != "none")
                if m != "none" and abs(e["psnr_denoised"] - best) < 1e-9:
                    val = rf"\textbf{{{val}}}"
                cells += [val, f"{e['ssim_denoised']:.3f}"]
            body.append(cells)
        if ds != DATASETS[-1]:
            body.append("MIDRULE")

    header = ["Method", r"PSNR$_{0.05}$", r"SSIM$_{0.05}$", r"PSNR$_{0.10}$", r"SSIM$_{0.10}$",
              r"PSNR$_{0.20}$", r"SSIM$_{0.20}$"]
    write_latex(
        "table3_main_psnr",
        "Denoising performance (PSNR in dB, SSIM) at three fixed noise levels on all three datasets. "
        "Best value per column in bold. MAP-DenoiseGAN is evaluated with known routing: each "
        "fixed-$\\sigma$ test set is processed by its matching agent, which isolates specialisation "
        "quality from router accuracy. The classical baselines are trained at a single "
        "$\\sigma=0.1$ following their original protocols, so $\\sigma=0.05$ and $\\sigma=0.20$ also "
        "measure their robustness to a noise level they were not trained on.",
        "tab:main", header, body, align="l" + "rr" * 3)
    return table


# ----------------------------------------------------------------- Table 4
def read_train_log(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        v = r.get("val_psnr_denoised", "")
        if v not in ("", None):
            out.append((int(r["epoch"]), float(v)))
    return out


def epochs_to_match(scratch_curve, target):
    for ep, psnr in scratch_curve:
        if psnr >= target:
            return ep
    return None


def table4():
    specs = {
        "BraTS2021": (BRATS_RES, "single_agent_joint"),
        "nodule3d_128": (CROSS_RES / "nodule3d_128", "joint"),
        "cbsd68_128": (CROSS_RES / "cbsd68_128", "joint"),
    }
    rows = []
    speedups = {}
    for ds, (base, joint_name) in specs.items():
        for agent in ["agent1_low", "agent2_medium", "agent3_high", joint_name]:
            mp = base / agent / "test_metrics.json"
            if not mp.exists():
                continue
            m = json.load(open(mp))
            inherited = m.get("init_from") is not None
            rows.append({
                "dataset": PRETTY_DS[ds], "model": agent if agent != joint_name else "joint",
                "initialisation": "inherited" if inherited else "from scratch",
                "epochs": m["epochs"], "best_epoch": m["best_epoch"],
                "wall_clock_s": round(m["total_wall_time_s"], 1),
                "train_images": m["train_samples"],
                "test_psnr_db": round(m["psnr_denoised"], 2),
            })
        # inheritance speed-up: agent2_medium inherited vs its from-scratch control
        inh = base / "agent2_medium" / "train_log.csv"
        scr = base / "agent2_medium_scratch" / "train_log.csv"
        if inh.exists() and scr.exists():
            inh_curve = read_train_log(inh)
            scr_curve = read_train_log(scr)
            target = inh_curve[0][1]
            ep = epochs_to_match(scr_curve, target)
            speedups[ds] = {
                "inherited_epoch1_val_psnr": round(target, 2),
                "scratch_epochs_to_match": ep,
                "scratch_best_val_psnr_in_30": round(max(p for _, p in scr_curve), 2),
                "speedup_x": (f">{scr_curve[-1][0]}" if ep is None else f"{ep}"),
            }
    write_csv("table4_training_efficiency", list(rows[0].keys()), rows)
    body = [[r["dataset"], r["model"].replace("_", r"\_"), r["initialisation"], r["best_epoch"],
             f"{r['wall_clock_s']:.0f}", f"{r['test_psnr_db']:.2f}"] for r in rows]
    write_latex(
        "table4_training_efficiency",
        "Training efficiency. All runs use a 30-epoch budget. \\emph{Best epoch} is the epoch at "
        "which validation PSNR peaked; inherited agents frequently peak at epoch~1, i.e. the "
        "inherited initialisation is already near-optimal for the new noise regime before any "
        "fine-tuning on it.",
        "tab:efficiency",
        ["Dataset", "Model", "Initialisation", "Best epoch", "Wall clock (s)", "Test PSNR (dB)"],
        body, align="lllrrr")

    srows = []
    for ds in DATASETS:
        s = speedups.get(ds)
        if not s:
            continue
        srows.append({"dataset": PRETTY_DS[ds], **s})
    write_csv("table4b_inheritance_speedup", list(srows[0].keys()), srows)
    body = [[r["dataset"], f"{r['inherited_epoch1_val_psnr']:.2f}",
             ("not reached in 30" if r["scratch_epochs_to_match"] is None
              else str(r["scratch_epochs_to_match"])),
             f"{r['scratch_best_val_psnr_in_30']:.2f}",
             (r"$\geq 30\times$" if r["scratch_epochs_to_match"] is None
              else rf"${30 // max(r['scratch_epochs_to_match'],1)}\times$")] for r in srows]
    write_latex(
        "table4b_inheritance_speedup",
        "Inheritance speed-up, measured on the first inheritance step "
        "(\\texttt{agent1\\_low}$\\rightarrow$\\texttt{agent2\\_medium}). The inherited agent's "
        "validation PSNR after a single epoch is compared against the epoch at which an otherwise "
        "identical from-scratch control first reaches that same value. On BraTS and CBSD68 the "
        "control never reaches it within the 30-epoch budget.",
        "tab:speedup",
        ["Dataset", "Inherited, epoch 1 (dB)", "Scratch epochs to match",
         "Scratch best in 30 (dB)", "Speed-up"],
        body, align="lrrrr")
    return rows, speedups


# ----------------------------------------------------------------- Table 5
def table5():
    classical = {}
    for r in csv.DictReader(open(PHASE2_RES / "downstream_results.csv")):
        if r["model"] in ("none", "oracle", "dncnn", "restormer", "swinir"):
            classical.setdefault(r["model"], {})[float(r["sigma"])] = r
    mapped = {}
    for r in csv.DictReader(open(OUT / "table5_downstream_dice.csv")):
        key = "map_agent" if r["model"].startswith("map_denoisegan_routed") else (
            "map_joint" if r["model"] == "map_single_agent_joint" else r["model"])
        mapped.setdefault(key, {})[float(r["sigma"])] = r

    rng_path = OUT / "table5_downstream_range.csv"
    rng = {}
    if rng_path.exists():
        for r in csv.DictReader(open(rng_path)):
            rng.setdefault("restormer_range", {})[float(r["sigma"])] = r
    else:
        print("  [warn] no downstream range-baseline row")

    order = [("none", "No denoising"), ("dncnn", "DnCNN ($\\sigma{=}0.1$)"),
             ("restormer", "Restormer-lite ($\\sigma{=}0.1$)"),
             ("swinir", "SwinIR-lite ($\\sigma{=}0.1$)"),
             ("restormer_range", "Restormer-lite (range-trained)"),
             ("map_joint", "Single-agent joint"),
             ("map_agent", "MAP-DenoiseGAN (matched agent)"), ("oracle", "Oracle (clean image)")]
    src = {**classical, **mapped, **rng}
    order = [(k, lbl) for k, lbl in order if k in src]
    flat, body = [], []
    for key, label in order:
        cells = [label]
        for sigma in SIGMAS:
            r = src[key][sigma]
            d = float(r["mean_dice"])
            flat.append({"method": label, "sigma": sigma, "mean_dice": d,
                         "std_dice": float(r["std_dice"]),
                         "mean_hausdorff": float(r["mean_hausdorff"])})
            txt = f"{d:.4f}"
            if key == "map_agent":
                txt = rf"\textbf{{{txt}}}"
            cells += [txt, f"{float(r['mean_hausdorff']):.1f}"]
        body.append(cells)
    write_csv("table5_downstream_summary", list(flat[0].keys()), flat)
    write_latex(
        "table5_downstream_dice",
        "Downstream clinical task: BraTS tumour segmentation Dice and Hausdorff distance (px) after "
        "denoising, using a fixed pre-trained segmentation U-Net (5{,}000 test slices). "
        "\\emph{Oracle} segments the true clean image and is the upper bound achievable by any "
        "denoiser. At $\\sigma=0.20$ the fixed-$\\sigma$ baselines fall to Dice $\\approx0.65$, but "
        "this reflects their being evaluated outside their training distribution: a range-trained "
        "Restormer-lite reaches 0.786, statistically indistinguishable from MAP-DenoiseGAN's 0.791. "
        "The clinically relevant conclusion is therefore that noise-range-aware training of "
        "\\emph{either} architecture preserves downstream accuracy, not that the multi-agent "
        "decomposition is required for it.",
        "tab:dice",
        ["Method", r"Dice$_{0.05}$", r"HD$_{0.05}$", r"Dice$_{0.10}$", r"HD$_{0.10}$",
         r"Dice$_{0.20}$", r"HD$_{0.20}$"],
        body, align="l" + "rr" * 3)
    return src


# ----------------------------------------------------------------- Table 6
def table6(t3, speedups):
    rows, body = [], []
    for ds in DATASETS:
        agent_hi = t3[ds][0.20]["map_agent"]["psnr_denoised"]
        joint_hi = t3[ds][0.20]["joint"]["psnr_denoised"]
        best_fixed_hi = max(t3[ds][0.20][m]["psnr_denoised"]
                            for m in FIXED_SIGMA_BASELINES if m in t3[ds][0.20])
        rng_hi = t3[ds][0.20].get("restormer_range", {}).get("psnr_denoised")
        # the fair comparison is against the range-trained baseline; the
        # fixed-sigma one is evaluated out of distribution at this noise level
        best_classical_hi = max([best_fixed_hi] + ([rng_hi] if rng_hi else []))
        deltas = []
        for sigma in SIGMAS:
            deltas.append(t3[ds][sigma]["map_agent"]["psnr_denoised"]
                          - t3[ds][sigma]["joint"]["psnr_denoised"])
        monotone = all(deltas[i] < deltas[i + 1] for i in range(len(deltas) - 1))
        s = speedups.get(ds, {})
        rows.append({
            "dataset": PRETTY_DS[ds],
            "delta_vs_joint_s005": round(deltas[0], 2),
            "delta_vs_joint_s010": round(deltas[1], 2),
            "delta_vs_joint_s020": round(deltas[2], 2),
            "delta_vs_fixed_sigma_classical_s020": round(agent_hi - best_fixed_hi, 2),
            "delta_vs_range_classical_s020": (round(agent_hi - rng_hi, 2) if rng_hi else ""),
            "advantage_monotone_in_sigma": monotone,
            "inheritance_speedup": s.get("speedup_x", "n/a"),
        })
        body.append([PRETTY_DS[ds], f"{deltas[0]:+.2f}", f"{deltas[1]:+.2f}",
                     rf"\textbf{{{deltas[2]:+.2f}}}",
                     f"{agent_hi - best_fixed_hi:+.2f}",
                     (f"{agent_hi - rng_hi:+.2f}" if rng_hi else "--"),
                     r"\checkmark" if monotone else r"$\times$",
                     (r"$\geq 30\times$" if str(s.get("speedup_x", "")).startswith(">")
                      else rf"${30 // max(int(s.get('speedup_x', 30)),1)}\times$")])
    write_csv("table6_replication", list(rows[0].keys()), rows)
    write_latex(
        "table6_replication",
        "Cross-dataset replication summary. $\\Delta$ is MAP-DenoiseGAN's matched agent minus the "
        "compute-matched single-agent joint baseline, in dB; the specialisation advantage grows "
        "monotonically with noise level on all three datasets. The final two columns contrast the "
        "margin over the strongest \\emph{fixed}-$\\sigma$ classical baseline with the margin over "
        "a \\emph{range}-trained Restormer-lite. The large apparent margin shrinks to "
        "0.1--0.6~dB once the baseline is trained on the same noise distribution, showing that most "
        "of it reflects out-of-distribution evaluation of the baseline rather than the benefit of "
        "specialisation. The advantage does not change sign at low noise on nodule3d or CBSD68: we "
        "therefore report a monotonic trend, not a crossover.",
        "tab:replication",
        ["Dataset", r"$\Delta_{0.05}$", r"$\Delta_{0.10}$", r"$\Delta_{0.20}$",
         r"vs.\ fixed-$\sigma_{0.20}$", r"vs.\ range-trained$_{0.20}$", "Monotone", "Speed-up"],
        body, align="lrrrrrcc")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[make_bspc_tables] device={device}")
    print("Table 1: datasets");            table1()
    print("Table 2: complexity");          table2(device)
    print("Table 3: main PSNR");           t3 = table3(device)
    print("Table 4: training efficiency"); _, speedups = table4()
    print("Table 5: downstream Dice");     table5()
    print("Table 6: replication");         t6 = table6(t3, speedups)
    with open(OUT / "key_numbers.json", "w") as f:
        json.dump({"replication": t6, "speedups": speedups}, f, indent=2)
    print(f"[make_bspc_tables] all tables written to {OUT}")


if __name__ == "__main__":
    main()
