"""
Task B1.1 numerical check for Proposition 2 (generalisation bound).

Zoubeirou a Mayaki (2026) bounds MoE estimation error by (i) a term in the
ACTIVE per-input parameter budget and (ii) a routing-overhead term obtained by
union-bounding over realisable routing patterns.

For MAP-DenoiseGAN under hard routing exactly one agent is active per input, so
term (i) is IDENTICAL to the single-agent joint model's -- both are one U-Net of
5,142,433 parameters. The entire difference between the two bounds is therefore
term (ii).

A generalisation bound carries unspecified constants, so "the bound holds"
cannot be checked directly. What IS checkable is its comparative prediction:

    Since active capacity is equal, MAP's excess generalisation gap over the
    joint model is controlled by the routing overhead alone. That overhead
    shrinks as routing becomes deterministic. Hence at high noise -- where we
    measure the matched agent to be per-image optimal for ~100% of inputs, so
    the realised routing collapses to essentially one fixed pattern -- MAP's
    generalisation gap should be no worse than the joint model's, while its
    approximation error is strictly better.

This script measures, per dataset and noise level:
  * train-set and test-set MSE for the matched agent and the joint model
  * the generalisation gap (test - train) for each
  * the excess gap (MAP - joint)
  * the realised routing entropy, as the empirical proxy for routing overhead

Inference only on existing checkpoints.
Output: paper/tables/table9_generalization.csv
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
from utils import add_noise  # noqa: E402

RES = Path(os.path.expanduser(RESULTS_ROOT + ""))
CROSS = RES / "map_project" / "cross_dataset"
CROSS_DATA = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
BRATS_FEAS = RES / "map_denoise_feasibility"
BRATS_DATA = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/map_denoise"))
BRATS_P2 = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
OUT = RES / "map_project" / "paper" / "tables"

DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]
PRETTY = {"BraTS2021": "BraTS 2021", "nodule3d_128": "nodule3d", "cbsd68_128": "CBSD68"}
SIGMAS = [0.05, 0.10, 0.20]
AGENTS = ["agent1_low", "agent2_medium", "agent3_high"]
SIGMA_TO_IDX = {0.05: 0, 0.10: 1, 0.20: 2}
MAX_TRAIN = 800           # cap train-set evaluation for tractability
BATCH = 100


def model_dir(ds, name):
    if ds == "BraTS2021":
        return BRATS_FEAS / ("single_agent_joint" if name == "joint" else name)
    return CROSS / ds / name


def train_clean(ds):
    d = BRATS_DATA if ds == "BraTS2021" else CROSS_DATA / ds
    return np.load(d / "train_joint_clean.npy").astype(np.float32)


def test_pair(ds, sigma):
    tag = f"{sigma:.2f}".replace("0.", "0")
    if ds == "BraTS2021":
        return (np.load(BRATS_P2 / "phase2_test_5000.npy").astype(np.float32),
                np.load(BRATS_P2 / f"degraded_sigma{tag}.npy").astype(np.float32))
    return (np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_clean.npy").astype(np.float32),
            np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32))


def denoise(model, noisy, device):
    outs = []
    t = torch.from_numpy(noisy).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), BATCH):
            outs.append(model(t[i:i + BATCH].to(device)).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def per_image_mse(pred, clean):
    return ((pred - clean) ** 2).reshape(len(clean), -1).mean(axis=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in DATASETS:
        models = {}
        for name in AGENTS + ["joint"]:
            m = UNet2D().to(device)
            m.load_state_dict(torch.load(model_dir(ds, name) / "checkpoints" / "generator_best.pt",
                                         map_location=device, weights_only=True))
            m.eval()
            models[name] = m
        n_active = sum(p.numel() for p in models["joint"].parameters())

        tr_clean_full = train_clean(ds)
        tr_clean = tr_clean_full[:MAX_TRAIN]
        print(f"[prop2] {ds}: active params/input = {n_active:,} "
              f"(identical for MAP and joint); train eval on {len(tr_clean)} images", flush=True)

        for sigma in SIGMAS:
            k = SIGMA_TO_IDX[sigma]
            agent = AGENTS[k]

            # train-set risk at this fixed sigma (same degradation as the test set)
            tr_noisy = add_noise(tr_clean, gaussian_sigma=sigma, seed=12345)
            tr_agent = per_image_mse(denoise(models[agent], tr_noisy, device), tr_clean).mean()
            tr_joint = per_image_mse(denoise(models["joint"], tr_noisy, device), tr_clean).mean()

            te_clean, te_noisy = test_pair(ds, sigma)
            r_all = np.stack([per_image_mse(denoise(models[a], te_noisy, device), te_clean)
                              for a in AGENTS])
            te_agent = r_all[k].mean()
            te_joint = per_image_mse(denoise(models["joint"], te_noisy, device), te_clean).mean()

            # realised routing: which agent is per-image optimal -> empirical
            # proxy for how many routing patterns the data actually exercises
            which = r_all.argmin(axis=0)
            counts = np.bincount(which, minlength=len(AGENTS)).astype(np.float64)
            p = counts / counts.sum()
            nz = p[p > 0]
            entropy = float(-(nz * np.log2(nz)).sum())
            eff_patterns = float(2 ** entropy)

            gap_agent = float(te_agent - tr_agent)
            gap_joint = float(te_joint - tr_joint)
            rows.append({
                "dataset": PRETTY[ds], "sigma": sigma,
                "active_params_map": n_active, "active_params_joint": n_active,
                "train_mse_map": float(tr_agent), "test_mse_map": float(te_agent),
                "gap_map": gap_agent,
                "train_mse_joint": float(tr_joint), "test_mse_joint": float(te_joint),
                "gap_joint": gap_joint,
                "excess_gap_map_minus_joint": gap_agent - gap_joint,
                "routing_entropy_bits": entropy,
                "effective_routing_patterns": eff_patterns,
                "matched_is_optimal_frac": float((which == k).mean()),
            })
            print(f"  sigma={sigma}: gap_MAP={gap_agent:+.3e} gap_joint={gap_joint:+.3e} "
                  f"excess={gap_agent - gap_joint:+.3e} | routing H={entropy:.3f} bits "
                  f"({eff_patterns:.2f} eff. patterns)", flush=True)

        for m in models.values():
            del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(OUT / "table9_generalization.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT / 'table9_generalization.csv'}")

    # the comparative prediction: excess gap should shrink as routing
    # becomes deterministic (entropy -> 0)
    ent = np.array([r["routing_entropy_bits"] for r in rows])
    exc = np.array([r["excess_gap_map_minus_joint"] for r in rows])
    corr = float(np.corrcoef(ent, exc)[0, 1])
    hi = [r for r in rows if r["sigma"] == 0.20]
    holds_hi = all(r["excess_gap_map_minus_joint"] <= 0 for r in hi)
    summary = {
        "active_params_identical": True,
        "pearson_r_entropy_vs_excess_gap": corr,
        "excess_gap_nonpositive_at_sigma020_all_datasets": bool(holds_hi),
        "sigma020_excess_gaps": {r["dataset"]: r["excess_gap_map_minus_joint"] for r in hi},
        "sigma020_routing_entropy": {r["dataset"]: r["routing_entropy_bits"] for r in hi},
    }
    json.dump(summary, open(OUT / "table9_prop2_check.json", "w"), indent=2)
    print(f"Pearson r(routing entropy, excess gap) = {corr:+.3f}")
    print(f"excess gap <= 0 at sigma=0.20 on all datasets: {holds_hi}")


if __name__ == "__main__":
    main()
