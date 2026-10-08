"""
Phase C1: measures the three terms of Theorem 2.4 (Bause et al., 2026,
arXiv:2605.25929) for MAP-DenoiseGAN on all three datasets.

IMPORTANT SCOPE NOTE, implemented here and stated in the manuscript rather
than glossed over: Theorem 2.4 is stated for the Brier loss
l(y,p) = ||p - e_y||^2 with beliefs on the probability simplex, i.e. a
classification setting. Denoising is continuous-valued regression, so the
theorem does not apply verbatim. What carries over is the algebra it rests on:
the ambiguity decomposition for squared Euclidean loss,

    r_ens = sum_j pi_j r_j  -  sum_j pi_j ||s_j - s_bar||^2 ,        (*)

which holds for ANY squared-error target space, not only the simplex. We
therefore restate the condition for per-pixel MSE denoising and VERIFY (*)
numerically here; the residual of that check is reported so the adaptation is
falsifiable rather than asserted.

Terms, per noise level sigma, with r_j(i) the per-image MSE of agent j:
  Specialization Gain SG = E_i[ r_joint(i) - min_j r_j(i) ]
  Local Diversity     D  = E_i[ sum_j pi_j ||s_j(i) - s_bar(i)||^2 ]
  Routing Regret      RR = E_i[ sum_j pi_j r_j(i) - min_j r_j(i) ]
  Condition           SG + D - RR > 0

Two routing policies are evaluated:
  * hard   - our deployed policy: pi is one-hot on the agent matching sigma.
             Then s_bar = s_j so D = 0 EXACTLY, and the condition collapses to
             E[r_joint] > E[r_routed]. We report this openly: under hard
             routing the condition is an identity, so it explains rather than
             independently predicts the observed win. Its value is that it
             decomposes that win into how much specialisation was available
             and how much of it routing gave back.
  * uniform - pi_j = 1/3, an ensemble average. This is the regime in which the
             diversity term is genuinely non-zero, so it tests the full
             three-term condition rather than a degenerate case.

Inference only on existing checkpoints. Nothing is trained.

Output: paper/tables/table8_theorem_terms.csv
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

RES = Path(os.path.expanduser(RESULTS_ROOT + ""))
CROSS = RES / "map_project" / "cross_dataset"
CROSS_DATA = Path(os.path.expanduser(DATA_ROOT + "/cross_dataset"))
BRATS_FEAS = RES / "map_denoise_feasibility"
BRATS_P2 = Path(os.path.expanduser(DATA_ROOT + "/wigan_datasets/phase2"))
OUT = RES / "map_project" / "paper" / "tables"

DATASETS = ["BraTS2021", "nodule3d_128", "cbsd68_128"]
PRETTY = {"BraTS2021": "BraTS 2021", "nodule3d_128": "nodule3d", "cbsd68_128": "CBSD68"}
SIGMAS = [0.05, 0.10, 0.20]
AGENTS = ["agent1_low", "agent2_medium", "agent3_high"]
SIGMA_TO_AGENT = {0.05: 0, 0.10: 1, 0.20: 2}   # index into AGENTS
EVAL_BATCH = 100


def model_dir(ds, name):
    if ds == "BraTS2021":
        return BRATS_FEAS / ("single_agent_joint" if name == "joint" else name)
    return CROSS / ds / name


def load_test(ds, sigma):
    tag = f"{sigma:.2f}".replace("0.", "0")
    if ds == "BraTS2021":
        clean = np.load(BRATS_P2 / "phase2_test_5000.npy").astype(np.float32)
        noisy = np.load(BRATS_P2 / f"degraded_sigma{tag}.npy").astype(np.float32)
    else:
        clean = np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_clean.npy").astype(np.float32)
        noisy = np.load(CROSS_DATA / ds / f"test_fixedsigma{tag}_noisy.npy").astype(np.float32)
    return clean, noisy


def denoise(model, noisy, device):
    outs = []
    t = torch.from_numpy(noisy).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(t), EVAL_BATCH):
            outs.append(model(t[i:i + EVAL_BATCH].to(device)).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def per_image_mse(pred, clean):
    return ((pred - clean) ** 2).reshape(len(clean), -1).mean(axis=1)


def mse_to_db(x):
    """Report terms in dB so they are commensurate with the PSNR tables."""
    return -10.0 * np.log10(np.maximum(x, 1e-12))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in DATASETS:
        models = {}
        for name in AGENTS + ["joint"]:
            ck = model_dir(ds, name) / "checkpoints" / "generator_best.pt"
            m = UNet2D().to(device)
            m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            m.eval()
            models[name] = m
        print(f"[theorem] {ds}: loaded {len(models)} models", flush=True)

        for sigma in SIGMAS:
            clean, noisy = load_test(ds, sigma)
            preds = np.stack([denoise(models[a], noisy, device) for a in AGENTS])   # (3,N,H,W)
            pred_joint = denoise(models["joint"], noisy, device)

            r = np.stack([per_image_mse(preds[j], clean) for j in range(len(AGENTS))])  # (3,N)
            r_joint = per_image_mse(pred_joint, clean)
            r_oracle = r.min(axis=0)                     # min_j r_j(i), per-image oracle
            which = r.argmin(axis=0)

            # --- hard routing (deployed policy): pi one-hot on the matching agent
            k = SIGMA_TO_AGENT[sigma]
            r_hard = r[k]
            sg = float((r_joint - r_oracle).mean())
            rr_hard = float((r_hard - r_oracle).mean())
            d_hard = 0.0                                  # exact: s_bar == s_k
            cond_hard = sg + d_hard - rr_hard

            # --- uniform routing: pi_j = 1/3, diversity genuinely non-zero
            s_bar = preds.mean(axis=0)
            r_ens = per_image_mse(s_bar, clean)
            d_i = np.stack([((preds[j] - s_bar) ** 2).reshape(len(clean), -1).mean(axis=1)
                            for j in range(len(AGENTS))]).mean(axis=0)
            r_bar = r.mean(axis=0)
            rr_unif = float((r_bar - r_oracle).mean())
            d_unif = float(d_i.mean())
            cond_unif = sg + d_unif - rr_unif

            # --- verify the ambiguity identity (*) that licenses the adaptation
            resid = float(np.abs(r_ens - (r_bar - d_i)).max())

            rows.append({
                "dataset": PRETTY[ds], "sigma": sigma,
                "specialization_gain_mse": sg,
                "diversity_hard_mse": d_hard,
                "routing_regret_hard_mse": rr_hard,
                "condition_hard_mse": cond_hard,
                "diversity_uniform_mse": d_unif,
                "routing_regret_uniform_mse": rr_unif,
                "condition_uniform_mse": cond_unif,
                "specialization_gain_db": float(mse_to_db(r_oracle.mean()) - mse_to_db(r_joint.mean())),
                "condition_hard_db": float(mse_to_db(r_hard.mean()) - mse_to_db(r_joint.mean())),
                "condition_uniform_db": float(mse_to_db(r_ens.mean()) - mse_to_db(r_joint.mean())),
                "psnr_joint_db": float(mse_to_db(r_joint.mean())),
                "psnr_routed_db": float(mse_to_db(r_hard.mean())),
                "psnr_oracle_db": float(mse_to_db(r_oracle.mean())),
                "psnr_uniform_ens_db": float(mse_to_db(r_ens.mean())),
                "oracle_picks_matching_agent_frac": float((which == k).mean()),
                "ambiguity_identity_max_resid": resid,
                "n_images": int(len(clean)),
            })
            print(f"  sigma={sigma}: SG={sg:.3e} RR_hard={rr_hard:.3e} "
                  f"cond_hard={cond_hard:+.3e} | D_unif={d_unif:.3e} "
                  f"RR_unif={rr_unif:.3e} cond_unif={cond_unif:+.3e} | "
                  f"oracle picks matching agent {100 * (which == k).mean():.1f}% | "
                  f"identity resid={resid:.2e}", flush=True)

        for m in models.values():
            del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(OUT / "table8_theorem_terms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r_ in rows:
            w.writerow(r_)
    print(f"\nwrote {OUT / 'table8_theorem_terms.csv'} ({len(rows)} rows)")

    max_resid = max(r_["ambiguity_identity_max_resid"] for r_ in rows)
    summary = {
        "ambiguity_identity_max_residual_over_all": max_resid,
        "identity_holds": bool(max_resid < 1e-6),
        "note": ("Theorem 2.4 is stated for Brier loss on the simplex; this is an "
                 "adaptation to per-pixel MSE, licensed by the ambiguity decomposition "
                 "for squared Euclidean loss, verified numerically here."),
    }
    json.dump(summary, open(OUT / "table8_theorem_check.json", "w"), indent=2)
    print(f"ambiguity identity max residual over all rows: {max_resid:.3e} "
          f"({'HOLDS' if max_resid < 1e-6 else 'FAILS'})")


if __name__ == "__main__":
    main()
