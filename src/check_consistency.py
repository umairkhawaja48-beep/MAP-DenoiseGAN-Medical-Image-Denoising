"""
BSPC Task C2.1: automated self-review. Verifies that numbers asserted in the
manuscript text actually match the generated tables / source result files,
that no forbidden claim appears, and that every figure and table is referenced.

Run:  python check_consistency.py <manuscript.tex>
Exit code 0 if all checks pass.
"""
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import re
import sys
from pathlib import Path

TAB = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
FIG = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/figures"))

failures, passes, warnings = [], [], []


def check(name, condition, detail=""):
    (passes if condition else failures).append(f"{name}: {detail}")
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def warn(name, detail):
    warnings.append(f"{name}: {detail}")
    print(f"  [WARN] {name} -- {detail}")


def load_t3():
    t = {}
    for r in csv.DictReader(open(TAB / "table3_main_psnr.csv")):
        t.setdefault(r["dataset"], {}).setdefault(float(r["sigma"]), {})[r["method"]] = {
            "psnr": float(r["psnr_db"]), "ssim": float(r["ssim"])}
    return t


def main():
    tex_path = Path(sys.argv[1])
    tex = tex_path.read_text(encoding="utf-8")
    t3 = load_t3()
    t6 = {r["dataset"]: r for r in csv.DictReader(open(TAB / "table6_replication.csv"))}
    t5 = {}
    for r in csv.DictReader(open(TAB / "table5_downstream_summary.csv")):
        t5.setdefault(r["method"], {})[float(r["sigma"])] = float(r["mean_dice"])
    t4b = {r["dataset"]: r for r in csv.DictReader(open(TAB / "table4b_inheritance_speedup.csv"))}
    t2 = {r["model"]: r for r in csv.DictReader(open(TAB / "table2_complexity.csv"))}

    AGENT = "MAP-DenoiseGAN (matched agent)"
    JOINT = "Single-agent joint"

    print("\n== 1. Headline PSNR deltas vs joint at sigma=0.20 ==")
    for ds, txt in [("BraTS 2021", "+0.63"), ("nodule3d", "+1.71"), ("CBSD68", "+1.53")]:
        true = float(t6[ds]["delta_vs_joint_s020"])
        check(f"delta vs joint {ds}", abs(true - float(txt)) < 0.005 and txt in tex,
              f"table={true:+.2f}, text contains '{txt}'={txt in tex}")

    print("\n== 2. Deltas vs FIXED-sigma classical at sigma=0.20 ==")
    for ds, txt in [("BraTS 2021", "4.82"), ("nodule3d", "4.86"), ("CBSD68", "1.84")]:
        true = float(t6[ds]["delta_vs_fixed_sigma_classical_s020"])
        check(f"delta vs fixed-sigma classical {ds}", abs(true - float(txt)) < 0.005 and txt in tex,
              f"table={true:+.2f}, text contains '{txt}'={txt in tex}")

    print("\n== 2b. Deltas vs RANGE-trained classical (the fair comparison) ==")
    for ds, txt in [("BraTS 2021", "0.11"), ("nodule3d", "0.49"), ("CBSD68", "0.62")]:
        true = float(t6[ds]["delta_vs_range_classical_s020"])
        check(f"delta vs range classical {ds}", abs(true - float(txt)) < 0.005 and txt in tex,
              f"table={true:+.2f}, text contains '{txt}'={txt in tex}")
    low = tex.lower()
    check("fixed-sigma margin described as an artefact",
          "range-trained" in low and ("artefact" in low or "artifact" in low),
          "manuscript must say the large margin is largely a protocol artefact")

    print("\n== 3. Absolute PSNR values quoted in Results ==")
    RESTORMER_FIXED = "Restormer-lite ($\\sigma{=}0.1$)"
    RESTORMER_RANGE = "Restormer-lite (range-trained)"
    for ds, method, txt in [("BraTS 2021", AGENT, "29.46"),
                             ("BraTS 2021", RESTORMER_FIXED, "24.63"),
                             ("BraTS 2021", RESTORMER_RANGE, "29.35"),
                             ("nodule3d", AGENT, "31.77"),
                             ("nodule3d", RESTORMER_FIXED, "26.91"),
                             ("nodule3d", RESTORMER_RANGE, "31.27"),
                             ("CBSD68", AGENT, "23.94"),
                             ("CBSD68", RESTORMER_RANGE, "23.31")]:
        true = t3[ds][0.20][method]["psnr"]
        check(f"PSNR {ds}/{method}", abs(true - float(txt)) < 0.005 and txt in tex,
              f"table={true:.2f}, text contains '{txt}'={txt in tex}")

    print("\n== 4. Monotonicity claim ==")
    for ds in ["BraTS 2021", "nodule3d", "CBSD68"]:
        d = [float(t6[ds][f"delta_vs_joint_s{s}"]) for s in ("005", "010", "020")]
        check(f"monotone {ds}", d[0] < d[1] < d[2], f"deltas={d}")

    print("\n== 5. Crossover claim must NOT be made ==")
    lowered = tex.lower()
    bad = [p for p in ["universal crossover", "crossover holds", "crossover replicat",
                       "confirms the crossover", "crossover is confirmed"] if p in lowered]
    check("no universal-crossover claim", not bad, f"found={bad}")
    check("crossover explicitly disclaimed",
          "no crossover claim" in lowered and "seed noise" in lowered,
          "text must attribute the apparent crossover to seed noise and disclaim it")

    print("\n== 6. Downstream Dice numbers ==")
    for method, sigma, txt in [(AGENT, 0.20, "0.7908"), (JOINT, 0.20, "0.7847"),
                                ("Oracle (clean image)", 0.20, "0.8243"),
                                ("No denoising", 0.20, "0.4873")]:
        true = t5[method][sigma]
        check(f"Dice {method}@{sigma}", abs(true - float(txt)) < 1e-6 and txt in tex,
              f"table={true}, text contains '{txt}'={txt in tex}")
    fixed_keys = [k for k in t5 if "\\sigma{=}0.1" in k]
    worst = max(t5[k][0.20] for k in fixed_keys)
    check("fixed-sigma Dice range quoted", "0.654" in tex and "0.660" in tex,
          f"best fixed-sigma baseline at 0.20 = {worst} over {len(fixed_keys)} baselines")
    gap = t5[AGENT][0.20] - worst
    check("Dice gap vs fixed-sigma +0.131", abs(gap - 0.131) < 0.001 and "0.131" in tex,
          f"gap={gap:.4f}")
    oracle_gap = t5["Oracle (clean image)"][0.20] - t5[AGENT][0.20]
    check("within 0.034 of oracle", abs(oracle_gap - 0.034) < 0.001 and "0.034" in tex,
          f"oracle gap={oracle_gap:.4f}")

    print("\n== 7. Attribution honesty (joint must be acknowledged) ==")
    check("joint's downstream parity acknowledged",
          "attributable to the adversarial" in lowered or "shared with" in lowered,
          "text attributes downstream gain to architecture, not specialisation alone")

    print("\n== 8. Inheritance speed-up ==")
    check("speedup range 10x to >=30x stated",
          ("10" in tex and "30" in tex and "matched quality" in lowered), "")
    for ds in ["BraTS 2021", "nodule3d", "CBSD68"]:
        row = t4b[ds]
        check(f"speedup data present {ds}", row["speedup_x"] != "",
              f"epoch1={row['inherited_epoch1_val_psnr']}, "
              f"scratch_epochs={row['scratch_epochs_to_match'] or 'not reached'}")
    check("BraTS epoch-1 PSNR 31.96 quoted",
          "31.96" in tex and abs(float(t4b["BraTS 2021"]["inherited_epoch1_val_psnr"]) - 31.96) < 0.005)
    check("BraTS scratch best 31.47 quoted",
          "31.47" in tex and abs(float(t4b["BraTS 2021"]["scratch_best_val_psnr_in_30"]) - 31.47) < 0.005)

    print("\n== 9. Complexity numbers ==")
    gen = t2["MAP-DenoiseGAN generator (U-Net)"]
    check("generator params 5,142,433", "5{,}142{,}433" in tex or "5.14" in tex,
          f"table={gen['parameters']}")
    check("generator latency 0.674 ms", "0.674" in tex, f"table={gen['inference_ms_per_image']}")

    print("\n== 10. Figures and tables all referenced ==")
    for i, lbl in enumerate(["fig:arch", "fig:curves", "fig:psnr", "fig:win", "fig:dice", "fig:qual"], 1):
        check(f"Fig {lbl} referenced", f"\\ref{{{lbl}}}" in tex)
    for lbl in ["tab:datasets", "tab:complexity", "tab:main", "tab:efficiency",
                "tab:speedup", "tab:dice", "tab:replication"]:
        check(f"Table {lbl} referenced", f"\\ref{{{lbl}}}" in tex)
    for f in ["fig1_architecture", "fig2_training_curves", "fig3_psnr_vs_noise",
              "fig4_high_noise_win", "fig5_downstream_dice", "fig6_qualitative"]:
        check(f"{f}.pdf exists", (FIG / f"{f}.pdf").exists())
        check(f"{f}.pdf included", f"{f}.pdf" in tex)
    for t in ["table1_datasets", "table2_complexity", "table3_main_psnr",
              "table4_training_efficiency", "table4b_inheritance_speedup",
              "table5_downstream_dice", "table6_replication"]:
        check(f"{t}.tex input", f"\\input{{{t}.tex}}" in tex)

    print("\n== 10b. Seed study ==")
    sp = TAB / "table7_seed_stats.json"
    check("seed stats file exists", sp.exists())
    if sp.exists():
        seed = json.load(open(sp))
        for ds, gap, sd in [("BraTS 2021", "0.81", "0.17"), ("nodule3d", "1.54", "0.12"),
                            ("CBSD68", "1.27", "0.44")]:
            e = seed[ds]["0.2"]
            ok = (abs(e["gap_mean"] - float(gap)) < 0.005
                  and abs(e["gap_std"] - float(sd)) < 0.005
                  and gap in tex and e["n_positive"] == e["n_seeds"])
            check(f"seed gap {ds}", ok,
                  f"{e['gap_mean']:+.3f}+-{e['gap_std']:.3f}, "
                  f"{e['n_positive']}/{e['n_seeds']} positive, p={e.get('paired_t_p'):.4f}")
        lo = seed["BraTS 2021"]["0.05"]
        check("low-noise null result stated",
              lo.get("paired_t_p", 0) > 0.05 and "0.93" in tex,
              f"BraTS sigma=0.05 p={lo.get('paired_t_p'):.3f} (not significant)")
        check("no crossover claim retained",
              "no crossover claim" in tex.lower())

    print("\n== 10c. Downstream range baseline ==")
    rp = TAB / "table5_downstream_range.csv"
    check("downstream range row exists", rp.exists())
    if rp.exists():
        rr = {float(r["sigma"]): float(r["mean_dice"]) for r in csv.DictReader(open(rp))}
        check("range Dice 0.7864 quoted", abs(rr[0.20] - 0.7864) < 1e-6 and "0.7864" in tex,
              f"range-trained Dice at sigma=0.20 = {rr[0.20]}")
        check("clinical collapse reading retracted",
              "retract" in tex.lower() or "artefact of the training protocol" in tex.lower(),
              "manuscript must withdraw the 'classical collapse' reading")

    print("\n== 11. Limitations completeness ==")
    for topic in ["synthetic", "routing", "over-smooth", "seed", "177"]:
        check(f"limitation '{topic}' present", topic in lowered)

    print("\n== 12. Baselines cited ==")
    for key in ["zhang2017dncnn", "zamir2022restormer", "liang2021swinir"]:
        check(f"cite {key}", f"\\cite{{{key}}}" in tex or f"{{{key}}}" in tex)
    cited = set(re.findall(r"\\cite\{([^}]*)\}", tex))
    cited = {c.strip() for grp in cited for c in grp.split(",")}
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
    check("no undefined citations", cited <= defined, f"undefined={sorted(cited - defined)}")
    unused = defined - cited
    if unused:
        warn("unused bibliography entries", f"{sorted(unused)}")

    print("\n" + "=" * 62)
    print(f"PASSED {len(passes)}   FAILED {len(failures)}   WARNINGS {len(warnings)}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  - " + f)
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
