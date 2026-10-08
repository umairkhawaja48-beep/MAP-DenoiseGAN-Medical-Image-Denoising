"""
Verifies the strengthening additions to both manuscripts against source data,
and checks that no unverifiable citation was introduced.

Usage: python check_additions.py <bspc.tex> <quantum.tex>
"""
import csv
import json
import os

from paths import DATA_ROOT, EXTERNAL_DIR, RESULTS_ROOT  # noqa: E402
import re
import sys
from pathlib import Path

MAP_TAB = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
MUON = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/muon_study"))
ok, bad = [], []


def check(name, cond, detail=""):
    (ok if cond else bad).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def muon_dpsnr(arm, agent, sigma):
    p = MUON / arm / agent / "test_metrics.json"
    return json.load(open(p))["fixed_sigma"][str(sigma)]["delta_psnr"]


def main():
    # LaTeX wraps prose across lines, so a phrase probe must be whitespace-
    # insensitive or it reports false failures on wrapped text.
    def flat(p):
        return re.sub(r"\s+", " ", Path(p).read_text(encoding="utf-8"))

    b_raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    q_raw = Path(sys.argv[2]).read_text(encoding="utf-8")
    b, q = flat(sys.argv[1]), flat(sys.argv[2])

    print("\n== BSPC: Muon A/B table matches measured values ==")
    pairs = [("adam", "agent1_low", 0.05, "-1.03"), ("muon_lr0.005", "agent1_low", 0.05, "+4.05"),
             ("adam", "joint", 0.05, "-5.91"),       ("muon_lr0.005", "joint", 0.05, "+2.02"),
             ("muon_lr0.005", "agent3_high", 0.20, "+10.06"),
             ("adam", "agent2_medium", 0.10, "+5.61")]
    for arm, ag, s, txt in pairs:
        v = muon_dpsnr(arm, ag, s)
        check(f"{arm}/{ag}@{s} = {txt}", abs(v - float(txt)) < 0.005 and txt.lstrip("+") in b,
              f"measured={v:+.2f}")
    check("Muon criterion: all Muon models positive at all sigmas",
          all(muon_dpsnr("muon_lr0.005", a, s) > 0
              for a in ["agent1_low", "agent2_medium", "agent3_high", "joint"]
              for s in [0.05, 0.10, 0.20]))
    check("Adam joint genuinely fails at 0.05", muon_dpsnr("adam", "joint", 0.05) < 0)
    check("LR sweep disclosed", "four-point sweep" in b or "four-point" in b)
    check("Muon/Adam non-comparability disclosed",
          "not directly comparable" in b or "should not be compared directly" in b)

    print("\n== BSPC: 4th dataset theorem terms ==")
    rows = list(csv.DictReader(open(MAP_TAB / "table8b_brain_tumor_muon.csv")))
    sg = [float(r["specialization_gain_mse"]) for r in rows]
    rr = [float(r["routing_regret_hard_mse"]) for r in rows]
    frac = [float(r["oracle_picks_matching_agent_frac"]) for r in rows]
    check("SG monotone on 4th dataset", sg[0] < sg[1] < sg[2], f"{[f'{x:.2e}' for x in sg]}")
    check("routing regret -> 0 on 4th dataset", rr[-1] == 0.0, f"{[f'{x:.2e}' for x in rr]}")
    check("matched-optimal rises to 100%", frac[0] < frac[-1] and frac[-1] == 1.0,
          f"{[f'{100*x:.1f}%' for x in frac]}")
    check("47.4% and 100% quoted", "47.4" in b and "100\\%" in b)
    check("dB non-monotonicity disclosed", "+2.03" in b and "+1.81" in b and "+1.86" in b)
    check("four of four claimed", "four of four" in b.lower())

    print("\n== BSPC: Proposition 2 honesty ==")
    g = json.load(open(MAP_TAB / "table9_prop2_check.json"))
    check("active params identical (fact)", g["active_params_identical"])
    check("5,142,433 quoted for both", "5{,}142{,}433" in b)
    check("routing entropy collapse quoted", "0.04" in b and "bits" in b)
    check("excess-gap prediction NOT claimed as confirmed",
          "not able to confirm" in b.lower() or "cannot be separated" in b.lower(),
          f"excess_gap<=0 on all at 0.20 = {g['excess_gap_nonpositive_at_sigma020_all_datasets']}")
    check("underfitting regime disclosed", "underfitting" in b.lower())
    check("weak correlation reported", "0.18" in b)
    check("only one of three disclosed", "one of three" in b.lower())

    print("\n== BSPC: MoE-for-restoration related work ==")
    for key in ["supmoe2026", "moediffsr2025", "mimdit2026", "uniuir2025", "mode2025"]:
        check(f"cites {key}", f"\\cite{{{key}}}" in b or f"{key}," in b or f"{{{key}}}" in b)
    check("no-theory claim present",
          "None of these MoE restoration methods provides a theoretical condition" in b)
    check("DP-MoE NOT cited (unverifiable)", "DP-MoE" not in b and "dpmoe" not in b.lower())

    print("\n== BSPC: decisive routing remark ==")
    check("routing-dynamics citation present",
          "moespecialize2025" in b or "moecluster2025" in b)
    check("states we have no learned router",
          "without a learned router" in b.lower())

    print("\n== QUANTUM: comparison table and corrections ==")
    check("comparison table present", "tab:compare" in q)
    check("Layered-QAS cited with real authors", "Kuete Meli" in q and "lqas2026" in q)
    check("growth paper attributed to Duffy et al.", "Duffy" in q and "duffy2024growth" in q)
    check("'Anonymous' placeholder removed", "Anonymous" not in q)
    check("miscitation correction disclosed",
          "described~\\cite{duffy2024growth} as a qubit-count-growth method" in q
          or "as a qubit-count-growth method" in q)
    check("depth-vs-qubit distinction stated", "It is not: it" in q or "grows circuit depth" in q)

    print("\n== QUANTUM: noise-resistant QAS paragraph ==")
    for key in ["naqas2026", "tfqas2026", "dqas2020", "afane2026"]:
        check(f"cites {key}", f"\\cite{{{key}}}" in q or f"{{{key}}}" in q)
    check("QNRS gap claim present", "fills that gap" in q)
    check("GEM-QAS NOT cited (unverifiable)", "GEM-QAS" not in q)

    print("\n== QUANTUM: morphism precursor + identity-init ==")
    check("network morphism cited in intro", "wei2016morphism" in q)
    check("quantum analogue phrasing", "quantum analogue of classical" in q.lower())
    check("Net2Net cited", "chen2016net2net" in q)
    check("identity-init subsection present", "identity initialisation" in q.lower())
    check("first machine-precision proof claim", "first machine-precision exactness proof" in q)

    print("\n== Citations resolve in both ==")
    for name, tex in [("BSPC", b_raw), ("TQE", q_raw)]:
        cited = {c.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", tex)
                 for c in grp.split(",")}
        defined = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
        check(f"{name}: no undefined citations", cited <= defined,
              f"undefined={sorted(cited - defined)}")
        unused = defined - cited
        check(f"{name}: no unused bib entries", not unused, f"unused={sorted(unused)}")

    print("\n" + "=" * 62)
    print(f"PASSED {len(ok)}   FAILED {len(bad)}")
    if bad:
        print("\nFAILURES:")
        for f in bad:
            print("  - " + f)
    print("=" * 62)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
